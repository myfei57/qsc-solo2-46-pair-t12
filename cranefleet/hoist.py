"""Hoist drum, hook lowering and the speed-feedback brake interlock."""

from __future__ import annotations

import json
import math
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from .audit import Audit
from .errors import CalibrationError, FeedbackNotConfirmed, HoistBusy
from .errors import StoreNotFound
from .store import Store

if TYPE_CHECKING:
    from .slew import SpeedFeedback

CABLE_DIAMETER = 0.02
DEFAULT_DRUM_WIDTH = 1.2

# Compensation defaults. Site feedback is that the displayed height drifts
# from the true hook height once the drum spools multiple layers, so every
# parameter the crew cannot measure reliably stays on the conservative side:
# a full cable diameter of radial growth per layer (nested spooling would
# grow less) and no rope-stretch allowance, so the display never under-reads
# how much cable a revolution actually pays out.
DEFAULT_LAYER_STEP = 1.0
DEFAULT_STRETCH = 0.0
DEFAULT_FALLS = 1

# A calibration correction outside this band means the tape reading or the
# drum parameters are wrong; the measurement is rejected instead of baking a
# blunder into every displayed height.
MIN_CORRECTION = 0.8
MAX_CORRECTION = 1.25


class FeedbackReader(Protocol):
    """The part of the speed feedback the brake interlock depends on."""

    def is_confirmed(self) -> bool:
        ...


class Hoist:
    """Mutable state of the hoisting gear."""

    def __init__(self, hoist_id: str) -> None:
        self._lock = threading.Lock()
        self.id = hoist_id
        self.lowered = False
        self.brake_engaged = True
        self.moving = False

    def hook_lowered(self) -> bool:
        with self._lock:
            return self.lowered

    def is_moving(self) -> bool:
        with self._lock:
            return self.moving

    def brake_on(self) -> bool:
        with self._lock:
            return self.brake_engaged


@dataclass(frozen=True)
class DrumCalibration:
    """One tape-measured calibration point kept as the drum's record."""

    sequence: int
    cable_out: float
    measured_height: float
    estimated_height: float
    correction: float

    def to_dict(self) -> dict[str, object]:
        return {
            "sequence": self.sequence,
            "cable_out": self.cable_out,
            "measured_height": self.measured_height,
            "estimated_height": self.estimated_height,
            "correction": self.correction,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "DrumCalibration":
        return cls(
            sequence=int(payload["sequence"]),
            cable_out=float(payload["cable_out"]),
            measured_height=float(payload["measured_height"]),
            estimated_height=float(payload["estimated_height"]),
            correction=float(payload["correction"]),
        )


class Drum:
    """Multi-layer winding geometry used to convert cable into hook height.

    The drum spools ``wraps_per_layer`` wraps of cable onto each layer; every
    new layer rides on the previous one, so each layer pays out a different
    length per revolution. Height is integrated layer by layer instead of
    converting the whole paid-out length at the current layer's diameter,
    which is what made the display drift once the drum spooled past the
    first layer.
    """

    def __init__(
        self,
        diameter: float,
        cable_diameter: float = CABLE_DIAMETER,
        width: float = DEFAULT_DRUM_WIDTH,
        falls: int = DEFAULT_FALLS,
        layer_step: float = DEFAULT_LAYER_STEP,
        stretch: float = DEFAULT_STRETCH,
    ) -> None:
        self._lock = threading.Lock()
        self.diameter = diameter
        self.cable_diameter = cable_diameter
        self.width = width
        self.falls = max(1, int(falls))
        self.layer_step = layer_step
        self.stretch = stretch
        self.layers = 1
        self.wound = 0.0
        self.revolutions = 0.0
        self.correction = 1.0
        self.calibrations = 0
        self.last_calibration: DrumCalibration | None = None

    def wraps_per_layer(self) -> int:
        return max(1, int(self.width / self.cable_diameter))

    def layer_diameter(self, layer: int) -> float:
        """Effective winding diameter of a 1-based layer."""

        return self.diameter + (2 * layer - 1) * self.cable_diameter * self.layer_step

    def layer_capacity(self, layer: int) -> float:
        """Cable length that fits on a 1-based layer."""

        return self.wraps_per_layer() * math.pi * self.layer_diameter(layer)

    def _winding(self, length: float) -> tuple[int, float]:
        """Return (layer, revolutions) for a paid-out cable ``length``."""

        remaining = max(0.0, length)
        layer = 1
        revolutions = 0.0
        wraps = self.wraps_per_layer()
        while True:
            capacity = self.layer_capacity(layer)
            if remaining < capacity:
                revolutions += remaining / (math.pi * self.layer_diameter(layer))
                return layer, revolutions
            remaining -= capacity
            revolutions += wraps
            layer += 1

    def _cable_from_revolutions(self, revolutions: float) -> float:
        """Invert the winding model: cable length for ``revolutions`` turns."""

        remaining = max(0.0, revolutions)
        wraps = self.wraps_per_layer()
        layer = 1
        length = 0.0
        while remaining > 0.0:
            taken = min(remaining, float(wraps))
            length += taken * math.pi * self.layer_diameter(layer)
            remaining -= taken
            layer += 1
        return length

    def _hook_travel(self, cable_out: float) -> float:
        return cable_out / self.falls * (1.0 + self.stretch) * self.correction

    def pay_out(self, length: float) -> None:
        with self._lock:
            self.wound = max(0.0, self.wound + length)
            self.layers, self.revolutions = self._winding(self.wound)

    def height_from_cable(self, cable_out: float) -> float:
        """Hook travel for a paid-out cable length, reeving included."""

        with self._lock:
            return self._hook_travel(cable_out)

    def height_from_revolutions(self, revolutions: float) -> float:
        """Height read off the drum encoder, integrated layer by layer."""

        with self._lock:
            return self._hook_travel(self._cable_from_revolutions(revolutions))

    def layer_count(self) -> int:
        with self._lock:
            return self.layers

    def wound_total(self) -> float:
        with self._lock:
            return self.wound

    def revolutions_now(self) -> float:
        with self._lock:
            return self.revolutions

    def correction_now(self) -> float:
        with self._lock:
            return self.correction

    def calibration_count(self) -> int:
        with self._lock:
            return self.calibrations

    def last_calibration_now(self) -> DrumCalibration | None:
        with self._lock:
            return self.last_calibration


def _hoist_key(hoist_id: str) -> str:
    return "hoist/" + hoist_id


def _drum_key(hoist_id: str) -> str:
    return "hoist/" + hoist_id + "/drum"


def calibrate_drum(drum: Drum, measured_height: float) -> DrumCalibration:
    """Apply a tape-measured hook height as the drum's correction.

    Recalibration means measuring again: the crew reads the true hook height
    and the ratio to the drum's own estimate becomes the correction factor.
    A ratio outside the plausible band is refused so a misread tape cannot
    silently bend every height the console shows.
    """

    if measured_height <= 0.0:
        raise CalibrationError("measured height must be positive")
    with drum._lock:  # noqa: SLF001 - module-level functions share drum state
        if drum.wound <= 0.0:
            raise CalibrationError("no cable paid out to calibrate against")
        estimate = drum.wound / drum.falls * (1.0 + drum.stretch)
        correction = measured_height / estimate
        if not MIN_CORRECTION <= correction <= MAX_CORRECTION:
            raise CalibrationError(
                f"correction {correction:.3f} outside plausible band; measure again"
            )
        drum.correction = correction
        drum.calibrations += 1
        record = DrumCalibration(
            sequence=drum.calibrations,
            cable_out=drum.wound,
            measured_height=measured_height,
            estimated_height=estimate,
            correction=correction,
        )
        drum.last_calibration = record
        return record


def persist_drum(drum: Drum, hoist_id: str, store: Store) -> None:
    with drum._lock:  # noqa: SLF001
        payload = json.dumps(
            {
                "wound": drum.wound,
                "correction": drum.correction,
                "calibrations": drum.calibrations,
                "last_calibration": (
                    drum.last_calibration.to_dict() if drum.last_calibration else None
                ),
            }
        ).encode("utf-8")
    store.save(_drum_key(hoist_id), payload)


def restore_drum(drum: Drum, hoist_id: str, store: Store) -> None:
    """Reload the drum snapshot so a restart keeps the calibrated height."""

    try:
        payload = store.load(_drum_key(hoist_id))
    except StoreNotFound:
        return
    snapshot = json.loads(payload.decode("utf-8"))
    with drum._lock:  # noqa: SLF001
        drum.wound = max(0.0, float(snapshot.get("wound", 0.0)))
        drum.layers, drum.revolutions = drum._winding(drum.wound)
        drum.correction = float(snapshot.get("correction", 1.0))
        drum.calibrations = int(snapshot.get("calibrations", 0))
        last = snapshot.get("last_calibration")
        drum.last_calibration = DrumCalibration.from_dict(last) if last else None


def lower(hoist: Hoist) -> None:
    """Lower the hook; refused while the drum is already turning."""

    with hoist._lock:  # noqa: SLF001 - module-level functions share hoist state
        if hoist.moving:
            raise HoistBusy("drum is busy")
        hoist.lowered = True
        hoist.moving = True


def persist_hoist(hoist: Hoist, store: Store) -> None:
    with hoist._lock:  # noqa: SLF001
        payload = json.dumps(
            {
                "lowered": hoist.lowered,
                "moving": hoist.moving,
                "brake_engaged": hoist.brake_engaged,
            }
        ).encode("utf-8")
    store.save(_hoist_key(hoist.id), payload)


def resume_lower(hoist: Hoist, store: Store) -> None:
    """Reload the hoist snapshot so a resumed cycle keeps its hook state."""

    try:
        payload = store.load(_hoist_key(hoist.id))
    except StoreNotFound:
        with hoist._lock:  # noqa: SLF001
            hoist.lowered = False
        return
    snapshot = json.loads(payload.decode("utf-8"))
    with hoist._lock:  # noqa: SLF001
        hoist.lowered = snapshot["lowered"]
        hoist.moving = snapshot["moving"]
        hoist.brake_engaged = snapshot["brake_engaged"]


def release_brake(hoist: Hoist, feedback: FeedbackReader) -> None:
    """Release the brake only once the speed feedback is confirmed."""

    with hoist._lock:  # noqa: SLF001
        if not feedback.is_confirmed():
            raise FeedbackNotConfirmed("speed feedback not confirmed")
        hoist.brake_engaged = False
        hoist.moving = True


def engage_brake(hoist: Hoist) -> None:
    with hoist._lock:  # noqa: SLF001
        hoist.brake_engaged = True
        hoist.moving = False


def resume_release(hoist: Hoist, feedback: "SpeedFeedback", store: Store) -> None:
    """Restore the feedback snapshot and then release the brake."""

    from .slew import restore_feedback

    restore_feedback(feedback, store)
    release_brake(hoist, feedback)


def stop(hoist: Hoist, recorder: Audit | None = None) -> None:
    with hoist._lock:  # noqa: SLF001
        hoist.moving = False
    if recorder is not None:
        recorder.record(hoist.id, "hoist-stop", "hoist stopped")


def raise_hook(hoist: Hoist, recorder: Audit | None = None) -> None:
    with hoist._lock:  # noqa: SLF001
        if hoist.moving:
            raise HoistBusy("drum is busy")
        hoist.moving = True
        hoist.brake_engaged = False
    if recorder is not None:
        recorder.record(hoist.id, "hoist-raise", "raising")


def limit_stop(hoist: Hoist) -> None:
    with hoist._lock:  # noqa: SLF001
        hoist.moving = False
