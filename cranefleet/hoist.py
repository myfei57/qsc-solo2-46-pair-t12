"""Hoist drum, hook lowering and the speed-feedback brake interlock."""

from __future__ import annotations

import json
import math
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Protocol

from .audit import Audit
from .errors import (
    DrumCalibrationError,
    FeedbackNotConfirmed,
    HoistBusy,
    StoreError,
    StoreNotFound,
)
from .store import Store

if TYPE_CHECKING:
    from .slew import SpeedFeedback

DEFAULT_DRUM_DIAMETER = 0.8
DEFAULT_DRUM_WIDTH = 1.2
CABLE_DIAMETER = 0.02
DEFAULT_ROPE_PARTS = 1
# Ideal close-packed radial pitch is sqrt(3)/2 of the rope diameter.  Before a
# site supplies a measured packing pitch, bias the effective radius low: the
# turn counter then converts to slightly less paid rope, keeping the displayed
# hook height on the lower, safe side.
DEFAULT_LAYER_COMPENSATION = 0.90
MIN_LAYER_COMPENSATION = 0.80
MAX_LAYER_COMPENSATION = 1.00
MAX_CALIBRATION_RECORDS = 20


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


class Drum:
    """Winding geometry used to convert paid-out cable into hook height.

    A real drum packs successive rope layers radially.  Each outer turn pays
    out slightly more cable than the turn below it, so the model integrates
    turn length layer by layer rather than using one fixed circumference.
    """

    def __init__(
        self,
        diameter: float = DEFAULT_DRUM_DIAMETER,
        width: float = DEFAULT_DRUM_WIDTH,
        cable_diameter: float = CABLE_DIAMETER,
        layer_compensation: float = DEFAULT_LAYER_COMPENSATION,
        rope_parts: int = DEFAULT_ROPE_PARTS,
    ) -> None:
        self._lock = threading.RLock()
        self.diameter = _positive(diameter, "drum diameter")
        self.width = _positive(width, "drum width")
        self.cable_diameter = _positive(cable_diameter, "cable diameter")
        self.rope_parts = _positive_int(rope_parts, "rope parts")
        self.layer_compensation = _compensation(layer_compensation)
        self.cable_out = 0.0
        self.total_turns = 0.0
        self.calibrations: list[DrumCalibration] = []

    def pay_out(self, length: float) -> None:
        """Pay out a measured length of cable (lowering)."""

        length = _non_negative(length, "cable length")
        with self._lock:
            self._apply_length(length)
            self.total_turns = self._turns_for_length(self.cable_out)

    def wind_in(self, length: float) -> None:
        """Wind a measured length of cable back onto the drum (raising)."""

        length = _non_negative(length, "cable length")
        with self._lock:
            self._apply_length(-length)
            self.total_turns = self._turns_for_length(self.cable_out)

    def report_turns(self, turns: float) -> None:
        """Update the paid cable reading from a cumulative drum-turn counter."""

        turns = _non_negative(turns, "drum turns")
        with self._lock:
            self.total_turns = turns
            self.cable_out = self._length_for_turns(turns)

    def turns_from_cable(self, cable_out: float) -> float:
        """Drum turns required to pay out a cable length through every layer."""

        with self._lock:
            return self._turns_for_length(_non_negative(cable_out, "cable length"))

    def height_from_cable(self, cable_out: float) -> float:
        """Vertical hook travel represented by a paid-cable reading."""

        with self._lock:
            return _non_negative(cable_out, "cable length") / self.rope_parts

    def height_from_turns(self, turns: float) -> float:
        """Vertical hook travel represented by a cumulative turn-counter reading."""

        with self._lock:
            return self._length_for_turns(_non_negative(turns, "drum turns")) / self.rope_parts

    def cable_length(self) -> float:
        with self._lock:
            return self.cable_out

    def turns_now(self) -> float:
        with self._lock:
            return self.total_turns

    def layer_count(self) -> int:
        with self._lock:
            return self._active_layer_index() + 1

    def layer_index(self) -> int:
        with self._lock:
            return self._active_layer_index()

    def wound_total(self) -> float:
        with self._lock:
            return self.cable_out

    def turns_per_layer(self) -> float:
        return self.width / self.cable_diameter

    def layer_diameter(self, layer: int) -> float:
        """Mean winding diameter of the zero-indexed rope ``layer``."""

        if layer < 0:
            raise DrumCalibrationError("layer must be non-negative")
        return self.diameter + (1.0 + 2.0 * layer * self.layer_compensation) * self.cable_diameter

    def turn_length(self, layer: int) -> float:
        """Length of one helical turn in ``layer``."""

        diameter = self.layer_diameter(layer)
        return math.hypot(math.pi * diameter, self.cable_diameter)

    def layer_length(self, layer: int) -> float:
        return self.turns_per_layer() * self.turn_length(layer)

    def hook_height(self) -> float | None:
        """Absolute hook height using the latest field calibration, if any."""

        with self._lock:
            if not self.calibrations:
                return None
            calibration = self.calibrations[-1]
            travel = (self.cable_out - calibration.cable_out) / calibration.rope_parts
            return calibration.reference_height - travel

    def calibrate(
        self,
        measured_height: float,
        cable_out: float | None = None,
        rope_parts: int | None = None,
        layer_compensation: float | None = None,
        operator: str = "",
        note: str = "",
        recorded_at: str | None = None,
    ) -> "DrumCalibration":
        """Record a fresh measured-height reference and optional geometry."""

        height = _number(measured_height, "measured height")
        with self._lock:
            if cable_out is None:
                cable_at_record = self.cable_out
            else:
                cable_at_record = _non_negative(cable_out, "cable length")
            parts = self.rope_parts if rope_parts is None else _positive_int(rope_parts, "rope parts")
            compensation = (
                self.layer_compensation
                if layer_compensation is None
                else _compensation(layer_compensation)
            )
            calibration = DrumCalibration(
                recorded_at=recorded_at or datetime.now(timezone.utc).isoformat(),
                reference_height=height,
                cable_out=cable_at_record,
                rope_parts=parts,
                layer_compensation=compensation,
                operator=str(operator or ""),
                note=str(note or ""),
            )
            self.rope_parts = parts
            self.layer_compensation = compensation
            self.calibrations.append(calibration)
            if len(self.calibrations) > MAX_CALIBRATION_RECORDS:
                del self.calibrations[: len(self.calibrations) - MAX_CALIBRATION_RECORDS]
            return calibration

    def latest_calibration(self) -> "DrumCalibration | None":
        with self._lock:
            return self.calibrations[-1] if self.calibrations else None

    def calibration_records(self) -> list["DrumCalibration"]:
        with self._lock:
            return list(self.calibrations)

    def to_dict(self) -> dict[str, object]:
        with self._lock:
            return {
                "diameter": self.diameter,
                "width": self.width,
                "cable_diameter": self.cable_diameter,
                "layer_compensation": self.layer_compensation,
                "rope_parts": self.rope_parts,
                "cable_out": self.cable_out,
                "total_turns": self.total_turns,
                "calibrations": [record.to_dict() for record in self.calibrations],
            }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "Drum":
        try:
            drum = cls(
                diameter=float(payload["diameter"]),
                width=float(payload["width"]),
                cable_diameter=float(payload["cable_diameter"]),
                layer_compensation=float(
                    payload.get("layer_compensation", DEFAULT_LAYER_COMPENSATION)
                ),
                rope_parts=int(payload.get("rope_parts", DEFAULT_ROPE_PARTS)),
            )
            raw_records = payload.get("calibrations", [])
            if not isinstance(raw_records, list):
                raise ValueError("calibrations must be a list")
            records = [DrumCalibration.from_dict(item) for item in raw_records]
        except (KeyError, TypeError, ValueError, DrumCalibrationError) as exc:
            raise DrumCalibrationError("invalid drum snapshot") from exc
        drum.cable_out = _non_negative(float(payload.get("cable_out", 0.0)), "cable length")
        raw_turns = payload.get("total_turns")
        drum.total_turns = (
            drum._turns_for_length(drum.cable_out)  # noqa: SLF001
            if raw_turns is None
            else _non_negative(float(raw_turns), "drum turns")
        )
        drum.calibrations = records[-MAX_CALIBRATION_RECORDS:]
        return drum

    def _apply_length(self, signed_length: float) -> None:
        self.cable_out = max(0.0, self.cable_out + signed_length)

    def _active_layer_index(self) -> int:
        per_layer = self.turns_per_layer()
        if self.total_turns <= 0.0:
            return 0
        return min(int(self.total_turns // per_layer), 10**6)

    def _turns_for_length(self, cable_length: float) -> float:
        remaining = cable_length
        layer = 0
        per_layer = self.turns_per_layer()
        while True:
            layer_capacity = per_layer * self.turn_length(layer)
            if remaining < layer_capacity:
                return layer * per_layer + remaining / self.turn_length(layer)
            remaining -= layer_capacity
            layer += 1

    def _length_for_turns(self, turns: float) -> float:
        per_layer = self.turns_per_layer()
        completed_layers = int(turns // per_layer)
        partial_turns = turns - completed_layers * per_layer
        total = 0.0
        for layer in range(completed_layers):
            total += self.layer_length(layer)
        return total + partial_turns * self.turn_length(completed_layers)


@dataclass(frozen=True)
class DrumCalibration:
    """A measured-height reference recorded during field recalibration."""

    recorded_at: str
    reference_height: float
    cable_out: float
    rope_parts: int = DEFAULT_ROPE_PARTS
    layer_compensation: float = DEFAULT_LAYER_COMPENSATION
    operator: str = ""
    note: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "recorded_at": self.recorded_at,
            "reference_height": self.reference_height,
            "cable_out": self.cable_out,
            "rope_parts": self.rope_parts,
            "layer_compensation": self.layer_compensation,
            "operator": self.operator,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "DrumCalibration":
        try:
            return cls(
                recorded_at=str(payload["recorded_at"]),
                reference_height=float(payload["reference_height"]),
                cable_out=float(payload["cable_out"]),
                rope_parts=int(payload.get("rope_parts", DEFAULT_ROPE_PARTS)),
                layer_compensation=float(
                    payload.get("layer_compensation", DEFAULT_LAYER_COMPENSATION)
                ),
                operator=str(payload.get("operator", "")),
                note=str(payload.get("note", "")),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise DrumCalibrationError("invalid drum calibration") from exc


def _number(value: object, name: str) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise DrumCalibrationError(f"{name} must be a number") from exc
    if not math.isfinite(number):
        raise DrumCalibrationError(f"{name} must be finite")
    return number


def _positive(value: object, name: str) -> float:
    number = _number(value, name)
    if number <= 0.0:
        raise DrumCalibrationError(f"{name} must be positive")
    return number


def _non_negative(value: object, name: str) -> float:
    number = _number(value, name)
    if number < 0.0:
        raise DrumCalibrationError(f"{name} must be non-negative")
    return number


def _positive_int(value: object, name: str) -> int:
    number = _positive(value, name)
    integer = int(number)
    if integer != number:
        raise DrumCalibrationError(f"{name} must be a whole number")
    return integer


def _compensation(value: object) -> float:
    number = _number(value, "layer compensation")
    if not MIN_LAYER_COMPENSATION <= number <= MAX_LAYER_COMPENSATION:
        raise DrumCalibrationError(
            f"layer compensation must be between {MIN_LAYER_COMPENSATION} and {MAX_LAYER_COMPENSATION}"
        )
    return number


DRUM_KEY = "hoist/drum"


def persist_drum(drum: Drum, store: Store) -> None:
    store.save(DRUM_KEY, json.dumps(drum.to_dict(), ensure_ascii=False).encode("utf-8"))


def restore_drum(drum: Drum, store: Store) -> None:
    try:
        payload = store.load(DRUM_KEY)
    except StoreNotFound:
        return
    try:
        restored = Drum.from_dict(json.loads(payload.decode("utf-8")))
    except (json.JSONDecodeError, DrumCalibrationError, UnicodeDecodeError):
        return
    drum.__dict__.update(restored.__dict__)


def _hoist_key(hoist_id: str) -> str:
    return "hoist/" + hoist_id


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
