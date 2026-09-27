"""Unit tests for slewing, trolley travel and hoisting."""

from __future__ import annotations

import unittest

from cranefleet.audit import Audit
from cranefleet.errors import (
    CalibrationError,
    FeedbackNotConfirmed,
    HoistBusy,
    HookNotLowered,
    SlewBlocked,
    SlewBusy,
    SlewIdle,
    StoreError,
)
from cranefleet.geometry import Point
from cranefleet.hoist import (
    Drum,
    Hoist,
    calibrate_drum,
    engage_brake,
    limit_stop as hoist_limit_stop,
    lower,
    persist_drum,
    persist_hoist,
    raise_hook,
    release_brake,
    restore_drum,
    resume_release,
    stop as hoist_stop,
)
from cranefleet.slew import (
    Ramp,
    Slew,
    SpeedFeedback,
    is_axis_stopped,
    limit_stop,
    persist_feedback,
    ramp_reached,
    release_latch_hold,
    release_wind_hold,
    restore_feedback,
    start,
    stop,
    stop_for_wind,
    stop_reason,
    stop_sequence,
    swing,
)
from cranefleet.store import MemoryStore
from cranefleet.trolley import Speedometer, Trolley
from cranefleet.trolley import limit_stop as trolley_limit_stop
from cranefleet.trolley import move, move_resume, stop as trolley_stop
from cranefleet.zone import ZoneState

BOUNDARY = [Point(0.0, 0.0), Point(100.0, 0.0), Point(100.0, 100.0), Point(0.0, 100.0)]


class RampTests(unittest.TestCase):
    def test_ramp_walks_to_target(self) -> None:
        ramp = Ramp()
        ramp.set_target(5.0)
        for _ in range(25):
            ramp.step(0.1)
        self.assertTrue(ramp_reached(ramp))
        self.assertEqual(ramp.target_now(), 5.0)
        ramp.set_target(0.0)
        ramp.step(0.1)
        self.assertAlmostEqual(ramp.velocity_now(), 4.8)


class SlewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.slew = Slew("crane-1")
        self.zone = ZoneState("zone-1", "zone", BOUNDARY)
        self.store = MemoryStore()
        self.audit = Audit(50)

    def test_start_persists_the_zone_first(self) -> None:
        start(self.slew, self.zone, self.store, self.audit)
        self.assertTrue(self.slew.is_moving())
        self.assertTrue(self.zone.is_durable())
        with self.assertRaises(SlewBusy):
            start(self.slew, self.zone, self.store, self.audit)

    def test_blocked_when_zone_cannot_persist(self) -> None:
        class BrokenStore(MemoryStore):
            def save(self, key: str, value: bytes) -> None:
                raise StoreError("disk full")

        with self.assertRaises(StoreError):
            start(self.slew, self.zone, BrokenStore(), self.audit)
        self.assertFalse(self.slew.is_moving())
        self.assertFalse(self.zone.is_durable())

    def test_swing_requires_moving(self) -> None:
        with self.assertRaises(SlewIdle):
            swing(self.slew, 5.0, self.audit)
        start(self.slew, self.zone, self.store, self.audit)
        swing(self.slew, 5.0, self.audit)
        self.assertAlmostEqual(self.slew.angle_now(), 5.0)

    def test_stop_clears_motion(self) -> None:
        start(self.slew, self.zone, self.store, self.audit)
        stop(self.slew, self.audit)
        self.assertFalse(self.slew.is_moving())
        self.assertTrue(self.slew.is_stopped())

    def test_wind_hold_blocks_start(self) -> None:
        status = stop_for_wind(self.slew, self.audit, 1)
        self.assertTrue(status.acknowledged())
        self.assertTrue(status.already_stopped)
        with self.assertRaises(SlewBlocked):
            start(self.slew, self.zone, self.store, self.audit)
        release_wind_hold(self.slew)
        start(self.slew, self.zone, self.store, self.audit)
        self.assertTrue(self.slew.is_moving())

    def test_wind_stop_reports_motion(self) -> None:
        start(self.slew, self.zone, self.store, self.audit)
        status = stop_for_wind(self.slew, self.audit, 2)
        self.assertTrue(status.stopped_now)
        self.assertFalse(status.already_stopped)
        self.assertEqual(self.slew.stop_count_now(), 1)
        self.assertEqual(self.slew.last_stop_sequence_now(), 2)

    def test_latch_hold_blocks_until_released(self) -> None:
        self.slew.latch_held = True
        with self.assertRaises(SlewBlocked):
            start(self.slew, self.zone, self.store, self.audit)
        release_latch_hold(self.slew)
        start(self.slew, self.zone, self.store, self.audit)

    def test_limit_stop_records_reason(self) -> None:
        start(self.slew, self.zone, self.store, self.audit)
        limit_stop(self.slew, "slew", 7)
        self.assertTrue(is_axis_stopped(self.slew, "slew"))
        self.assertEqual(stop_reason(self.slew, "slew"), "limit")
        self.assertEqual(stop_sequence(self.slew, "slew"), 7)
        self.assertFalse(is_axis_stopped(self.slew, "trolley"))

    def test_feedback_round_trip(self) -> None:
        feedback = SpeedFeedback("crane-1")
        self.assertFalse(feedback.is_confirmed())
        feedback.confirm(1.5)
        self.assertTrue(feedback.is_confirmed())
        self.assertEqual(feedback.speed_now(), 1.5)
        persist_feedback(feedback, self.store)
        reloaded = SpeedFeedback("crane-1")
        restore_feedback(reloaded, self.store)
        self.assertTrue(reloaded.is_confirmed())
        self.assertEqual(reloaded.speed_now(), 1.5)
        missing = SpeedFeedback("crane-9")
        restore_feedback(missing, self.store)
        self.assertFalse(missing.is_confirmed())


class TrolleyTests(unittest.TestCase):
    def test_move_requires_lowered_hook(self) -> None:
        trolley = Trolley("crane-1")
        hoist = Hoist("crane-1")
        with self.assertRaises(HookNotLowered):
            move(trolley, hoist)
        lower(hoist)
        move(trolley, hoist)
        self.assertTrue(trolley.is_moving())
        trolley_stop(trolley)
        self.assertFalse(trolley.is_moving())
        trolley_limit_stop(trolley)
        self.assertFalse(trolley.is_moving())

    def test_resume_reads_hoist_snapshot(self) -> None:
        trolley = Trolley("crane-1")
        hoist = Hoist("crane-1")
        store = MemoryStore()
        lower(hoist)
        persist_hoist(hoist, store)
        resumed = Hoist("crane-1")
        move_resume(trolley, resumed, store)
        self.assertTrue(resumed.hook_lowered())
        self.assertTrue(trolley.is_moving())

    def test_speedometer(self) -> None:
        speedometer = Speedometer()
        self.assertEqual(speedometer.sample(0.0, 0.0), 0.0)
        self.assertAlmostEqual(speedometer.sample(5.0, 2.0), 2.5)
        self.assertAlmostEqual(speedometer.last_speed(), 2.5)


class HoistTests(unittest.TestCase):
    def test_lower_and_busy(self) -> None:
        hoist = Hoist("crane-1")
        lower(hoist)
        self.assertTrue(hoist.hook_lowered())
        self.assertTrue(hoist.is_moving())
        with self.assertRaises(HoistBusy):
            lower(hoist)

    def test_brake_requires_feedback(self) -> None:
        hoist = Hoist("crane-1")
        feedback = SpeedFeedback("crane-1")
        with self.assertRaises(FeedbackNotConfirmed):
            release_brake(hoist, feedback)
        feedback.confirm(1.0)
        release_brake(hoist, feedback)
        self.assertFalse(hoist.brake_on())
        engage_brake(hoist)
        self.assertTrue(hoist.brake_on())

    def test_resume_release_restores_feedback(self) -> None:
        store = MemoryStore()
        feedback = SpeedFeedback("crane-1")
        feedback.confirm(2.0)
        persist_feedback(feedback, store)
        hoist = Hoist("crane-1")
        fresh = SpeedFeedback("crane-1")
        resume_release(hoist, fresh, store)
        self.assertFalse(hoist.brake_on())

    def test_raise_and_stop(self) -> None:
        hoist = Hoist("crane-1")
        raise_hook(hoist)
        self.assertTrue(hoist.is_moving())
        hoist_stop(hoist)
        self.assertFalse(hoist.is_moving())
        hoist_limit_stop(hoist)
        self.assertFalse(hoist.is_moving())

    def test_drum_winding(self) -> None:
        drum = Drum(0.8)
        drum.pay_out(1.5)
        self.assertGreater(drum.wound_total(), 0.0)
        self.assertGreaterEqual(drum.layer_count(), 1)
        self.assertGreater(drum.height_from_cable(10.0), 0.0)

    def test_layer_advances_only_after_a_full_layer(self) -> None:
        drum = Drum(0.8)
        capacity = drum.layer_capacity(1)
        self.assertGreater(drum.layer_capacity(2), capacity)
        drum.pay_out(capacity * 0.5)
        self.assertEqual(drum.layer_count(), 1)
        drum.pay_out(capacity)
        self.assertEqual(drum.layer_count(), 2)
        self.assertAlmostEqual(drum.wound_total(), capacity * 1.5)

    def test_height_tracks_cable_across_layers(self) -> None:
        drum = Drum(0.8)
        # Spool well into the third layer, then every height reading must
        # still agree with the paid-out cable instead of drifting with the
        # layer count.
        drum.pay_out(drum.layer_capacity(1) + drum.layer_capacity(2) + 25.0)
        self.assertGreaterEqual(drum.layer_count(), 3)
        wound = drum.wound_total()
        self.assertAlmostEqual(drum.height_from_cable(wound), wound)
        self.assertAlmostEqual(
            drum.height_from_revolutions(drum.revolutions_now()),
            drum.height_from_cable(wound),
        )

    def test_height_respects_reeving(self) -> None:
        drum = Drum(0.8, falls=2)
        drum.pay_out(30.0)
        self.assertAlmostEqual(drum.height_from_cable(drum.wound_total()), 15.0)

    def test_calibrate_drum_applies_measured_height(self) -> None:
        drum = Drum(0.8)
        drum.pay_out(40.0)
        record = calibrate_drum(drum, 41.2)
        self.assertAlmostEqual(record.correction, 1.03)
        self.assertAlmostEqual(drum.height_from_cable(drum.wound_total()), 41.2)
        self.assertEqual(drum.calibration_count(), 1)
        self.assertEqual(drum.last_calibration_now(), record)

    def test_calibrate_drum_rejects_bad_measurements(self) -> None:
        drum = Drum(0.8)
        with self.assertRaises(CalibrationError):
            calibrate_drum(drum, 10.0)
        drum.pay_out(40.0)
        with self.assertRaises(CalibrationError):
            calibrate_drum(drum, 0.0)
        with self.assertRaises(CalibrationError):
            calibrate_drum(drum, 100.0)
        self.assertEqual(drum.correction_now(), 1.0)
        self.assertEqual(drum.calibration_count(), 0)

    def test_drum_snapshot_round_trip(self) -> None:
        store = MemoryStore()
        drum = Drum(0.8)
        drum.pay_out(drum.layer_capacity(1) + 12.0)
        calibrate_drum(drum, drum.wound_total() * 1.02)
        persist_drum(drum, "crane-1", store)
        restored = Drum(0.8)
        restore_drum(restored, "crane-1", store)
        self.assertAlmostEqual(restored.wound_total(), drum.wound_total())
        self.assertEqual(restored.layer_count(), drum.layer_count())
        self.assertAlmostEqual(restored.revolutions_now(), drum.revolutions_now())
        self.assertAlmostEqual(restored.correction_now(), drum.correction_now())
        self.assertEqual(restored.calibration_count(), 1)
        self.assertAlmostEqual(
            restored.height_from_cable(restored.wound_total()),
            drum.height_from_cable(drum.wound_total()),
        )
        fresh = Drum(0.8)
        restore_drum(fresh, "crane-9", store)
        self.assertEqual(fresh.wound_total(), 0.0)
        self.assertEqual(fresh.correction_now(), 1.0)


if __name__ == "__main__":
    unittest.main()
