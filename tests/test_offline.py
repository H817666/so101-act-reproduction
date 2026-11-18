"""Pure-function checks; never connects to a robot or camera."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from so101_infer import bounded_goal, startup_check
from so101_record import features, quality


class OfflineChecks(unittest.TestCase):
    def test_good_recording_timing(self):
        samples = [{"state_time": i / 20, "camera_age_s": 0.01} for i in range(40)]
        self.assertTrue(quality(samples)["accepted"])

    def test_short_episode_rejected(self):
        samples = [{"state_time": i / 20, "camera_age_s": 0.01} for i in range(20)]
        self.assertFalse(quality(samples)["accepted"])

    def test_dual_camera_quality_and_features(self):
        samples = [{"state_time": i / 20, "camera_age_s": 0.01,
                    "wrist_camera_age_s": 0.02, "camera_pair_skew_s": 0.01}
                   for i in range(40)]
        self.assertTrue(quality(samples)["accepted"])
        self.assertIn("observation.images.wrist", features(include_wrist=True))
        self.assertNotIn("observation.images.wrist", features())

    def test_dual_camera_missing_or_skewed_frames_rejected(self):
        samples = [{"state_time": i / 20, "camera_age_s": 0.01,
                    "wrist_camera_age_s": 0.02, "camera_pair_skew_s": 0.01}
                   for i in range(40)]
        samples[10]["camera_pair_skew_s"] = 0.08
        self.assertFalse(quality(samples)["accepted"])
        del samples[10]["wrist_camera_age_s"]
        self.assertFalse(quality(samples)["accepted"])

    def test_startup_jump_rejected(self):
        with self.assertRaises(RuntimeError):
            startup_check(np.array([16., 0., 0., 0., 0., 0.]), np.zeros(6))

    def test_joint_speed_bounded(self):
        goal = bounded_goal(np.full(6, 50.), np.zeros(6), np.zeros(6),
                            np.full(6, -90.), np.full(6, 90.), 0.05)
        np.testing.assert_allclose(goal[:5], 1.5)
        self.assertAlmostEqual(goal[5], 0.75)


if __name__ == "__main__":
    unittest.main()
