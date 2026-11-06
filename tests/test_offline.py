"""Pure-function checks; never connects to a robot or camera."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from so101_infer import bounded_goal, startup_check
from so101_record import quality


class OfflineChecks(unittest.TestCase):
    def test_good_recording_timing(self):
        samples = [{"state_time": i / 20, "camera_age_s": 0.01} for i in range(40)]
        self.assertTrue(quality(samples)["accepted"])

    def test_short_episode_rejected(self):
        samples = [{"state_time": i / 20, "camera_age_s": 0.01} for i in range(20)]
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
