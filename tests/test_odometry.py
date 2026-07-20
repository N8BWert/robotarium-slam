"""
Unittests for the odometry calculations in odometry.py
"""

import unittest
import gtsam
import numpy as np
from rps.robotarium import Robotarium

from odometry import calculate_odometry_from_encoders, odometry_jacobian, propogate_pose

class TestOdometryCalculations(unittest.TestCase):
    def test_calculate_odometry_from_encoders_no_motion(self):
        previous_encoder_values = np.array([[0, 0], [0, 0]])
        current_encoder_values = np.array([[0, 0], [0, 0]])
        odometry = calculate_odometry_from_encoders(previous_encoder_values, current_encoder_values)
        self.assertEqual(len(odometry), 2)
        for pose in odometry:
            self.assertAlmostEqual(pose.x(), 0.0)
            self.assertAlmostEqual(pose.y(), 0.0)
            self.assertAlmostEqual(pose.theta(), 0.0)

    def test_calculate_odometry_from_encoders_forward_motion(self):
        previous_encoder_values = np.array([[0, 0], [0, 0]])
        current_encoder_values = np.array([[1, 1], [1, 1]])
        odometry = calculate_odometry_from_encoders(previous_encoder_values, current_encoder_values)
        self.assertEqual(len(odometry), 2)
        for pose in odometry:
            self.assertAlmostEqual(pose.x(), 2 * np.pi * Robotarium.WHEEL_RADIUS / Robotarium.ENCODER_COUNTS_PER_REVOLUTION / Robotarium.MOTOR_GEAR_RATIO)
            self.assertAlmostEqual(pose.y(), 0.0)
            self.assertAlmostEqual(pose.theta(), 0.0)

    def test_calculate_odometry_from_encoders_rotation(self):
        previous_encoder_values = np.array([[0, 0], [0, 0]])
        current_encoder_values = np.array([[1, -1], [-1, 1]])
        odometry = calculate_odometry_from_encoders(previous_encoder_values, current_encoder_values)
        self.assertEqual(len(odometry), 2)
        for i, pose in enumerate(odometry):
            self.assertAlmostEqual(pose.x(), 0.0)
            self.assertAlmostEqual(pose.y(), 0.0)
            sign = (-1) ** (i + 1)
            self.assertAlmostEqual(pose.theta(), sign * 2.0 * 2.0 * np.pi * Robotarium.WHEEL_RADIUS / Robotarium.ENCODER_COUNTS_PER_REVOLUTION / Robotarium.MOTOR_GEAR_RATIO / Robotarium.BASE_LENGTH)

    def test_odometry_jacobian(self):
        pose_estimate = gtsam.Pose2(0.0, 0.0, 0.0)
        odometry_delta = gtsam.Pose2(0.1, 0.1, np.pi/2)
        jacobian_prev, jacobian_odom = odometry_jacobian(pose_estimate, odometry_delta)
        self.assertEqual(jacobian_prev.shape, (3, 3))
        self.assertEqual(jacobian_odom.shape, (3, 3))
        np.testing.assert_array_almost_equal(
            jacobian_prev,
            np.array([[1.0, 0.0, -0.1 * np.sin(0.0) - 0.1 * np.cos(0.0)],
                      [0.0, 1.0, 0.1 * np.cos(0.0) - 0.1 * np.sin(0.0)],
                      [0.0, 0.0, 1.0]])
        )
        np.testing.assert_array_almost_equal(
            jacobian_odom,
            np.array([[1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0]])
        )

    def test_propogate_pose(self):
        previous_estimate = gtsam.Pose2(0.0, 0.0, 0.0)
        odometry_delta = gtsam.Pose2(1.0, 0.0, np.pi/2)
        previous_covariance = np.eye(3) * 0.1
        updated_pose, updated_covariance = propogate_pose(
            previous_estimate,
            odometry_delta,
            previous_covariance
        )
        self.assertIsInstance(updated_pose, gtsam.Pose2)
        self.assertEqual(updated_covariance.shape, (3, 3))
        expected_pose = previous_estimate.compose(odometry_delta)
        self.assertAlmostEqual(updated_pose.x(), expected_pose.x())
        self.assertAlmostEqual(updated_pose.y(), expected_pose.y())
        self.assertAlmostEqual(updated_pose.theta(), expected_pose.theta())
        np.testing.assert_array_almost_equal(
            updated_covariance,
            np.array([[0.1625, 0.0, 0.0],
                      [0.0, 0.2625, 0.1],
                      [0.0, 0.1, 0.1625]])
        )

if __name__ == "__main__":
    unittest.main()
