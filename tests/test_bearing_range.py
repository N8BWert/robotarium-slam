"""
Unittests for hte bearing and range calculations in bearing_range.py
"""

import unittest
import gtsam
import numpy as np
from rps.robotarium import Robotarium

from bearing_range import LandmarkCandidate, BearingRangeDetector


class TestBearingRangeCalculations(unittest.TestCase):
    def test_landmark_candidate_check_past_max_observation_window(self):
        candidate = LandmarkCandidate(np.array([1.0, 1.0]), 0)
        self.assertEqual(candidate.hit_count, 1)
        self.assertEqual(candidate.last_observation_timestamp, 0)
        for i in range(candidate.MAX_OBSERVATION_WINDOW + 1):
            self.assertFalse(candidate.check(np.array([-1.0, -1.0]), i + 1))
        self.assertEqual(candidate.hit_count, 0)
        self.assertTrue(candidate.landmark_elapsed(candidate.MAX_OBSERVATION_WINDOW + 1))

    def test_landmark_candidate_check_within_tolerance_in_a_row(self):
        candidate = LandmarkCandidate(np.array([1.0, 1.0]), 0)
        self.assertEqual(candidate.hit_count, 1)
        self.assertEqual(candidate.last_observation_timestamp, 0)
        for i in range(1, candidate.MIN_HITS):
            self.assertTrue(candidate.check(np.array([1.01, 1.01]), i ))
        self.assertEqual(candidate.hit_count, candidate.MIN_HITS)
        self.assertTrue(candidate.is_landmark())

    def test_landmark_candidate_check_within_tolerance_not_in_a_row(self):
        candidate = LandmarkCandidate(np.array([1.0, 1.0]), 0)
        self.assertEqual(candidate.hit_count, 1)
        self.assertEqual(candidate.last_observation_timestamp, 0)
        for i in range(1, candidate.MIN_HITS):
            if i % 2 == 0:
                self.assertTrue(candidate.check(np.array([1.01, 1.01]), i))
            else:
                self.assertFalse(candidate.check(np.array([-1.0, -1.0]), i))
        self.assertEqual(candidate.hit_count, candidate.MIN_HITS // 2 + 1)
        self.assertFalse(candidate.is_landmark())

    def test_landmark_candidate_is_landmark(self):
        candidate = LandmarkCandidate(np.array([1.0, 1.0]), 0)
        self.assertEqual(candidate.hit_count, 1)
        self.assertEqual(candidate.last_observation_timestamp, 0)
        for i in range(1, candidate.MIN_HITS):
            self.assertTrue(candidate.check(np.array([1.01, 1.01]), i))
        self.assertEqual(candidate.hit_count, candidate.MIN_HITS)
        self.assertTrue(candidate.is_landmark())

    def test_landmark_candidate_observation_location_valid_landmark(self):
        candidate = LandmarkCandidate(np.array([1.0, 1.0]), 0)
        self.assertEqual(candidate.hit_count, 1)
        self.assertEqual(candidate.last_observation_timestamp, 0)
        for i in range(1, candidate.MIN_HITS):
            self.assertTrue(candidate.check(np.array([1.01, 1.01]), i))
        self.assertEqual(candidate.hit_count, candidate.MIN_HITS)
        self.assertTrue(candidate.is_landmark())
        expected_location = np.mean(candidate.location, axis=0)
        np.testing.assert_array_almost_equal(candidate.location, expected_location)

    def test_convert_distance_to_world_coordinates(self):
        pose = np.array([[0.0], [0.0], [0.0]])
        distances = np.array([[-1.0], [-1.0], [-1.0], [0.5], [-1.0], [-1.0], [-1.0]])
        detector = BearingRangeDetector()
        world_coordinates = detector.convert_distance_readings_to_world_coordinates(pose, distances)
        self.assertEqual(world_coordinates.shape, (2, 7, 1))
        np.testing.assert_array_almost_equal(world_coordinates[:, 3, 0], np.array([0.5 + Robotarium.ROBOT_DIAMETER / 2.0 + 0.05, 0.0]))
        self.assertTrue(np.all(np.isnan(world_coordinates[:, :3, 0])))
        self.assertTrue(np.all(np.isnan(world_coordinates[:, 4:, 0])))

    def test_convert_distance_readings_to_world_coordinates_outside_of_boundaries(self):
        pose = np.array([[Robotarium.BOUNDARIES[1] - 0.2], [0.0], [0.0]])
        distances = np.array([[-1.0], [-1.0], [-1.0], [0.5], [-1.0], [-1.0], [-1.0]])
        detector = BearingRangeDetector()
        world_coordinates = detector.convert_distance_readings_to_world_coordinates(pose, distances)
        self.assertEqual(world_coordinates.shape, (2, 7, 1))
        self.assertTrue(np.all(np.isnan(world_coordinates[:, :, 0])))

    def test_calculate_ranges(self):
        poses = np.array([[0.0], [0.0], [0.0]])
        world_coordinates = np.array([[[1.0], [1.0]], [[0.0], [0.0]]])
        detector = BearingRangeDetector()
        ranges = detector.calculate_ranges(poses, world_coordinates)
        self.assertEqual(ranges.shape, (1, 2))
        np.testing.assert_array_almost_equal(ranges, np.array([[1.0, 1.0]]))

    def test_calculate_ranges_with_nan_values(self):
        poses = np.array([[0.0], [0.0], [0.0]])
        world_coordinates = np.array([[[np.nan]], [[np.nan]]])
        detector = BearingRangeDetector()
        ranges = detector.calculate_ranges(poses, world_coordinates)
        self.assertEqual(ranges.shape, (1, 1))
        self.assertTrue(np.isnan(ranges[0, 0]))

    def test_calculate_bearings(self):
        poses = np.array([[0.0], [0.0], [0.0]])
        world_coordinates = np.array([[[1.0], [0.0]], [[0.0], [1.0]]])
        detector = BearingRangeDetector()
        bearings = detector.calculate_bearings(poses, world_coordinates)
        self.assertEqual(bearings.shape, (1, 2))
        np.testing.assert_array_almost_equal(bearings, np.array([[0.0, np.pi/2]]))

    def test_calculate_bearings_with_nan_values(self):
        poses = np.array([[0.0], [0.0], [0.0]])
        world_coordinates = np.array([[[np.nan]], [[np.nan]]])
        detector = BearingRangeDetector()
        bearings = detector.calculate_bearings(poses, world_coordinates)
        self.assertEqual(bearings.shape, (1, 1))
        self.assertTrue(np.isnan(bearings[0, 0]))

    def test_compute_bearing_range_jacobian_wrt_pose(self):
        pose = np.array([0.0, 0.0, 0.0])
        landmark = np.array([1.0, 1.0])
        detector = BearingRangeDetector()
        jacobian = detector.compute_bearing_range_jacobian_wrt_pose(pose, landmark)
        self.assertEqual(jacobian.shape, (2, 3))
        np.testing.assert_array_almost_equal(jacobian, np.array([[1/2, -1/2, -1.0], [-1/np.sqrt(2), -1/np.sqrt(2), 0.0]]))

    def test_compute_bearing_range_jacobian_wrt_landmark(self):
        pose = np.array([0.0, 0.0, 0.0])
        landmark = np.array([1.0, 1.0])
        detector = BearingRangeDetector()
        jacobian = detector.compute_bearing_range_jacobian_wrt_landmark(pose, landmark)
        self.assertEqual(jacobian.shape, (2, 2))
        np.testing.assert_array_almost_equal(jacobian, np.array([[-1/2, 1/2], [1/np.sqrt(2), 1/np.sqrt(2)]]))

    def test_gate_observation_inside_gate(self):
        detector = BearingRangeDetector()
        measurement = np.array([np.pi/4, np.sqrt(2)])
        predicted_measurement = np.array([np.pi/4, np.sqrt(2)])
        pose_covariance = np.eye(3)
        landmark_covariance = np.eye(2)
        H_pose = np.array([[1/2, -1/2, -1.0], [-1/np.sqrt(2), -1/np.sqrt(2), 0.0]])
        H_landmark = np.array([[-1/2, 1/2], [1/np.sqrt(2), 1/np.sqrt(2)]])
        self.assertTrue(detector.gate_observation(measurement, predicted_measurement, pose_covariance, landmark_covariance, H_pose, H_landmark))

    def test_gate_observation_outside_gate(self):
        detector = BearingRangeDetector()
        measurement = np.array([np.pi/2, 0.5])
        predicted_measurement = np.array([0.0, 1.0])
        pose_covariance = np.array([[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]])
        landmark_covariance = np.array([[0.1, 0.0], [0.0, 0.1]])
        H_pose = np.array([[1/2, -1/2, -1.0], [-1/np.sqrt(2), -1/np.sqrt(2), 0.0]])
        H_landmark = np.array([[-1/2, 1/2], [1/np.sqrt(2), 1/np.sqrt(2)]])
        self.assertFalse(detector.gate_observation(measurement, predicted_measurement, pose_covariance, landmark_covariance, H_pose, H_landmark))


if __name__ == "__main__":
    unittest.main()