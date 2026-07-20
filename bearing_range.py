"""
Bearing Range Factor utilities for the SLAM Demo
"""

import numpy as np
import gtsam
from scipy.stats import chi2
from functools import partial

from rps.robotarium import Robotarium

from utilities import get_robot_key

# The maximum distance between a distance sensor reading and a robot to be considered valid for inter-robot association
ROBOT_ASSOCIATION_TOLERANCE = 0.15
# The maximum distance between a distance sensor reading and a landmark to be considered valid
LANDMARK_ASSOCIATION_TOLERANCE = 0.225
# The standard deviation for the bearing part of the bearing range factor
BEARING_NOISE_STD = 0.85
# The standard deviation for the range part of the bearing range factor
DISTANCE_NOISE_STD = 0.85
# The noise model for the bearing range factors.
BASE_NOISE_MODEL = gtsam.noiseModel.Diagonal.Sigmas(np.array([BEARING_NOISE_STD, DISTANCE_NOISE_STD]))
# The huber threshold for the huber noise model
HUBER_THRESHOLD = 1.345
# The huber noise model for the bearing range factors
HUBER_NOISE_MODEL = gtsam.noiseModel.Robust.Create(
    gtsam.noiseModel.mEstimator.Huber.Create(HUBER_THRESHOLD),
    BASE_NOISE_MODEL
)
# The standard deviation for noise on the landmark position
LANDMARK_POSITION_NOISE_STD = 0.5
# The mahalanobis distance threshold for the chi-squared test for outlier rejection
MAHALANOBIS_THRESHOLD = chi2.ppf(0.95, df=2)


def bearing_range_robot_error(
    measurement: np.ndarray,
    this: gtsam.CustomFactor,
    values: gtsam.Values,
    jacobians
):
    """
    Custom error function for the bearing range factor between two robots.

    Args:
        measurement: The bearing and range measurement (2,)
        this: The gtsam.CustomFactor object
        values: The gtsam.Values object containing the estimated poses of the robots and landmarks
        jacobians: The jacobians of the error function with respect to the poses of the robots
    Returns:
        The error vector (2,) 
    """
    key_observer = this.keys()[0]
    key_observed = this.keys()[1]

    pose_i = values.atPose2(key_observer)
    pose_i = np.array([pose_i.x(), pose_i.y(), pose_i.theta()])
    pose_j = values.atPose2(key_observed)
    landmark_point = np.array([pose_j.x(), pose_j.y()])

    delta = landmark_point - pose_i[:2]
    predicted_range = np.linalg.norm(delta)
    predicted_bearing = np.arctan2(delta[1], delta[0]) - pose_i[2]

    bearing_error = gtsam.Rot2(predicted_bearing - measurement[0]).theta()
    range_error = predicted_range - measurement[1]
    error = np.array([bearing_error, range_error])

    if jacobians is not None:
        H_pose_i = BearingRangeDetector.compute_bearing_range_jacobian_wrt_pose(
            pose_i,
            landmark_point
        )
        H_landmark = BearingRangeDetector.compute_bearing_range_jacobian_wrt_landmark(
            pose_i,
            landmark_point
        )
        H_pose_j = np.hstack([H_landmark, np.zeros((2, 1))])

        jacobians[0] = H_pose_i
        jacobians[1] = H_pose_j

    return error


class LandmarkCandidate:
    """
    A landmark candidate is a potential landmark detection by a seeking robot.  It is used to 
    track the detections of a specific landmark and, eventually, promote them into true landmarks. 
    """

    # The number of times the landmark candidate has to be observed before it is added to the
    # graph
    MIN_HITS: int = 12
    # The window of timestamps between which a landmark candidate has to be observed before
    # it is deemed to be invalid
    MAX_OBSERVATION_WINDOW: int = 15

    def __init__(self, observation_location: np.ndarray, timestamp: int):
        self.observation_locations = [observation_location]
        self.location = observation_location
        self.last_observation_timestamp = timestamp
        self.hit_count = 1

    def check(
        self,
        observation_location: np.ndarray,
        timestamp: int,
        tolerance: float = LANDMARK_ASSOCIATION_TOLERANCE
    ) -> bool:
        """
        Check if the landmark candidate has been observed again and update the hit count and last observation timestamp

        Args:
            observation_location: The location of the observation (2,)
            timestamp: The timestamp of the observation
        Returns:
            True if the landmark candidate has been observed enough times to be added to the graph, False otherwise
        """
        if timestamp - self.last_observation_timestamp > self.MAX_OBSERVATION_WINDOW:
            self.hit_count = 0
            return False
        delta = observation_location - self.location
        if np.linalg.norm(delta) < tolerance:
            self.hit_count += 1
            self.last_observation_timestamp = timestamp
            self.observation_locations.append(observation_location)
            self.location = np.mean(self.observation_locations, axis=0)
            return True
        return False
    
    def is_landmark(self) -> bool:
        """
        Check if the landmark candidate has been observed enough times to be added to the graph

        Returns:
            True if the landmark candidate has been observed enough times to be added to the graph, False otherwise
        """
        return self.hit_count >= self.MIN_HITS
    
    def landmark_elapsed(self, timestamp: int) -> bool:
        """
        Check if the landmark candidate has not been observed for a long time

        Returns:
            True if the landmark candidate has not been observed for a long time, False otherwise
        """
        return timestamp - self.last_observation_timestamp > self.MAX_OBSERVATION_WINDOW
    
    def landmark_stationary(self, max_spread: float = Robotarium.ROBOT_DIAMETER) -> bool:
        positions = np.array(self.observation_locations)
        spread = np.max(np.linalg.norm(positions - positions.mean(axis=0), axis=1))
        return spread < max_spread
    
    def observation_location(self) -> np.ndarray:
        """
        Get the average observation location of the landmark candidate

        Returns:
            The average observation location of the landmark candidate (2,)
        """
        return self.location


class BearingRangeDetector:
    def __init__(self):
        self.landmark_candidates: list[LandmarkCandidate] = []

    def update(
        self,
        poses: np.ndarray,
        pose_covariances: np.ndarray,
        world_coordinates: np.ndarray,
        new_graph: gtsam.NonlinearFactorGraph,
        new_values: gtsam.Values,
        estimates: gtsam.Values,
        isam2: gtsam.ISAM2,
        timestamp: int
    ):
        """
        Update the landmark candidates based on the distance readings from the seeking robots

        Args:
            poses: The current pose estimates of the seeking robots (3, N)
            pose_covariances: The covariances of the pose estimates (N, 3, 3)
            world_coordinates: The world coordinates of the observations (2, M, N)
            new_graph: The gtsam.NonlinearFactorGraph object to add new factors to
            new_values: The gtsam.Values object to add new values to
            estimates: The gtsam.Values object containing the estimated poses of the robots and landmarks
            isam2: The gtsam.isam2 object to use for calculating marginal covariances
            timestamp: The timestamp of the estimates
        """
        _, M, N = world_coordinates.shape

        measured_ranges = BearingRangeDetector.calculate_ranges(poses, world_coordinates)
        measured_bearings = BearingRangeDetector.calculate_bearings(poses, world_coordinates)
        measurements = np.stack([measured_bearings, measured_ranges], axis=2)

        # Find landmark positions and covariances
        landmark_positions, landmark_covariances = BearingRangeDetector.find_landmarks(estimates, isam2)
        landmarks = 0
        if len(landmark_positions) > 0:
            landmarks = landmark_positions.shape[1]
        landmark_key = landmarks
        landmark_predicted_measurements = None
        if len(landmark_positions) > 0:
            landmarks_repeated = np.repeat(landmark_positions[:, :, np.newaxis], N, axis=2)
            predicted_ranges = BearingRangeDetector.calculate_ranges(poses, landmarks_repeated)
            predicted_bearings = BearingRangeDetector.calculate_bearings(poses, landmarks_repeated)
            landmark_predicted_measurements = np.stack([predicted_bearings, predicted_ranges], axis=2)

        for robot_id in range(N):
            # Check for any valid measurements
            if np.isnan(world_coordinates[:, :, robot_id]).all():
                continue

            # Check for distance associations
            for sensor_id in range(M):
                # Skip sensors that didn't see anything
                if np.isnan(world_coordinates[:, sensor_id, robot_id]).any():
                    continue

                # Check other robots for potential association if no existing landmark was found
                found_association = False
                measurement = measurements[robot_id, sensor_id]
                for other_robot_id in range(N):
                    if other_robot_id == robot_id:
                        continue
                    world_position = world_coordinates[:, sensor_id, robot_id]
                    robot_position = poses[:2, other_robot_id]
                    if np.linalg.norm(world_position - robot_position) < ROBOT_ASSOCIATION_TOLERANCE:
                        # print(f"Adding bearing range factor for robot {robot_id} and robot {other_robot_id} at timestamp {timestamp}")
                        new_graph.add(gtsam.CustomFactor(
                            HUBER_NOISE_MODEL,
                            [get_robot_key(robot_id, timestamp), get_robot_key(other_robot_id, timestamp)],
                            partial(bearing_range_robot_error, measurement)
                        ))
                        found_association = True
                        break

                # Check existing landmarks for association
                if not found_association:
                    if landmark_predicted_measurements is not None:
                        for landmark_id in range(landmarks):
                            world_position = world_coordinates[:, sensor_id, robot_id]
                            landmark_position = landmark_positions[:, landmark_id]
                            if np.linalg.norm(world_position - landmark_position) < LANDMARK_ASSOCIATION_TOLERANCE:
                                # print(f"Adding bearing range factor for robot {robot_id} and landmark {landmark_id} at timestamp {timestamp}")
                                new_graph.add(gtsam.BearingRangeFactor2D(
                                    get_robot_key(robot_id, timestamp),
                                    gtsam.symbol("l", landmark_id),
                                    gtsam.Rot2(measurement[0]),
                                    measurement[1],
                                    BASE_NOISE_MODEL
                                ))
                                found_association = True
                                break
                        # predicted_measurement = landmark_predicted_measurements[robot_id, landmark_id]
                        # H_pose = BearingRangeDetector.compute_bearing_range_jacobian_wrt_pose(
                        #     poses[:, robot_id],
                        #     landmark_positions[:, landmark_id]
                        # )
                        # H_landmark = BearingRangeDetector.compute_bearing_range_jacobian_wrt_landmark(
                        #     poses[:, robot_id],
                        #     landmark_positions[:, landmark_id]
                        # )
                        # pose_covariance = pose_covariances[robot_id, :, :]
                        # if BearingRangeDetector.gate_observation(
                        #     measurement,
                        #     predicted_measurement,
                        #     pose_covariance,
                        #     landmark_covariances[landmark_id],
                        #     H_pose,
                        #     H_landmark
                        # ):
                        #     print(f"Adding bearing range factor for robot {robot_id} and landmark {landmark_id} at timestamp {timestamp}")
                        #     # new_graph.add(gtsam.BearingRangeFactor2D(
                        #     #     get_robot_key(robot_id, timestamp),
                        #     #     gtsam.symbol("l", landmark_id),
                        #     #     gtsam.Rot2(measurement[0]),
                        #     #     measurement[1],
                        #     #     HUBER_NOISE_MODEL
                        #     # ))
                        #     found_association = True
                        #     break

                # if not found_association:
                #     for other_robot_id in range(N):
                #         if other_robot_id == robot_id:
                #             continue
                #         predicted_measurement = inter_robot_measurements[robot_id, other_robot_id]
                #         H_pose = BearingRangeDetector.compute_bearing_range_jacobian_wrt_pose(
                #             poses[:, robot_id],
                #             poses[:2, other_robot_id]
                #         )
                #         H_landmark = BearingRangeDetector.compute_bearing_range_jacobian_wrt_landmark(
                #             poses[:, robot_id],
                #             poses[:2, other_robot_id]
                #         )
                #         pose_covariance = pose_covariances[robot_id, :, :]
                #         landmark_covariance = pose_covariances[other_robot_id, :2, :2]
                #         if BearingRangeDetector.gate_observation(
                #             measurement,
                #             predicted_measurement,
                #             pose_covariance,
                #             landmark_covariance,
                #             H_pose,
                #             H_landmark
                #         ):
                #             print(f"Adding bearing range factor for robot {robot_id} and robot {other_robot_id} at timestamp {timestamp}")
                #             # new_graph.add(gtsam.CustomFactor(
                #             #     HUBER_NOISE_MODEL,
                #             #     [get_robot_key(robot_id, timestamp), get_robot_key(other_robot_id, timestamp)],
                #             #     partial(bearing_range_robot_error, measurement)
                #             # ))
                #             found_association = True
                #             break

                # Check the landmark candidates for association if no existing landmark was found
                # print("Checking for Landmark Candidates")
                if not found_association:
                    candidate_exists = False
                    for candidate in self.landmark_candidates:
                        if candidate.check(world_coordinates[:, sensor_id, robot_id], timestamp):
                            candidate_exists = True
                            if candidate.is_landmark():
                                # print(f"Promoting landmark candidate to landmark for robot {robot_id} at timestamp {timestamp}")
                                new_key = gtsam.symbol("l", landmark_key)
                                # new_graph.add(gtsam.PriorFactorPoint2(
                                #     new_key,
                                #     gtsam.Point2(candidate.observation_location()),
                                #     gtsam.noiseModel.Diagonal.Sigmas(np.array([LANDMARK_POSITION_NOISE_STD, LANDMARK_POSITION_NOISE_STD]))
                                # ))
                                new_values.insert(
                                    new_key,
                                    gtsam.Point2(candidate.observation_location())
                                )
                                new_graph.add(gtsam.BearingRangeFactor2D(
                                    get_robot_key(robot_id, timestamp),
                                    new_key,
                                    gtsam.Rot2(measurement[0]),
                                    measurement[1],
                                    BASE_NOISE_MODEL
                                ))
                                self.landmark_candidates.remove(candidate)
                                landmark_key += 1
                            break
                    if not candidate_exists:
                        # print(f"Adding new landmark candidate for robot {robot_id} at timestamp {timestamp}")
                        self.landmark_candidates.append(LandmarkCandidate(world_coordinates[:, sensor_id, robot_id], timestamp))

        # Remove stale landmark candidates
        for i in reversed(range(len(self.landmark_candidates))):
            if self.landmark_candidates[i].landmark_elapsed(timestamp):
                self.landmark_candidates.pop(i)
    
    @staticmethod
    def find_landmarks(
        estimates: gtsam.Values,
        isam2: gtsam.ISAM2
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Find the positions and covariances of the landmarks in the graph

        Args:
            estimates: The gtsam.Values object containing the estimated poses of the robots and landmarks
            isam2: The gtsam.isam2 object to use for calculating marginal covariances
        Returns:
            A tuple containing the positions of the landmarks (2, M) and their covariances (M, 2, 2)
        """
        landmark_positions = []
        landmark_covariances = []
        i = 0
        while True:
            landmark_key = gtsam.symbol("l", i)
            if not estimates.exists(landmark_key):
                break
            landmark_positions.append(estimates.atPoint2(landmark_key))
            landmark_covariances.append(isam2.marginalCovariance(landmark_key))
            i += 1
        return np.array(landmark_positions).T, np.array(landmark_covariances)


    @staticmethod
    def convert_distance_readings_to_world_coordinates(
        poses: np.ndarray,
        distance_readings: np.ndarray
    ) -> np.ndarray:
        """
        Convert the distance readings from the seeking robots to world coordinates

        Args:
            poses: The poses of the seeking robots (3, N)
            distance_readings: The distance readings from the seeking robots (M, N)
        Returns:
            The world coordinates of the distance readings (2, M, N)
        """
        M, N = distance_readings.shape
        distance_readings = distance_readings.copy()
        distance_readings[distance_readings > 0.75] = np.nan
        # add half of a robot's width to the distance readings to account for the fact that the distance readings
        # are taken from the outside of the robot, not the center which we care about
        distance_readings = distance_readings.copy() + Robotarium.ROBOT_DIAMETER / 2.0

        # Convert the distance readings to world coordinates
        world_coordinates = np.full((2, M, N), np.nan)
        for i in range(N):
            R = np.array([[np.cos(poses[2, i]), -np.sin(poses[2, i])],
                          [np.sin(poses[2, i]), np.cos(poses[2, i])]])
            for j in range(M):
                if distance_readings[j, i] >= 0:
                    start_point = R @ Robotarium.DISTANCE_SENSORS_ORIENTATION[:2, j] + poses[:2, i]
                    end_point = start_point + \
                        distance_readings[j, i] * \
                        np.array([np.cos(poses[2, i] + Robotarium.DISTANCE_SENSORS_ORIENTATION[2, j]), np.sin(poses[2, i] + Robotarium.DISTANCE_SENSORS_ORIENTATION[2, j])])
                    # Check to make sure the end point is within the bounds of the arena
                    if Robotarium.BOUNDARIES[0] <= end_point[0] <= Robotarium.BOUNDARIES[1] and \
                        Robotarium.BOUNDARIES[2] <= end_point[1] <= Robotarium.BOUNDARIES[3]:
                        world_coordinates[:, j, i] = end_point
        return world_coordinates
    
    @staticmethod
    def calculate_ranges(
        poses: np.ndarray,
        world_coordinates: np.ndarray
    ) -> np.ndarray:
        """
        Calculates the range from a set of given poses to a set of given world coordinates

        Args:
            poses: The poses of the seeking robots (3, N)
            world_coordinates: The world coordinates of the distance readings (2, M, N)
        Returns:
            The ranges from the seeking robots to the landmarks (N, M)
        """
        _, M, N = world_coordinates.shape
        ranges = np.full((N, M), np.nan)
        for i in range(N):
            for j in range(M):
                if not np.isnan(world_coordinates[0, j, i]):
                    ranges[i, j] = np.linalg.norm(world_coordinates[:, j, i] - poses[:2, i])
        return ranges
    
    @staticmethod
    def calculate_bearings(
        poses: np.ndarray,
        world_coordinates: np.ndarray
    ) -> np.ndarray:
        """
        Calculates the bearing from a set of given poses to a set of given world coordinates

        Args:
            poses: The poses of the seeking robots (3, N)
            world_coordinates: The world coordinates of the distance readings (2, M, N)
        Returns:
            The bearings from the seeking robots to the landmarks (N, M)
        """
        _, M, N = world_coordinates.shape
        bearings = np.full((N, M), np.nan)
        for i in range(N):
            for j in range(M):
                if not np.isnan(world_coordinates[0, j, i]):
                    delta = world_coordinates[:, j, i] - poses[:2, i]
                    bearings[i, j] = np.arctan2(delta[1], delta[0]) - poses[2, i]
        return bearings
    
    @staticmethod
    def compute_bearing_range_jacobian_wrt_pose(
        pose: np.array,
        landmark_point: np.array
    ) -> np.ndarray:
        """
        Computes the Jacobian of the bearing range factor with respect to the pose of the seeking robot

        Args:
            pose: The pose of the seeking robot (3,)
            landmark_point: The position of the landmark (2,)
        Returns:
            The Jacobian of the bearing range factor (2, 3)
        """
        dx = landmark_point[0] - pose[0]
        dy = landmark_point[1] - pose[1]
        q = dx**2 + dy**2
        r = np.sqrt(q)

        return np.array([
            [dy/q, -dx/q, -1.0],
            [-dx/r, -dy/r, 0.0]
        ])
    
    @staticmethod
    def compute_bearing_range_jacobian_wrt_landmark(
        pose: np.ndarray,
        landmark_point: np.ndarray
    ) -> np.ndarray:
        """
        Computes the Jacobian of the bearing range factor with respect to the landmark position

        Args:
            pose: The pose of the seeking robot (3,)
            landmark_point: The position of the landmark (2,)
        Returns:
            The Jacobian of the bearing range factor with respect to the landmark position (2, 2)
        """
        dx = landmark_point[0] - pose[0]
        dy = landmark_point[1] - pose[1]
        q = dx**2 + dy**2
        r = np.sqrt(q)

        return np.array([
            [-dy/q, dx/q],
            [dx/r, dy/r]
        ])
    
    @staticmethod
    def gate_observation(
        measurement: np.ndarray,
        predicted_measurement: np.ndarray,
        pose_covariance: np.ndarray,
        landmark_covariance: np.ndarray,
        H_pose: np.ndarray,
        H_landmark: np.ndarray
    ) -> bool:
        """
        Gate an observation based on the Mahalanobis distance between the measurement and the predicted measurement

        Args:
            measurement: The measurement (2,)
            predicted_measurement: The predicted measurement (2,)
            pose_covariance: The covariance of the observing pose (3, 3)
            landmark_covariance: The covariance of the landmark (2, 2)
            H_pose: The Jacobian of the measurement model with respect to the pose (2, 3)
            H_landmark: The Jacobian of the measurement model with respect to the landmark (2, 2)
        Returns:
            True if the observation is within the gate, False otherwise
        """
        innovation = measurement - predicted_measurement
        innovation[0] = gtsam.Rot2(innovation[0]).theta()

        S = (H_pose @ pose_covariance @ H_pose.T
             + H_landmark @ landmark_covariance @ H_landmark.T
             + BASE_NOISE_MODEL.covariance())
        
        d_sq = innovation.T @ np.linalg.inv(S) @ innovation
        return d_sq < MAHALANOBIS_THRESHOLD
