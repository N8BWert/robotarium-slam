"""
Odometry functionality for the SLAM Demo
"""

import numpy as np
import gtsam
from rps.robotarium import Robotarium

from utilities import get_robot_key

# The wheel radius in meters
WHEEL_RADIUS: float = 0.016
# The number of encoder counts per revolution
ENCODER_COUNTS_PER_REVOLUTION: float = 28.0
# The gear ratio
MOTOR_GEAR_RATIO: float = 100.0
# The base length
BASE_LENGTH: float = 0.1045

# The noise model for encoder odometry factors.
ENCODER_NOISE_MODEL = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.01, 0.01, 0.01]))


def ticks_to_distance(ticks: np.ndarray) -> np.ndarray:
    """
    Convert encoder ticks to distance traveled by the robot

    Args:
        ticks: The encoder ticks for the robot (2,N)
    Returns:
        The distance traveled by the robot (N,)
    """
    return ticks * 2 * np.pi * WHEEL_RADIUS / (ENCODER_COUNTS_PER_REVOLUTION * MOTOR_GEAR_RATIO)

def calculate_odometry_from_encoders(
    previous_encoder_values: np.ndarray,
    current_encoder_values: np.ndarray
) -> list[gtsam.Pose2]:
    """
    Calculate the odometry from the encoder values

    Args:
        previous_encoder_values: The encoder values for the robot at the previous timestamp (2,N)
        current_encoder_values: The encoder values for the robot at the current timestamp (2,N)
    Returns:
        A list of gtsam.Pose2 objects representing the odometry for each robot
    """
    delta_encoders = current_encoder_values - previous_encoder_values
    delta_arc = ticks_to_distance(delta_encoders)
    delta_s = (delta_arc[0, :] + delta_arc[1, :]) / 2.0
    delta_theta = (delta_arc[1, :] - delta_arc[0, :]) / BASE_LENGTH

    dxs = delta_s * np.cos(delta_theta / 2.0)
    dys = delta_s * np.sin(delta_theta / 2.0)

    return [gtsam.Pose2(dxs[i], dys[i], delta_theta[i]) for i in range(dxs.shape[0])]


def add_odometry_factor(
    graph: gtsam.NonlinearFactorGraph,
    odometry_deltas: list[gtsam.Pose2],
    timestamp: int
):
    """
    Adds an odometry factor to the graph based on the encoder values and timestamp

    Args:
        graph: The factor graph to add the odometry factor to
        odometry_deltas: list[gtsam.Pose2],
        timestamp: int
    """
    for robot_id, odometry in enumerate(odometry_deltas):
        graph.add(gtsam.BetweenFactorPose2(
            get_robot_key(robot_id, timestamp - 1),
            get_robot_key(robot_id, timestamp),
            odometry,
            ENCODER_NOISE_MODEL
        ))

def odometry_jacobian(
    previous_estimate: gtsam.Pose2,
    odometry_delta: gtsam.Pose2
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the Jacobian of the odometry factor with respect to the previous estimate and the odometry delta

    Args:
        previous_estimate: The previous estimate of the robot's pose
        odometry_delta: The odometry delta from the encoder values
    Returns:
        A tuple containing the Jacobian with respect to the previous estimate and the Jacobian with respect to the odometry delta
    """
    theta = previous_estimate.theta()
    cs, ss = np.cos(theta), np.sin(theta)
    dx, dy = odometry_delta.x(), odometry_delta.y()

    # Jacobian with respect to the previous estimate
    J_prev = np.array([
        [1, 0, -dx * ss - dy * cs],
        [0, 1, dx * cs - dy * ss],
        [0, 0, 1]
    ])

    # Jacobian with respect to the odometry delta
    J_odom = np.array([
        [cs, -ss, 0],
        [ss, cs, 0],
        [0, 0, 1]
    ])

    return J_prev, J_odom

def propogate_pose(
    previous_estimate: gtsam.Pose2,
    odometry_delta: gtsam.Pose2,
    previous_covariance: np.ndarray,
) -> tuple[gtsam.Pose2, np.ndarray]:
    """
    Propagate the pose and its covariance through the odometry update.

    Args:
        previous_estimate: The previous estimate of the robot's pose
        odometry_delta: The odometry delta from the encoder values
        previous_covariance: The previous covariance of the robot's pose

    Returns:
        A tuple containing the updated pose and its covariance
    """
    # Calculate the Jacobians
    J_prev, J_odom = odometry_jacobian(previous_estimate, odometry_delta)

    # Update the pose
    updated_pose = previous_estimate.compose(odometry_delta)

    # Update the covariance
    updated_covariance = J_prev @ previous_covariance @ J_prev.T + J_odom @ ENCODER_NOISE_MODEL.covariance() @ J_odom.T

    return updated_pose, updated_covariance
