"""
Orientation gtsam functionality for the SLAM Demo
"""

import numpy as np
import gtsam
from rps.robotarium import Robotarium

from utilities import get_robot_key

ORIENTATION_NOISE_MODEL = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e6, 1e6, Robotarium.ORIENTATION_NOISE_STD]))

def add_orientation_factor(
    graph: gtsam.NonlinearFactorGraph,
    headings: np.ndarray,
    timestamp: int
):
    """
    Adds an orientation factor to the graph based on the IMU headings and timestamp

    Args:
        graph: The factor graph to add the orientation factor to
        headings: The IMU headings for the robot at the current timestamp (N,)
        timestamp: The timestamp of the IMU readings
    """
    headings = np.deg2rad(headings)
    for robot_id, heading in enumerate(headings):
        graph.add(gtsam.PriorFactorPose2(
            get_robot_key(robot_id, timestamp),
            gtsam.Pose2(0.0, 0.0, heading),
            ORIENTATION_NOISE_MODEL
        ))
