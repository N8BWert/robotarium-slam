"""
Utility functions for use in the SLAM Demo
"""

import gtsam

def get_robot_key(robot_id: int, timestamp: int) -> gtsam.Symbol:
    """
    Get the gtsam key for a robot at a given timestamp

    Args:
        robot_id: The id of the robot
        timestamp: The timestamp of the robot's pose
    Returns:
        The gtsam key for the robot's pose at the given timestamp
    """
    return gtsam.symbol("x", robot_id * 100000 + timestamp)
