"""
Utilties for the SLAM demo
"""

import numpy as np


def generate_initial_conditions(N: int) -> np.ndarray:
    """
    Generate the initial conditions for the SLAM demo (there are 1 seekers and 9 landmarkers)
    """
    accepted = [np.array([0.0, 0.0])]
    while len(accepted) < N:
        candidate = np.array([
            (np.random.rand() - 0.5) * 3.0,
            (np.random.rand() - 0.5) * 1.8
        ])

        if np.all(np.linalg.norm(np.array(accepted) - candidate, axis=1) >= 0.5):
            accepted.append(candidate)

    poses = np.zeros((3, N))
    poses[:2, :] = np.array(accepted).T
    poses[2, :] = np.random.rand(N) * 2 * np.pi - np.pi
    return poses
