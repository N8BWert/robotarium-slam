"""
The controller for the SLAM Demo
"""

import math
import numpy as np
from typing import Optional
from cvxopt import matrix, sparse
from cvxopt.solvers import qp, options
import random
import gtsam
from abc import ABC

from rps.robotarium import Robotarium
from rps.utilities.transformations import create_si_to_uni_mapping, create_uni_to_si_mapping
from rps.utilities.controllers import create_uni_position_controller

from utilities import get_robot_key

options["show_progress"] = False
options["reltol"] = 1e-2
options["feastol"] = 1e-2
options["maxiters"] = 50

class SLAMDemoBarrierCertificate:
    """
    The barrier certificate used for the SLAM Demo 
    """
    
    def __init__(
        self,
        barrier_gain: float = 1.0,
        safety_radius: float = 0.15,
        projection_distance: float = 0.05,
        velocity_magnitude_limit: float = 0.15
    ):
        self.barrier_gain = barrier_gain
        self.safety_radius = safety_radius
        self.projection_distance = projection_distance
        self.magnitude_limit = velocity_magnitude_limit
        self.si_uni_dyn, self.uni_si_states = create_si_to_uni_mapping(projection_distance=self.projection_distance)
        self.uni_si_dyn, _ = create_uni_to_si_mapping(projection_distance=self.projection_distance)

    def h(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """
        The barrier function for the barrier certificate

        Args:
            x1: The position of the first robot (2,)
            x2: The position of the second robot (2,)
        Returns:
            The value of the barrier function
        """
        diff = x1[:2] - x2[:2]
        return np.dot(diff, diff) - self.safety_radius ** 2
    
    def grad_h(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """
        The gradient of the barrier function for the barrier certificate

        Args:
            x1: The position of the first robot (2,)
            x2: The position of the second robot (2,)
        Returns:
            The gradient of the barrier function
        """
        diff = x1[:2] - x2[:2]
        return 2.0 * diff

    def apply(
        self,
        velocities: np.ndarray,
        poses: np.ndarray,
        obstacles: Optional[np.ndarray]
    ) -> np.ndarray:
        """
        Apply the barrier certificate to the desired velocities

        Args:
            velocities: The desired velocities for the robots (2, N)
            poses: The current poses of the robots (3, N)
            obstacles: The positions of the obstacles (2, M)
        Returns:
            The safe velocities for the robots (2, N)
        """
        N = velocities.shape[1]
        if obstacles is not None:
            num_constraints = math.comb(N, 2) + 8 * N + obstacles.shape[1] * N
        else:
            num_constraints = math.comb(N, 2) + 8 * N
        A = np.zeros((num_constraints, 2 * N))
        b = np.zeros(num_constraints)

        x_si = self.uni_si_states(poses)
        dx_si = self.uni_si_dyn(velocities, poses)

        # Apply inter-robot constraints
        constraint = 0
        for i in range(N-1):
            for j in range(i+1, N):
                h = self.h(x_si[:, i], x_si[:, j])
                grad_h = self.grad_h(x_si[:, i], x_si[:, j])
                A[constraint, 2*i:2*i+2] = -grad_h
                A[constraint, 2*j:2*j+2] = grad_h
                b[constraint] = self.barrier_gain * h
                constraint += 1

        # Apply velocity magnitude constraints
        for i in range(N):
            # vx <= magnitude_limit
            A[constraint, 2*i] = 1.0
            b[constraint] = self.magnitude_limit * np.cos(np.pi / 8)
            constraint += 1

            # 1/sqrt(2) * (vx + vy) <= magnitude_limit
            A[constraint, 2*i:2*i+2] = [1.0 / np.sqrt(2), 1.0 / np.sqrt(2)]
            b[constraint] = self.magnitude_limit * np.cos(np.pi / 8)
            constraint += 1

            # vy <= magnitude_limit
            A[constraint, 2*i+1] = 1.0
            b[constraint] = self.magnitude_limit * np.cos(np.pi / 8)
            constraint += 1

            # 1/sqrt(2) * (-vx + vy) <= magnitude_limit
            A[constraint, 2*i:2*i+2] = [-1.0 / np.sqrt(2), 1.0 / np.sqrt(2)]
            b[constraint] = self.magnitude_limit * np.cos(np.pi / 8)
            constraint += 1

            # -vx <= magnitude_limit
            A[constraint, 2*i] = -1.0
            b[constraint] = self.magnitude_limit * np.cos(np.pi / 8)
            constraint += 1

            # 1/sqrt(2) * (-vx - vy) <= magnitude_limit
            A[constraint, 2*i:2*i+2] = [-1.0 / np.sqrt(2), -1.0 / np.sqrt(2)]
            b[constraint] = self.magnitude_limit * np.cos(np.pi / 8)
            constraint += 1

            # -vy <= magnitude_limit
            A[constraint, 2*i+1] = -1.0
            b[constraint] = self.magnitude_limit * np.cos(np.pi / 8)
            constraint += 1

            # 1/sqrt(2) * (vx - vy) <= magnitude_limit
            A[constraint, 2*i:2*i+2] = [1.0 / np.sqrt(2), -1.0 / np.sqrt(2)]
            b[constraint] = self.magnitude_limit * np.cos(np.pi / 8)
            constraint += 1

        # Apply obstacle / landmark constraints
        if obstacles is not None:
            for i in range(N):
                for j in range(obstacles.shape[1]):
                    h = self.h(x_si[:, i], obstacles[:, j])
                    grad_h = self.grad_h(x_si[:, i], obstacles[:, j])
                    A[constraint, 2*i:2*i+2] = -grad_h
                    b[constraint] = self.barrier_gain * h
                    constraint += 1

        safe_velocities = self._solve_qp(N, dx_si, A, b)
        if safe_velocities is None:
            return np.zeros((2, N))
        else:
            return self.si_uni_dyn(safe_velocities, poses)

    def _solve_qp(self, N: int, vhat: np.ndarray, A: np.ndarray, b: np.ndarray) -> Optional[np.ndarray]:
        """
        Solve the QP for the barrier certificate

        Args:
            N: The number of robots
            vhat: The desired velocities for the robots (2, N)
            A: The A matrix for the QP (N*(N-1)/2, 2*N)
            b: The b vector for the QP (N*(N-1)/2,)
        Returns:
            The safe velocities for the robots (2, N) or None if the QP is infeasible
        """
        H = sparse(matrix(2.0 * np.eye(2 * N)))
        f = matrix(-2.0 * vhat.reshape(-1, order="F"))

        try:
            sol = qp(H, f, matrix(A), matrix(b))
            if sol["status"] == "optimal":
                return np.reshape(sol["x"], (2, N), order="F")
        except Exception:
            pass

        return None
    
class SLAMBehavior(ABC):
    """
    The base class for the SLAM Demo behaviors
    """

    def apply(
        self,
        estimates: np.ndarray,
    ) -> np.ndarray:
        """
        Apply the behavior to the robots

        Args:
            estimates: The current estimates of the robot poses
        Returns:
            The desired velocities for the robots (2, N)
        """
        ...

    def done(self) -> bool:
        """
        Check if the behavior is done

        Returns:
            True if the behavior is done, False otherwise
        """
        ...

    def reset(self):
        """
        Reset the behavior to its initial state
        """
        ...

class WaypointingBehavior(SLAMBehavior):
    """
    The waypointing behavior for the SLAM Demo
    """

    def __init__(
        self,
        seeking_ids: list[int] = [0, 1, 2],
        x_columns: int = 4,
        y_columns: int = 3,
        waypointing_length: int = 5 * 30
    ):
        self.controller = create_uni_position_controller(
            x_velocity_gain=0.8,
            y_velocity_gain=0.8,
            velocity_magnitude_limit=0.15,
            projection_distance=0.05
        )
        self.seeking_ids = seeking_ids
        self.waypoints = WaypointingBehavior.generate_waypoints(x_columns, y_columns)
        self.waypoint_idxs = [i for i in range(len(self.waypoints))]
        self.waypoint_assignments = random.sample(self.waypoint_idxs, len(self.seeking_ids))
        self.counter = 0
        self.waypointing_length = waypointing_length

    def apply(
        self,
        estimates: np.ndarray,
    ) -> np.ndarray:
        """
        Apply the behavior to the robots

        Args:
            estimates: The current estimates of the robot poses
        Returns:
            The desired velocities for the robots (2, N)
        """
        self.counter += 1
        desired_poses = self.waypoints[self.waypoint_assignments]
        return self.controller(estimates, np.array(desired_poses).T)

    def done(self) -> bool:
        """
        Check if the behavior is done

        Returns:
            True if the behavior is done, False otherwise
        """
        return self.counter >= self.waypointing_length
    
    def reset(self):
        """
        Reset the behavior to its initial state
        """
        self.counter = 0
        self.waypoint_assignments = random.sample(self.waypoint_idxs, len(self.seeking_ids))
    
    @staticmethod
    def generate_waypoints(x_columns: int = 4, y_columns: int = 3):
        """
        Generate a set of waypoints for the robots to drive to

        Args:
            x_columns: The number of columns of waypoints in the x direction
            y_columns: The number of columns of waypoints in the y direction
        Returns:
            A list of waypoints for the robots to drive to
        """
        x_waypoints = np.linspace(
            Robotarium.BOUNDARIES[0] + 0.2,
            Robotarium.BOUNDARIES[1] - 0.2,
            x_columns,
        )
        y_waypoints = np.linspace(
            Robotarium.BOUNDARIES[2] + 0.2,
            Robotarium.BOUNDARIES[3] - 0.2,
            y_columns
        )
        xv, yv = np.meshgrid(x_waypoints, y_waypoints)
        return np.array([[x, y] for x, y in zip(xv.flatten(), yv.flatten())])
    
class ScanningBehavior(SLAMBehavior):
    """
    The scanning behavior for the SLAM Demo
    """

    def __init__(
        self,
        seeking_ids: list[int] = [0, 1, 2],
        scanning_length: int = 1 * 30,
    ):
        self.scanning_length = scanning_length
        self.counter = 0
        self.seeking_ids = seeking_ids

    def apply(
        self,
        estimates: np.ndarray,
    ) -> np.ndarray:
        """
        Apply the behavior to the robots

        Args:
            estimates: The current estimates of the robot poses
        Returns:
            The desired velocities for the robots (2, N)
        """
        dxu = np.zeros((2, len(self.seeking_ids)))
        if self.counter < self.scanning_length // 2:
            for i in range(len(self.seeking_ids)):
                dxu[:, i] = np.array([0.0, np.pi / 4])
        else:
            for i in range(len(self.seeking_ids)):
                dxu[:, i] = np.array([0.0, -np.pi / 4])
        self.counter += 1
        return dxu

    def done(self) -> bool:
        """
        Check if the behavior is done

        Returns:
            True if the behavior is done, False otherwise
        """
        return self.counter >= self.scanning_length
    
    def reset(self):
        """
        Reset the behavior to its initial state
        """
        self.counter = 0

class SLAMDemoController:
    """
    The controller for the SLAM Demo 
    """

    def __init__(
        self,
        seeking_ids: list = [0, 1, 2], x_columns: int = 4, y_columns: int = 3):
        # The barrier certificate to use for preventing crashes
        self.barrier_certificate = SLAMDemoBarrierCertificate(
            barrier_gain=1.0,
            safety_radius=0.15,
            projection_distance=0.05,
            velocity_magnitude_limit=0.15
        )
        # The unicycle position controller to use to drive the robot
        self.controller = create_uni_position_controller(
            x_velocity_gain=0.8,
            y_velocity_gain=0.8,
            velocity_magnitude_limit=0.15,
            projection_distance=0.05
        )
        self.waypointing_behavior = WaypointingBehavior(
            seeking_ids,
            x_columns, 
            y_columns
        )
        self.scanning_behavior = ScanningBehavior(
            seeking_ids=seeking_ids,
            scanning_length = 2 * 30
        )
        self.waypointing = True
        self.counter = 0
        self.seeking_ids = seeking_ids
    
    def apply(
        self,
        estimates: gtsam.Values,
        timestamp: int
    ) -> np.ndarray:
        """
        Apply the controller to the robots

        Args:
            estimates: The current estimates of the robot poses and landmark positions
        Returns:
            The safe velocities for the robots (2, N)
        """
        # Get the current poses of the robots
        robot_poses = np.zeros((3, len(self.seeking_ids)))
        for i in self.seeking_ids:
            robot_key = get_robot_key(i, timestamp)
            if estimates.exists(robot_key):
                pose = estimates.atPose2(robot_key)
                x = pose.x()
                y = pose.y()
                theta = pose.theta()
                robot_poses[:, i] = np.array([x, y, theta])
            else:
                raise ValueError(f"Robot {i} does not exist in the estimates at timestamp {timestamp}")

        # Get the current positions of the landmarks
        i = 0
        landmark_positions = []
        while True:
            landmark_key = gtsam.symbol("l", i)
            if estimates.exists(landmark_key):
                position = estimates.atPoint2(landmark_key)
                landmark_positions.append(np.array([position[0], position[1]]))
                i += 1
            else:
                break
        landmark_positions = np.array(landmark_positions).T if landmark_positions else None

        # Apply the current behavior (waypointing or scanning)
        if self.waypointing:
            dxu = self.waypointing_behavior.apply(robot_poses)
            if self.waypointing_behavior.done():
                self.waypointing = True
                self.waypointing_behavior.reset()
        else:
            dxu = self.scanning_behavior.apply(robot_poses)
            if self.scanning_behavior.done():
                self.waypointing = True
                self.scanning_behavior.reset()

        # Apply Barrier Certificate to prevent collisions
        dxu = self.barrier_certificate.apply(dxu, robot_poses, landmark_positions)
        return dxu


