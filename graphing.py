"""
Graphing Utilities for the slam demo.
"""

import matplotlib.patches as patches
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
import gtsam
import numpy as np

from utilities import get_robot_key

ROBOT_COLORS = ["#1f77b4", "#2ca02c", "#9467bd", "#d62728", "#17becf"]
 
GT_LANDMARK_COLOR = "#c0392b"
EST_LANDMARK_COLOR = "#e67e22"
DETECTION_COLOR = "#f39c12"


class Grapher:
    """
    A class for graphing the ground truth and estimated poses of the robots and landmarks in the SLAM demo
    """

    def __init__(
        self,
        axes: Axes,
        seeking_ids: list[int] = [0, 1, 2],
        N: int = 10,
    ):
        self.axes = axes
        self.seeking_ids = seeking_ids
        self._legend_built = False

        self._style_axes()

        # Ground truth landmarks
        self.gt_landmark_patches = []
        for _ in range(N - len(seeking_ids)):
            patch = patches.Circle(
                (np.nan, np.nan),
                0.1,
                facecolor=GT_LANDMARK_COLOR,
                edgecolor="black",
                linewidth=0.6,
                alpha=0.30,
                zorder=2
            )
            self.gt_landmark_patches.append(patch)
            self.axes.add_patch(patch)

        # Ground truth trajectories (thin dashed lines, one per seeker)
        self.gt_trajectory_lines: dict[int, Line2D] = {}
        for idx, robot_id in enumerate(self.seeking_ids):
            color = ROBOT_COLORS[idx % len(ROBOT_COLORS)]
            (line,) = self.axes.plot(
                [],
                [],
                linestyle="--",
                linewidth=1.3,
                color=color,
                alpha=0.45,
                zorder=1,
                label=f"Robot {robot_id} (ground truth)"
            )
            self.gt_trajectory_lines[robot_id] = line

        # Estimated trajectories (solid lines)
        self.est_trajectory_lines: dict[int, Line2D] = {}
        for idx, robot_id in enumerate(self.seeking_ids):
            color = ROBOT_COLORS[idx % len(ROBOT_COLORS)]
            (line,) = self.axes.plot(
                [],
                [],
                linestyle="-",
                linewidth=2.2,
                color=color,
                alpha=0.95,
                zorder=3,
                label=f"Robot {robot_id} (estimate)"
            )
            self.est_trajectory_lines[robot_id] = line

        # Current-pose markers (traingles rotated to show heading)
        self.est_pose_markers: dict[int, Line2D] = {}
        for idx, robot_id in enumerate(self.seeking_ids):
            color = ROBOT_COLORS[idx % len(ROBOT_COLORS)]
            (marker,) = self.axes.plot(
                [],
                [],
                marker=(3, 0, 0),
                markersize=30,
                markerfacecolor=color,
                markeredgecolor="black",
                markeredgewidth=0.9,
                linestyle="",
                zorder=4
            )
            self.est_pose_markers[robot_id] = marker
        
        # Estimated landmarks (orange circles, gro dynamically as promoted)
        self.est_landmark_patches = []

        # Distance sensor detections (small transient markers)
        self.distance_patches = []
        for _ in range(len(self.seeking_ids)):
            p = []
            for _ in range(7):
                patch = patches.Circle(
                    (np.nan, np.nan),
                    0.05,
                    facecolor=DETECTION_COLOR,
                    edgecolor="none",
                    alpha=0.65,
                    zorder=5
                )
                p.append(patch)
                self.axes.add_patch(patch)
            self.distance_patches.append(p)

    def _style_axes(self):
        """
        Apply consistent demo-friendly styling to the plot.
        """
        self.axes.set_facecolor("#fafafa")
        self.axes.grid(True, linestyle=":", linewidth=0.6, alpha=0.5)
        for spine in self.axes.spines.values():
            spine.set_color("#888888")

    def _build_legend(self):
        """
        Build the legend for the plot.
        """
        handles, labels = self.axes.get_legend_handles_labels()
        if handles:
            self.axes.legend(loc="lower right", fontsize=8, framealpha=0.9, ncol=2)
            self._legend_built = True

    def show_ground_truth(
        self,
        seeking_trajectories: list[np.ndarray],
        landmark_positions: np.ndarray
    ):
        """
        Show the ground truth poses of the robots and landmarks in the experiment

        Args:
            seeking_trajectories: A list of the ground truth trajectories of the robots in the experiment
            landmark_positions: The ground truth positions of the landmarks in the experiment (2, M)
        """
        trajectories = np.asarray(seeking_trajectories)
        for idx, robot_id in enumerate(self.seeking_ids):
            xs = trajectories[:, 0, idx]
            ys = trajectories[:, 1, idx]
            self.gt_trajectory_lines[robot_id].set_data(xs, ys)

        num_landmarks = landmark_positions.shape[1]
        for i in range(num_landmarks):
            if i >= len(self.gt_landmark_patches):
                patch = patches.Circle(
                    (np.nan, np.nan),
                    0.1,
                    facecolor=GT_LANDMARK_COLOR,
                    edgecolor="black",
                    linewidth=0.6,
                    alpha=0.30,
                    zorder=2
                )
                self.gt_landmark_patches.append(patch)
                self.axes.add_patch(patch)
            self.gt_landmark_patches[i].center = (landmark_positions[0, i], landmark_positions[1, i])

        self._build_legend()

    def show_estimates(self, estimates: gtsam.Values, timestamp: int):
        """
        Show the estimated poses of the robots and landmarks in the experiment

        Args:
            estimates: The gtsam.Values object containing the estimated poses of the robots and landmarks
            timestamp: The timestamp of the estimates to show
        """
        # TODO: Draw the robot estimate trajectories for the seeking robots
        for robot_id in self.seeking_ids:
            xs, ys = [], []
            for t in range(timestamp + 1):
                key = get_robot_key(robot_id, t)
                if not estimates.exists(key):
                    continue
                pose = estimates.atPose2(key)
                xs.append(pose.x())
                ys.append(pose.y())
            
            if not xs:
                continue

            self.est_trajectory_lines[robot_id].set_data(xs, ys)

            latest_key = get_robot_key(robot_id, timestamp)
            if estimates.exists(latest_key):
                pose = estimates.atPose2(latest_key)
                heading_deg = np.degrees(pose.theta()) - 90
                marker = self.est_pose_markers[robot_id]
                marker.set_marker((3, 0, heading_deg))
                marker.set_data([pose.x()], [pose.y()])

        i = 0
        while True:
            landmark_key = gtsam.symbol("l", i)
            if not estimates.exists(landmark_key):
                break
            position = estimates.atPoint2(landmark_key)
            if i >= len(self.est_landmark_patches):
                patch = patches.Circle(
                    (np.nan, np.nan),
                    0.1,
                    facecolor=EST_LANDMARK_COLOR,
                    edgecolor="black",
                    linewidth=0.7,
                    alpha=0.55,
                    zorder=6
                )
                self.est_landmark_patches.append(patch)
                self.axes.add_patch(patch)
            self.est_landmark_patches[i].center = (position[0], position[1])
            i += 1

        self._build_legend()

    def draw_detections(self, world_coordinates: np.ndarray):
        """
        Draw the detections of the robots in the experiment

        Args:
            world_coordinates: The world coordinates of the detections (2,7,N)
        """
        for robot_id in range(world_coordinates.shape[2]):
            for sensor_id in range(world_coordinates.shape[1]):
                if not np.isnan(world_coordinates[:, sensor_id, robot_id]).any():
                    self.distance_patches[robot_id][sensor_id].center = (
                        world_coordinates[0, sensor_id, robot_id],
                        world_coordinates[1, sensor_id, robot_id]
                    )
                else:
                    self.distance_patches[robot_id][sensor_id].center = (np.nan, np.nan)


