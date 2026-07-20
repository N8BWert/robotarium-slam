"""
Robtoarium SLAM Demo

This is a simple demo of conducting multi-agent SLAM in the Robotarium using onboard sensors.

This demo has 1 robots "seeking" and 9 robots "landmarking".
Seeking robots move around the robotarium adding encoder odometry factors and heading
priors based on the IMU.  The landmarking robots are stationary at different locations
in the Robotarium.  The seeking robots also use their distance sensors to detect the
landmarking robots and other seeking robots.
"""

from rps.robotarium import Robotarium
from rps.utilities.barrier_certificates import create_uni_barrier_certificate
import numpy as np
import gtsam

from graphing import Grapher
from odometry import add_odometry_factor, calculate_odometry_from_encoders, propogate_pose
from orientation import add_orientation_factor
from utilities import get_robot_key
from controller import SLAMDemoController
from bearing_range import BearingRangeDetector
from utils import generate_initial_conditions

# The ids of the seeking robots
SEEKING_IDS = [0]
# The ids of the landmarking robots
LANDMARKING_IDS = [1, 2, 3, 4, 5, 6]
# The number of robots in the experiment
N = len(SEEKING_IDS) + len(LANDMARKING_IDS)
# The number of iterations in the experiment
ITERATIONS = 5000

def main():
    r = Robotarium(
        number_of_robots = N,
        show_figure = True,
        initial_conditions = generate_initial_conditions(N),
        use_distance_sensors = True,
        sim_in_real_time = True,
        show_arena_boundaries = True,
        show_distance_endpoints = False,
        skip_initialization = True
    )

    axes = r._axes_handle

    # Initialize the controller
    controller = SLAMDemoController(
        seeking_ids=SEEKING_IDS,
        x_columns=4,
        y_columns=3,
    )
    barrier = create_uni_barrier_certificate()
    grapher = Grapher(axes, SEEKING_IDS)
    detector = BearingRangeDetector()

    # Initial Pass
    x = r.get_poses()
    initial_poses = x[:, SEEKING_IDS]
    initial_encoders = r.get_encoders()[:, SEEKING_IDS]
    initial_headings = r.get_orientations()[SEEKING_IDS]
    r.set_velocities(np.arange(N), np.zeros((2, N)))
    r.step()

    gt_trajectories = []
    encoder_readings = []
    heading_readings = []
    encoder_readings.append(initial_encoders.copy())
    gt_trajectories.append(initial_poses.copy())
    heading_readings.append(initial_headings.copy())

    # Initialize the ISAM2 Solver
    params = gtsam.ISAM2Params()
    params.setRelinearizeThreshold(0.01)
    params.relinearizeSkip = 1
    isam = gtsam.ISAM2(params)

    # Initialize GTSAM Factor Graph
    estimates = gtsam.Values()
    initial_factors = gtsam.NonlinearFactorGraph()
    initial_estimates = gtsam.Values()
    for robot_id in SEEKING_IDS:
        initial_factors.add(gtsam.PriorFactorPose2(
            get_robot_key(robot_id, 0),
            gtsam.Pose2(initial_poses[0, robot_id], initial_poses[1, robot_id], initial_poses[2, robot_id]),
            gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-6, 1e-6, 1e-6]))
        ))
        initial_estimates.insert(
            get_robot_key(robot_id, 0),
            gtsam.Pose2(initial_poses[0, robot_id], initial_poses[1, robot_id], initial_poses[2, robot_id])
        )
    isam.update(initial_factors, initial_estimates)

    previous_encoders = initial_encoders.copy()
    for iteration in range(1, ITERATIONS+1):
        # Initialize the new factors and values to add
        new_factors = gtsam.NonlinearFactorGraph()
        new_values = gtsam.Values()

        # Get the current poses
        x = r.get_poses()
        gt_trajectories.append(x[:, SEEKING_IDS].copy())

        # Add odometry factors to the graph
        encoders = r.get_encoders()[:, SEEKING_IDS]
        encoder_readings.append(encoders.copy())
        odometry_deltas = calculate_odometry_from_encoders(previous_encoders, encoders)
        add_odometry_factor(
            new_factors,
            odometry_deltas,
            iteration
        )

        previous_encoders = encoders.copy()

        # Add heading factors to the graph
        headings = r.get_orientations()[SEEKING_IDS]
        heading_readings.append(headings.copy())
        add_orientation_factor(
            new_factors,
            headings,
            iteration
        )

        # Calculate the robot pose estimates from the previous iteration
        current_pose_estimates = []
        current_pose_covariances =  []
        for robot_id in SEEKING_IDS:
            key = get_robot_key(robot_id, iteration - 1)
            previous_pose = isam.calculateEstimatePose2(key)
            previous_covariance = isam.marginalCovariance(key)
            odometry = odometry_deltas[robot_id]
            updated_pose, updated_covariance = propogate_pose(
                previous_pose,
                odometry,
                previous_covariance
            )
            current_pose_estimates.append(updated_pose)
            current_pose_covariances.append(updated_covariance)

        poses_np = np.array([[pose.x(), pose.y(), pose.theta()] for pose in current_pose_estimates]).T
        pose_covariances_np = np.array(current_pose_covariances)

        # Add distance detection factors to the graph
        distances = r.get_distances()[:, SEEKING_IDS]
        world_coordinates = detector.convert_distance_readings_to_world_coordinates(
            poses_np,
            distances
        )
        detector.update(
            poses_np,
            pose_covariances_np,
            world_coordinates,
            new_factors,
            new_values,
            estimates,
            isam,
            iteration
        )

        # Provide initial guesses for new keys
        for robot_id in SEEKING_IDS:
            new_values.insert(get_robot_key(robot_id, iteration), current_pose_estimates[robot_id])

        # Update iSAM2 estimate
        isam.update(new_factors, new_values)

        # Update Visualization
        estimates = isam.calculateEstimate()
        grapher.show_estimates(estimates, iteration)
        grapher.show_ground_truth(gt_trajectories, x[:2, LANDMARKING_IDS])
        grapher.draw_detections(world_coordinates)

        # Send Controls
        dxu = controller.apply(estimates, iteration)
        controls = np.zeros((2, N))
        controls[:, SEEKING_IDS] = dxu
        # controls = barrier(controls, x)
        r.set_velocities(np.arange(N), controls)

        # Update the Robotarium
        r.step()

    trajectory_estimate = []
    for timestamp in range(ITERATIONS):
        trajectory_timestamp = []
        for robot_id in SEEKING_IDS:
            key = get_robot_key(robot_id, timestamp)
            if estimates.exists(key):
                pose = estimates.atPose2(key)
                trajectory_timestamp.append([pose.x(), pose.y(), pose.theta()])
        trajectory_estimate.append(np.array(trajectory_timestamp).T)

    np.save("gt_trajectories.npy", np.array(gt_trajectories))
    np.save("trajectory_estimate.npy", np.array(trajectory_estimate))
    np.save("heading_readings.npy", np.array(heading_readings))
    np.save("encoder_readings.npy", np.array(encoder_readings))
    r.debug()
    

if __name__ == "__main__":
    main()
