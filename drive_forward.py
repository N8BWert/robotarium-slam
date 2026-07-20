"""
Test file for driving forward to get the standard deviation of error on the encoders
"""

from rps.robotarium import Robotarium
import numpy as np
import time


TIME_STEP = 1.0 / 15.0
ITERATIONS = 20 * 15


def main():
    r = Robotarium(
        number_of_robots = 1,
        show_figure = True,
        initial_conditions = np.array([[-1.0], [0.0], [0.0]]),
        skip_initialization = True
    )

    for i in range(15):
        r.get_poses()
        r.set_velocities(np.arange(1), np.array([[0.1 / (15 - i)], [0.0]]))
        r.step()

    ground_truth = np.zeros((ITERATIONS, 3))
    encoder_values = np.zeros((ITERATIONS, 2))
    start_time = time.time()
    last_time = start_time

    for i in range(ITERATIONS):
        while time.time() - last_time <= TIME_STEP:
            continue
        last_time = time.time()
        print(f"Elapsed Time: {last_time - start_time}")
        ground_truth[i] = r.get_poses()[:, 0]
        encoders = r.get_encoders()[:, 0]
        encoder_values[i] = encoders
        r.set_velocities(np.arange(1), np.array([[0.1], [0.0]]))
        r.step()

    r.get_poses()
    r.set_velocities(np.arange(1), np.zeros((2, 1)))
    r.step()

    gt_deltas = np.zeros((ITERATIONS - 1, 3))
    for i in range(ITERATIONS - 1):
        gt_deltas[i] = ground_truth[i + 1] - ground_truth[i]

    encoder_deltas = np.zeros((ITERATIONS - 1, 2))
    for i in range(ITERATIONS - 1):
        encoder_deltas[i] = encoder_values[i + 1] - encoder_values[i]

    print(f"Mean Encoder Deltas: {np.mean(encoder_deltas, axis=0)}")
    print(f"Std Encoder Deltas: {np.std(encoder_deltas, axis=0)}")

    np.save("encoders.npy", encoder_values)
    np.save("ground_truth_deltas.npy", gt_deltas)
    np.save("encoder_deltas.npy", encoder_deltas)
    np.save("ground_truth.npy", ground_truth)

    r.debug()

if __name__ == "__main__":
    main()

