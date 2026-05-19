import numpy as np
from helperfunctions import add_pose_from_global, add_landmark_measurement_from_global
import gtsam
from gtsam.symbol_shorthand import L, X

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))  # (x, y, theta)
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))  # (dx, dy, dtheta)
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))  # (bearing, range)

def add_pose(graph, initial_estimate, pose_5):
    # Adding the initial estimate for the 5th pose using our helper function `add_pose_from_global` which also adds the odometry factor between X(4) and X(5).
    pose_4 = initial_estimate.atPose2(X(4))
    graph, initial_estimate = add_pose_from_global(
        graph=graph,
        initial_estimate=initial_estimate,
        prev_key=X(4),
        new_key=X(5),
        prev_pose=pose_4,
        new_pose_global=pose_5,
        odom_noise=ODOMETRY_NOISE
    )
    return graph, initial_estimate

def add_landmark_measurement(graph, result, pose_5, landmark):
    # Adding the measurement from X(5) to the chosen landmark using our helper function `add_landmark_measurement_from_global` which calculates the correct bearing and range from the global poses.``
    landmark_point = result.atPoint2(L(landmark))
    graph = add_landmark_measurement_from_global(
        graph=graph,
        pose_key=X(5),
        pose=pose_5,
        landmark_key=L(landmark),
        landmark_point=landmark_point,
        measurement_noise=MEASUREMENT_NOISE
    )
    return graph

def optimize(graph, initial_estimate):
    # TODO: Initialize the optimizer 
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate)

    # TODO: Perform the optimization and print the result
    result = optimizer.optimize()
    print(result)

    return result

def minimize_marginals(graph, initial_estimate, pose_options):
    #TODO: try different pose and landmark options here, and keep the one with the lowest sum of marginals.

    best_pose = None
    best_landmark = None
    lowest_marginals = np.inf

    for pose_name, pose_5 in pose_options.items():
        for landmark in [1, 2]:

            # Copy graph and estimates so originals stay unchanged
            temp_graph = gtsam.NonlinearFactorGraph(graph)
            temp_initial = gtsam.Values(initial_estimate)

            # Add pose
            temp_graph, temp_initial = add_pose(temp_graph, temp_initial, pose_5)

            # Optimize after adding pose
            result = optimize(temp_graph, temp_initial)

            # Add landmark measurement
            temp_graph = add_landmark_measurement(
                temp_graph,
                result,
                pose_5,
                landmark
            )

            # Optimize again
            result = optimize(temp_graph, temp_initial)

            # TODO: Calculate marginal covariances for the relevant variables and visualize the updated factor graph with covariances
            marginals = gtsam.Marginals(temp_graph, result)

            # The sum of the marginals for each landmark can be computed using marginals.marginalCovariance(L(x)).sum()
            sum_of_marginals = (
                marginals.marginalCovariance(L(1)).sum()
                + marginals.marginalCovariance(L(2)).sum()
            )

            if sum_of_marginals < lowest_marginals:
                lowest_marginals = sum_of_marginals
                best_pose = pose_name
                best_landmark = landmark

    return best_pose, best_landmark, lowest_marginals #sum_of_marginals

def minimize_errors(graph, initial_estimate, pose_options):
    #TODO: try different pose and landmark options here, and keep the one with the lowest resulting error.

    best_pose = None
    best_landmark = None
    lowest_error = np.inf

    for pose_name, pose_5 in pose_options.items():
        for landmark in [1, 2]:

            # Copy graph and estimates
            temp_graph = gtsam.NonlinearFactorGraph(graph)
            temp_initial = gtsam.Values(initial_estimate)

            # Add pose
            temp_graph, temp_initial = add_pose(temp_graph, temp_initial, pose_5)

            # Optimize
            result = optimize(temp_graph, temp_initial)

            # Add landmark measurement
            temp_graph = add_landmark_measurement(
                temp_graph,
                result,
                pose_5,
                landmark
            )

            # Optimize again
            result = optimize(temp_graph, temp_initial)

            # TODO: create a list of errors (each index corresponds to a pose) and add the error of each pose to the list
            list_of_errors = []

            for i in range(temp_graph.size()):
                error = temp_graph.at(i).error(result)
                list_of_errors.append(error)

            # TODO: compute the sum of the errors and return it along with the best pose and landmark
            sum_of_errors = sum(list_of_errors)

            if sum_of_errors < lowest_error:
                lowest_error = sum_of_errors
                best_pose = pose_name
                best_landmark = landmark

    return best_pose, best_landmark, lowest_error #sum_of_errors 