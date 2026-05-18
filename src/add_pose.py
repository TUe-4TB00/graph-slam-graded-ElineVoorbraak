
import math
import numpy as np
import gtsam
from gtsam.symbol_shorthand import L, X

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))  # (x, y, theta)
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))  # (dx, dy, dtheta)
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))  # (bearing, range)

def add_pose(graph, initial_estimate):

    angle_45 = np.deg2rad(45)
    rel_x = 2.0 * np.cos(angle_45)
    rel_y = 2.0 * np.sin(angle_45)
    rel_theta = np.deg2rad(90)

    odometry = gtsam.Pose2(rel_x, rel_y, rel_theta)

    # This connects X(3) to X(4)
    graph.add(gtsam.BetweenFactorPose2(X(3), X(4), odometry, ODOMETRY_NOISE))
    
    # Add an initial guess for X(4)
    pose_3_guess = initial_estimate.atPose2(X(3))
    pose_4_guess = pose_3_guess.compose(odometry)
    initial_estimate.insert(X(4), pose_4_guess)

    return graph, initial_estimate