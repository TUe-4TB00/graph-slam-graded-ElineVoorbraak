
import math
import numpy as np
import gtsam
from gtsam.symbol_shorthand import L, X

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))  # (x, y, theta)
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))  # (dx, dy, dtheta)
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))  # (bearing, range)


def add_pose(graph, initial_estimate):

    angle_rad = math.radians(45)
    dx = 2.0 * math.cos(angle_rad)
    dy = 2.0 * math.sin(angle_rad)
    dtheta = math.radians(90) 
    
    odometry_delta = gtsam.Pose2(dx, dy, dtheta)

    # Add the BetweenFactor to the graph
    # This connects the previous pose X(3) to the new pose X(4)
    graph.add(gtsam.BetweenFactorPose2(X(3), X(4), odometry_delta, ODOMETRY_NOISE))

    initial_estimate.insert(X(4), gtsam.Pose2(5.4, 1.4, 1.57))
    
    return graph, initial_estimate