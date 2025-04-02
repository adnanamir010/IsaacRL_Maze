# global_vars.py
import numpy as np

# Initialize global variables
body_pose = np.array([0.0, 0.0, 0.0], float)  # x, y, theta
lidar_data = np.zeros(20)
clash_sum = 0
image = None
image_for_clash_calc = None