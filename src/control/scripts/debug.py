import numpy as np
from environment import VectorizedDDEnv
import time

# Test the environment rendering directly (without vectorization)
env = VectorizedDDEnv(render_mode='human', obstacle_shape="square")
obs, _ = env.reset()

# Apply brighter colors for obstacles and regions
for i in range(100):
    action = np.array([0.1])  # Slight right turn
    obs, reward, terminated, truncated, _ = env.step(action)
    time.sleep(0.05)
    if terminated or truncated:
        obs, _ = env.reset()

env.close()