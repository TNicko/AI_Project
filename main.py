import sys
import numpy as np
import math
import random
import gym_grid
import gym

env = gym.make("Grid-v0", map_name="5x5")
observation = env.reset()
T = 20

for _ in range(T):
    action = env.action_space.sample()
    env.render()
    observation, reward, term, trunc, info = env.step(action)
    # print(observation, reward, term, info)
    if term:
        observation = env.reset()

env.close()