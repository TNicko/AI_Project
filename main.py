import sys
import numpy as np
import math
import random
import gym_grid
import gym

env = gym.make("Grid-v0", map_name="3x3")
episodes = 3
for episode in range(episodes):
    observation = env.reset()
    terminate = False
    count = 0
    total_reward = 0

    while not terminate:
        action = env.action_space.sample()
        env.render()
        observation, reward, terminate, trunc, info = env.step(action)
        count += 1
        total_reward += reward
        # print(observation, reward, term, info)
    print(f"Episode: {episode}, Steps: {count}, Reward: {total_reward}")

env.close()