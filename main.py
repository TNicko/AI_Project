import sys
import numpy as np
import gym_grid.models as gmodels
import math
import random
import gym_grid
import gym

env = gym.make("Grid-v0", map_name="3x3")
episodes = 10
states = env.observation_space.n
actions = env.action_space.n

#list to contain total rewards
rList = []

# Set learning params
lr = .8
y = .95

#Initialize table with all zeros
Q = np.zeros([states, actions])

for episode in range(episodes):
    s = env.reset()
    print()
    terminate = False
    count = 0
    total_reward = 0

    while not terminate:
        env.render()

        #Choose an action by greedily picking from Q table
        a = np.argmax(Q[s,:] + np.random.randn(1, actions)*(1./(episode+1)))

        #Get new state and reward from environment
        s1, reward, terminate, trunc, info = env.step(a)

        #Update Q-Table with new knowledge
        Q[s,a] = Q[s,a] + lr*(reward + y*np.max(Q[s1,:]) - Q[s,a])

        count += 1
        total_reward += reward
        s = s1

        if terminate == True:
            break
    print(f"Episode: {episode}, Steps: {count}, Reward: {total_reward}")
    # print(Q)
    rList.append(total_reward)

env.close()