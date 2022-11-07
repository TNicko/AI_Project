import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam


# class RLModel():

#     def __init__(self, states, actions):
        
#         self.states = states
#         self.actions = actions

#         print(states, actions)

def build_model(states, actions):
    model = Sequential()
    model.add(Dense(24, activation='relu', input_shape=states))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    model = build_model(states, actions)
    model.summary()
    return model



