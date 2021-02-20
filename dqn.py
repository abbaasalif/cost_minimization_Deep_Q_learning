#artificial intelligence for buisness - case study 2
#implementing Deep q-learning with experience replay

# importing the libraries
import numpy as np

class DQN(object):

    #Introduction and initialization of all parameters and variables of Deep Q Network
    def __init__(self, max_memory = 100, discount = 0.9):
        self.memory = list()
        self.max_memory = max_memory
        self.discount = discount

    #Making a method that builds the memory in Experience Replay

    # Making a method that builds two batches of 10 inputs and 10 targets by extracting 10 transitions
