#artificial intelligence for buisness - case study 2
#Training the AI


import os
import nummpy as np
import random as rn
import environment
import brain
import dqn

#Seeds
os.environ["PYTHONHASHSEED"] = '0'
np.random.seed(42)
rn.seed(12345)

#Setting the parameters
# epsilon is exploration parameter
epsilon = 0.3
number_actions=5
direction_boundary = (number_actions -1)/2
number_epochs = 1000
max_memory = 3000
batch_size = 512
temperature_step = 1.5

# building the environment by simply creating an object of the environment class
env = environment.Environment(optimal_temperature = (18.0,24.0), initial_month = 0, initial_number_users = 20, inital_rate_data = 30)
# building the brain by simpy creating an object of the brain class
brain = brain.Brain(learning_rate = 0.00001, number_actions = number_actions)
# building the DQN model by simpy creating an object of the DQN class
dqn = dqn.DQN(max_memory = max_memory, discount = 0.9)
# choosing the mode
train = True

