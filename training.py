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

# training the AI
env.train = train
model = brain.model
if (env.train):
    for epoch in range(1, number_epochs):
        #initializing all the variables of both the environment and the training loop
        total_reward = 0
        loss = 0.0
        new_month = np.random.randint(0,12)
        env.reset(new_month - new_month)
        game_over = False
        current_state, _, _ = env.observe()
        #1minute
        timestep = 0
        while ((not game_over) and (timestep <= 5 * 30 * 24 *60)):
            # playing the next action by exploration
            if np.random.rand() <= epsilon:
                action = np.random.randint(0,number_actions)
                if (action - direction_boundary < 0):
                    direction = -1
                else:
                    direction = 1
                energy_ai = abs(action - direction_boundary) + temperature_step
            # playing the next action by inference
            

            #updating the environment and reaching the next state

            # gathering in two separate batches the inputs and the targets

            # computing the loss over the two whole batches of inputs and targets







