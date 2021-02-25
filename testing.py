#testing the AI

import os
import numpy as np
import random as rn
import environment
from keras.models import load_model

#Seeds
os.environ["PYTHONHASHSEED"] = '0'
np.random.seed(42)
rn.seed(12345)

#Setting the parameters
# epsilon is exploration parameter
number_actions=5
direction_boundary = (number_actions-1)/2
temperature_step = 1.5

# building the environment by simply creating an object of the environment class
env = environment.Environment(optimal_temperature = (18.0,24.0), initial_month = 0, initial_number_users = 20, initial_rate_data = 30)

#loading a pretrained model

model = load_model("model.h5")

# choosing the mode
train = False

# Running a 1 year simulation in inference mode
env.train = train
current_state,_,_ = env.observe()
for timestep in range(0, 12 * 30 * 24 * 60):
    q_values = model.predict(current_state)
    action = np.argmax(q_values[0])
    if (action - direction_boundary < 0):
        direction = -1
    else:
        direction = 1
    energy_ai = abs(action - direction_boundary) + temperature_step
    next_state, reward, game_over = env.update_env(direction, energy_ai, int(timestep/(30*24*60)))
    current_state = next_state

#printing the training results
print("\n")
print("Total energy spent with an AI: {:.0f}".format(env.total_energy_ai))
print("Total energy spent without an AI: {:.0f}". format(env.total_energy_noai))
print("ENERGY SAVED: {:.0f} %".format((env.total_energy_noai - env.total_energy_ai) / env.total_energy_noai * 100))










