#Artificial Intelligence for Buisness - Case Study 2
#Building the environment

#importing the libraries
import numpy as np

# BULIDING THE ENVIRONMENT IN A CLASS

class Environment(object):
    
    
    #INTRODUCING AND INITIALIZING ALL THE PARAMETERS AND VARIABLES OFF THE ENVIRONMENT
    def __init__(self, optimal_temperature = (18.0,24.0), initial_month = 0, initial_number_users = 10, inital_rate_data = 60):
        
        self.monthly_atmospheric_temperatures = [1.0,5.0,7.0,10.0,11.0, 20.0, 23.0, 24.0 , 22.0 , 10.0, 5.0, 1.0]
        self.initial_month = initial_month
        self.atmospheric_temperature = self.monthly_atmospheric_temperatures
        self.optimal_temperature = optimal_temperature
        self.min_temperature = -20
        self.max_temperature = 80
        self.min_number_users = 10
        self.max_number_users = 100
        self.max_update_users = 5
        self.min_rate_data=20
        self.max_rate_data = 300
        self.max_update_data = 10
        self.initial_number_users = initial_number_users
        self.current_number_users = initial_number_users
        self.initial_rate_data = initial_rate_data
        self.current_rate_data = inital_rate_data
        self.intrinsic_temperature = self.atmospheric_temperature + 1.25 * self.current_number_users + 1.25 * self.current_rate_data
        self.temperature_ai = self.intrinsic_temperature
        self.temperature_noai = (self.optimal_temperature[0] + self.optimal_temperature[1])/2.0
        self.total_energy_ai = 0.0
        self.total_energy_noai = 0.0
        self.reward = 0.0
        self.game_over = 0
        self.train = 1
        
    
    #MAKING A METHOD THAT UPDATES THE ENVIRONMENT RIGHT AFTER THE AI PLAYS AN ACTION
    def update_env(self, direction, energy_ai, month):
        #getting the reward 
        #computing the energy spent by the system when there is no Ai
        energy_noai=0
        if(self.temperature_noai < self.optimal_temperature[0]):
            energy_noai = self.optimal_temperature[0] - self.temperature_noai
            self.temperature_noai = self.optimal_temperature[0]
        elif(self.temperature_noai>self.optimal_temperature[1]):
            energy_noai = self.temperature_noai- self.optimal_temperature[1]
            self.temperature_noai = self.optimal_temperature[1]
        #computing the Reward
        self.reward = energy_noai - energy_ai
        #Scaling the reward
        self.reward = 1e-3 * self.reward
        #Getting the next state
        
        #updating the atmospheric temerature
        self.atmospheric_temperature = self.monthly_atmospheric_temperatures[month]

        #updating the number of users
        self.current_number_users += np.random.randint(-self.max_update_users,self.max_update_users)
        if(self.current_number_users > self.max_number_users):
            self.current_number_users = self.max_number_users
        elif(self.current_number_users<self.min_number_users):
            self.current_number_users = self.min_number_users
            
            
        #updating the rate of data     
        self.current_rate_data += np.random.randint(-self.max_update_data,self.max_update_data)
        if(self.current_rate_data > self.max_rate_data):
            self.current_rate_data = self.max_rate_data
        elif(self.current_rate_data < self.min_rate_data):
            self.current_rate_data = self.min_rate_data
        #computing delta of intrinsic temperature
        past_intrinsic_temperature = self.intrinsic_temperature
        self.intrinsic_temperature = self.atmospheric_temperature + 1.25 * self.current_number_users + 1.25 * self.current_rate_data
        delta_intrinsic_temperature = self.intrinsic_temperature - past_intrinsic_temperature
        
        #computing the Delta of  temperature caused by AI
        if (direction == -1):
            delta_temperature_ai = -energy_ai
        elif(direction == 1):
            delta_temperature_ai = energy_ai
        #updating the new servers temperature when there is AI
        self.temperature_ai += delta_intrinsic_temperature + delta_temperature_ai    
        #updating the server tempreature when there is no AI
        self.temperature_noai += delta_intrinsic_temperature 
        
        #Getting game over
        
        if (self.temperature_ai < self.min_temperature):
            if (self.train == 1):
                self.game_over = 1
            else:
                self.temperature_ai = self.optimal_temperature[0]
                self.total_energy_ai += self.optimal_temperature[0] - self.temperature_ai
                
        elif (self.temperature_ai > self.max_temperature):
            if (self.train == 1):
                self.game_over=1
            else:
                self.temperature_ai = self.optimal_temperature[1]
                self.total_energy_ai += self.temperature_ai - self.optimal_temperature[1]
        #updating the scores
        
        #updating the Total energy spent by AI
        self.total_energy_ai += energy_ai      
        #updating the Total energy spent by the server's cooling system where there is no AI
        self.total_energy_noai += energy_noai 
          
        #Scaling the next state
        
        scaled_temperature_ai = (self.temperature_ai - self.min_temperature) / (self.max_temperature - self.min_temperature)
        scaled_number_users = (self.current_number_users - self.min_number_users) / (self.max_number_users - self.min_number_users)
        scaled_rate_data = (self.current_rate_data - self.min_rate_data) / (self.max_rate_data - self.min_rate_data)
        next_state = np.matrix([scaled_temperature_ai, scaled_number_users , scaled_rate_data])

        # Returning to the next state, The reward, and game over
        return next_state, self.reward, self.game_over        
    
    #MAKING A METHOD THAT RESETS THE ENVIRONMENT

    def reset(self, new_month):
        self.atmospheric_temperature = self.monthly_atmospheric_temperatures[new_month]
        self.initial_month = new_month
        self.current_number_users = self.initial_number_users
        self.current_rate_data = self.initial_rate_data
        self.intrinsic_temperature = self.atmospheric_temperature + 1.25 * self.current_number_users + 1.25 * self.current_rate_data
        self.temperature_ai = self.intrinsic_temperature 
        self.temperature_noai = (self.optimal_temperature[0] + self.optimal_temperature[1])/2.0
        self.total_energy_ai = 0.0
        self.total_energy_noai = 0.0
        self.reward = 0.0
        self.game_over = 0
        self.train = 1

    
    #MAKING A METHOD THAT GIVES US ANY TIME THE CURRENT STATE , THE LAST REWARD AND WHETHER THE GAME IS OVER

    def 
