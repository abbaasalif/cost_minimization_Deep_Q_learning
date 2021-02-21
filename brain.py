#building the brain

# importing the libraries

from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam


# building the brain

class Brain(object):

    def __init__(self, learning_rate = 0.001, number_actions = 5):
        self.learning_rate = learning_rate
        states = Input(shape = (3,))
        x = Dense(64, activation='sigmoid')(states)
        y = Dense(32, activation='sigmoid')(x)
        q_values = Dense(number_actions, activation="softmax")(y)
        self.model = Model(inputs=states, outputs=q_values)
        self.model.compile(loss = "mse", optimizer=Adam(lr = learning_rate))
        