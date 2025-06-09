import torch
from model import model

# define true function we are trying to approximate
def hidden_function(x, y):
    p = 1  # manually change this to 1 or 0.25 as needed
    return ((x)**2 + 2 * (y)**2) ** p

# defines function that takes learned weights (c,d) and learned frequency vectors (z1, z2)
# returns new fully configured function withtrained parameters
def get_learned_function(c, d, z1, z2):
    # defines another function that evaluates learned function on pts (point is not to rely on global state)
    def learned(x_input, y_input):
        return model(x_input, y_input, c, d, z1, z2)
    return learned