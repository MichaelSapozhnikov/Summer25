# torch library part of PyTorch, a machine learning framework 
import torch
# model() is the approximating function with tunable parameters
# x and y : are input coordinates (vectors)
# c, d, z1, z2 : are model parameters that will be learned 
# Define model: sum_j c_j * cos(z1_j x + z2_j y) + d_j * sin(z1_j x + z2_j y)
def model(x, y, c, d, z1, z2):
    #XZ is an N x M matrix with each entry a combination of z1, z2 with x, y
    XZ = x[:, None] * z1[None, :] + y[:, None] * z2[None, :] 
    return torch.cos(XZ) @ c + torch.sin(XZ) @ d 
    # returns a sum of matrix of cosines multipled by vector with weight c plus sum of matrix of sines multiplied by vector of weight d 
    # sum represents an estimated value for the function at each (x_i, y_i)