import torch
from model import model
from functions import hidden_function

# function trains the model with M (model complexity; # trignometric terms) and N (# of training data points)
# epochs = how many times to repeat training 
def run_experiment(M, N, epochs=500):
    # generating N random values for x and y between 0 and 1
    x = torch.rand(N)
    y = torch.rand(N)
    # computes actual values of hidden function at (x,y) pts 
    f_vals = hidden_function(x, y)

    # c and d are weights of cosine/sine terms
    # z1, z2 are frequencies for how to mix x/y
    # initializing each as a vector of M random values from a normal distribution 
    # requires_grad=True tracks changes to the variable so gradients can be computed later
    c = torch.randn(M, requires_grad=True)
    d = torch.randn(M, requires_grad=True)
    z1 = torch.randn(M, requires_grad=True)
    z2 = torch.randn(M, requires_grad=True)

    # Adam = Adapative Moment Estimation = gradient-based optimization algorithm in PyTorch
    # Adam updates the parameters to reduce error over time 
    # lr = 0.01 is the learning rate (how large parameter updates are in each step)
    optimizer = torch.optim.Adam([c, d, z1, z2], lr=0.01)

    # Runs training for number of set epochs
    for _ in range(epochs):
        optimizer.zero_grad() # clearing old gradients
        pred = model(x, y, c, d, z1, z2) # computing prediction at curr x,y & parameters
        loss = torch.mean((pred - f_vals) ** 2) # computing MSE 
        loss.backward() # commputes gradients of the loss w.r.t. each parameter
        optimizer.step() # updates parameters using gradients

    # Returns trained paremters and final loss
    # detach() gives raw values no longer connected to gradient tracking, .item() turns tensor to number
    return c.detach(), d.detach(), z1.detach(), z2.detach(), loss.item()