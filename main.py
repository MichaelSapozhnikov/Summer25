# Summary: training trignometric model to approximate hidden function with M paremters, N training pts

# torch library part of PyTorch, a machine learning framework 
import torch
# import plotting functions
from resultplot import plot_scatter
from plot_hidden_v_prediction import plot_scatter_true_vs_pred
from plot_comparison import plot_true_vs_learned_3d

# define true function we are trying to approximate
def hidden_function(x, y):
    return x**2 + 2 * y**2

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


# defines new function using learned paremters from training run
def learned_function(x_input, y_input):
    XZ = x_input[:, None] * z1[None, :] + y_input[:, None] * z2[None, :]
    return torch.cos(XZ) @ c + torch.sin(XZ) @ d

# empty lists to collect data on M, N, and loss
M_vals = []
N_vals = []
Losses = []
# values of M and N to test
M_values = list(range(2, 22, 2))     # 2, 4, ..., 20 (10 values)
N_values = list(range(50, 301, 50))  # 50, 100, ..., 300 (6 values)

# Runs training for various M and N, saves loss 
results = {}
for M in M_values:
    for N in N_values:
        _, _, _, _, loss = run_experiment(M, N)
        results[(M, N)] = loss
        M_vals.append(M)
        N_vals.append(N)
        Losses.append(loss)

# Print header
print("\nLoss Table (rows = M, columns = N):")
header = "     " + "".join(f"N={N:<7}" for N in N_values)
print(header)
for M in M_values:
    row = f"M={M:<2} " + "".join(f"{results[(M, N)]:<8.4f}" for N in N_values)
    print(row)
   
# uses a fixed M and N for one-time training run used for visualizations
M_plot = 10
N_plot = 50
# trains model given fixed values
c, d, z1, z2, _ = run_experiment(M_plot, N_plot)

# 3D scatterplot of M, N, Loss
## plot_scatter(M_vals, N_vals, Losses)
# Plot comparing hidden function and model's learned function
## plot_scatter_true_vs_pred(hidden_function, learned_function)

# plot the true vs learned function in 3D
## plot_true_vs_learned_3d(hidden_function, learned_function)
print("done")