import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Evaluates both true function and learned approximation over grid of points in [0,1] [0,1\ region
# Plots both surfaces side by side in 3D space
def plot_true_vs_learned_3d(hidden_function, learned_function):
    # Create a grid over [0,1]^2 of test points
    grid_size = 20 
    x_vals = torch.linspace(0, 1, grid_size)
    y_vals = torch.linspace(0, 1, grid_size)
    X, Y = torch.meshgrid(x_vals, y_vals, indexing='ij') # creates 2d tensors of (x,y) pairs to evaluate both functions

    # Flatten grid for learned_funexpction which expects vector inputs
    X_flat = X.reshape(-1)
    Y_flat = Y.reshape(-1)

    # Evaluate both functions on each (x,y) in meshgrid
    # Converts tensor into array so matplotlib can plot
    Z_true = hidden_function(X, Y).numpy()
    Z_learned = learned_function(X_flat, Y_flat).reshape(grid_size, grid_size).detach().numpy()

    # Plotting
    fig = plt.figure(figsize=(14, 6))

    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.plot_surface(X.numpy(), Y.numpy(), Z_true, cmap='viridis')
    ax1.set_title("Hidden Function")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("f(x, y)")

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.plot_surface(X.numpy(), Y.numpy(), Z_learned, cmap='plasma')
    ax2.set_title("Learned Approximation")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("Prediction")

    plt.tight_layout()
    plt.show()