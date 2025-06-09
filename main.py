# Summary: training trignometric model to approximate hidden function with M paremters, N training pts

# torch library part of PyTorch, a machine learning framework 
import torch
# import plotting functions
from resultplot import plot_scatter
from plot_hidden_v_prediction import plot_scatter_true_vs_pred
from plot_comparison import plot_true_vs_learned_3d
from table import run_loss_grid
from train import run_experiment
from functions import hidden_function, get_learned_function
from comparison import compare_methods, compute_mse_error

# values of M and N to test
M_values = list(range(2, 22, 2))     # 2, 4, ..., 20 (10 values)
N_values = list(range(50, 301, 50))  # 50, 100, ..., 300 (6 values)

# uses a fixed M and N for one-time training run used for visualizations
M_plot, N_plot = 10, 50

# trains model given fixed values
c, d, z1, z2, _ = run_experiment(M_plot, N_plot)
learned_function = get_learned_function(c, d, z1, z2)

print("Comparing methods...")
adaptive_error, fourier_error, _, _ = compare_methods(M_plot, N_plot)
print(f"Our algorithm's error: {adaptive_error}")
print(f"Fourier series error:  {fourier_error}")

# Prints loss table
## M_vals, N_vals, Losses, results = run_loss_grid(M_values, N_values)

# Scatterplot of M, N, Loss
## plot_scatter(M_vals, N_vals, Losses)
# Plot comparing hidden function and model's learned function
## plot_scatter_true_vs_pred(hidden_function, learned_function)

# plot the true vs learned function in 3D
## plot_true_vs_learned_3d(hidden_function, learned_function)
print("done")