import matplotlib.pyplot as plt
import torch

def plot_scatter_true_vs_pred(hidden_func, learned_func, resolution=20):
    # Create grid points in [0, 1] x [0, 1]
    x_grid, y_grid = torch.meshgrid(
        torch.linspace(0, 1, resolution),
        torch.linspace(0, 1, resolution),
        indexing="ij"
    )
    x_flat = x_grid.flatten()
    y_flat = y_grid.flatten()

    # Evaluate both functions on the grid
    true_vals = hidden_func(x_flat, y_flat)
    pred_vals = learned_func(x_flat, y_flat)

    # Scatter plot of predicted vs true
    plt.figure(figsize=(6, 6))
    plt.scatter(true_vals, pred_vals, alpha=0.6, s=10, color='blue')
    plt.plot([true_vals.min(), true_vals.max()], [true_vals.min(), true_vals.max()], 'r--')  # y = x
    plt.xlabel("True Value (f(x, y))")
    plt.ylabel("Predicted Value (p(x, y))")
    plt.title("Prediction vs Hidden Function")
    plt.grid(True)
    plt.axis("equal")
    plt.tight_layout()
    plt.show()
