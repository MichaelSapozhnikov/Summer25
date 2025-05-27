import torch
from resultplot import plot_scatter
from plot_hidden_v_prediction import plot_scatter_true_vs_pred

# Define hidden function
def hidden_function(x, y):
    return x**2 + 2 * y**2

# Define model: sum_j c_j * cos(z1_j x + z2_j y) + d_j * sin(z1_j x + z2_j y)
def model(x, y, c, d, z1, z2):
    XZ = x[:, None] * z1[None, :] + y[:, None] * z2[None, :]
    return torch.cos(XZ) @ c + torch.sin(XZ) @ d

# Training function for a single (M, N)
def run_experiment(M, N, epochs=2000):
    # Generate data
    x = torch.rand(N)
    y = torch.rand(N)
    f_vals = hidden_function(x, y)

    # Initialize trainable parameters
    c = torch.randn(M, requires_grad=True)
    d = torch.randn(M, requires_grad=True)
    z1 = torch.randn(M, requires_grad=True)
    z2 = torch.randn(M, requires_grad=True)

    optimizer = torch.optim.Adam([c, d, z1, z2], lr=0.01)

    # Train
    for _ in range(epochs):
        optimizer.zero_grad()
        pred = model(x, y, c, d, z1, z2)
        loss = torch.mean((pred - f_vals) ** 2)
        loss.backward()
        optimizer.step()

    return c.detach(), d.detach(), z1.detach(), z2.detach(), loss.item()

# Using a fixed M and N for visualization
M_plot = 10
N_plot = 200

# Train a model on that M and N
c, d, z1, z2, _ = run_experiment(M_plot, N_plot)

def learned_function(x_input, y_input):
    XZ = x_input[:, None] * z1[None, :] + y_input[:, None] * z2[None, :]
    return torch.cos(XZ) @ c + torch.sin(XZ) @ d

# Run experiments for different M and N
M_vals = []
N_vals = []
Losses = []

M_values = list(range(2, 22, 2))     # 2, 4, ..., 20 (10 values)
N_values = list(range(50, 301, 50))  # 50, 100, ..., 300 (6 values)

"""
for M in M_values:
    for N in N_values:
        loss = run_experiment(M, N)
        M_vals.append(M)
        N_vals.append(N)
        Losses.append(loss)
"""   

## plot_scatter(M_vals, N_vals, Losses)
plot_scatter_true_vs_pred(hidden_function, learned_function)
