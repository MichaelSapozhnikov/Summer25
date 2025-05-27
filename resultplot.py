import matplotlib.pyplot as plt

def plot_scatter(M_vals, N_vals, Losses):
    plt.figure(figsize=(9, 6))
    scatter = plt.scatter(M_vals, N_vals, c=Losses, cmap='plasma', s=80)
    plt.xlabel("Model Complexity (M)")
    plt.ylabel("Data Size (N)")
    plt.title("Final Loss (MSE) by M and N")
    plt.colorbar(scatter, label="Final Loss (MSE)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()