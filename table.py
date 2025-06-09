from train import run_experiment

# Runs training for various M and N, saves loss 
def run_loss_grid(M_values, N_values):
    results = {}
    M_vals, N_vals, Losses = [], [], []
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
    return M_vals, N_vals, Losses, results