import torch
import math
from train import run_experiment
from fourier import train_fourier_series
from functions import hidden_function, get_learned_function
import matplotlib.pyplot as plt

def compute_mse_error(func1, func2, n_test_points=1000):
    """
    Compute MSE between two functions over [0,1]^2 using random sampling
    This approximates the integral: ∫∫[0,1]² (func1(x,y) - func2(x,y))² dx dy
    """
    # Generate random test points
    x_test = torch.rand(n_test_points)
    y_test = torch.rand(n_test_points)
    
    # Evaluate both functions
    vals1 = func1(x_test, y_test)
    vals2 = func2(x_test, y_test)
    
    # Compute MSE
    mse = torch.mean((vals1 - vals2) ** 2).item()
    return mse

def compare_methods(M, N, n_test_points=1000):
    """
    Compare adaptive frequency method vs classical Fourier series
    Returns: (adaptive_error, fourier_error, training_loss_adaptive, training_loss_fourier)
    """
    print(f"\nComparing methods with M={M} terms, N={N} data points")
    
    # Train adaptive frequency method
    print("Training adaptive frequency method...")
    c, d, z1, z2, adaptive_train_loss = run_experiment(M, N)
    adaptive_func = get_learned_function(c, d, z1, z2)
    
    # Train classical Fourier series
    print("Training classical Fourier series...")
    c_coeffs, d_coeffs, fourier_train_loss = train_fourier_series(M, N)
    
    # Create Fourier function for evaluation
    def fourier_func(x, y):
        from fourier import fourier_series_model
        return fourier_series_model(x, y, c_coeffs, d_coeffs)
    
    # Compute errors against true hidden function
    adaptive_error = compute_mse_error(hidden_function, adaptive_func, n_test_points)
    fourier_error = compute_mse_error(hidden_function, fourier_func, n_test_points)
    
    print(f"Results:")
    print(f"  Adaptive method - Training loss: {adaptive_train_loss:.6f}, Test error: {adaptive_error:.6f}")
    print(f"  Fourier series  - Training loss: {fourier_train_loss:.6f}, Test error: {fourier_error:.6f}")
    
    return adaptive_error, fourier_error, adaptive_train_loss, fourier_train_loss