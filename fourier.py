# Summary: uses fixed frequencies 2pij, 2pik, instead of learning z1, z2
# Coefficients computed with formula instead of gradient descent
# No optimization loop, direct calculation instead

import torch
import math 

# computes classical Fourier series coefficients
# takes in input coordinates (N pts), function values at these N pts, with M terms
# returns coefficient matrices (M x M each)
def compute_fourier_coefficients(x_data, y_data, f_data, M):

    N = len(x_data)

    # Initialize coefficient matrices, two MxM matrices to store coefficients 
    # c_coeffs stores cosine coefficients c, d_coeffs stores sine coefficients d
    # M x M because classical Fourier uses frequencies (j,k) where j and k range from 1 to M
    c_coeffs = torch.zeros(M, M)
    d_coeffs = torch.zeros(M, M)

    # Compute coefficients using data approximation
    # Double loop to calculate coeffients for each frequency pair (j,k)
    for j in range(1, M + 1):  # j = 1, 2, ..., M
        for k in range(1, M + 1):  # k = 1, 2, ..., M
            # Approximate integrals using data points
            # c_{j,k} ≈ (1/N) * Σ f(x_h, y_h) * cos(2πjx_h + 2πky_h)
            # Compute 2pijx + 2piky for all data points, compute average of integrals to approximate integral
            cos_terms = torch.cos(2 * math.pi * j * x_data + 2 * math.pi * k * y_data)
            c_coeffs[j-1, k-1] = torch.mean(f_data * cos_terms)
            
            # d_{j,k} ≈ (1/N) * Σ f(x_h, y_h) * sin(2πjx_h + 2πky_h)
            sin_terms = torch.sin(2 * math.pi * j * x_data + 2 * math.pi * k * y_data)
            d_coeffs[j-1, k-1] = torch.mean(f_data * sin_terms)
    # Return computed coefficient matrices
    return c_coeffs, d_coeffs

# Evaluates Fourier at given pts
# Takes input coordinates x, y; coefficient matrices c_coeffs, d_coeffs
def fourier_series_model(x, y, c_coeffs, d_coeffs):
    # gets M from coefficient matrix size
    M = c_coeffs.shape[0]
    # creates result tensor with same shape as x, initialized to 0s
    result = torch.zeros_like(x, dtype=torch.float32)
    
    # double loop over all frequency pairs (j,k)
    for j in range(1, M + 1):
        for k in range(1, M + 1):
            # Add c_{j,k} * cos(2πjx + 2πky) term
            cos_term = c_coeffs[j-1, k-1] * torch.cos(2 * math.pi * j * x + 2 * math.pi * k * y)
            result += cos_term
            
            # Add d_{j,k} * sin(2πjx + 2πky) term  
            sin_term = d_coeffs[j-1, k-1] * torch.sin(2 * math.pi * j * x + 2 * math.pi * k * y)
            result += sin_term
    # returns final Fourier series evaluation
    return result

# Returns function that evaluates Fourier series with learned coefficients
# Point is to create closure to remember learned coefficients, return function to call later
def get_fourier_function(c_coeffs, d_coeffs):
    def fourier_func(x_input, y_input):
        return fourier_series_model(x_input, y_input, c_coeffs, d_coeffs)
    return fourier_func

# Trains Fourier approximation
# Takes in number of frequency terms and number of data pts
# Returns learned coefficient matrices and mse error on data
def train_fourier_series(M, N):
    from functions import hidden_function
    
    # Generate random training data
    x_data = torch.rand(N)
    y_data = torch.rand(N)
    f_data = hidden_function(x_data, y_data)
    
    # Compute Fourier coefficients
    c_coeffs, d_coeffs = compute_fourier_coefficients(x_data, y_data, f_data, M)
    
    # Evaluate trained model on training data to compute loss
    predictions = fourier_series_model(x_data, y_data, c_coeffs, d_coeffs)
    mse_loss = torch.mean((predictions - f_data) ** 2).item()
    
    # Returns learned coefficients and loss
    return c_coeffs, d_coeffs, mse_loss