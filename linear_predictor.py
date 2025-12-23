#!/usr/bin/env python3
"""
2nd-Order Linear Predictor using LMS and RLS Algorithms

This script performs 2nd-order linear prediction on a signal from data5.mat
using both LMS (Least Mean Squares) and RLS (Recursive Least Squares) algorithms.
"""

import numpy as np
import scipy.io
import matplotlib.pyplot as plt


def load_signal(filename='data5.mat'):
    """
    Load signal from .mat file.
    
    Args:
        filename: Path to the .mat file
        
    Returns:
        1D numpy array containing the signal
    """
    # Load the mat file
    data = scipy.io.loadmat(filename)
    
    # Find the signal variable (exclude metadata keys)
    signal_key = None
    for key in data.keys():
        if not key.startswith('__'):
            signal_key = key
            break
    
    if signal_key is None:
        raise ValueError("No signal variable found in .mat file")
    
    # Extract and flatten the signal
    signal = data[signal_key].flatten()
    
    print(f"Loaded signal '{signal_key}' with {len(signal)} samples")
    print(f"Signal dtype: {signal.dtype}")
    
    return signal


def lms_predictor(signal, order=2, mu=0.0001, num_iterations=None):
    """
    2nd-order LMS (Least Mean Squares) predictor.
    
    Implements: x_hat(n) = w1*x(n-1) + w2*x(n-2)
    Update rule: w(n+1) = w(n) + mu * e(n) * conj(x(n))
    
    Args:
        signal: Input signal (1D array)
        order: Order of the predictor (default=2)
        mu: Step size / learning rate
        num_iterations: Number of iterations (default=len(signal)-order)
        
    Returns:
        weights_history: Array of weight values over time (shape: [iterations, order])
        errors: Prediction errors over time
        predictions: Predicted values
    """
    if num_iterations is None:
        num_iterations = len(signal) - order
    
    # Initialize weights to zero
    weights = np.zeros(order, dtype=signal.dtype)
    
    # Storage for history
    weights_history = np.zeros((num_iterations, order), dtype=signal.dtype)
    errors = np.zeros(num_iterations, dtype=signal.dtype)
    predictions = np.zeros(num_iterations, dtype=signal.dtype)
    
    # LMS algorithm
    for n in range(order, order + num_iterations):
        # Get input vector [x(n-1), x(n-2)]
        x = signal[n-order:n][::-1]  # [x(n-1), x(n-2), ...]
        
        # Prediction
        x_hat = np.dot(weights, x)
        predictions[n - order] = x_hat
        
        # Error
        e = signal[n] - x_hat
        errors[n - order] = e
        
        # Store weights
        weights_history[n - order] = weights.copy()
        
        # Update weights: w(n+1) = w(n) + mu * e(n) * conj(x(n))
        weights = weights + mu * e * np.conj(x)
    
    return weights_history, errors, predictions


def rls_predictor(signal, order=2, lambda_factor=0.99, delta=100.0, num_iterations=None):
    """
    2nd-order RLS (Recursive Least Squares) predictor.
    
    Implements: x_hat(n) = w1*x(n-1) + w2*x(n-2)
    
    Args:
        signal: Input signal (1D array)
        order: Order of the predictor (default=2)
        lambda_factor: Forgetting factor (0 < lambda <= 1)
        delta: Initialization parameter for P matrix (larger for complex signals)
        num_iterations: Number of iterations (default=len(signal)-order)
        
    Returns:
        weights_history: Array of weight values over time (shape: [iterations, order])
        errors: Prediction errors over time
        predictions: Predicted values
    """
    if num_iterations is None:
        num_iterations = len(signal) - order
    
    # Initialize weights to zero
    weights = np.zeros(order, dtype=signal.dtype)
    
    # Initialize inverse correlation matrix P (larger delta for stability with complex signals)
    P = np.eye(order, dtype=signal.dtype) * delta
    
    # Storage for history
    weights_history = np.zeros((num_iterations, order), dtype=signal.dtype)
    errors = np.zeros(num_iterations, dtype=signal.dtype)
    predictions = np.zeros(num_iterations, dtype=signal.dtype)
    
    # RLS algorithm
    for n in range(order, order + num_iterations):
        # Get input vector [x(n-1), x(n-2)]
        x = signal[n-order:n][::-1]  # [x(n-1), x(n-2), ...]
        
        # Prediction
        x_hat = np.dot(weights, x)
        predictions[n - order] = x_hat
        
        # A priori error
        e = signal[n] - x_hat
        errors[n - order] = e
        
        # Store weights
        weights_history[n - order] = weights.copy()
        
        # Compute gain vector using proper RLS formula for complex signals
        Px = np.dot(P, np.conj(x))
        denominator = lambda_factor + np.dot(x, Px)
        # Add small regularization to prevent numerical instability
        k = Px / (denominator + 1e-8)
        
        # Update weights
        weights = weights + k * e
        
        # Update inverse correlation matrix (Joseph form for numerical stability)
        P = (P - np.outer(k, np.dot(x, P))) / lambda_factor
    
    return weights_history, errors, predictions


def plot_learning_curves(errors_lms, errors_rls, filename='learning_curves.png'):
    """
    Plot learning curves (squared prediction error) for LMS and RLS.
    
    Args:
        errors_lms: Prediction errors from LMS
        errors_rls: Prediction errors from RLS
        filename: Output filename
    """
    # Compute squared errors
    squared_errors_lms = np.abs(errors_lms) ** 2
    squared_errors_rls = np.abs(errors_rls) ** 2
    
    # Smooth the curves using moving average for better visibility
    window_size = 50
    def moving_average(data, window):
        return np.convolve(data, np.ones(window)/window, mode='valid')
    
    smoothed_lms = moving_average(squared_errors_lms, window_size)
    smoothed_rls = moving_average(squared_errors_rls, window_size)
    
    # Create figure
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Raw squared errors (log scale)
    plt.subplot(1, 2, 1)
    plt.semilogy(squared_errors_lms, alpha=0.3, label='LMS (raw)', color='blue')
    plt.semilogy(squared_errors_rls, alpha=0.3, label='RLS (raw)', color='red')
    plt.xlabel('Iteration')
    plt.ylabel('Squared Error (log scale)')
    plt.title('Learning Curves - Raw Data')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Smoothed curves
    plt.subplot(1, 2, 2)
    plt.plot(range(window_size-1, len(squared_errors_lms)), smoothed_lms, 
             label='LMS (smoothed)', color='blue', linewidth=2)
    plt.plot(range(window_size-1, len(squared_errors_rls)), smoothed_rls, 
             label='RLS (smoothed)', color='red', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Squared Error')
    plt.title(f'Learning Curves - Smoothed (window={window_size})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved learning curves to {filename}")
    plt.close()


def plot_weight_convergence(weights_lms, weights_rls, filename='weight_convergence.png'):
    """
    Plot weight convergence trajectories for LMS and RLS.
    
    Args:
        weights_lms: Weight history from LMS (shape: [iterations, order])
        weights_rls: Weight history from RLS (shape: [iterations, order])
        filename: Output filename
    """
    order = weights_lms.shape[1]
    
    # Create figure with subplots for each weight
    fig, axes = plt.subplots(order, 2, figsize=(14, 4*order))
    
    if order == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(order):
        # Plot real part
        ax_real = axes[i, 0]
        ax_real.plot(np.real(weights_lms[:, i]), label=f'LMS w{i+1} (real)', 
                     color='blue', alpha=0.7)
        ax_real.plot(np.real(weights_rls[:, i]), label=f'RLS w{i+1} (real)', 
                     color='red', alpha=0.7)
        ax_real.set_xlabel('Iteration')
        ax_real.set_ylabel(f'w{i+1} (Real Part)')
        ax_real.set_title(f'Weight w{i+1} Convergence - Real Part')
        ax_real.legend()
        ax_real.grid(True, alpha=0.3)
        
        # Plot imaginary part
        ax_imag = axes[i, 1]
        ax_imag.plot(np.imag(weights_lms[:, i]), label=f'LMS w{i+1} (imag)', 
                     color='blue', alpha=0.7)
        ax_imag.plot(np.imag(weights_rls[:, i]), label=f'RLS w{i+1} (imag)', 
                     color='red', alpha=0.7)
        ax_imag.set_xlabel('Iteration')
        ax_imag.set_ylabel(f'w{i+1} (Imaginary Part)')
        ax_imag.set_title(f'Weight w{i+1} Convergence - Imaginary Part')
        ax_imag.legend()
        ax_imag.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved weight convergence to {filename}")
    plt.close()


def main():
    """Main function to run the linear predictor analysis."""
    print("=" * 60)
    print("2nd-Order Linear Predictor: LMS vs RLS")
    print("=" * 60)
    
    # Load signal
    signal = load_signal('data5.mat')
    
    # Parameters
    order = 2
    mu = 0.0001  # LMS step size (small for complex signals)
    lambda_factor = 0.99  # RLS forgetting factor
    delta = 100.0  # RLS initialization parameter (larger for complex signals)
    
    print(f"\nParameters:")
    print(f"  Order: {order}")
    print(f"  LMS step size (mu): {mu}")
    print(f"  RLS forgetting factor (lambda): {lambda_factor}")
    print(f"  RLS delta: {delta}")
    
    # Run LMS
    print("\nRunning LMS algorithm...")
    weights_lms, errors_lms, predictions_lms = lms_predictor(
        signal, order=order, mu=mu
    )
    mse_lms = np.mean(np.abs(errors_lms) ** 2)
    print(f"  LMS MSE: {mse_lms:.6f}")
    
    # Run RLS
    print("\nRunning RLS algorithm...")
    weights_rls, errors_rls, predictions_rls = rls_predictor(
        signal, order=order, lambda_factor=lambda_factor, delta=delta
    )
    mse_rls = np.mean(np.abs(errors_rls) ** 2)
    print(f"  RLS MSE: {mse_rls:.6f}")
    
    # Create plots
    print("\nGenerating plots...")
    plot_learning_curves(errors_lms, errors_rls)
    plot_weight_convergence(weights_lms, weights_rls)
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)
    
    # Print final weights
    print("\nFinal weights:")
    print(f"  LMS: w1={weights_lms[-1, 0]:.4f}, w2={weights_lms[-1, 1]:.4f}")
    print(f"  RLS: w1={weights_rls[-1, 0]:.4f}, w2={weights_rls[-1, 1]:.4f}")


if __name__ == "__main__":
    main()
