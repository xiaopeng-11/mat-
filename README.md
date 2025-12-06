# Linear Predictor

This repository contains a Python implementation of 2nd-order linear prediction using LMS (Least Mean Squares) and RLS (Recursive Least Squares) algorithms.

## Overview

The `linear_predictor.py` script performs adaptive 2nd-order linear prediction on a complex-valued signal stored in `data5.mat`. It implements both LMS and RLS algorithms and generates visualization plots comparing their performance.

## Requirements

- Python 3.x
- numpy
- scipy
- matplotlib

Install dependencies with:
```bash
pip install numpy scipy matplotlib
```

## Usage

Simply run the script:
```bash
python linear_predictor.py
```

The script will:
1. Load the signal from `data5.mat`
2. Apply 2nd-order LMS and RLS predictors
3. Generate two visualization plots:
   - `learning_curves.png`: Shows the squared prediction error over iterations
   - `weight_convergence.png`: Shows how the weights (w1, w2) converge over time

## Algorithm Details

### 2nd-Order Linear Predictor
The predictor estimates the current sample based on the previous two samples:
```
x̂(n) = w₁·x(n-1) + w₂·x(n-2)
```

### LMS (Least Mean Squares)
- **Update rule**: `w(n+1) = w(n) + μ·e(n)·conj(x(n))`
- **Step size**: μ = 0.0001
- Adapts gradually with a fixed learning rate

### RLS (Recursive Least Squares)
- **Forgetting factor**: λ = 0.99
- **Initialization**: δ = 100.0
- Converges faster than LMS due to optimal gain computation

## Output

The script displays:
- Signal information (length, data type)
- Algorithm parameters
- Final MSE (Mean Squared Error) for both algorithms
- Final converged weights

### Sample Output
```
============================================================
2nd-Order Linear Predictor: LMS vs RLS
============================================================
Loaded signal 'data5' with 1024 samples
Signal dtype: complex128

Parameters:
  Order: 2
  LMS step size (mu): 0.0001
  RLS forgetting factor (lambda): 0.99
  RLS delta: 100.0

Running LMS algorithm...
  LMS MSE: 27.153432

Running RLS algorithm...
  RLS MSE: 6.963423

Generating plots...
Saved learning curves to learning_curves.png
Saved weight convergence to weight_convergence.png

============================================================
Analysis complete!
============================================================

Final weights:
  LMS: w1=0.2812+1.4270j, w2=0.9101-0.3871j
  RLS: w1=0.3103+1.4507j, w2=0.9084-0.4050j
```

## Visualization

### Learning Curves
Shows the squared prediction error over iterations for both algorithms:
- Left plot: Raw squared errors on a logarithmic scale
- Right plot: Smoothed curves (50-sample moving average)

RLS typically converges faster and achieves lower error than LMS.

### Weight Convergence
Shows the evolution of weights w₁ and w₂ over iterations:
- Separate plots for real and imaginary parts (complex signal)
- Blue lines: LMS weights
- Red lines: RLS weights

RLS weights converge more quickly and with less noise than LMS.

## Features

- ✅ Automatic signal loading and preprocessing
- ✅ Support for complex-valued signals
- ✅ Adaptive LMS algorithm with configurable step size
- ✅ Recursive Least Squares (RLS) with numerical stability
- ✅ Comprehensive visualization with both raw and smoothed curves
- ✅ Performance metrics (MSE) for algorithm comparison
- ✅ High-quality plot generation (150 DPI)

## License

This project is provided as-is for educational and research purposes.
