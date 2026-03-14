from scipy.optimize import minimize, fsolve
from langchain.tools import tool
from scipy.integrate import quad
from scipy.stats import binom, poisson, chi2, t, norm, gaussian_kde, zscore
from scipy.linalg import inv, solve, eig
from scipy.interpolate import interp1d
from scipy.spatial.distance import euclidean, pdist
from scipy.signal import savgol_filter, find_peaks
from scipy.fft import fft

import numpy as np
import sympy as sp

@tool
def minimize_function(function: str, guess: float):
    """
    Finds minimum of a mathematical function.
    Automatically detects the variable used in the string.
    """
    expr = sp.sympify(function)
    
    vars = list(expr.free_symbols)
    if not vars:
        return "Error: No variable found in function."
    
    f = sp.lambdify(vars[0], expr, "numpy")
    result = minimize(f, guess)

    return {
        "minimum_x": float(result.x[0]),
        "minimum_value": float(result.fun)
    }

@tool
def solve_nonlinear_equation(equation: str, guess: float):
    """
    Solves nonlinear equation numerically (f(x) = 0).
    """
    expr = sp.sympify(equation)
    vars = list(expr.free_symbols)
    if not vars:
        return "Error: No variable found in equation."

    f = sp.lambdify(vars[0], expr, "numpy")
    root = fsolve(f, guess)

    return float(root[0])

@tool
def solve_equation_system(equations: list):
    """
    Solves system of algebraic equations. 
    Dynamically detects all variables (e.g., x, y, z or a, b, c).
    
    Example: ["x + y = 20", "90*x + 150*y = 2100"]
    """
    sympy_eqs = []
    all_symbols = set()

    for eq in equations:
        if "=" not in eq:
            continue
        left, right = eq.split("=")
        lhs = sp.sympify(left.strip())
        rhs = sp.sympify(right.strip())
        expr = lhs - rhs
        sympy_eqs.append(expr)
        all_symbols.update(expr.free_symbols)

    
    symbol_list = sorted(list(all_symbols), key=lambda s: s.name)
    solution = sp.solve(sympy_eqs, symbol_list)

   
    if isinstance(solution, dict):
        return {str(k): float(v) if v.is_number else str(v) for k, v in solution.items()}
    return str(solution)

@tool
def integrate_function(function: str, a: float, b: float):
    """
    Computes definite integral of a function.
    """
    expr = sp.sympify(function)
    vars = list(expr.free_symbols)
    if not vars:
        return "Error: No variable found."
    
    f = sp.lambdify(vars[0], expr, "numpy")
    result, _ = quad(f, a, b)
    return float(result)

@tool
def normal_distribution_cdf(x: float):
    """Normal distribution cumulative probability."""
    return float(norm.cdf(x))

@tool
def binomial_probability(n: int, p: float, k: int):
    """Probability of k successes in n trials."""
    return float(binom.pmf(k, n, p))

@tool
def poisson_probability(mu: float, k: int):
    """Poisson probability."""
    return float(poisson.pmf(k, mu))

@tool
def t_distribution_probability(x: float, df: int):
    """T distribution cumulative probability."""
    return float(t.cdf(x, df))

@tool
def chi_square_probability(x: float, df: int):
    """Chi-square cumulative probability."""
    return float(chi2.cdf(x, df))

@tool
def matrix_inverse(A: list):
    """Computes matrix inverse."""
    return inv(np.array(A)).tolist()

@tool
def eigen_values(A: list):
    """Computes eigenvalues and eigenvectors."""
    values, vectors = eig(np.array(A))
    return {
        "eigenvalues": values.tolist(),
        "eigenvectors": vectors.tolist()
    }

@tool
def solve_linear_system(A: list, b: list):
    """Solves Ax=b linear system."""
    solution = solve(np.array(A), np.array(b))
    return solution.tolist()

@tool
def interpolate_data(x: list, y: list, value: float):
    """Estimates missing data points."""
    f = interp1d(x, y)
    return float(f(value))

@tool
def euclidean_distance(p1: list, p2: list):
    """Euclidean distance between two points."""
    return float(euclidean(p1, p2))

@tool
def pairwise_distance(points: list):
    """Pairwise distance between multiple points."""
    return pdist(points).tolist()

@tool
def detect_peaks(data: list):
    """Detects peaks in signal data."""
    peaks, _ = find_peaks(data)
    return peaks.tolist()

@tool
def smooth_signal(data: list):
    """Smooth noisy signal using Savitzky-Golay filter."""
    return savgol_filter(data, 5, 2).tolist()

@tool
def fourier_transform(data: list):
    """Fourier transform of signal."""
    
    res = fft(data)
    return [[float(val.real), float(val.imag)] for val in res]

@tool
def statistical_mean(data: list):
    """Average of dataset."""
    return float(np.mean(data))

@tool
def standard_deviation(data: list):
    """Standard deviation of dataset."""
    return float(np.std(data))

@tool
def compute_zscore(data: list):
    """Standardized values of dataset."""
    return zscore(data).tolist()

@tool
def density_estimation(data: list, point: float):
    """Kernel density estimation."""
    kde = gaussian_kde(data)
    return float(kde(point))

scipy_tools = [
    minimize_function,
    solve_nonlinear_equation,
    solve_equation_system,
    integrate_function,
    normal_distribution_cdf,
    binomial_probability,
    poisson_probability,
    t_distribution_probability,
    chi_square_probability,
    matrix_inverse,
    eigen_values,
    solve_linear_system,
    interpolate_data,
    euclidean_distance,
    pairwise_distance,
    detect_peaks,
    smooth_signal,
    fourier_transform,
    statistical_mean,
    standard_deviation,
    compute_zscore,
    density_estimation,
]