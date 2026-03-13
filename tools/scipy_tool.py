from scipy.optimize import minimize
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

    Example:
    function="x**2 + 3*x + 2"
    guess=0
    """

    x = sp.symbols("x")
    expr = sp.sympify(function)
    f = sp.lambdify(x, expr, "numpy")

    result = minimize(f, guess)

    return {
        "minimum_x": float(result.x[0]),
        "minimum_value": float(result.fun)
    }


@tool
def solve_nonlinear_equation(equation: str, guess: float):
    """
    Solves nonlinear equation numerically.

    Example:
    equation="x**3 - x - 2"
    guess=1
    """

    from scipy.optimize import fsolve

    x = sp.symbols("x")
    expr = sp.sympify(equation)

    f = sp.lambdify(x, expr, "numpy")

    root = fsolve(f, guess)

    return float(root[0])


@tool
def solve_equation_system(equations: list):
    """
    Solves system of algebraic equations.

    Example:
    ["a-6*b+6*c=4", "6*a+3*b-3*c=50"]
    """

    a, b, c = sp.symbols("a b c")

    sympy_eqs = []

    for eq in equations:
        left, right = eq.split("=")
        sympy_eqs.append(sp.sympify(left) - sp.sympify(right))

    solution = sp.solve(sympy_eqs, (a, b, c))

    return solution


@tool
def integrate_function(function: str, a: float, b: float):
    """
    Computes definite integral of a function.

    Example:
    function="x**2"
    a=0
    b=2
    """

    x = sp.symbols("x")
    expr = sp.sympify(function)
    f = sp.lambdify(x, expr, "numpy")

    result, _ = quad(f, a, b)

    return float(result)


@tool
def normal_distribution_cdf(x: float):
    """
    Normal distribution cumulative probability.
    """
    return float(norm.cdf(x))


@tool
def binomial_probability(n: int, p: float, k: int):
    """
    Probability of k successes in n trials.
    """
    return float(binom.pmf(k, n, p))


@tool
def poisson_probability(mu: float, k: int):
    """
    Poisson probability.
    """
    return float(poisson.pmf(k, mu))


@tool
def t_distribution_probability(x: float, df: int):
    """
    T distribution cumulative probability.
    """
    return float(t.cdf(x, df))


@tool
def chi_square_probability(x: float, df: int):
    """
    Chi-square cumulative probability.
    """
    return float(chi2.cdf(x, df))


@tool
def matrix_inverse(A: list):
    """
    Computes matrix inverse.

    Example:
    [[1,2],[3,4]]
    """
    return inv(np.array(A)).tolist()


@tool
def eigen_values(A: list):
    """
    Computes eigenvalues and eigenvectors.
    """
    values, vectors = eig(np.array(A))

    return {
        "eigenvalues": values.tolist(),
        "eigenvectors": vectors.tolist()
    }


@tool
def solve_linear_system(A: list, b: list):
    """
    Solves Ax=b linear system.
    """

    solution = solve(np.array(A), np.array(b))

    return solution.tolist()


@tool
def interpolate_data(x: list, y: list, value: float):
    """
    Estimates missing data points.
    """

    f = interp1d(x, y)

    return float(f(value))


@tool
def euclidean_distance(p1: list, p2: list):
    """
    Euclidean distance between two points.
    """

    return float(euclidean(p1, p2))


@tool
def pairwise_distance(points: list):
    """
    Pairwise distance between multiple points.
    """

    return pdist(points).tolist()


@tool
def detect_peaks(data: list):
    """
    Detects peaks in signal data.
    """

    peaks, _ = find_peaks(data)

    return peaks.tolist()


@tool
def smooth_signal(data: list):
    """
    Smooth noisy signal.
    """

    return savgol_filter(data, 5, 2).tolist()


@tool
def fourier_transform(data: list):
    """
    Fourier transform of signal.
    """

    return fft(data).tolist()


@tool
def statistical_mean(data: list):
    """
    Average of dataset.
    """

    return float(np.mean(data))


@tool
def standard_deviation(data: list):
    """
    Standard deviation of dataset.
    """

    return float(np.std(data))


@tool
def compute_zscore(data: list):
    """
    Standardized values of dataset.
    """

    return zscore(data).tolist()


@tool
def density_estimation(data: list, point: float):
    """
    Kernel density estimation.
    """

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