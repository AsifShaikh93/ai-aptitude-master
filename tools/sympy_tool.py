import io, math, contextlib
from sympy.stats import Binomial, P
from langchain.tools import tool
from sympy.geometry import Point, Line, Circle, Triangle
from sympy import gcd, factorial, symbols, Eq, solve, sympify, diff, integrate, summation
import sympy as sp
from sympy.functions.combinatorial.factorials import binomial

@tool
def arithmetic_solver(expression: str):
    """
    Evaluate arithmetic expressions and simplify symbolic expressions.
    Use this for: simplifying math, basic arithmetic, or symbolic expansion.
    """
    try:
        
        clean_expr = expression.replace("^", "**")
        expr = sympify(clean_expr)
        return str(expr)
    except Exception as e:
        return f"Math parsing error: {str(e)}"

@tool
def solve_algebra_with_equal_to(equation: str):
    """
    Solve algebraic equations containing an equal sign "=".
    Supports any variable (x, y, z, etc.).
    Example: "2*y + 3 = 7"
    """
    try:
        equation = equation.replace("^", "**")
        left, right = equation.split("=")
        lhs = sympify(left.strip())
        rhs = sympify(right.strip())
        
        
        eq = Eq(lhs, rhs)
        vars = list(eq.free_symbols)
        
        if not vars:
            return "No variables found to solve."
            
        solution = solve(eq, vars)
        return str(solution)
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def quadratic_solver(equation: str):
    """
    Solve quadratic equations (ax^2 + bx + c = 0).
    Automatically detects the variable used.
    """
    try:
        equation = equation.replace("^", "**")
        if "=" in equation:
            lhs, rhs = equation.split("=")
            eq = Eq(sympify(lhs), sympify(rhs))
        else:
            eq = sympify(equation)
            
        vars = list(eq.free_symbols)
        solutions = solve(eq, vars[0] if vars else symbols('x'))
        return str(solutions)
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def permutations(n: int) -> str:
    """Compute permutations (n!) for arranging n objects."""
    return str(factorial(n))

@tool
def combinations(n: int, r: int) -> str:
    """Compute combinations (nCr) where order does not matter."""
    return str(binomial(n, r))

@tool
def solve_algebra(expression: str):
    """
    Find the roots of an expression (assumes expression = 0).
    Example: "x**2 - 5*x + 6"
    """
    try:
        expr = sympify(expression.replace("^", "**"))
        vars = list(expr.free_symbols)
        solution = solve(expr, vars[0] if vars else symbols('x'))
        return str(solution)
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def probability_binomial(n: int, p: float, k: int):
    """
    Compute binomial probability P(X=k) for n trials with probability p.
    """
    try:
        X = Binomial("X", n, p)
        result = P(Eq(X, k))
        return str(float(result.evalf()))
    except Exception as e:
        return f"Probability Error: {str(e)}"

@tool
def series_sum(expression: str, var: str, start: int, end: int):
    """
    Compute summation of a series.
    expression: "x**2", var: "x", start: 1, end: 10
    """
    try:
        s = symbols(var)
        expr = sympify(expression.replace("^", "**"))
        result = summation(expr, (s, start, end))
        return str(result)
    except Exception as e:
        return f"Summation Error: {str(e)}"

@tool
def derivative(expr: str):
    """
    Compute the derivative of a function. Detects variable automatically.
    """
    try:
        e = sympify(expr.replace("^", "**"))
        vars = list(e.free_symbols)
        var = vars[0] if vars else symbols('x')
        return str(diff(e, var))
    except Exception as e:
        return f"Calculus Error: {str(e)}"

@tool
def integrate_expression(expr: str):
    """
    Compute the indefinite integral of an expression.
    """
    try:
        e = sympify(expr.replace("^", "**"))
        vars = list(e.free_symbols)
        var = vars[0] if vars else symbols('x')
        return str(integrate(e, var))
    except Exception as e:
        return f"Calculus Error: {str(e)}"

@tool
def distance_between_points(x1: float, y1: float, x2: float, y2: float):
    """Compute Euclidean distance between (x1,y1) and (x2,y2)."""
    p1, p2 = Point(x1, y1), Point(x2, y2)
    return str(p1.distance(p2).evalf())

@tool
def midpoint(x1: float, y1: float, x2: float, y2: float):
    """Compute the midpoint of a line segment."""
    p1, p2 = Point(x1, y1), Point(x2, y2)
    return str(p1.midpoint(p2))

@tool
def triangle_area(x1: float, y1: float, x2: float, y2: float, x3: float, y3: float):
    """Compute the area of a triangle given three vertices."""
    t = Triangle(Point(x1, y1), Point(x2, y2), Point(x3, y3))
    return str(t.area.evalf())

@tool
def gcd_tool(a: int, b: int):
    """Compute the Greatest Common Divisor."""
    return str(gcd(a, b))

@tool
def line_slope(x1: float, y1: float, x2: float, y2: float):
    """Compute the slope of a line between two points."""
    try:
        p1, p2 = Point(x1, y1), Point(x2, y2)
        l = Line(p1, p2)
        return str(l.slope)
    except Exception as e:
        return "Vertical Line (Infinite Slope)"

@tool
def circle_equation(cx: float, cy: float, r: float):
    """Generate the equation of a circle given center and radius."""
    c = Circle(Point(cx, cy), r)
    return str(c.equation())

@tool
def intersection_of_lines(x1, y1, x2, y2, x3, y3, x4, y4):
    """Find the intersection point(s) of two lines."""
    l1 = Line(Point(x1, y1), Point(x2, y2))
    l2 = Line(Point(x3, y3), Point(x4, y4))
    return str(l1.intersection(l2))

@tool
def angle_between_lines(x1, y1, x2, y2, x3, y3, x4, y4):
    """Compute the angle between two lines in radians."""
    l1 = Line(Point(x1, y1), Point(x2, y2))
    l2 = Line(Point(x3, y3), Point(x4, y4))
    return str(l1.angle_between(l2).evalf())

@tool
def square_root(number: float) -> float:
    """Compute the square root of a number."""
    return math.sqrt(number)

sympy_tools = [
    arithmetic_solver, quadratic_solver, solve_algebra_with_equal_to, 
    permutations, combinations, solve_algebra, probability_binomial, 
    series_sum, derivative, integrate_expression, distance_between_points, 
    midpoint, triangle_area, gcd_tool, line_slope, circle_equation, 
    intersection_of_lines, angle_between_lines, square_root
]