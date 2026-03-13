import io, math, contextlib
from sympy.stats import Binomial, P
from langchain.tools import tool
from sympy.geometry import Point, Line, Circle, Triangle
from sympy import gcd, factorial
import sympy as sp
from sympy.functions.combinatorial.factorials import binomial

@tool
def arithmetic_solver(expression: str):
    """
    Evaluate arithmetic expressions, symbolic expressions.

    Use this tool when:
    - The task requires evaluating arithmetic expressions.
    - Simplifying mathematical expressions.
    - Performing symbolic computations.

    Examples:
    Arithmetic:
    input: 2*(5+7)

    Symbolic:
    input: x + (1/3)*x

    Power expressions must use Python format:
    x^2 → x**2
    (1-x)^2 → (1-x)**2

    """

    try:
        expr = sp.sympify(expression)
        return str(expr)

    except Exception as e:
        return f"Math parsing error: {str(e)}"


@tool
def solve_algebra_with_equal_to(equation: str):
    """
    Solve algebraic equations that contain an equal sign "=".

    Use this tool when:
    - The problem contains an equation with LHS = RHS.
    - You need to solve for unknown variable x.
    - The arithmetic_solver tool cannot directly evaluate the expression.

    Important rules:
    - Convert powers to Python format.
    x^2 → x**2

    Example inputs:
    2*x + 3 = 7
    BD*DC = AD**2
    44100*(10000-X**2)**2/10000**2 = 1944.81 * X**2

    This tool converts the equation into a symbolic SymPy equation
    and solves it algebraically.
    """

    x = sp.symbols('x')

    left, right = equation.split("=")

    eq = sp.Eq(sp.sympify(left), sp.sympify(right))

    solution = sp.solve(eq, x)

    return str(solution)


@tool
def quadratic_solver(equation: str):
    """
    Solve quadratic equations.

    Use when:
    - The equation contains x^2 terms.
    - It follows the quadratic form ax^2 + bx + c = 0.

    Example:
    2*x**2 + 3*x - 2 = 0

    The tool automatically finds all possible roots of the quadratic equation.
    """

    x = sp.symbols('x')

    lhs, rhs = equation.split("=")

    lhs = sp.sympify(lhs.replace("^", "**"))
    rhs = sp.sympify(rhs.replace("^", "**"))

    eq = sp.Eq(lhs, rhs)

    solutions = sp.solve(eq, x)

    return solutions


@tool
def permutations(n: int) -> str:
    """
    Compute permutations using factorial.

    Use this tool when:
    - The problem involves permutations.
    - Arranging objects in order.
    - Counting number of ways objects can be arranged.

    Formula used:
    n!  (factorial)

    Example:
    Number of ways to arrange 5 people in a line.

    Input:
    permutations(5)

    Output:
    120
    """

    return str(factorial(n))


@tool
def combinations(n: int, r: int) -> str:
    """
    Compute combinations (nCr).

    Use this tool when:
    - Selecting items from a group.
    - Order does NOT matter.
    - Choosing r objects from n objects.

    Formula:
    nCr = n! / (r! * (n-r)!)

    Example:
    Choosing 3 students from 10 students.

    combinations(10,3)
    """

    return str(binomial(n, r))


@tool
def solve_algebra(equation: str):
    """
    Solve algebraic expressions without equal sign.

    Use when:
    - Expression is set equal to zero implicitly.
    - Example form: x**2 - 5*x + 6

    The tool finds roots of the expression.

    Example:
    x**2 - 5*x + 6
    """

    x = sp.symbols('x')
    expr = sp.sympify(equation)

    solution = sp.solve(expr, x)

    return str(solution)


@tool
def probability_binomial(n: int, p: float, k: int):
    """
    Compute binomial probability.

    Use when:
    - Repeated independent trials occur.
    - Each trial has success probability p.
    - Need probability of exactly k successes.

    Formula:
    P(X=k) = C(n,k) * p^k * (1-p)^(n-k)

    Example:
    Probability of getting exactly 3 heads in 5 coin tosses.

    probability_binomial(5,0.5,3)
    """

    X = Binomial("X", n, p)

    return str(P(X == k))


@tool
def series_sum(expression: str, var: str, start: int, end: int):
    """
    Compute summation of a sequence or series.

    Use when:
    - Calculating sum of sequences.
    - Arithmetic series.
    - Polynomial summations.

    Example:
    Sum of squares from 1 to 10

    expression: x**2
    variable: x
    start: 1
    end: 10
    """

    x = sp.symbols(var)

    expr = sp.sympify(expression)

    result = sp.summation(expr, (x, start, end))

    return str(result)


@tool
def derivative(expr: str):
    """
    Compute derivative of a function with respect to x.

    Use when:
    - Finding rate of change.
    - Calculus problems.
    - Optimization problems.

    Example:
    derivative("x**3 + 2*x")
    """

    x = sp.symbols('x')

    return str(sp.diff(expr, x))


@tool
def integrate_expression(expr):
    """
    Compute indefinite integral of an expression.

    Use when:
    - Calculating area under curves.
    - Solving calculus integration problems.

    Example:
    integrate_expression("x**2")
    """

    x = sp.symbols('x')

    return str(sp.integrate(expr, x))


@tool
def distance_between_points(x1,y1,x2,y2):
    """
    Compute Euclidean distance between two points.

    Formula:
    distance = sqrt((x2-x1)^2 + (y2-y1)^2)

    Use when:
    - Geometry problems.
    - Coordinate distance calculation.

    Example:
    distance_between_points(0,0,3,4)
    """

    p1 = Point(x1,y1)
    p2 = Point(x2,y2)

    return str(p1.distance(p2))


@tool
def midpoint(x1,y1,x2,y2):
    """
    Compute midpoint of a line segment.

    Formula:
    midpoint = ((x1+x2)/2 , (y1+y2)/2)

    Use when:
    - Geometry coordinate problems.
    """

    p1 = Point(x1,y1)
    p2 = Point(x2,y2)

    return str(p1.midpoint(p2))


@tool
def triangle_area(x1, y1, x2, y2, x3, y3):
    """
    Compute area of triangle from coordinates.

    Use when:
    - Given coordinates of triangle vertices.
    - Geometry area problems.
    """

    p1 = Point(x1, y1)
    p2 = Point(x2, y2)
    p3 = Point(x3, y3)

    tri = Triangle(p1, p2, p3)

    return str(tri.area)


@tool
def gcd_tool(a:int,b:int):
    """
    Compute Greatest Common Divisor (GCD).

    Use when:
    - Simplifying fractions.
    - Number theory problems.
    """

    return str(gcd(a,b))


@tool
def line_slope(x1, y1, x2, y2):
    """
    Compute slope of a line between two points.

    Formula:
    slope = (y2 - y1) / (x2 - x1)

    Use when:
    - Coordinate geometry problems.
    """

    p1 = Point(x1, y1)
    p2 = Point(x2, y2)

    line = Line(p1, p2)

    return str(line.slope)


@tool
def circle_equation(cx, cy, r):
    """
    Generate equation of a circle.

    Standard form:
    (x-h)^2 + (y-k)^2 = r^2

    Where:
    h,k = center coordinates
    r = radius
    """

    center = Point(cx, cy)

    circle = Circle(center, r)

    return str(circle.equation())


@tool
def intersection_of_lines(x1,y1,x2,y2,x3,y3,x4,y4):
    """
    Find intersection point of two lines.

    Use when:
    - Two lines are given by coordinates.
    - Need their intersection point.
    """

    p1 = Point(x1,y1)
    p2 = Point(x2,y2)

    p3 = Point(x3,y3)
    p4 = Point(x4,y4)

    l1 = Line(p1,p2)
    l2 = Line(p3,p4)

    return str(l1.intersection(l2))


@tool
def angle_between_lines(x1,y1,x2,y2,x3,y3,x4,y4):
    """
    Compute angle between two lines.

    Use when:
    - Geometry problems involving angles between lines.
    """

    p1 = Point(x1,y1)
    p2 = Point(x2,y2)

    p3 = Point(x3,y3)
    p4 = Point(x4,y4)

    l1 = Line(p1,p2)
    l2 = Line(p3,p4)

    return str(l1.angle_between(l2))


@tool
def square_root(number: float) -> float:
    """
    Compute square root of a number.

    Formula:
    sqrt(x)

    Use when:
    - Basic arithmetic square root calculation.

    Example:
    square_root(49) → 7
    """

    return math.sqrt(number)


sympy_tools=[arithmetic_solver, quadratic_solver , solve_algebra_with_equal_to , permutations, combinations, solve_algebra, 
probability_binomial, series_sum, derivative, integrate_expression, 
distance_between_points, midpoint,triangle_area, gcd_tool, line_slope, circle_equation, 
intersection_of_lines, angle_between_lines, square_root]