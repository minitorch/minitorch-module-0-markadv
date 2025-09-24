"""Collection of the core mathematical operators used throughout the code base."""

import math
from typing import Callable, Iterable, TypeVar, List, Any

# ## Task 0.1

# Implementation of a prelude of elementary functions.

# Mathematical functions:

def id(a: float) -> float:
    """Identity function."""
    return a


def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


def mul(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


def neg(a: float) -> float:
    """Negate a number."""
    return -a


def lt(a: float, b: float) -> bool:
    """Check if a is less than b."""
    return a < b


def eq(a: float, b: float) -> bool:
    """Check if a equals b."""
    return a == b


def max(a: float, b: float) -> float:
    """Return the maximum of two numbers."""
    return a if a > b else b


def is_close(a: float, b: float) -> bool:
    """Check if two numbers are close within a tolerance of 1e-2."""
    return abs(a - b) < 1e-2


def sigmoid(a: float) -> float:
    """Sigmoid activation function."""
    if a >= 0:
        return 1.0 / (1.0 + math.exp(-a))
    else:
        e = math.exp(a)
        return e / (1.0 + e)


def relu(a: float) -> float:
    """ReLU activation function."""
    return max(0.0, a)


def log(a: float) -> float:
    """Natural logarithm."""
    return math.log(a)


def exp(a: float) -> float:
    """Exponential function."""
    return math.exp(a)


def log_back(a: float, b: float) -> float:
    """Derivative of log."""
    return 1.0 / a if a != 0 else float('inf')


def inv(a: float) -> float:
    """Inverse of a number."""
    return 1.0 / a if a != 0 else float('inf')


def inv_back(a: float, b: float) -> float:
    """Derivative of inverse."""
    return -1.0 / (a * a) if a != 0 else float('inf')


def relu_back(a: float, b: float) -> float:
    """Derivative of ReLU."""
    return 1.0 if a > 0 else 0.0


# ## Task 0.3
# Higher-order functions

def map(fn: Callable[[float], float], xs: Iterable[float]) -> List[float]:
    """Apply fn to each element in xs."""
    return [fn(x) for x in xs]


def zip_with(fn: Callable[[float, float], float], xs: Iterable[float], ys: Iterable[float]) -> List[float]:
    """Apply fn to pairs of elements from xs and ys."""
    return [fn(x, y) for x, y in zip(xs, ys)]

# Alias for backward compatibility
zipWith = zip_with


def reduce(fn: Callable[[float, float], float], xs: Iterable[float], start: float = 0.0) -> float:
    """Reduce xs using fn, starting with start."""
    result = start
    for x in xs:
        result = fn(result, x)
    return result


# List operations using higher-order functions

def neg_list(xs: Iterable[float]) -> List[float]:
    """Negate each element in xs."""
    return map(neg, xs)

# Alias for backward compatibility
negList = neg_list


def add_lists(xs: Iterable[float], ys: Iterable[float]) -> List[float]:
    """Add corresponding elements of xs and ys."""
    return zip_with(add, xs, ys)

# Alias for backward compatibility
addLists = add_lists


def sum(xs: Iterable[float]) -> float:
    """Sum all elements in xs."""
    return reduce(add, xs, 0.0)


def prod(xs: Iterable[float]) -> float:
    """Multiply all elements in xs."""
    return reduce(mul, xs, 1.0)
