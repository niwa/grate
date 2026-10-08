import sys
import pathlib
import argparse
import numpy as np
import scipy


def resolved_path(rpath):
    """Return full path, works in development or deployed code

    Parameters
    ----------
    rpath: pathlib.Path
        Relative path of file

    Returns
    -------
    pathlib.Path
        Resolved full path
    """

    if getattr(sys, "frozen", False) or "__compiled__" in globals():
        base = pathlib.Path(sys.executable).parent
    else:
        base = pathlib.Path(__file__).parent.parent

    return base / rpath


def positive_float(value):
    value = float(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def try_to_num(val: str):
    """Convert string to int or float if possible, otherwise leave as string."""
    try:
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        pass
    return val


def get_bracket_for_monotonic(f, x0, lower, max_growth: float | None = None, maxits=50):
    """Get bracket for monotonic f

    Parameters
    ----------
    f: function of x
        f must be monotonic and defined for x>lower

    x0: float
        Initial point > lower

    lower: float
        Lower bound for domain

    max_growth: float or None
        If defined, don't search when x > lower + max_growth

    maxits: int
        Maximum number of doubling or halving from x0 to find bracket

    Returns
    -------
    tuple or None:
        None if a bracket cannot be found
        (a, ) if f(x0) == 0
        (a, b) where a <= x0 <= b and f(a)f(b) < 0

    Exceptions
    ----------
    ValueError
        if x0 <= lower

    """

    if x0 <= lower:
        raise ValueError(f"x0 must be > {lower}, got {x0}")

    # maybe already at zero
    fx0 = f(x0)
    if fx0 == 0:
        return (x0,)

    # check if increasing or decreasing
    dx = 0.1 * (x0 - lower)
    x = x0 + dx
    fx = f(x)
    if fx == fx0:
        raise ValueError(f"Cannot determine f' at x0={x0} because f is flat")

    # go left if f(x0) positive and increasing, or negative and decreasing
    increasing = fx > fx0
    search_left = (fx0 > 0) == increasing

    # possibly found a bracket already
    # if (fx <= 0) if fx0 > 0 else (fx >= 0):
    #     return (x, x0) if search_left else (x0, x)

    # as a quick and dirty use the derivative and jump twice the distance
    fprime = (fx - fx0) / dx
    x = x0 - 2 * fx0 / fprime
    if x > lower:
        fx = f(x)
        if (fx <= 0) if fx0 > 0 else (fx >= 0):
            return (x, x0) if search_left else (x0, x)

    x = x0
    for _ in range(maxits):
        x = (x + lower) / 2 if search_left else lower + 2 * (x - lower)
        if max_growth and x > lower + max_growth:
            return None
        fx = f(x)
        if (fx <= 0) if fx0 > 0 else (fx >= 0):
            return (x, x0) if search_left else (x0, x)

    return None


def get_bracket(f, x0, lower, steps=100, mult=10):
    """Find a bracket by uniformly sampling around x0

    Parameters
    ----------
    f: function of x
        f defined for x>lower

    x0: float
        Initial point > lower

    lower: float
        Lower bound for domain

    steps: int
        How many points will be tried between lower and x0

    mult: int
        This multiplied by x0 is the largest point tried

    Returns
    -------
    tuple or None:
        None if a bracket cannot be found
        (a, ) if f(x0) == 0
        (a, b) where a <= x0 <= b and f(a)f(b) < 0

    Exceptions
    ----------
    ValueError
        if x0 <= lower

    """

    if x0 <= lower:
        raise ValueError(f"x0 must be > {lower}, got {x0}")

    fx0 = f(x0)
    if fx0 == 0:
        return (x0,)

    dx = (x0 - lower) / steps
    x_prev, f_prev = x0, fx0

    # Search left, then right, using the same spacing
    for x in np.linspace(x0 - dx, lower + dx, steps):
        # print(f"{lower=} {x=} {f(x)=}")
        fx = f(x)
        if fx == 0:
            return (x,)
        if (fx > 0) != (f_prev > 0):
            return (x, x_prev)
        x_prev, f_prev = x, fx

    x_prev, f_prev = x0, fx0
    for x in np.linspace(x0 + dx, mult * x0, steps * mult):
        # print(f"{lower=} {x=} {f(x)=}")
        fx = f(x)
        if fx == 0:
            return (x,)
        if (fx > 0) != (f_prev > 0):
            return (x_prev, x)
        x_prev, f_prev = x, fx

    return None


def find_root(f, x0, lower: float, max_growth: float | None = None):
    """Finds root of f, where f is only defined for x > lower

    Parameters
    ----------
    f: function of x
        f must be defined for x > lower.

    x0: float
        Initial point > lower

    lower: float
        Lower bound for domain

    max_growth: float or None
        If defined, don't search when x > lower + max_growth
    """

    # try to get a bracket
    b = get_bracket_for_monotonic(f, x0, lower, max_growth)

    # if f isn't monotonic or something else gone wrong, try uniform grid
    if b is None:
        sys.stderr.write(f"WARNING: can't find bracket assuming monotonic f at {x0}\n")
        b = get_bracket(f, x0, lower)
        if b is None:
            raise ValueError(f"Cannot find bracket for root finding at {x0}")
        sys.stderr.write("Got a bracket using the exhaustive approach\n")

    # f(x0) == 0
    if len(b) == 1:
        return b[0]

    a, b = b
    return scipy.optimize.brentq(f, a, b)
