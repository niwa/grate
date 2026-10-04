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


def get_bracket_for_monotonic(f, x0, maxits=50):
    """Get bracket for monotonic f

    Parameters
    ----------
    f: function of x
        f must be monotonic and defined for x>0

    x0: float
        Initial point > 0

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
        if x0 <= 0

    """

    if x0 <= 0:
        raise ValueError(f"x0 must be positive, got {x0}")

    # maybe already at zero
    fx0 = f(x0)
    if fx0 == 0:
        return (x0,)

    # check if increasing or decreasing
    x = 2 * x0
    fx = f(x)
    if fx == fx0:
        raise ValueError(f"Cannot determine f' at x0={x0} because f is flat")

    # go left if f(x0) positive and increasing, or negative and decreasing
    increasing = fx > fx0
    search_left = (fx0 > 0) == increasing
    x = x0

    for _ in range(maxits):
        x = x / 2 if search_left else x * 2
        fx = f(x)
        if (fx <= 0) if fx0 > 0 else (fx >= 0):
            return (x, x0) if search_left else (x0, x)

    return None


def get_bracket(f, x0, steps=100, mult=10):
    """Find a bracket by uniformly sampling around x0

    Parameters
    ----------
    f: function of x
        f defined for x>0

    x0: float
        Initial point > 0

    steps: int
        How many points will be tried between 0 and x0

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
        if x0 <= 0

    """

    if x0 <= 0:
        raise ValueError(f"x0 must be positive, got {x0}")

    fx0 = f(x0)
    if fx0 == 0:
        return (x0,)

    dx = x0 / steps
    x_prev, f_prev = x0, fx0

    # Search left, then right, using the same spacing
    for x in np.linspace(x0 - dx, np.finfo(float).eps, steps):
        fx = f(x)
        if fx == 0:
            return (x,)
        if (fx > 0) != (f_prev > 0):
            return (x, x_prev)
        x_prev, f_prev = x, fx

    x_prev, f_prev = x0, fx0
    for x in np.linspace(x0 + dx, mult * x0, steps * mult):
        fx = f(x)
        if fx == 0:
            return (x,)
        if (fx > 0) != (f_prev > 0):
            return (x_prev, x)
        x_prev, f_prev = x, fx

    return None


def find_root(f, x0):
    """Finds root of f, where f is only defined for x > 0

    Parameters
    ----------
    f: function of x
        f must be defined for x > 0.

    x0: float
        Initial point > 0

    """

    # try to get a bracket
    b = get_bracket_for_monotonic(f, x0)

    # if f isn't monotonic or something else gone wrong, try uniform grid
    if b is None:
        sys.stderr.write(f"WARNING: can't find bracket assuming monotonic f at {x0}\n")
        b = get_bracket(f, x0)
        if b is None:
            raise ValueError(f"Cannot find bracket for root finding at {x0}")

    # f(x0) == 0
    if len(b) == 1:
        return b[0]

    a, b = b
    return scipy.optimize.brentq(f, a, b)
