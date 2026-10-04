import sys
import pathlib
import argparse
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


def find_root(f, x0):
    """Finds root of monotonic f

    x is restricted to being positive
    """

    if x0 <= 0:
        raise ValueError(f"x0 must be positive, got {x0}")

    # get the bracket
    fx0 = f(x0)
    if fx0 == 0:
        return x0

    # check if increasing or decreasing
    x = 2 * x0
    fx = f(x)
    if fx == fx0:
        raise ValueError(f"Cannot determine slope at x0={x0}")

    increasing = fx > fx0

    if increasing:
        if fx0 > 0:
            x = x0 / 2
            while f(x) > 0:
                x /= 2
            a, b = x, x0
        else:
            x = 2 * x0
            while f(x) < 0:
                x *= 2
            a, b = x0, x
    else:
        if fx0 > 0:
            x = 2 * x0
            while f(x) > 0:
                x *= 2
            a, b = x0, x
        else:
            x = x0 / 2
            while f(x) < 0:
                x /= 2
            a, b = x, x0

    return scipy.optimize.brentq(f, a, b)
