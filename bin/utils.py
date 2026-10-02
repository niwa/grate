import sys
import pathlib
import argparse
from scipy.optimize import root_scalar


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


#
# def newton(func, x0, fprime=None, tol=1.48e-8, maxiter=50):
#     """Save pulling in scipy just to get newton.
#
#     x cannot go below 0
#     """
#
#     x = float(x0)
#
#     if fprime is not None:
#         # Newton-Raphson
#         for _ in range(maxiter):
#             fx = func(x)
#             dfx = fprime(x)
#
#             if dfx == 0:
#                 raise RuntimeError("Derivative was zero")
#
#             dx = fx / dfx
#
#             if x - dx <= 0:
#                 x /= 2.0
#             else:
#                 x -= dx
#
#             if abs(dx) <= tol:
#                 return x
#
#     else:
#         # Secant method
#         x_prev = x
#         x = x + 1e-4 if x == 0 else x * (1 + 1e-4)
#
#         f_prev = func(x_prev)
#
#         for _ in range(maxiter):
#             f = func(x)
#             denominator = f - f_prev
#
#             if denominator == 0:
#                 raise RuntimeError("Zero denominator in secant iteration")
#
#             x_new = x - f * (x - x_prev) / denominator
#
#             if x_new <= 0:
#                 x_new = x / 2
#
#             if abs(x_new - x) <= tol:
#                 return x_new
#
#             x_prev, f_prev = x, f
#             x = x_new
#
#     raise RuntimeError("Newton iteration did not converge")
#


def find_root(f, x0, maxiter=500):
    a = b = x0
    fa = fb = f(x0)

    if fa == 0:
        return x0

    step = max(abs(x0) * 0.01, 1e-8)

    for _ in range(maxiter):
        a = max(1e-8, x0 - step)
        b = x0 + step

        fa = f(a)
        fb = f(b)

        if fa == 0:
            return a
        if fb == 0:
            return b

        if fa * fb < 0:
            sol = root_scalar(f, bracket=(a, b), method="brentq")
            if not sol.converged:
                raise RuntimeError("Root finding failed")
            return sol.root

        if a == 1e-8:
            # No further expansion possible in the negative direction
            pass

        step *= 2

    # FIXME.  try again but print out
    a = b = x0
    fa = fb = f(x0)

    if fa == 0:
        return x0

    step = max(abs(x0) * 0.1, 1e-8)

    for _ in range(maxiter):
        a = max(1e-8, x0 - step)
        b = x0 + step

        print(f"{a},{b},{fa},{fb}")

        fa = f(a)
        fb = f(b)

        if fa == 0:
            return a
        if fb == 0:
            return b

        if fa * fb < 0:
            sol = root_scalar(f, bracket=(a, b), method="brentq")
            if not sol.converged:
                raise RuntimeError("Root finding failed")
            return sol.root

        if a == 1e-8:
            # No further expansion possible in the negative direction
            pass

        step *= 2

    #
    raise RuntimeError(f"Could not bracket root around x0={x0}")
