import sys
import pathlib


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


def newton(func, x0, fprime=None, tol=1.48e-8, maxiter=50):
    """Save pulling in scipy just to get newton."""

    x = float(x0)

    if fprime is not None:
        # Newton-Raphson
        for _ in range(maxiter):
            fx = func(x)
            dfx = fprime(x)

            if dfx == 0:
                raise RuntimeError("Derivative was zero")

            dx = fx / dfx
            x -= dx

            if abs(dx) <= tol:
                return x

    else:
        # Secant method
        x_prev = x
        x = x + 1e-4 if x == 0 else x * (1 + 1e-4)

        f_prev = func(x_prev)

        for _ in range(maxiter):
            f = func(x)
            denominator = f - f_prev

            if denominator == 0:
                raise RuntimeError("Zero denominator in secant iteration")

            x_new = x - f * (x - x_prev) / denominator

            if abs(x_new - x) <= tol:
                return x_new

            x_prev, f_prev = x, f
            x = x_new

    raise RuntimeError("Newton iteration did not converge")
