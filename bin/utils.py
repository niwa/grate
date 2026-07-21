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
