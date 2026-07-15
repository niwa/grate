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
        print("Frozen")
        base = pathlib.Path(sys.executable).parent
    else:
        print("Not frozen")
        base = pathlib.Path(__file__).parent.parent

    print(f"Base is {base}")

    return base / rpath


def testit():
    print("In utils.textit")
