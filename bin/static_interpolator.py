import numpy as np


class StaticInterpolator:
    """Set up interpolator using y values in profile"""

    def __init__(self, f, yvals, max_depth=1000):
        """
        f is the function we are interpoling
        yvals is a list of depths for channel, or elevations for banks
        """

        # these are the x values (depth or elevation) to plug into f.  these
        # come from the profile y values.  sort them and add on one big one on
        # the end.
        self.xs = np.array(sorted(yvals) + [max_depth])
        self.xs = np.unique(self.xs)
        self.fs = np.array([f(x) for x in self.xs])
        self.xmin, self.xmax = self.xs.min(), self.xs.max()

    def eval(self, x):
        if x <= self.xmin:
            return 0

        # Sometimes we get x way outside range, np.interp will clamp to max f
        # which is probably fine, so skip the assert
        # assert self.xmin <= x <= self.xmax, f"{self.xmin} {x} {self.xmax}"
        return np.interp(x, self.xs, self.fs)

    def get_np(self):
        return np.stack([self.xs, self.fs], axis=1)
