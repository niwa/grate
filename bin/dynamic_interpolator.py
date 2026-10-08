import numpy as np


class DynamicInterpolatorOld:
    """Keep track of evaluations, after enough evaluations we can just start
    interpolating results
    """

    def __init__(self, f, atol=1e-1, rtol=1e-2):
        self.f = f
        self.atol = atol
        self.rtol = rtol
        self.tab = np.empty((0, 2), dtype=float)

    def eval(self, x):
        if self.tab.shape[0] == 0:
            y = self.f(x)
            self.tab = np.array([[x, y]], dtype=float)
            return y

        xmin = self.tab[0, 0]
        xmax = self.tab[-1, 0]

        # print(f"TAB has {len(self.tab)} entries")

        if x < xmin:
            y = self.f(x)
            new_points = self._refine(x, y, xmin, self.tab[0, 1])
            self.tab = np.vstack((new_points, self.tab[1:]))
            return y

        if x > xmax:
            y = self.f(x)
            new_points = self._refine(xmax, self.tab[-1, 1], x, y)
            self.tab = np.vstack((self.tab[:-1], new_points))
            print(f"{x} > {xmax} so increasing tab to {xmin} to {self.tab[-1, 0]}")
            return y

        return np.interp(
            x,
            self.tab[:, 0],
            self.tab[:, 1],
        )

    def _refine(self, x0, y0, x1, y1):
        """Return points needed to interpolate accurately between endpoints."""

        points = [(x0, y0), (x1, y1)]
        intervals = [(x0, y0, x1, y1)]

        # print(f"Refining on {x0} {y0} {x1} {y1}")

        while intervals:
            xa, ya, xb, yb = intervals.pop()

            xm = (xa + xb) / 2
            if xm == xa or xm == xb:
                continue

            ym = self.f(xm)
            yinterp = ya + (yb - ya) * (xm - xa) / (xb - xa)

            # print(f"Refining {xm} {ym} {yinterp}")

            if abs(ym - yinterp) > self.atol + self.rtol * abs(ym):
                # print(f"Not good enought, appending points")
                points.append((xm, ym))
                intervals.append((xa, ya, xm, ym))
                intervals.append((xm, ym, xb, yb))

        points.sort()
        return points


class DynamicInterpolator:
    """Dynamically extend an interpolation range as required."""

    def __init__(self, f, xmin, n=100):
        self.f = f
        self.xmin = xmin
        self.n = n
        self.tab = np.empty((0, 2), dtype=float)

    def eval(self, x):
        if x <= self.xmin:
            # can have water level elevation below elevation
            return 0

        if self.tab.shape[0] == 0:
            xmax = x
            self.tab = self._make_tab(self.xmin, xmax)

        else:
            xmin = self.tab[0, 0]
            xmax = self.tab[-1, 0]

            if x < xmin:
                new_xmin = max(self.xmin, x - (xmax - xmin))
                self.tab = self._make_tab(new_xmin, xmax)

            elif x > xmax:
                new_xmax = x + (xmax - xmin)
                self.tab = self._make_tab(xmin, new_xmax)

        return np.interp(x, self.tab[:, 0], self.tab[:, 1])

    def _make_tab(self, xmin, xmax):
        if xmin == self.xmin:
            xmin = np.nextafter(xmin, xmax)

        xs = np.linspace(xmin, xmax, self.n)
        ys = np.array([self.f(x) for x in xs])
        return np.column_stack((xs, ys))
