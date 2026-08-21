import math
import numpy as np
import pandas as pd
from gin import GrateConfig

# p = Profile(df)
# p.B(10 - p.bed_level)
# p.P(10 - p.bed_level)
# p.area(9 - p.bed_level)


class CrossSection:
    """Currently just the x/y points across cross-section profile"""

    def __init__(self, xs: GrateConfig.CrossSectionProfile, default_formrf, wallrf):
        self._from_points(
            xs,
            pd.read_csv(xs.profile),
            xs.formrf if xs.formrf is not None else default_formrf,
            wallrf,
        )

    def _from_points(self, xs: GrateConfig.CrossSectionProfile, df, formrf, wallrf):
        self._xs = xs
        self._formrf = formrf
        self._wallrf = wallrf

        # get roughness in for each segment in the profile
        self.df = df
        if "relrf" in self.df.columns:
            self.df["roughness"] = self.df["relrf"].fillna(1) * self._formrf
        else:
            self.df["roughness"] = self._formrf

        # break into left, channel and right
        self.left, self.channel, self.right = self._split_pts_into_three(self.df)

        self.mean_bed_level = self._calculate_mean_bed_level()
        self.bed_level = self._calculate_bed_level()

    def shifted(self, dy: float) -> "CrossSection":
        xs = self._xs.model_copy(deep=True)
        df = self.df.copy()
        df["y"] += dy
        me = CrossSection.__new__(CrossSection)
        me._from_points(xs, df, self._formrf, self._wallrf)
        return me

    def get_formrf(self):
        return self._formrf

    def get_wallrf(self):
        return self._wallrf

    def _split_pts_into_three(self, df):
        """Return three dataframes, left, channel and right bank"""
        if "ob" not in df.columns:
            return (df.iloc[:0], df, df.iloc[:0])

        i1 = df.index[df["ob"] == 1]
        i2 = df.index[df["ob"] == 2]
        i1 = None if i1.empty else i1[0]
        i2 = None if i2.empty else i2[0]

        if i1 is not None and i2 is not None:
            assert i1 <= i2, "ob==1 must occur before ob==2"

        if i1 is None:
            left = df.iloc[:0]
            if i2 is None:
                channel = df
                right = df.iloc[:0]
            else:
                channel = df.iloc[: i2 + 1]
                right = df.iloc[i2:]
        else:
            left = df.iloc[: i1 + 1]
            if i2 is None:
                channel = df.iloc[i1:]
                right = df.iloc[:0]
            else:
                channel = df.iloc[i1 : i2 + 1]
                right = df.iloc[i2:]

        return (left, channel, right)

    def _calculate_mean_bed_level(self):
        """Return weighted mean of the bed elevations in channel"""
        x = self.channel["x"].to_numpy()
        y = self.channel["y"].to_numpy()
        match len(x):
            case 0:
                raise ValueError("Channel profile has no points.")
            case 1:
                return y[0]
            case 2:
                return (y[0] + y[1]) / 2
            case _:
                dx = (x[2:] - x[:-2]) / 2
                return np.sum(dx * y[1:-1]) / np.sum(dx)

    def _calculate_bed_level(self):
        """The minimum bed level."""
        return self.channel["y"].min()

    def B(self, h: float):
        """Return water surface width for depth h above the channel bed."""
        water_level = self.bed_level + h

        x = self.df["x"].to_numpy()
        y = self.df["y"].to_numpy()

        width = 0.0

        for x0, x1, y0, y1 in zip(x[:-1], x[1:], y[:-1], y[1:]):
            # seg is above
            if y0 > water_level and y1 > water_level:
                continue

            # seg below water
            if y0 <= water_level and y1 <= water_level:
                width += x1 - x0
                continue

            # seg crosses
            f = (water_level - y0) / (y1 - y0)
            xc = x0 + f * (x1 - x0)

            if y0 <= water_level:
                # heading down
                width += xc - x0
            else:
                width += x1 - xc

        return width

    def P(self, h: float):
        """Wetted perimeter for given water level"""
        water_level = self.bed_level + h

        x = self.df["x"].to_numpy()
        y = self.df["y"].to_numpy()

        peri = 0.0

        for x0, x1, y0, y1 in zip(x[:-1], x[1:], y[:-1], y[1:]):
            # seg is above
            if y0 > water_level and y1 > water_level:
                continue

            # seg below water
            if y0 <= water_level and y1 <= water_level:
                peri += np.hypot(x1 - x0, y1 - y0)
                continue

            # seg crosses
            f = (water_level - y0) / (y1 - y0)
            xc = x0 + f * (x1 - x0)

            if y0 <= water_level:
                # heading down
                peri += math.hypot(xc - x0, water_level - y0)
            else:
                peri += math.hypot(x1 - xc, y1 - water_level)

        return peri

    def area(self, h: float):
        """Area of water below this height"""
        water_level = self.bed_level + h

        x = self.df["x"].to_numpy()
        y = self.df["y"].to_numpy()

        a = 0.0

        for x0, x1, y0, y1 in zip(x[:-1], x[1:], y[:-1], y[1:]):
            # seg is above
            if y0 > water_level and y1 > water_level:
                continue

            # seg below water
            if y0 <= water_level and y1 <= water_level:
                a += (x1 - x0) * (water_level - (y0 + y1) / 2)
                continue

            # seg crosses
            f = (water_level - y0) / (y1 - y0)
            xc = x0 + f * (x1 - x0)

            if y0 <= water_level:
                # heading up
                a += 0.5 * (xc - x0) * (water_level - y0)
            else:
                # heading down
                a += 0.5 * (x1 - xc) * (water_level - y1)

        return a


class Channel:
    """Flume, river, braided channel"""

    def __init__(self, cfg: GrateConfig):
        self._cfg = cfg
        self.dc = self._cfg.discretisation.dc
        self.cs = self._chainpts()
        self.nc = len(self.cs)
        self.xss = self._get_interpolated_cross_sections()

    def _chainpts(self):
        """Return chain points"""
        d = self._cfg.discretisation
        return np.arange(d.chainage_min, d.chainage_max + self.dc / 2, self.dc)

    def _get_cross_sections(self):
        """Load the cross sections"""
        xss = {}

        formrf = self._cfg.cross_sections.formrf
        wallrf = self._cfg.cross_sections.wallrf

        for xs in self._cfg.cross_sections.profiles:
            c = xs.chainage
            xss[c] = CrossSection(xs, formrf, wallrf)

        # check min/max chainage
        assert min(xss.keys()) <= self.cs[0], (
            f"Minimum chainage ({self.cs[0]}) isn't at least the minimum cross section chainage"
        )
        assert self.cs[-1] <= max(xss.keys()), (
            f"Maximum chainage ({self.cs[-1]}) is more than the maximum cross section chainage"
        )
        return xss

    def _get_interpolated_cross_sections(self):
        """Interpolate cross section profiles onto our chainage points"""
        xss = self._get_cross_sections()
        xs_chainpts = sorted(xss)

        ixss = []

        for c in self.cs:
            # Exact cross section
            if c in xss:
                ixss.append(xss[c])
                continue

            # Find surrounding cross sections
            for c0, c1 in zip(xs_chainpts[:-1], xs_chainpts[1:]):
                if c0 <= c <= c1:
                    p0 = xss[c0]
                    p1 = xss[c1]
                    break
            else:
                raise ValueError(f"No cross sections surrounding chainage {c}")

            f = (c - c0) / (c1 - c0)
            target_bed = p0.mean_bed_level + f * (p1.mean_bed_level - p0.mean_bed_level)

            # Use whichever source cross section is closer.
            p = p0 if f < 0.5 else p1
            ixss.append(p.shifted(target_bed - p.mean_bed_level))

        return ixss

    def beta(self):
        """Momentum correction factor"""
        return 1

    def d90(self, c: int):
        """90th percentile of the grain diameter.

        Referred to in Eq 8.4

        The grain diameter of the surface layer, and in the case of the
        floodplain use bank d90 from config file if specified else surface
        layer
        """
        raise NotImplementedError(f"d90({c})")

    def ng(self, c: int):
        """Grain roughness at given chainage"""
        return 0.044 * self.d90(c) ** (1 / 6)

    def nf(self, c: int, h: float):
        """Form roughness"""
        raise NotImplementedError(f"nf({c}, {h})")

    def area(self, c: int, h: float):
        """Return area of water between bed and h"""
        return self.xss[c].area(h)

    def P(self, c: int, h: float):
        """Wetted perimeter at chainage"""
        return self.xss[c].P(h)

    def mean_bed_level(self, c: int):
        """Return mean bed level of profile at chainage c

        For a flume this is the bed_level, for other channels it is some sort
        of average of the cross-section profile
        """
        return self.xss[c].mean_bed_level

    def S0(self, c: int):
        """Bed slope at c

        Bed slope is the slope between chainages of mean_bed_level
        """
        assert c < self.nc - 1, (
            f"Cannot calculate S0({c}), likely because this is the most downstream point"
        )
        return (self.mean_bed_level(c) - self.mean_bed_level(c + 1)) / self.dc

    def B(self, c: int, h: float):
        """Water surface width"""
        return self.xss[c].B(h)

    def bed_level(self, c: int):
        """Return deepest part of the cross-section"""
        return self.xss[c].bed_level


class Flume(Channel):
    def __init__(self, cfg: GrateConfig):
        super().__init__(cfg)
        self.active_layer_thickness = np.full(self.nc, self._cfg.morphological.la)
        self.storage_layer_thickness = np.full(self.nc, self._cfg.morphological.layer)

    def d90(self, c: int):
        """90 percentile of grain diameter of surface layer."""
        _ = c
        # FIXME
        return self._cfg.bankd90

    def nf(self, c: int, h: float):
        """Form roughness

        formrf * width + wallrf * 2h
        -------------------------------
            width + 2h
        """
        p = self.xss[c]
        B = p.B(h)
        return (p.get_formrf() * B + 2 * p.get_wallrf() * h) / (B + 2 * h)


class River(Channel):
    def nf(self, c: int, h: float):
        """Form roughness

        nfc * sum_k (r_k * p_k) / pc

        where nfc is default form roughness of cross-section
        rk and pk are relative roughness factory wetted perimeter FIXME
        pc FIXME

        """
        raise NotImplementedError(f"nf({c} {h})")
