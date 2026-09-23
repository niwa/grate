import math
import numpy as np
import pandas as pd
from enum import Enum
from functools import lru_cache
from gin import GrateConfig, CrossSectionProfile
from layers import LayerStack


class Loc(Enum):
    LEFT = "left"
    CHANNEL = "channel"
    RIGHT = "right"


class CrossSection:
    """A loaded CrossSectionProfile, includes metadata, xy points, derived props"""

    def __init__(
        self,
        xs: CrossSectionProfile,
        cfg: GrateConfig,
        default_formrf,
        wallrf,
    ):
        self.chainage = xs.chainage
        self.chainidx = None  # this will get filled in when interped
        self.topoid = xs.topoid
        self.river_name = xs.river_name

        self.formrf = xs.formrf if xs.formrf is not None else default_formrf
        self.wallrf = wallrf

        self.bankd90 = xs.bankd90
        self.bedrock_rl = xs.bedrock_rl
        self.qsfact = xs.qsfact
        self.lsf = xs.lsf

        self.df = pd.read_csv(xs.profile)
        self._set_points(self.df)

        self.layers = LayerStack(xs, cfg)

    def __str__(self):
        return (
            f"CrossSection at chainage={self.chainage} (idx={self.chainidx})\n{self.df}"
        )

    def _set_points(self, df):
        """Set profile points and calculate derived properties."""
        self.df = df

        # get roughness in for each segment in the profile
        if "relrf" in self.df.columns:
            self.df["roughness"] = self.df["relrf"].fillna(1) * self.formrf
        else:
            self.df["roughness"] = self.formrf

        # break into left, channel and right
        self.profile, self.left, self.channel, self.right = self._split_pts_into_three(
            self.df
        )

        self.mean_bed_level = self._calculate_mean_bed_level()
        self.min_bed_level = self._calculate_min_bed_level()

    def interpolate(
        self, other: "CrossSection", f: float, chainidx: int
    ) -> "CrossSection":
        """Return an approximate interpolation between two cross sections.

        Geometry is taken from the closer cross section and shifted to the
        interpolated mean bed level. Numeric properties adn sediment-layer
        properties linearly interpolated.

        Parameters
        ----------
        other: CrossSection
            Other cross section to interpolate with

        f: float
            interpolation proportion 0 to 1.  0 means self, 1 means other

        chainidx: int
            chain index to assign to the interpolant

        """

        def interp(a, b, f):
            if a is None or b is None:
                return None
            return a + f * (b - a)

        assert 0 <= f <= 1, f"Interpolation fraction must be between 0 and 1, got {f}"

        cs = self.__class__.__new__(self.__class__)

        cs.chainage = self.chainage + f * (other.chainage - self.chainage)
        cs.chainidx = chainidx

        # These aren't really interpolated.
        cs.topoid = self.topoid if f < 0.5 else other.topoid
        cs.river_name = self.river_name if f < 0.5 else other.river_name

        cs.formrf = interp(self.formrf, other.formrf, f)
        cs.wallrf = interp(self.wallrf, other.wallrf, f)
        cs.bankd90 = interp(self.bankd90, other.bankd90, f)
        cs.bedrock_rl = interp(self.bedrock_rl, other.bedrock_rl, f)
        cs.qsfact = interp(self.qsfact, other.qsfact, f)
        cs.lsf = interp(self.lsf, other.lsf, f)

        # Interpolate the mean bed level, then use the nearer geometry.
        target_bed = self.mean_bed_level + f * (
            other.mean_bed_level - self.mean_bed_level
        )

        source = self if f < 0.5 else other
        df = source.df.copy()
        df["y"] += target_bed - source.mean_bed_level
        cs._set_points(df)
        cs.layers = source.layers.interpolate(other.layers, f, chainidx)

        return cs

    def get_formrf(self):
        return self.formrf

    def get_wallrf(self):
        return self.wallrf

    def d90(self, loc: Loc):
        if loc == Loc.CHANNEL:
            return self.layers.d90()
        else:
            return self.bankd90

    def f_interface(self, aggrading: bool, p: float):
        return self.layers.f_interface(aggrading, p)

    def _split_pts_into_three(self, df):
        """Return entire profile, left, channel and right profiles as NumPy arrays.

        Each array has columns: x, y, roughness.
        """

        # FIXME, can remove after a while when I've remembered what is allowed
        # in profile csv
        assert set(df.columns) <= {"x", "y", "roughness", "ob"}, (
            f"Unexpected profile columns: "
            f"{sorted(set(df.columns) - {'x', 'y', 'roughness', 'ob'})}"
        )

        if "ob" not in df.columns:
            data = df[["x", "y", "roughness"]].to_numpy()
            return data, data[:0], data, data[:0]

        data = df[["x", "y", "roughness", "ob"]].to_numpy()

        i1 = np.flatnonzero(data[:, 3] == 1)
        i2 = np.flatnonzero(data[:, 3] == 2)

        i1 = None if len(i1) == 0 else i1[0]
        i2 = None if len(i2) == 0 else i2[0]

        if i1 is not None and i2 is not None:
            assert i1 <= i2, "ob==1 must occur before ob==2"

        # We don't need ob in the resulting arrays.
        data = data[:, :3]

        if i1 is None:
            left = data[:0]
            if i2 is None:
                channel = data
                right = data[:0]
            else:
                channel = data[: i2 + 1]
                right = data[i2:]
        else:
            left = data[: i1 + 1]
            if i2 is None:
                channel = data[i1:]
                right = data[:0]
            else:
                channel = data[i1 : i2 + 1]
                right = data[i2:]

        return data, left, channel, right

    def _calculate_mean_bed_level(self):
        """Return weighted mean of the bed elevations in channel"""
        x = self.channel[:, 0]
        y = self.channel[:, 1]
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

    def _calculate_min_bed_level(self):
        """The minimum bed level."""
        return self.channel[:, 1].min()

    def aggrade_bed(self, dy):
        """Add dy to the channel bed level (dy can be negative)."""
        self.channel[:, 1] += dy
        self.min_bed_level += dy
        self.mean_bed_level += dy

    def grain_stress(self, t: pd.Timestamp, hydro):
        return self.layers.grain_stress(t, hydro)

    @lru_cache(maxsize=200)
    def _wetted_segments(self, h: float, loc: Loc | None, min_bed_level: float):
        """Yield roughness, perimeter, width and area for each wetted segment."""
        water_level = min_bed_level + h

        profile = {
            Loc.LEFT: self.left,
            Loc.CHANNEL: self.channel,
            Loc.RIGHT: self.right,
            None: self.profile,
        }[loc]

        x = profile[:, 0]
        y = profile[:, 1]
        r = profile[:, 2]

        result = []

        for x0, x1, y0, y1, rough in zip(x[:-1], x[1:], y[:-1], y[1:], r[1:]):
            # seg is above
            if y0 > water_level and y1 > water_level:
                continue

            # seg below water
            if y0 <= water_level and y1 <= water_level:
                peri = math.hypot(x1 - x0, y1 - y0)
                width = x1 - x0
                area = width * (water_level - (y0 + y1) / 2)

            # seg crosses
            else:
                f = (water_level - y0) / (y1 - y0)
                xc = x0 + f * (x1 - x0)

                if y0 <= water_level:
                    # heading up
                    dx = xc - x0
                    dy = water_level - y0
                else:
                    # heading down
                    dx = x1 - xc
                    dy = water_level - y1

                peri = math.hypot(dx, dy)
                width = dx
                area = 0.5 * dx * dy

            result.append((rough, peri, width, area))

        return tuple(result)

    def Bchan(self):
        """Return channel width"""
        return self.channel[-1, 0] - self.channel[0, 0]

    def Bwet(self, d: float):
        """Return water surface width for given depth."""
        return sum(
            w for _, _, w, _ in self._wetted_segments(d, None, self.min_bed_level)
        )

    def P(self, d: float, loc: Loc | None = None):
        """Wetted perimeter for given water level."""
        return sum(
            p for _, p, _, _ in self._wetted_segments(d, loc, self.min_bed_level)
        )

    @lru_cache(maxsize=200)
    def _P_cached(self, d: float, loc: Loc | None, min_bed_level):
        """Wetted perimeter for given water level."""
        return sum(p for _, p, _, _ in self._wetted_segments(d, loc, min_bed_level))

    def area(self, d: float, loc: Loc | None = None):
        """Area of water below this height."""
        return sum(
            a for _, _, _, a in self._wetted_segments(d, loc, self.min_bed_level)
        )

    @lru_cache(maxsize=200)
    def _area_cached(self, d: float, loc: Loc | None, min_bed_level):
        """Area of water below this height."""
        return sum(a for _, _, _, a in self._wetted_segments(d, loc, min_bed_level))

    def nf(self, d: float, loc: Loc):
        """Return form roughness for the wetted cross-section.

        formrf * sum_k (r_k * p_k) / P

        formrf is the default form roughness of cross-section
        rk and pk are relative roughness and wetted perimeter
        P is the wetted perimeter

        """
        peri = 0.0
        weighted_p = 0.0

        for rough, p, _, _ in self._wetted_segments(d, loc, self.min_bed_level):
            peri += p
            weighted_p += rough * p

        # don't need to multiply by formrf since roughness already done that
        return weighted_p / peri

    @lru_cache(maxsize=200)
    def _nf_cached(self, d: float, loc: Loc, min_bed_level):
        """Return form roughness for the wetted cross-section.

        formrf * sum_k (r_k * p_k) / P

        formrf is the default form roughness of cross-section
        rk and pk are relative roughness and wetted perimeter
        P is the wetted perimeter

        """
        peri = 0.0
        weighted_p = 0.0

        for rough, p, _, _ in self._wetted_segments(d, loc, min_bed_level):
            peri += p
            weighted_p += rough * p

        # don't need to multiply by formrf since roughness already done that
        return weighted_p / peri

    def Qb_jli(self, t: pd.Timestamp, hydro):
        """Return bed material transport rate 2darray

        Parameters
        ----------
        t: pd.Timestamp
            Time

        Returns
        -------
        np.array:
            nbins x nlith 2d array.  (j, li) element is bed transport for li
            lith group and j proportion size
        """
        return self.layers.qb_jli(t, hydro) * self.Bwet(hydro.d[self.chainidx])

    def update_alayer_proportions(self, df: np.ndarray):
        self.layers.acfd += df

    def ng(self, loc: Loc):
        """Grain roughness in left/channel/right"""
        return 0.044 * self.d90(loc) ** (1 / 6)

    def conveyance(self, d: float):
        """K conveyance

        sum over left, main, right of
            A * R^(2/3) / (ng + nf)

        where
            A is the area
            R is A/P
            ng is grain roughness
            nf is form roughness
        """
        return self._conveyance_cached(d, self.min_bed_level)

    @lru_cache(maxsize=200)
    def _conveyance_cached(self, d: float, min_bed_level: float):
        """K conveyance

        sum over left, main, right of
            A * R^(2/3) / (ng + nf)

        where
            A is the area
            R is A/P
            ng is grain roughness
            nf is form roughness
        """
        # FIXME, not sure if ng is cacheable, grainsize distribution...
        K = 0
        for loc in (Loc.LEFT, Loc.CHANNEL, Loc.RIGHT):
            A = self._area_cached(d, loc, min_bed_level)
            P = self._P_cached(d, loc, min_bed_level)
            if P == 0:
                # no water in this part of channel
                continue
            R = A / P
            K += (
                A
                * R ** (2 / 3)
                / (self.ng(loc) + self._nf_cached(d, loc, min_bed_level))
            )

        assert K > 0, "No water"

        return K

    def R(self, d: float):
        """Hydraulic radius A/P over entire xsection"""
        return self._R_cached(d, self.min_bed_level)

    @lru_cache(maxsize=200)
    def _R_cached(self, d: float, min_bed_level: float):
        """Hydraulic radius A/P over entire xsection"""
        return self._area_cached(d, None, min_bed_level) / self._P_cached(
            d, None, min_bed_level
        )
