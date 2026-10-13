import math
import numpy as np
import pandas as pd
from enum import Enum
from functools import lru_cache
from gin import CrossSectionProfile
from static_interpolator import StaticInterpolator


class Loc(Enum):
    LEFT = "left"
    CHANNEL = "channel"
    RIGHT = "right"
    ALL = "all"


# properties that might be undefined and need interpolating
XS_INTERP_PROPS = ["formrf", "bankd90", "bedrock_rl", "qsfact", "lsf"]


class CrossSection:
    """A loaded CrossSectionProfile, includes metadata, xy points, derived props

    Geometry and scalar properties only.  The grain-size layers for every
    chainage live in Channel._layers (a layer_arrays.LayerArrays).

    Currently handles Flume or River cross sections.  Braided channel will need
    subclass
    """

    def __init__(
        self,
        xs: CrossSectionProfile,
        default_formrf,
        # wallrf,
    ):
        self.chainage = xs.chainage
        self.chainidx = None  # this will get filled in when interped
        self.topoid = xs.topoid
        self.river_name = xs.river_name

        # might be "Interp" in which case we need to interpolate from neighbours
        self.formrf = xs.formrf if xs.formrf is not None else default_formrf
        self.bankd90 = xs.bankd90
        self.bedrock_rl = xs.bedrock_rl
        self.qsfact = xs.qsfact
        self.lsf = xs.lsf

        self.df = pd.read_csv(xs.profile)

        # can't set points yet since above variables might not be set
        # self._set_points(self.df)

    def __setup_interpolators(self):

        # bank interpolators take elevations, and channel takes depths
        #
        # but all the lambdas have to take elevation, so for channel add on
        # min_bed_level
        #
        bl = self.calc_min_bed_level()
        for name in ("area", "P", "Bwet", "nf"):
            setattr(
                self,
                f"_{name}_interpolator",
                {
                    Loc.LEFT: StaticInterpolator(
                        lambda wl: getattr(self, f"_{name}")(wl, Loc.LEFT),
                        self.left[:, 1],
                    )
                    if len(self.left)
                    else None,
                    Loc.CHANNEL: StaticInterpolator(
                        lambda depth: getattr(self, f"_{name}")(
                            depth + bl, Loc.CHANNEL
                        ),
                        self.channel[:, 1] - self.channel[:, 1].min(),
                    ),
                    Loc.RIGHT: StaticInterpolator(
                        lambda wl: getattr(self, f"_{name}")(wl, Loc.RIGHT),
                        self.right[:, 1],
                    )
                    if len(self.right)
                    else None,
                },
            )

    @classmethod
    def resolve_interpolated_properties(cls, xss: dict):
        """Resolve properties marked 'Interp' using neighbouring cross sections."""

        cs = sorted(xss)

        def interp(cidx: int, prop: str):
            """Interpolate at chainidx the given property."""

            i = cs.index(cidx)

            left = None
            for c in reversed(cs[:i]):
                val = getattr(xss[c], prop)
                if not (isinstance(val, str) and val == "Interp"):
                    left = c
                    break

            right = None
            for c in cs[i + 1 :]:
                val = getattr(xss[c], prop)
                if not (isinstance(val, str) and val == "Interp"):
                    right = c
                    break

            if left is None or right is None:
                raise ValueError(
                    f"Can't interp {prop!r} at chainage {cidx}: no valid cross-sections"
                )

            left_value = getattr(xss[left], prop)
            right_value = getattr(xss[right], prop)

            f = (cidx - left) / (right - left)

            return left_value + f * (right_value - left_value)

        for c in cs:
            for prop in XS_INTERP_PROPS:
                if getattr(xss[c], prop) == "Interp":
                    setattr(xss[c], prop, interp(c, prop))

        for xs in xss.values():
            xs._set_points(xs.df)

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

        self.mean_bed_level = self.calc_mean_bed_level()
        self.min_bed_level = self.calc_min_bed_level()

        self.__setup_interpolators()

    def interpolate(
        self, other: "CrossSection", f: float, chainidx: int
    ) -> "CrossSection":
        """Return an approximate interpolation between two cross sections.

        Geometry is taken from the closer cross section and shifted to the
        interpolated mean bed level. Numeric properties linearly interpolated.
        (Sediment layers are interpolated in Channel, see LayerArrays.from_config.)

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
        # cs._wallrf_del = interp(self._wallrf_del, other._wallrf_del, f)
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
        return cs

    def _split_pts_into_three(self, df):
        """Return entire profile, left, channel and right profiles as NumPy arrays.

        Each array has columns: x, y, roughness.
        """

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

    def calc_mean_bed_level(self):
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

    def calc_min_bed_level(self):
        """The minimum bed level."""
        return self.channel[:, 1].min()

    def aggrade_bed(self, dy):
        """Add dy to the channel bed level (dy can be negative)."""
        self.channel[:, 1] += dy
        self.min_bed_level += dy
        self.mean_bed_level += dy

    def _wetted_segments(self, wl: float, loc: Loc):
        """Yield roughness, perimeter, width and area for each wetted segment."""

        profile = {
            Loc.LEFT: self.left,
            Loc.CHANNEL: self.channel,
            Loc.RIGHT: self.right,
        }[loc]

        x = profile[:, 0]
        y = profile[:, 1]
        r = profile[:, 2]

        result = []

        for x0, x1, y0, y1, rough in zip(x[:-1], x[1:], y[:-1], y[1:], r[1:]):
            # seg is above
            if y0 > wl and y1 > wl:
                continue

            # seg below water
            if y0 <= wl and y1 <= wl:
                peri = math.hypot(x1 - x0, y1 - y0)
                width = x1 - x0
                area = width * (wl - (y0 + y1) / 2)

            # seg crosses
            else:
                f = (wl - y0) / (y1 - y0)
                xc = x0 + f * (x1 - x0)

                if y0 <= wl:
                    # heading up
                    dx = xc - x0
                    dy = wl - y0
                else:
                    # heading down
                    dx = x1 - xc
                    dy = wl - y1

                peri = math.hypot(dx, dy)
                width = dx
                area = 0.5 * dx * dy

            result.append((rough, peri, width, area))

        return tuple(result)

    def calc_Bchan(self):
        """Return channel width"""
        return self.channel[-1, 0] - self.channel[0, 0]

    def calc_Bwet(self, wl: float):
        """Return water surface width for given water level."""
        ret = 0
        for loc in (Loc.LEFT, Loc.CHANNEL, Loc.RIGHT):
            interp = self._Bwet_interpolator[loc]
            if interp is not None:
                ret += interp.eval(
                    wl - self.min_bed_level if loc is Loc.CHANNEL else wl
                )
        return ret

    def _Bwet(self, wl: float, loc: Loc):
        """Return water surface width for given water level."""
        return sum(w for _, _, w, _ in self._wetted_segments(wl, loc))

    def calc_area(self, wl: float, loc: Loc):
        """Area of water below this water level and above min_bed_level."""
        if loc == Loc.ALL:
            return sum(
                self.calc_area(wl, loc) for loc in (Loc.LEFT, Loc.CHANNEL, Loc.RIGHT)
            )

        interp = self._area_interpolator[loc]
        if interp is None:
            return 0.0

        return interp.eval(wl - self.min_bed_level if loc is Loc.CHANNEL else wl)

    def _area(self, wl: float, loc: Loc):
        return sum(a for _, _, _, a in self._wetted_segments(wl, loc))

    def calc_P(self, wl: float, loc: Loc):
        """Wetted perimeter for given water level."""

        interp = self._P_interpolator[loc]
        if interp is None:
            return 0.0
        return interp.eval(wl - self.min_bed_level if loc is Loc.CHANNEL else wl)

    def _P(self, wl: float, loc: Loc):
        """Wetted perimeter for given water level."""
        return sum(p for _, p, _, _ in self._wetted_segments(wl, loc))

    def calc_nf(self, wl: float, loc: Loc):
        """Return form roughness for the wetted cross-section.

        formrf * sum_k (r_k * p_k) / P

        formrf is the default form roughness of cross-section
        rk and pk are relative roughness and wetted perimeter
        P is the wetted perimeter

        """
        interp = self._nf_interpolator[loc]
        if interp is None:
            return 0.0
        return interp.eval(wl - self.min_bed_level if loc is Loc.CHANNEL else wl)

    def _nf(self, wl: float, loc: Loc):
        """Return form roughness for the wetted cross-section.

        formrf * sum_k (r_k * p_k) / P

        formrf is the default form roughness of cross-section
        rk and pk are relative roughness and wetted perimeter
        P is the wetted perimeter

        """
        peri = 0.0
        weighted_p = 0.0

        for rough, p, _, _ in self._wetted_segments(wl, loc):
            peri += p
            weighted_p += rough * p

        # don't need to multiply by formrf since roughness already done that
        return weighted_p / peri if peri != 0 else 0

    def calc_conveyance(self, A: np.array, P: np.array, nf: np.array, ng: np.array):
        """Calculate conveyance using known A, P, nf, ng

        Parameters
        ----------
        A: np.array
            area of left, channel and right

        P: np.array
            wetted periment of left, channel and right

        nf: np.array
            roughness of left, channel and right

        ng: np.array
            grain roughness of left, channel and right
        """

        mask = P != 0
        K = np.sum(A[mask] * (A[mask] / P[mask]) ** (2 / 3) / (ng[mask] + nf[mask]))
        assert K > 0, "No water"
        return K

    def calc_conveyance_using_wl(self, wl: float, ng_array):
        """K conveyance

        sum over left, main, right of
            A * R^(2/3) / (ng + nf)

        where
            A is the area
            R is A/P
            ng is grain roughness
            nf is form roughness
        """
        K = 0
        for loc, ng in zip((Loc.LEFT, Loc.CHANNEL, Loc.RIGHT), ng_array):
            A = self.calc_area(wl, loc)
            P = self.calc_P(wl, loc)
            if P == 0:
                # no water in this part of channel
                continue
            R = A / P
            K += A * R ** (2 / 3) / (ng + self.calc_nf(wl, loc))

        if K <= 0:
            raise ValueError(
                f"No water {K=} {wl=} {ng_array=} {self.left=} {self.channel=} {self.right=}"
            )

        return K

    # def R(self, wl: float):
    #     """Hydraulic radius A/P over entire xsection"""
    #     return self.area(wl, Loc.ALL) / self._P(wl, Loc.ALL)
