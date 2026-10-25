import math
import copy
import numpy as np
import pandas as pd
from enum import Enum
from gin import CrossSectionProfile

from static_interpolator import StaticInterpolator
from grainprofile import (
    get_grain_props,
    # get_representative_grain_sizes,
    # get_grain_size_bin_uppers,
)


class Loc(Enum):
    LEFT = "left"
    CHANNEL = "channel"
    RIGHT = "right"


# properties that might be undefined and need interpolating
XS_INTERP_PROPS = [
    "formrf",
    "bankd90",
    "bedrock_rl",
    "qsfact",
    "lsf",
    "_aprop",
    "_sprop",
]


class CrossSection:
    """A loaded CrossSectionProfile, includes metadata, xy points, derived props

    Currently handles Flume or River cross sections.  Braided channel will need
    subclass
    """

    def __init__(self, xs: CrossSectionProfile, cfg):
        self.chainage = xs.chainage
        # self.chainidx = None  # this must be filled in later

        # might be "Interp" in which case we need to interpolate from neighbours
        self.formrf = xs.formrf if xs.formrf is not None else cfg.cross_sections.formrf
        self.bankd90 = xs.bankd90
        self.bedrock_rl = xs.bedrock_rl
        self.qsfact = xs.qsfact
        self.lsf = xs.lsf

        # get profile, but can't set roughness, might be Interp
        self.df = pd.read_csv(xs.profile)

        # nbins x nlith
        gs = cfg.grain_size_profiles
        # if groups are good, we can setup aprop and sprop
        if xs.active_layer_group != "Interp":
            self._aprop = get_grain_props(
                xs.active_layer_group - 1, gs.grain_size_cfds, gs.lithfractions
            )
        else:
            self._aprop = "Interp"

        if xs.storage_layer_group != "Interp":
            self._sprop = get_grain_props(
                xs.storage_layer_group - 1, gs.grain_size_cfds, gs.lithfractions
            )
        else:
            self._sprop = "Interp"

    def _split_pts_into_three(self, df):
        """Return left, channel and right profiles as NumPy arrays.

        Each array has columns: x, y, roughness.
        """

        if "ob" not in df.columns:
            data = df[["x", "y", "roughness"]].to_numpy()
            return data[:0], data, data[:0]

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

        return left, channel, right

    def _calc_mean_bed_level(self):
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

    def _calc_min_bed_level(self):
        """The minimum bed level."""
        return self.channel[:, 1].min()

    @classmethod
    def spaced_chainpts(cls, cfg):
        """Return probably spaced chain points

        The chainage's hit the cross section profiles in cfg, but potentially
        introduce new ones if they are too widely spaced
        """
        c = np.sort([xs.chainage for xs in cfg.cross_sections.profiles])

        # number of intervals between each chainage
        gaps = np.diff(c)
        ints = np.ceil(gaps / cfg.discretisation.max_dc).astype(int)

        return np.concatenate(
            [np.linspace(a, b, n + 1)[:-1] for a, b, n in zip(c[:-1], c[1:], ints)]
            + [c[-1:]]
        )

    @classmethod
    def load_interped_cross_sections(cls, cfg):
        """Get cross_sections, interp them onto new chainage"""
        xss = cls.load_cross_sections(cfg)
        cls.resolve_interped_props(xss)
        cs = cls.spaced_chainpts(cfg)
        return cls.interp_onto_new_chainages(xss, cs)

    @classmethod
    def load_cross_sections(cls, cfg):
        """Return un-interpolated cross sections from cfg"""

        return {
            xs.chainage: CrossSection(xs, cfg) for xs in cfg.cross_sections.profiles
        }

    @classmethod
    def resolve_interped_props(cls, xss: dict):
        """Resolve Interp properties using neighbours, modifying xss in place

        Returns
        -------
        None
        """

        # sorted chainages
        cidxs = sorted(xss)

        def interp(cidx: int, prop: str):
            """Interpolate at chainidx the given property."""

            i = cidxs.index(cidx)

            left = None
            for c in reversed(cidxs[:i]):
                val = getattr(xss[c], prop)
                if not (isinstance(val, str) and val == "Interp"):
                    left = c
                    break

            right = None
            for c in cidxs[i + 1 :]:
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

        for cidx in sorted(xss):
            for prop in XS_INTERP_PROPS:
                val = getattr(xss[cidx], prop)
                if isinstance(val, str) and val == "Interp":
                    setattr(xss[cidx], prop, interp(cidx, prop))

        # now we can set roughness since formrf done
        for xs in xss.values():
            if "relrf" in xs.df.columns:
                xs.df["roughness"] = xs.df["relrf"].fillna(1) * xs.formrf
            else:
                xs.df["roughness"] = xs.formrf

        # can split df into left/channel/right and calculate bed levels
        for xs in xss.values():
            # break into left, channel and right
            xs.left, xs.channel, xs.right = xs._split_pts_into_three(xs.df)
            xs.mean_bed_level = xs._calc_mean_bed_level()
            xs.min_bed_level = xs._calc_min_bed_level()

    @classmethod
    def interp_onto_new_chainages(cls, xss: dict, cs: np.array) -> list:
        """Make new CrossSections on the given chainages by interpolation

        Parameters
        ----------
        xss: dict
            Maps chainage to existing CrossSections.  There must be a CrossSection
            at cs.min and cs.max.

        cs: np.array
            Sorted chainage points

        Returns
        -------
        list:
            New CrossSections fully interpolated
        """

        def __interp_two(left, right, f: float) -> CrossSection:
            """Return an approximate interpolation between two cross sections.

            Geometry is taken from the closer cross section and shifted to the
            interpolated mean bed level. Numeric properties linearly interpolated.

            Parameters
            ----------
            left: CrossSection
                Cross section to the left to interpolate with

            right: CrossSection
                Cross section to the right to interpolate with

            f: float
                interpolation proportion 0 to 1.  0 means left, 1 means right
            """

            def interp(a, b, f):
                # some of the props can be None, they are valid
                if a is None:
                    if b is None:
                        return None
                    else:
                        return b
                elif b is None:
                    return a

                return a + f * (b - a)

            assert 0 <= f <= 1, (
                f"Interpolation fraction must be between 0 and 1, got {f}"
            )

            # the new one
            source = left if f < 0.5 else right
            me = copy.deepcopy(source)
            # this isn't perfectly accurate, so skip, set it outside interp
            # me.chainage = left.chainage + f * (right.chainage - left.chainage)

            for prop in XS_INTERP_PROPS:
                newval = interp(getattr(left, prop), getattr(right, prop), f)
                setattr(me, prop, newval)

            # Interpolate the mean bed level, then use the nearer geometry
            t = left.mean_bed_level + f * (right.mean_bed_level - left.mean_bed_level)
            me.df["y"] += t - source.mean_bed_level
            return me

        # sorted current chainages
        xs_chainpts = sorted(xss)

        newxss = []
        for chainage in cs:
            # Exact cross section
            if chainage in xss:
                newxss.append(xss[chainage])
                continue

            # Find surrounding cross sections
            for c0, c1 in zip(xs_chainpts[:-1], xs_chainpts[1:]):
                if c0 <= chainage <= c1:
                    left = xss[c0]
                    right = xss[c1]
                    break
            else:
                raise ValueError(f"No cross sections surrounding chainage {chainage}")

            f = (chainage - c0) / (c1 - c0)
            new = __interp_two(left, right, f)
            new.chainage = chainage
            new.left, new.channel, new.right = new._split_pts_into_three(new.df)
            new.mean_bed_level = new._calc_mean_bed_level()
            new.min_bed_level = new._calc_min_bed_level()
            newxss.append(new)

        return newxss

    def calc_Bchan(self):
        """Return channel width"""
        return self.channel[-1, 0] - self.channel[0, 0]

    def get_profile_interpolators(self, name):
        """Profile interpolator for area, P, Bwet, nf

        Parameters
        ----------
        name: str
            area|P|Bwet|nf

            P is wetted perimeter
            Bwet is wetted surface
            nf is roughness for the wetted cross section

        Returns
        -------
        tuple:
            [StaticInterpolator|None, StaticInterpolator|None, StaticInterpolator|None]
            These give the values of name for the left/channel/right.

            If the interpolator is None, there is no left/channel/right, you
            can treat the value as 0.0.

            The left and right interpolators take wl, the channel takes depth
        """

        # all the lambdas have to take elevation, so for channel add on
        # min_bed_level.  The lambdas are used to create the tables, after that
        # the x values are wl, depth, wl
        #
        bl = self.min_bed_level
        return [
            StaticInterpolator(
                lambda wl: getattr(self, f"_{name}")(wl, Loc.LEFT), self.left[:, 1]
            )
            if len(self.left)
            else None,
            StaticInterpolator(
                lambda depth: getattr(self, f"_{name}")(depth + bl, Loc.CHANNEL),
                self.channel[:, 1] - bl,
            ),
            StaticInterpolator(
                lambda wl: getattr(self, f"_{name}")(wl, Loc.RIGHT), self.right[:, 1]
            )
            if len(self.right)
            else None,
        ]

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

    def _Bwet(self, wl: float, loc: Loc):
        """Return water surface width for given water level."""
        return sum(w for _, _, w, _ in self._wetted_segments(wl, loc))

    def _area(self, wl: float, loc: Loc):
        return sum(a for _, _, _, a in self._wetted_segments(wl, loc))

    def _P(self, wl: float, loc: Loc):
        """Wetted perimeter for given water level."""
        return sum(p for _, p, _, _ in self._wetted_segments(wl, loc))

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
