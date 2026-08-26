import math
import numpy as np
import pandas as pd
from gin import GrateConfig, CrossSectionProfile, GrainSizeProfiles
from layers import LayerStack
from hydrodynamics_models import HydroDynamicModel

# p = Profile(df)
# p.B(10 - p.bed_level)
# p.P(10 - p.bed_level)
# p.area(9 - p.bed_level)


class CrossSection:
    """A loaded CrossSectionProfile, includes metadata, xy points, derived props"""

    def __init__(
        self,
        xs: CrossSectionProfile,
        gs: GrainSizeProfiles,
        hydro: HydroDynamicModel,
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

        self.layers = LayerStack(self.chainidx, xs, gs, hydro)

    def _set_points(self, df):
        """Set profile points and calculate derived properties."""
        self.df = df

        # get roughness in for each segment in the profile
        if "relrf" in self.df.columns:
            self.df["roughness"] = self.df["relrf"].fillna(1) * self.formrf
        else:
            self.df["roughness"] = self.formrf

        # break into left, channel and right
        self.left, self.channel, self.right = self._split_pts_into_three(self.df)

        self.mean_bed_level = self._calculate_mean_bed_level()
        self.bed_level = self._calculate_bed_level()

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
        cs.layers = source.layers.interpolate(other.layers, f)
        cs.layers.chainidx = chainidx

        return cs

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

    def _wetted_segments(self, h: float):
        """Yield roughness, perimeter, width and area for each wetted segment."""
        water_level = self.bed_level + h

        x = self.df["x"].to_numpy()
        y = self.df["y"].to_numpy()
        r = self.df["roughness"].to_numpy()

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

            yield rough, peri, width, area

    def B(self, h: float):
        """Return water surface width for given depth."""
        return sum(w for _, _, w, _ in self._wetted_segments(h))

    def P(self, h: float):
        """Wetted perimeter for given water level."""
        return sum(p for _, p, _, _ in self._wetted_segments(h))

    def area(self, h: float):
        """Area of water below this height."""
        return sum(a for _, _, _, a in self._wetted_segments(h))

    def nf(self, h: float):
        """Return form roughness for the wetted cross-section.

        formrf * sum_k (r_k * p_k) / P

        formrf is the default form roughness of cross-section
        rk and pk are relative roughness and wetted perimeter
        P is the wetted perimeter

        """
        peri = 0.0
        weighted_p = 0.0

        for rough, p, _, _ in self._wetted_segments(h):
            peri += p
            weighted_p += rough * p

        return self.formrf * weighted_p / peri


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

    def _get_cross_sections(self) -> dict:
        """Return chainage point to CrossSection at that point"""
        xss = {}

        formrf = self._cfg.cross_sections.formrf
        wallrf = self._cfg.cross_sections.wallrf

        for xs in self._cfg.cross_sections.profiles:
            c = xs.chainage
            xss[c] = CrossSection(xs, self._cfg, formrf, wallrf)

        # check min/max chainage
        assert min(xss.keys()) <= self.cs[0], (
            f"Minimum chainage ({self.cs[0]}) isn't at least the minimum cross section chainage"
        )
        assert self.cs[-1] <= max(xss.keys()), (
            f"Maximum chainage ({self.cs[-1]}) is more than the maximum cross section chainage"
        )
        return xss

    def _get_interpolated_cross_sections(self):
        """Return a list of CrossSection at self.cs"""
        xss = self._get_cross_sections()
        xs_chainpts = sorted(xss)

        ixss = []

        for i, cpt in enumerate(self.cs):
            # Exact cross section
            if cpt in xss:
                xss[cpt].chainidx = i
                ixss.append(xss[cpt])
                continue

            # Find surrounding cross sections
            for c0, c1 in zip(xs_chainpts[:-1], xs_chainpts[1:]):
                if c0 <= cpt <= c1:
                    p0 = xss[c0]
                    p1 = xss[c1]
                    break
            else:
                raise ValueError(f"No cross sections surrounding chainage {cpt}")

            f = (cpt - c0) / (c1 - c0)
            ixss.append(p0.interpolate(p1, f, i))

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
        return self.xss[c].nf(h)

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
        xs = self.xss[c]
        B = xs.B(h)
        return (xs.get_formrf() * B + 2 * xs.get_wallrf() * h) / (B + 2 * h)


class River(Channel):
    def nf(self, c: int, h: float):
        """Form roughness

        nfc * sum_k (r_k * p_k) / pc

        where nfc is default form roughness of cross-section
        rk and pk are relative roughness factory wetted perimeter FIXME
        pc FIXME

        """
        raise NotImplementedError(f"nf({c} {h})")
