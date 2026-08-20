import numpy as np
import pandas as pd
from gin import GrateConfig


class Profile:
    """Currently just the x/y points across cross-section profile"""

    def __init__(self, points: pd.DataFrame, mbl: float | None = None):
        """FIXME, maybe don't bother passing in mbl"""

        self.points = points[["x", "y"]].copy()
        self.mean_bed_level = self._calculate_mean_bed_level() if mbl is None else mbl
        self.bed_level = self._calculate_bed_level()

    def _calculate_mean_bed_level(self):
        """Return weighted mean of the bed elevations"""
        x = self.points["x"].to_numpy()
        y = self.points["y"].to_numpy()
        dx = (x[2:] - x[:-2]) / 2
        return np.sum(dx * y[1:-1]) / np.sum(dx)

    def _calculate_bed_level(self):
        """The minimum bed level."""
        return self.points["y"].min()

    def shifted(self, dy: float) -> "Profile":
        points = self.points.copy()
        points["y"] += dy
        return Profile(points, mbl=self.mean_bed_level + dy)


class Channel:
    """Flume, river, braided channel"""

    def __init__(self, cfg: GrateConfig):
        self._cfg = cfg
        self.dc = self._cfg.discretisation.dc
        self.cs = self._chainpts()
        self.nc = len(self.cs)
        self.profiles = self._get_interpolated_profiles()

    def _chainpts(self):
        """Return chain points"""
        d = self._cfg.discretisation
        return np.arange(d.chainage_min, d.chainage_max + self.dc / 2, self.dc)

    def _get_cross_section_profiles(self):
        """Load the cross section csv profile files.

        FIXME: currently ignoring all info _except_ profile shape

        """
        profiles = {}
        for xs in self._cfg.cross_sections.profiles:
            c = xs.chainage
            df = pd.read_csv(xs.profile)
            profiles[c] = Profile(df[["x", "y"]])

        # check min/max chainage
        assert min(self._profiles.keys()) <= self.cs[0], (
            f"Minimum chainage ({self.cs[0]}) isn't at least the minimum cross section chainage"
        )
        assert self.cs[-1] <= max(self._profiles.keys()), (
            f"Maximum chainage ({self.cs[-1]}) is more than the maximum cross section chainage"
        )
        return profiles

    def _get_interpolated_profiles(self):
        """Interpolate cross section profiles onto our chainage points"""
        profs = self._get_cross_section_profiles()
        profile_cs = sorted(profs)

        iprofs = []

        for c in self.cs:
            # Exact profile
            if c in profs:
                iprofs.append(profs[c])
                continue

            # Find surrounding profiles
            for c0, c1 in zip(profile_cs[:-1], profile_cs[1:]):
                if c0 <= c <= c1:
                    p0 = profs[c0]
                    p1 = profs[c1]
                    break
            else:
                raise ValueError(f"No profiles surrounding chainage {c}")

            f = (c - c0) / (c1 - c0)
            target_bed = p0.mean_bed_level + f * (p1.mean_bed_level - p0.mean_bed_level)

            # Use whichever source profile is closer.
            p = p0 if f < 0.5 else p1
            iprofs.append(p.shifted(target_bed - p.mean_bed_level))

        return iprofs

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

    def area(self, c: int, h: float):
        """Return area of water between active layer surface and h"""
        raise NotImplementedError(f"area({c}, {h})")

    def nf(self, c: int):
        """Form roughness"""
        raise NotImplementedError(f"nf({c})")

    def P(self, c: int, h: float):
        """Wetted perimeter at chainage"""
        raise NotImplementedError(f"P({c}, {h})")

    def mean_bed_level(self, c: int):
        """Return mean bed level of profile at chainage c

        For a flume this is the bed_level, for other channels it is some sort
        of average of the cross-section profile
        """
        return self.profiles[c].mean_bed_level

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
        raise NotImplementedError(f"B({c}, {h})")

    def bed_level(self, c: int):
        """Return deepest part of the cross-section"""
        return self.profiles[c].bed_level


class Flume(Channel):
    def __init__(self, cfg: GrateConfig):
        super().__init__(self, cfg)
        self.active_layer_thickness = np.full(self.cs, self._cfg.morphological.la)
        self.storage_layer_thickness = np.full(self.cs, self._cfg.morphological.layer)

    def area(self, c: int, h: float):
        """Return area of water between water level and bed_level

        Parameters:
        -----------
        h: float
            The height of the water above bed_level
        """
        width = self.profiles[c].points["x"].max() - self.profiles[c].points["x"].min()
        return h * width

    def d90(self, c: int):
        """90 percentile of grain diameter of surface layer."""
        _ = c
        # FIXME
        return self._cfg.bankd90

    def nf(self, c: int):
        """Form roughness"""
        # FIXME
        _ = c
        return self._cfg.formrf


class River(Channel):
    def nf(self, c: int):
        """Form roughness

        nfc * sum_k (r_k * p_k) / pc

        where nfc is default form roughness of cross-section
        rk and pk are relative roughness factory wetted perimeter FIXME
        pc FIXME

        """
        raise NotImplementedError(f"nf({c})")
