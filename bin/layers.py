import math
import numpy as np
import scipy
import pandas as pd
from gin import CrossSectionProfile, GrainSizeProfiles
from hydrodynamics_models import HydroDynamicModel

KAPPA = 0.4
GRAVITY = 9.81
WATER_DENSITY = 1000


class LayerStack:
    """Holds active and storage layer grain profiles for a given cross
    section"""

    def __init__(
        self, xs: CrossSectionProfile, gs: GrainSizeProfiles, hydro: HydroDynamicModel
    ):
        self.chainage = xs.chainage
        self.chainidx = None  # will be sorted when interpolated
        self.nlith = gs.nlith

        # nlith in length
        self.abrasion_coeffs = gs.abrasion_coeffs
        self.sediment_densities = gs.sediment_densities

        # number of bins in length
        self.rgsizes = self._get_representative_grain_sizes(gs.grain_size_cfds)

        # nlith in length of nbins
        self.acfd = self._get_props(gs.grain_size_cfds, xs.active_layer_group - 1)
        self.scfd = self._get_props(gs.grain_size_cfds, xs.storage_layer_group - 1)

        self.hydro = hydro

        # phi, representative bin value (Dj) in mm
        # 2** (( log(bot) + log(top) ) / 2)

    def _get_represenative_grain_sizes(self, cfds):
        """Return the representative grain sizes

        2^( (log(bot) + log(top))/2 )
        """
        # get first column, the sizes from cfds.
        p = [i[0] for i in cfds]
        return [
            2 ** ((math.log(bot) + math.log(top)) / 2)
            for bot, top in zip(p[:-1], p[1:])
        ]

    def _get_props(self, cfds, pid: int):
        """Return the fractions or proportions for given profile index.

        Parameters
        ----------
        pid: int
            The grain profile 1-based index

        Returns
        -------
        list of length nlith:
            For each lithology we have a list each element is the fraction of
            grains of this representative size.
        """
        # this is the cummulative frequency for this profile
        cfd = [i[pid] for i in cfds]

        # convert to a proportion fraction
        total = cfd[-1]
        prop = [cfd[0] / total, *[(b - a) / total for a, b in zip(cfd[:-1], cfd[1:])]]

        if (nlith := self.nlith) == 1:
            return [prop]

        # need the pid'th column of lithfractions
        lfracs = [lf[pid] for lf in self.grain_size_profiles.lithfractions]

        # number of bins x nlith
        weights = np.array(lfracs).reshape(-1, nlith)
        weights = weights / weights.sum(axis=1, keepdims=1)

        return (np.expand_dims(prop, 1) * weights).T.tolist()

    def interpolate(self, other: "LayerStack", f: float) -> "LayerStack":
        """Return a new layer stack that is interped between me and other"""

        result = object.__new__(LayerStack)

        result.chainage = self.chainage + f * (other.chainage - self.chainage)
        result.c = "FIXME"
        result.nlith = self.nlith
        result.rgsizes = self.rgsizes
        result.hydro = self.hydro

        for k in ["abrasion_coeffs", "sediment_densities", "acfd", "scfd"]:
            m = np.array(getattr(self, k))
            o = np.array(getattr(other, k))
            setattr(result, k, (m + f * (o - m)).tolist())

        return result

    def grain_shear_velocity(self, t: pd.Timestamp):
        """Return grain shear velocity, u^*

        From equation 10.3

        u / u^* = 1/kappa ln(11 * hs / ks)

        u is channel water velocity
        hs is flow depth attributable to grain roughness
        kappa is Von Kalman's constant, 0.4
        ks is the equivalent sand grain roughness = 2 d90
            where d90 is the 90th percentile of grain sizes in active layer

        This is used in the Wilcock & Crowe (2003) formula for qb_jc
        """

        u = self.hydro.u(t, self.chainidx)
        Sf = self.hydro.Sf(t, self.chainidx)
        ks = self.d90()
        h = self.hydro.h[self.chainidx]

        def f(ustar):
            hs = ustar**2 / GRAVITY / Sf
            return ustar - u * KAPPA / math.log(11 * hs / ks)

        init = u * KAPPA / math.log(11 * h / ks)
        ustar, info = scipy.optimize.newton(f, init, full_output=True)
        if not info.converged:
            raise ValueError(f"Can't solve grain_shear_velocity. {info=}")
        return ustar

    def grain_stress(self, t: pd.Timestamp):
        """Graint stress tau_g

        From equation 10.5

        rho grain_shear_velocity^2
        """
        ustar = self.grain_shear_velocity(t)
        return WATER_DENSITY * ustar**2

    def grain_proportion_size(self, j: int, li: int):
        """Proportion of grain of size j"""
        return self.acfd[li][j]

    def dj(self, j: int):
        """Representative diameter of jth grain size in active layer."""
        return self.rgsizes[j]

    def sediment_density(self, li: int):
        """Density of sediment lith type li"""
        # cfg.grain_size_profiles.sediment_densities
        return self.sediment_densities[li]

    def sand_fraction(self):
        """Fraction of grains with size < 2mm in active layer"""
        # FIXME: need the grain sizes in active layer
        pass

    def d90(self, li: int):
        """90th percentile of grain sizes in active layer for lith li."""
        # FIXME, maybe not per lith
        return self._percentile(0.9, self.acdf[li])

    def dsm(self, li: int):
        """Median of grain sizes in active layer."""
        # FIXME, maybe not per lith
        return self._percentile(0.5, self.acdf[li])

    def qb_jc(self, t: pd.Timestamp, j: int):
        """Volumetric transport rate per unit width

        Wilcock & Crowe (2003), equation 9.33, qb_jc
        """

        Fs = self.sand_fraction()
        phirm = 0.021 + 0.015 * math.exp(-20 * Fs)
        s = self.sediment_density(j) / WATER_DENSITY
        dsm = self.dsm()
        dj = self.dj(j)
        tau_rm = phirm * (s - 1) * WATER_DENSITY * GRAVITY * dsm
        b = 0.67 / (1 + math.exp(1.5 - dj / dsm))
        tau_rj = tau_rm * (dj / dsm) ** b
        phi = self.grain_stress(t) / tau_rj
        Fj = self.grain_proportion_size(j)
        ustar = self.grain_shear_velocity(t)

        q = Fj * ustar**3 / (s - 1) / GRAVITY
        if phi < 1.35:
            q *= 0.002 * phi**7.5
        else:
            q *= 14 * (1 - 0.894 / math.sqrt(phi)) ** 4.5

        return q
