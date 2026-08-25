import math
import scipy
import numpy as np
import pandas as pd
from gin import GrateConfig
from hydrodynamic_model import HydroDynamicModel
from channel import CrossSection

KAPPA = 0.4
GRAVITY = 9.81
WATER_DENSITY = 1000


class LayerInfo:
    """Holds active and storage layer grain profiles for a given cross
    section"""

    def __init__(self, chainpt: int, hydro: HydroDynamicModel):
        self.chainpt = chainpt
        self.hydro = hydro

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

        u = self.hydro.u(t, self.chainpt)
        Sf = self.hydro.Sf(t, self.chainpt)
        ks = self.d90()
        h = self.hydro.h[self.chainpt]

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

    def sand_fraction(self):
        """Fraction of grains with size < 2mm in active layer at given chainage"""
        # FIXME: need the grain sizes in active layer
        pass

    def grain_proportion_size(self, j: int):
        """Proportion of grain of size j"""
        pass

    def sediment_density(self, j: int):
        """Density of sediment grain j"""
        # this gives you density per lith group, need to map that to grain size
        # groups
        # cfg.grain_size_profiles.sediment_densities
        pass

    def d90(self):
        """90th percentile of grain sizes in active layer."""
        pass

    def dsm(self):
        """Median of grain sizes in active layer."""
        pass

    def dj(self, j: int):
        """Diameter of jth grain size in active layer."""
        pass

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
