import math
import numpy as np
import scipy.optimize
import pandas as pd
from gin import CrossSectionProfile, GrateConfig
from grainprofile import get_representative_grain_sizes, get_grain_props

KAPPA = 0.4  #  Von Kalmans constant
GRAVITY = 9.81
WATER_DENSITY = 1000


class LayerStack:
    """Holds active and storage layer grain profiles for a given cross
    section"""

    def __init__(self, xs: CrossSectionProfile, cfg: GrateConfig):
        self.chainage = xs.chainage
        self.chainidx = None  # will be sorted when interpolated
        gs = cfg.grain_size_profiles
        self.chi = cfg.morphological.chi
        self.nlith = gs.nlith

        # nlith in length
        self.abrasion_coeffs = gs.abrasion_coeffs
        self.sediment_densities = np.array(gs.sediment_densities)

        # nbins in length
        self.rgsizes = get_representative_grain_sizes(gs.grain_size_cfds)

        # nbins x nlith
        self.acfd = get_grain_props(
            xs.active_layer_group - 1, gs.grain_size_cfds, gs.lithfractions
        )
        self.scfd = get_grain_props(
            xs.storage_layer_group - 1, gs.grain_size_cfds, gs.lithfractions
        )

        # phi, representative bin value (Dj) in mm
        # 2** (( log(bot) + log(top) ) / 2)

    def interpolate(self, other: "LayerStack", f: float, chainidx: int) -> "LayerStack":
        """Return a new layer stack that is interped between me and other"""

        result = object.__new__(LayerStack)

        result.chainage = self.chainage + f * (other.chainage - self.chainage)
        result.chainidx = chainidx
        result.chi = self.chi
        result.nlith = self.nlith
        result.rgsizes = self.rgsizes

        for k in ["abrasion_coeffs", "sediment_densities", "acfd", "scfd"]:
            m = np.array(getattr(self, k))
            o = np.array(getattr(other, k))
            setattr(result, k, m + f * (o - m))

        return result

    def grain_shear_velocity(self, t: pd.Timestamp, hydro):
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

        u = hydro.u(t, self.chainidx)
        Sf = hydro.Sf(t, self.chainidx)
        ks = self.d90()
        h = hydro.h[self.chainidx]

        def f(ustar):
            hs = ustar**2 / GRAVITY / Sf
            return ustar - u * KAPPA / math.log(11 * hs / ks)

        init = u * KAPPA / math.log(11 * h / ks)
        ustar, info = scipy.optimize.newton(f, init, full_output=True)
        if not info.converged:
            raise ValueError(f"Can't solve grain_shear_velocity. {info=}")
        return ustar

    def grain_stress(self, t: pd.Timestamp, hydro):
        """Graint stress tau_g

        From equation 10.5

        rho grain_shear_velocity^2
        """
        ustar = self.grain_shear_velocity(t, hydro)
        return WATER_DENSITY * ustar**2

    def d90(self):
        """90th percentile of grain sizes in active layer."""
        return self._grain_size_percentile(0.9)

    def dsm(self):
        """Median of grain sizes in active layer."""
        return self._grain_size_percentile(0.5)

    def sand_fraction(self):
        """Fraction of grains with size < 2mm in active layer"""
        return self._grain_proportion_small_than(2)

    def _grain_size_percentile(self, x: float):
        """Grain size in active layer over all lith at this percentile

        Parameters
        ----------
        x: float
            Percentile between 0 and 1

        Returns
        -------
        float:
            The representative grain size over all lith at this percentile
        """

        assert 0 <= x <= 1

        # sum over lith groups and get cumulative sum
        prop = self.acfd.sum(axis=1)
        cf = np.cumsum(prop)

        # interpolate x in cf to find where we are in phi = -log(rgsizes)
        phi = np.interp(x, cf, -np.log(self.rgsizes))

        return np.exp(-phi)

    def _grain_proportion_small_than(self, x: float):
        """Proportion of grains (over all lith) smaller than given size

        Parameters
        ----------
        x: float
            Grain size in mm

        Returns
        -------
        float:
            Proportion of grains small than x
        """

        assert 0 < x

        # sum over lith groups and get cumulative sum
        prop = self.acfd.sum(axis=1)
        cf = np.cumsum(prop)

        # interpolate -log(x) in -log(rgsizes) to find where we are cf
        # we must reverse since np.interp expects x-coord to increase
        return np.interp(-np.log(x), -np.log(self.rgsizes)[::-1], cf[::-1])

    def qb_jli(self, t: pd.Timestamp, hydro):
        """Volumetric transport rate per unit width

        Wilcock & Crowe (2003), equation 9.33, qb_jc

        Returns
        -------
        np.array:
            nbins x nlith array.  (j, li) element is transport rate for
            jth proportion and li lith group.
        """

        rgsizes = np.expand_dims(self.rgsizes, 1)  # (nbins, 1)

        Fs = self.sand_fraction()
        phirm = 0.021 + 0.015 * np.exp(-20 * Fs)
        s = self.sediment_densities / WATER_DENSITY  # (nlith, )
        dsm = self.dsm()
        tau_rm = phirm * (s - 1) * WATER_DENSITY * GRAVITY * dsm
        b = 0.67 / (1 + np.exp(1.5 - rgsizes / dsm))  # (nbins, 1 )
        tau_rj = tau_rm * (rgsizes / dsm) ** b  # (nbins, nlith)
        phi = self.grain_stress(t, hydro) / tau_rj  # (nbins, nlith)
        Fj = self.acfd  # (nbins, nlith)
        ustar = self.grain_shear_velocity(t, hydro)

        q = Fj * ustar**3 / (s - 1) / GRAVITY  # (nbins, nlith)

        q *= np.where(
            phi < 1.35,
            0.002 * phi**7.5,
            14 * (1 - 0.894 / np.sqrt(phi)) ** 4.5,
        )

        return q

    def f_interface(self, aggrading: bool, p: float):
        """Interface distribution, how much is moving INTO active layer.

        Parameters
        ----------
        aggrading: bool
            If delta y > 0

        p: float
            Qb_jli / Qb for this cross section.  Sediment transfer in bed
        """

        if aggrading:
            return self.chi * self.acfd + (1 - self.chi) * p
        else:
            return self.scfd

    def update_grains_in_alayer(self, deltaf_jli: np.array):
        """Update proportion grain sizes

        Parameters
        ----------
        deltaf_jli: np.array
            nbins x nlith change in grain proportions
        """
        self.acfd += deltaf_jli
