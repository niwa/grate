import numpy as np

import scipy
import pandas as pd
from gin import CrossSectionProfile, GrateConfig
from grainprofile import get_representative_grain_sizes, get_grain_props

KAPPA = 0.4  #  Von Kalmans constant
GRAVITY = 9.81
WATER_DENSITY = 1000
SAND_SIZE = 0.002  # sand grain size in metres


class LayerStack:
    """Holds active and storage layer grain profiles for a given cross
    section"""

    def __init__(self, xs: CrossSectionProfile, cfg: GrateConfig):
        self.chainage = xs.chainage
        self.chainidx = None  # will be sorted when interpolated
        gs = cfg.grain_size_profiles
        self.chi = cfg.morphological.chi
        self.nlith = gs.num_lith

        # nlith in length
        self.abrasion_coeffs = tuple(gs.abrasion_coeffs)
        self.sediment_densities = np.array(gs.sediment_densities)

        # nbins in length
        self.rgsizes = get_representative_grain_sizes(gs.grain_size_cfds)

        # nbins x nlith
        # if groups are good, we can setup acfd and scfd
        if xs.active_layer_group != "Interp":
            self._acfd = get_grain_props(
                xs.active_layer_group - 1, gs.grain_size_cfds, gs.lithfractions
            )
        else:
            self._acfd = "Interp"

        if xs.storage_layer_group != "Interp":
            self._scfd = get_grain_props(
                xs.storage_layer_group - 1, gs.grain_size_cfds, gs.lithfractions
            )
        else:
            self._scfd = "Interp"

    def __str__(self):
        return f"Layer {self.chainage=} {self.chainidx=}"

    def _set_grain_props(self):
        self._sand_fraction = self._grain_proportion_smaller_than(SAND_SIZE)
        self._d90 = self._grain_size_percentile(0.9)
        self._dsm = self._grain_size_percentile(0.5)

    def interpolate(self, other: "LayerStack", f: float, chainidx: int) -> "LayerStack":
        """Return a new layer stack that is interped between me and other"""

        result = object.__new__(LayerStack)

        result.chainage = self.chainage + f * (other.chainage - self.chainage)
        result.chainidx = chainidx
        result.chi = self.chi
        result.nlith = self.nlith
        result.rgsizes = self.rgsizes

        for k in ["_acfd", "_scfd"]:
            m = np.array(getattr(self, k))
            o = np.array(getattr(other, k))
            setattr(result, k, m + f * (o - m))

        # these are constant
        result.abrasion_coeffs = self.abrasion_coeffs
        result.sediment_densities = self.sediment_densities

        # recalculate sand_fraction, dsm and d90, not interpolate
        result._sand_fraction = result._grain_proportion_smaller_than(SAND_SIZE)
        result._d90 = result._grain_size_percentile(0.9)
        result._dsm = result._grain_size_percentile(0.5)

        return result

    def grain_shear_velocity(self, hydro):
        """Return grain shear velocity, u^*

        From equation 10.3

        u / u^* = 1/kappa ln(11 * hs / ks)

        u is channel water velocity
        hs is flow depth attributable to grain roughness and is u^*^2/g/Sf
        kappa is Von Kalman's constant, 0.4
        ks is the equivalent sand grain roughness = 2 d90
            where d90 is the 90th percentile of grain sizes in active layer

        Used in the Wilcock & Crowe (2003) formula for qb_jc

        Let x be u^*, so we have to solve

            u KAPPA = x ln(11 * x^2 / g / Sf / ks)

        Let
            a = u KAPPA
            b = 11 / (g * Sf * ks)
        So the equation is
            x ln (b x^2) - a = 0

        This can be solved `analytically`, the solution is
            x = a / 2 / W(a sqrt(b) / 2)
        where W is the lambertw
        """

        Sf = hydro.get_Sf(self.chainidx)
        ks = 2 * self._d90

        a = KAPPA * hydro.get_u(self.chainidx)
        b = 11 / GRAVITY / Sf / ks

        x = a * np.sqrt(b) / 2
        assert x > 0, "ustar calc, arg to lambertw <= 0, possibly complex solutions"
        return (a / 2 / scipy.special.lambertw(x)).real

    def grain_stress(self, hydro):
        """Grain stress tau_g

        From equation 10.5

        rho grain_shear_velocity^2
        """
        ustar = self.grain_shear_velocity(hydro)
        return WATER_DENSITY * ustar**2

    def get_d90(self):
        """90th percentile of grain sizes in active layer."""
        return self._d90

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
        prop = self._acfd.sum(axis=1)
        cf = np.cumsum(prop)

        # interpolate x in cf to find where we are in phi = -log(rgsizes)
        phi = np.interp(x, cf, -np.log(self.rgsizes))

        return np.exp(-phi)

    def _grain_proportion_smaller_than(self, x: float):
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
        prop = self._acfd.sum(axis=1)
        cf = np.cumsum(prop)

        # interpolate -log(x) in -log(rgsizes) to find where we are cf
        # we must reverse since np.interp expects x-coord to increase
        return np.interp(-np.log(x), -np.log(self.rgsizes)[::-1], cf[::-1])

    def qb_jli(self, hydro):
        """Volumetric transport rate per unit width

        Wilcock & Crowe (2003), equation 9.33, qb_jc

        Returns
        -------
        np.array:
            nbins x nlith array.  (j, li) element is transport rate for
            jth proportion and li lith group.
        """

        rgsizes = np.expand_dims(self.rgsizes, 1)  # (nbins, 1)

        Fs = self._sand_fraction
        phirm = 0.021 + 0.015 * np.exp(-20 * Fs)
        s = self.sediment_densities / WATER_DENSITY  # (nlith, )
        dsm = self._dsm
        tau_rm = phirm * (s - 1) * WATER_DENSITY * GRAVITY * dsm
        b = 0.67 / (1 + np.exp(1.5 - rgsizes / dsm))  # (nbins, 1 )
        tau_rj = tau_rm * (rgsizes / dsm) ** b  # (nbins, nlith)
        phi = self.grain_stress(hydro) / tau_rj  # (nbins, nlith)
        Fj = self._acfd  # (nbins, nlith)
        ustar = self.grain_shear_velocity(hydro)

        q = Fj * ustar**3 / (s - 1) / GRAVITY  # (nbins, nlith)

        # print(
        #     f"gs={self.grain_stress(t, hydro)} {self.sediment_densities=}  {dsm=} {tau_rm=} tau_rj={tau_rj} phi={phi}"
        # )

        result = np.empty_like(phi)
        mask = phi < 1.35
        result[mask] = 0.002 * phi[mask] ** 7.5
        result[~mask] = 14 * (1 - 0.894 / np.sqrt(phi[~mask])) ** 4.5

        # q *= np.where(
        #     phi < 1.35,
        #     0.002 * phi**7.5,
        #     14 * (1 - 0.894 / np.sqrt(phi)) ** 4.5,
        # )
        q *= result

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
            return self.chi * self._acfd + (1 - self.chi) * p
        else:
            return self._scfd

    def add_to_acfd(self, df):
        self._acfd += df
        self._sand_fraction = self._grain_proportion_smaller_than(SAND_SIZE)
        self._dsm = self._grain_size_percentile(0.5)
        self._d90 = self._grain_size_percentile(0.9)
