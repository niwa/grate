import scipy
import numpy as np
import pandas as pd
from gin import GrateConfig
from cross_section import CrossSection
from grainprofile import (
    get_representative_grain_sizes,
    get_grain_size_bin_uppers,
)
from storage_layers import StorageLayers

SAND_SIZE = -np.log(0.002)  # sand grain size in phi units
KAPPA = 0.4  # Von Karman's constant
GRAVITY = 9.81
WATER_DENSITY = 1000


class Channel:
    """Covers Flume and River.  Braided channel needs a subclass

    A design choice is that getters should return cached array info, functions
    starting with _calc calculate variables
    """

    def __init__(self, cfg: GrateConfig):
        self._cfg = cfg
        self.poro = cfg.morphological.poro
        self._init_dt()
        self.chi = cfg.morphological.chi
        self.alayer_max_thickness = cfg.morphological.alayer_max_thickness
        self.slayer_max_thickness = cfg.morphological.slayer_max_thickness
        self.num_slayers = cfg.morphological.num_slayers

        # cross sections
        xss = CrossSection.load_interped_cross_sections(cfg)
        self._cs = np.array([xs.chainage for xs in xss])
        self._nc = len(self._cs)

        # None (use channel d90 for banks too) becomes nan
        self._bankd90 = np.array(
            [np.nan if xs.bankd90 is None else xs.bankd90 for xs in xss]
        )

        # this don't change over time
        self._Bchan = np.array([xs.calc_Bchan() for xs in xss])
        gs = cfg.grain_size_profiles
        self._sediment_densities = np.asarray(gs.sediment_densities)
        self._rgsizes = get_representative_grain_sizes(gs.grain_size_cfds)
        self._uppersizes = get_grain_size_bin_uppers(gs.grain_size_cfds)
        self._phi = -np.log(self._uppersizes)
        self._bedrock = np.array([xs.bedrock_rl for xs in xss], dtype=np.float64)
        self._bedrock = np.nan_to_num(self._bedrock, nan=-np.inf)

        # these should get updated at sediment transport aggradation
        self._min_bed_level = np.array([xs.min_bed_level for xs in xss])
        self._mean_bed_level = np.array([xs.mean_bed_level for xs in xss])

        # these are set after propogate sediment
        nbins = cfg.grain_size_profiles.num_bins
        nlith = cfg.grain_size_profiles.num_lith
        self._trans_rate_jli = np.zeros((self._nc, nbins, nlith))
        self._cum_transport = np.zeros(self._trans_rate_jli.shape)
        self._gsv = np.ones(self._nc)
        self._sand_fraction = np.empty(self._nc)
        self._dsm = np.empty(self._nc)
        self._d90 = np.empty((self._nc, 3))
        self._ng = np.empty((self._nc, 3))

        self._aprop = np.array([xs._aprop.copy() for xs in xss])  # (nc, nbins, nlith)
        self._athickness = np.minimum(
            self._min_bed_level - self._bedrock, self.alayer_max_thickness
        )

        self._storage_layers = StorageLayers(
            cfg, xss, self._min_bed_level - self._bedrock - self._athickness
        )

        # the profile interpolators for A, P, Bwet, and nf
        self._profile_tabs = [
            {
                name: xs.get_profile_interpolators(name)
                for name in ("area", "P", "Bwet", "nf")
            }
            for xs in xss
        ]

        self.set_grain_arrays()

    def _init_dt(self):
        # start out at max dt which the cfg has set under dt
        self._dt = self.max_dt = self._cfg.max_dt

        # start out 10% of max rate
        self.cdt = self._cfg.simulation_time.cdt
        self.max_dydt = 0.1 * self.cdt / self._dt

    def _set_next_dt(self):
        self.max_dydt = max(1e-10, self.max_dydt)
        self._dt = min(self.cdt / self.max_dydt, self.max_dt)

    def get_dt(self):
        return self._dt

    def set_dt(self, dt):
        self._dt = dt

    def get_Qb_jli(self):
        return self._trans_rate_jli

    def get_trans_rate_jli(self):
        return self._trans_rate_jli

    def get_cum_transport(self):
        return self._cum_transport

    def get_grain_shear_velocity(self):
        return self._gsv

    def get_gsv(self):
        return self._gsv

    def get_sand_fraction(self):
        return self._sand_fraction

    def get_dsm(self):
        return self._dsm

    def get_d90(self):
        return self._d90

    def get_mean_bed_level(self):
        """Return mean bed level array

        For a flume this is the bed_level, for other channels it is some sort
        of average of the cross-section profile
        """
        return self._mean_bed_level

    def get_min_bed_level(self):
        """Return deepest part of the cross-section"""
        return self._min_bed_level

    def get_layer_thickness(self):
        """Return thickness of active and storage for chainages

        Returns
        -------
        np.array
            (nc, 1 + num_slayers).  The first layer is active layer, then top
            storage etc.  Any layer
            can have zero thickness, and if the wl is close to the bedrock the
            lower storage layers often do have zero thickness.  The lowest
            storage layer can have inf thickness.  Storage layers that don't
            exist have NaN thickness
        """
        return np.column_stack([self._athickness, self._storage_layers.get_thickness()])

    def get_layer_composition(self):
        """Return active and storage layer proportions

        Returns:
        -------
        np.array
            (nc, 1 + num_slayers, nbin, nlith)
            The first layer is the active layer, then storage layers from the
            top down.  Storage layers that don't exist are filled with nan.
        """

        return np.concatenate(
            [self._aprop[:, None, :, :], self._storage_layers.get_sprop()], axis=1
        )

    def get_bedrock(self):
        """Return bedrock elevation for each chainage"""
        return self._bedrock

    def calc_area(self, c: int, wl: float):
        """Return area of water between bed and wl"""
        tab = self._profile_tabs[c]["area"]
        d = wl - self._min_bed_level[c]
        return np.array(
            [
                tab[0].eval(wl) if tab[0] else 0,
                tab[1].eval(d) if tab[1] else 0,
                tab[2].eval(wl) if tab[2] else 0,
            ]
        )

    def _calc_P(self, c: int, wl: float):
        """Return wetted perimeter"""
        tab = self._profile_tabs[c]["P"]
        d = wl - self._min_bed_level[c]
        return np.array(
            [
                tab[0].eval(wl) if tab[0] else 0,
                tab[1].eval(d) if tab[1] else 0,
                tab[2].eval(wl) if tab[2] else 0,
            ]
        )

    def _calc_Bwet(self, c: int, wl: float) -> float:
        """Return water width"""
        tab = self._profile_tabs[c]["Bwet"]
        d = wl - self._min_bed_level[c]
        return (
            (tab[0].eval(wl) if tab[0] else 0)
            + (tab[1].eval(d) if tab[1] else 0)
            + (tab[2].eval(wl) if tab[2] else 0)
        )

    def _calc_nf(self, c: int, wl: float) -> float:
        """Return roughness"""
        tab = self._profile_tabs[c]["nf"]
        d = wl - self._min_bed_level[c]
        return np.array(
            [
                tab[0].eval(wl) if tab[0] else 0,
                tab[1].eval(d) if tab[1] else 0,
                tab[2].eval(wl) if tab[2] else 0,
            ]
        )

    def calc_conveyance(self, c: int, wl: float) -> float:
        """K conveyance

        sum over left, main, right of
            A * R^(2/3) / (ng + nf)

        where
            A is the area
            R is A/P
            ng is grain roughness
            nf is form roughness

        Parameters
        ----------
        wl: float
            Water level

        ng: np.array
            grain roughness in left, channel, right

        """
        A = self.calc_area(c, wl)
        P = self._calc_P(c, wl)
        nf = self._calc_nf(c, wl)
        mask = P != 0
        K = np.sum(
            A[mask] * (A[mask] / P[mask]) ** (2 / 3) / (self._ng[c, mask] + nf[mask])
        )
        assert K > 0, (
            f"K=0 at c={c}, wl={wl}, A={A}, P={P}, nf={nf}, ng={self._ng[c]}, min_bed_level={self._min_bed_level[c]}"
        )
        return K

    def set_transport_rates(self, wl):
        """Set the bed material transport rate for all chaianges

        m^3/s
        """

        Bwet = np.array([self._calc_Bwet(c, w) for c, w in zip(range(self._nc), wl)])
        self._trans_rate_jli[:] = self.qb_jli() * Bwet[:, None, None]

        # FIXME
        # np.savetxt("/tmp/python_array.txt", self.qb_jli().ravel(), fmt="%.16e")
        # import sys

        # sys.exit(0)
        self._cum_transport += self._trans_rate_jli * self._dt

    def qb_jli(self):
        """Wilcock & Crowe (2003) eq 9.33 for every chainage at once.

        Returns (nc, nbins, nlith), per unit width (multiply by Bwet outside).

        Volumetric bed-load transport rate per unit width
        m^3/s / m
        """

        with np.errstate(over="raise"):
            phirm = 0.021 + 0.015 * np.exp(-20 * self._sand_fraction)  # (nc,)

        gsv = self._gsv[:, None, None]
        s = self._sediment_densities / WATER_DENSITY  # (nlith,)
        tau_rm = (
            phirm[:, None] * (s - 1) * WATER_DENSITY * GRAVITY * self._dsm[:, None]
        )  # (nc, nlith)

        ratio = self._rgsizes[None, :] / self._dsm[:, None]  # (nc, nbins)
        b = 0.67 / (1 + np.exp(1.5 - ratio))  # (nc, nbins)
        tau_rj = tau_rm[:, None, :] * (ratio**b)[:, :, None]  # (nc, nbins, nlith)

        phi = (WATER_DENSITY * gsv**2) / tau_rj  # (nc, nbins, nlith)
        q = (
            self._aprop
            * (self._athickness[:, None, None] / self.alayer_max_thickness)
            * gsv**3
            / (s - 1)
            / GRAVITY
        )  # (nc, nbins, nlith)

        # np.where would evaluate both branches everywhere (nan/overflow
        # warnings), so fill through a mask
        w = np.empty_like(phi)
        mask = phi < 1.35
        w[mask] = 0.002 * phi[mask] ** 7.5
        w[~mask] = 14 * (1 - 0.894 / np.sqrt(phi[~mask])) ** 4.5
        return q * w

    def propagate_sediment(self, t: pd.Timestamp):

        # sediment rates at all chainages, nc x nbins x nlith
        bdy_sed_rate = self._cfg._processed_sediment_boundary.value_at(t)

        # sediment transport rates
        my_Qb_jli = self._trans_rate_jli
        up_Qb_jli = np.concatenate(
            [np.zeros_like(my_Qb_jli[:1]), my_Qb_jli[:-1]], axis=0
        )
        up_Qb_jli += bdy_sed_rate

        dc = np.diff(self._cs)
        dc = np.insert(dc, 0, dc[0])
        fact = 1.0 / dc / (1 - self.poro) / self._Bchan

        # dy by dt to figure out next dt
        dydt = fact * (up_Qb_jli.sum(axis=(1, 2)) - my_Qb_jli.sum(axis=(1, 2)))
        self.max_dydt = max(np.abs(dydt))
        self._set_next_dt()

        # now get the dys
        fact *= self._dt
        dy = dydt * self._dt

        # np.savetxt("/tmp/python_array.txt", dy.ravel(), fmt="%.16e")

        # unitless ratio of transport of a particular fraction relative to all
        p = np.zeros_like(my_Qb_jli)
        denom = my_Qb_jli.sum(axis=(1, 2))[:, None, None]
        np.divide(my_Qb_jli, denom, out=p, where=denom != 0)

        # units are m^3 / s  times s / m^2 = m
        dy_jli = (up_Qb_jli - my_Qb_jli) * fact[:, None, None]

        # aggrade/degrade cross-section channel
        self._min_bed_level += dy
        self._mean_bed_level += dy

        # available change in active thickness (nc, )
        space = np.where(
            dy >= 0,
            np.maximum(self.alayer_max_thickness - self._athickness, 0),
            np.maximum(self._athickness, 0),
        )

        # change that will occur in active layer (nc, )
        change_alayer = np.sign(dy) * np.minimum(space, np.abs(dy))

        # remaining must change storage layer (nc, ), same sign as dy
        shift_alayer = dy - change_alayer

        # change thickness and proportion in active where final thickness not 0
        # The denom can be 0 when no active layer and degrading
        f_int = self.f_interface(shift_alayer > 0, p)
        denom = self._athickness[:, None, None] + change_alayer[:, None, None]
        numer = (
            self._aprop * self._athickness[:, None, None]
            + dy_jli
            - self.f_interface(shift_alayer > 0, p) * shift_alayer[:, None, None]
        )
        np.divide(numer, denom, out=self._aprop, where=denom != 0)
        self._athickness += change_alayer

        # shift in y must go into storage layer (nc, )
        self._storage_layers.update(f_int, shift_alayer)

        # now we can update all the cache variables
        self.set_grain_arrays()

    def f_interface(self, aggrading, p):
        """Interface distribution

        We subtract this from what moves INTO active layer

        aggrading: (n,) bool
        p: (n, nbins, nlith)

        NB: When degrading more than is available in top storage layer, this
        function returns incorrect value, we should return a combination of
            self._storage_layers.get_top_sprop()[1:],
        and
            self._storage_layers.get_below_top_sprop()[1:],

        Currently we will get a small mass balance error

        """
        aggrading = np.asarray(aggrading)[:, None, None]
        return np.where(
            aggrading,
            self.chi * self._aprop + (1 - self.chi) * p,
            self._storage_layers.get_top_sprop(),
        )

    def set_grain_arrays(self):
        """Update dsm and d90, sand_fraction, and ng"""

        # d90 as [left, channel, right]; banks use the channel value if
        # the cross section has no bankd90
        d90_channel = self.percentile(0.9)
        bank = np.where(np.isnan(self._bankd90), d90_channel, self._bankd90)
        self._d90[:] = np.column_stack([bank, d90_channel, bank])

        self._ng[:] = 0.044 * self._d90 ** (1 / 6)
        self._dsm[:] = self.percentile(0.5)
        self._sand_fraction[:] = self.sand_fraction()

    def _cumfreq(self):
        """(nc, nbins) cumulative proportion, summed over lith"""
        return np.cumsum(self._aprop.sum(axis=2), axis=1)

    def sand_fraction(self):
        """(nc,) proportion of active layer smaller than SAND_SIZE"""
        cf = self._cumfreq()
        xp = self._phi[::-1]  # np.interp wants x increasing
        return np.array([np.interp(SAND_SIZE, xp, row[::-1]) for row in cf])

    def percentile(self, x: float):
        """(nc,) grain size at percentile x (0..1).  Row-wise
        np.interp(x, cf, -log(rgsizes)); like np.interp this assumes each cf
        row is non-decreasing."""
        assert 0 <= x <= 1
        cf = self._cumfreq()  # (nc, nbins)
        phi = self._phi
        n = cf.shape[1]  # nbins

        j = (cf <= x).sum(axis=1) - 1  # last bin with cf <= x, -1 if none
        jc = np.clip(j, 0, n - 2)
        rows = np.arange(self._nc)
        c0, c1 = cf[rows, jc], cf[rows, jc + 1]
        p0, p1 = phi[jc], phi[jc + 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            mid = (p1 - p0) / (c1 - c0) * (x - c0) + p0
        # outside the table np.interp clamps to the end values
        out = np.where(j < 0, phi[0], np.where(x >= cf[:, -1], phi[-1], mid))
        return np.exp(-out)

    def update_gsv(self, Sf, velocity):
        """(nc,) grain shear velocity u*.  See LayerStack.calc_gsv for the
        derivation (Lambert W solution).  All arguments are (nc,) arrays."""

        d90 = self._d90[:, 1]
        ks = 2 * d90
        a = KAPPA * velocity
        b = 11 / GRAVITY / Sf / ks
        x = a * np.sqrt(b) / 2
        bad = ~(x > 0)  # also catches nan
        if bad.any():
            raise ValueError(
                "ustar calc, arg to lambertw <= 0, possibly complex solutions, "
                f"at chainage indices {np.flatnonzero(bad)}"
            )
        self._gsv[:] = (a / 2 / scipy.special.lambertw(x)).real


class River(Channel):
    pass


class BraidedChannel(Channel):
    def _init_dt(self):
        raise NotImplementedError("BraidedChannel is unusable")
