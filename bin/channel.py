import numpy as np
import pandas as pd
from gin import GrateConfig
from cross_section import CrossSection, Loc
from layer_arrays import LayerArrays


class Channel:
    """Covers Flume and River.  Braided channel needs a subclass

    A design choice is that getters should return cached array info, functions
    starting with _calc calculate variables
    """

    def __init__(self, cfg: GrateConfig):
        self._cfg = cfg
        self.max_dc = cfg.discretisation.max_dc
        self.poro = cfg.morphological.poro
        self._cs = self._chainpts()
        self._nc = len(self._cs)
        self._xss = self._get_interpolated_cross_sections()
        self._init_layer_arrays()
        self._init_dt()

        # these don't change over time
        self._Bchan = np.array([xs.calc_Bchan() for xs in self._xss])

        # these are set each iteration AFTER wl known
        self._Bwet = np.empty(len(self._cs))
        self._min_bed_level = np.empty(len(self._cs))
        self._mean_bed_level = np.empty(len(self._cs))
        self._area = np.empty((len(self._cs), 3))
        self._P = np.empty((len(self._cs), 3))
        self._nf = np.empty((len(self._cs), 3))
        self._conveyance = np.empty(len(self._cs))

        # these are set after propogate sediment
        nbins = cfg.grain_size_profiles.num_bins
        nlith = cfg.grain_size_profiles.num_lith
        self._Qb_jli = np.empty((len(self._cs), nbins, nlith))
        self._grain_shear_velocity = np.empty(len(self._cs))
        self._sand_fraction = np.empty(len(self._cs))
        self._dsm = np.empty(len(self._cs))
        self._d90 = np.empty((len(self._cs), 3))
        self._ng = np.empty((len(self._cs), 3))

    def _init_layer_arrays(self):
        """Grain-size layer state for every chainage point, as arrays with a
        leading chainage axis (acfd, scfd: nc x nbins x nlith).  This does the
        "Interp" resolution and interpolation to self._cs that used to be done
        per cross section by LayerStack.
        """
        self._layers = LayerArrays.from_config(self._cfg, self._cs)

        # None (use channel d90 for banks too) becomes nan
        self._bankd90 = np.array(
            [np.nan if xs.bankd90 is None else xs.bankd90 for xs in self._xss]
        )

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

    def get_Bwet(self):
        return self._Bwet

    def get_area(self):
        return self._area

    def get_P(self):
        return self._P

    def get_nf(self):
        return self._nf

    def get_Qb_jli(self):
        return self._Qb_jli

    def get_grain_shear_velocity(self):
        return self._grain_shear_velocity

    def get_sand_fraction(self):
        return self._sand_fraction

    def get_dsm(self):
        return self._dsm

    def get_d90(self):
        return self._d90

    def _chainpts(self):
        """Return chain points"""
        c = np.array([xs.chainage for xs in self._cfg.cross_sections.profiles])

        # number of intervals between each chainage
        gaps = np.diff(c)
        ints = np.ceil(gaps / self.max_dc).astype(int)

        return np.concatenate(
            [np.linspace(a, b, n + 1)[:-1] for a, b, n in zip(c[:-1], c[1:], ints)]
            + [c[-1:]]
        )

    def _get_cross_sections(self) -> dict:
        """Return chainage point to CrossSection at that point"""
        xss = {}

        formrf = self._cfg.cross_sections.formrf

        for xs in self._cfg.cross_sections.profiles:
            c = xs.chainage
            xss[c] = CrossSection(xs, formrf)  # , wallrf)

        # check min/max chainage
        assert min(xss.keys()) <= self._cs[0], (
            f"Minimum chainage ({self._cs[0]}) isn't at least the minimum cross section chainage"
        )
        assert self._cs[-1] <= max(xss.keys()), (
            f"Maximum chainage ({self._cs[-1]}) is more than the maximum cross section chainage"
        )

        # some variables in CrossSections might need interpolated values
        CrossSection.resolve_interpolated_properties(xss)

        return xss

    def _get_interpolated_cross_sections(self):
        """Return a list of CrossSection at self.cs"""
        xss = self._get_cross_sections()
        xs_chainpts = sorted(xss)

        ixss = []

        for i, cpt in enumerate(self._cs):
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

    def get_mean_bed_level(self):
        """Return mean bed level of profile at chainage c

        For a flume this is the bed_level, for other channels it is some sort
        of average of the cross-section profile
        """
        return self._mean_bed_level

    def get_min_bed_level(self):
        """Return deepest part of the cross-section"""
        return self._min_bed_level

    def calc_area(self, c: int, wl: float, loc: Loc):
        """Return area of water between bed and wl"""
        return self._xss[c].calc_area(wl, loc)

    def calc_Qb_jli(self):
        """Return bed material transport rate for ALL chainages

        Returns
        -------
        np.array:
            nc x nbins x nlith 3d array.  (c, j, li) element is bed transport
            at chainage index c for li lith group and j proportion size
        """
        return (
            self._layers.qb_jli(
                self._grain_shear_velocity, self._sand_fraction, self._dsm
            )
            * self._Bwet[:, None, None]
        )

    def calc_conveyance(self, c: int, wl: float):
        try:
            return self._xss[c].calc_conveyance_using_wl(wl, self._ng[c])
        except Exception as e:
            raise ValueError(
                f"Couldn't cal convey for cross-section {c}, {self._min_bed_level[c]=}: {e}"
            )
        # return self._xss[c].calc_conveyance_using_wl(wl, self._ng[c])

    def set_transport_rates(self):
        self._Qb_jli[:] = self.calc_Qb_jli()

    def propogate_sediment(self, t: pd.Timestamp, hydro):

        # sediment rates at all chainages, nc x nbins x nlith
        bdy_sed_rate = self._cfg._processed_sediment_boundary.value_at(t)

        # sediment transport rates
        up_Qb_jli = self._Qb_jli[:-1] + bdy_sed_rate[:-1]
        my_Qb_jli = self._Qb_jli[1:]

        dc = np.diff(self._cs)
        fact = 1.0 / dc / (1 - self.poro) / self._Bchan[1:]

        # dy by dt to figure out next dt
        dydt = fact * (up_Qb_jli.sum(axis=(1, 2)) - my_Qb_jli.sum(axis=(1, 2)))
        self.max_dydt = max(np.abs(dydt))
        self._set_next_dt()

        # now get the dys
        fact *= self._dt
        dy = dydt * self._dt

        # Vectorise the sediment proportions and flux differences.
        p = my_Qb_jli / my_Qb_jli.sum(axis=(1, 2))[:, None, None]
        flux_diff = (up_Qb_jli - my_Qb_jli) * fact[:, None, None]

        # aggrade/degrade cross-section channel
        for i, c in enumerate(range(1, self._nc)):
            self._xss[c].aggrade_bed(dy[i])

        # Layer state is one array: update every chainage in a single shot
        self._layers.update_active_layer(
            dy, flux_diff, p, self._cfg.morphological.la, rows=slice(1, None)
        )

        # FIXME, change storage layer f_jli

        # now we can update all the cache variables
        self.set_layer_arrays(hydro)

    def set_layer_arrays(self, hydro, do_gsv: bool = True):
        """don't do gsb on init since Sf and vel wrong so can't calculate gsm"""

        lay = self._layers

        # d90 as [left, channel, right]; banks use the channel value if
        # the cross section has no bankd90
        ch = lay.d90()
        bank = np.where(np.isnan(self._bankd90), ch, self._bankd90)
        self._d90[:] = np.column_stack([bank, ch, bank])
        self._ng[:] = 0.044 * self._d90 ** (1 / 6)

        if do_gsv:
            self._grain_shear_velocity[:] = lay.gsv(
                hydro.get_Sf(), hydro.get_vel(), self._d90[:, 1]
            )
        else:
            self._grain_shear_velocity[:] = 1
        self._sand_fraction[:] = lay.sand_fraction()
        self._dsm[:] = lay.dsm()

        # geometry, still per cross section
        for c in range(self._nc):
            self._min_bed_level[c] = self._xss[c].calc_min_bed_level()
            self._mean_bed_level[c] = self._xss[c].calc_mean_bed_level()

    def set_profile_arrays(self, wl: np.array):
        """Hydrodynamics model has sorted wl, this is to update all the profile
        arrays that rely on wl"""

        for c, w in zip(range(self._nc), wl):
            self._Bwet[c] = self._xss[c].calc_Bwet(w)
            for j, loc in enumerate((Loc.LEFT, Loc.CHANNEL, Loc.RIGHT)):
                self._area[c, j] = self.calc_area(c, w, loc)
                self._P[c, j] = self._xss[c].calc_P(w, loc)
                self._nf[c, j] = self._xss[c].calc_nf(w, loc)
            # self._conveyance[c] = self._xss[c].calc_conveyance(
            #     self._area[c], self._P[c], self._nf[c], self._ng[c]
            # )


class River(Channel):
    pass


class BraidedChannel(Channel):
    def _init_dt(self):
        raise NotImplementedError("BraidedChannel is unusable")
