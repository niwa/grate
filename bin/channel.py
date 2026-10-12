import numpy as np
import pandas as pd
from gin import GrateConfig
from cross_section import CrossSection, Loc


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
        self._init_dt()

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

    def _init_dt(self):
        # start out at max dt which the cfg has set under dt
        self._dt = self.max_dt = self._cfg.max_dt

        # start out 10% of max rate
        self.cdt = self._cfg.simulation_time.cdt
        self.max_deta_over_dt = 0.1 * self.cdt / self._dt

    def _set_next_dt(self):
        self.max_deta_over_dt = max(1e-10, self.max_deta_over_dt)
        self._dt = min(self.cdt / self.max_deta_over_dt, self.max_dt)

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
            xss[c] = CrossSection(xs, self._cfg, formrf)  # , wallrf)

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
                xss[cpt].layers.chainidx = i
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

    def calc_Qb_jli(self, c: int):
        """Return bed material transport rate 2darray

        Parameters
        ----------
        t: pd.Timestamp
            Time

        Returns
        -------
        np.array:
            nbins x nlith 2d array.  (j, li) element is bed transport for li
            lith group and j proportion size
        """
        return (
            self._xss[c].layers.qb_jli(
                self._grain_shear_velocity[c], self._sand_fraction[c], self._dsm[c]
            )
            * self._Bwet[c]
        )

    def calc_conveyance(self, c: int, wl: float):
        return self._xss[c].calc_conveyance_using_wl(wl, self._ng[c])

    def propogate_sediment(self, t: pd.Timestamp, hydro):

        # keep track of dy's to update max_deta_over_dt
        dys = []

        for c in range(1, self._nc):
            dc = self._cs[c] - self._cs[c - 1]

            # nbins x nlith rate of sediment coming in from boundary
            # FIXME, want this like inflow bc
            bdy_sediment_rate = sum(
                sb.value_at(t)
                for sb in self._cfg._processed_sediment_boundary
                if self._cs[c - 1] <= sb.ordinate < self._cs[c]
            )

            up_Qb_jli = self._Qb_jli[c - 1] + bdy_sediment_rate
            my_Qb_jli = self._Qb_jli[c]

            # FIXME, we should return these.  Then do output.
            # calculate dy_by_dt:
            #
            # fact = 1 / dc / (1 - self.poro) / self._xss[c].Bchan()
            # dy_by_dt = (up_Qb_jli.sum() - my_Qb_jli.sum()) * fact
            #
            # now set next dt using
            # dt = min(max_dt, max over all crosssections(dy_by_dt) / cdt)
            #
            # now can calculate dy and df to update bed and the acfd/scfd

            # constants...
            fact = self._dt / dc / (1 - self.poro) / self._xss[c].Bchan()

            dy = (up_Qb_jli.sum() - my_Qb_jli.sum()) * fact
            self._xss[c].aggrade_bed(dy)
            dys.append(dy)

            p = my_Qb_jli / my_Qb_jli.sum()
            df = (
                (up_Qb_jli - my_Qb_jli) * fact
                - self._xss[c].f_interface(dy > 0, p) * dy
            ) / self._cfg.morphological.la

            self._xss[c].update_alayer_proportions(df)

            # FIXME, change storage layer f_jli

        self.max_deta_over_dt = max(dys)
        self._set_next_dt()

        # now we can update all the cache variables
        self.set_layer_arrays(hydro)

    def set_layer_arrays(self, hydro, do_gsv: bool = True):
        """don't do gsb on init since Sf and vel wrong so can't calculate gsm"""

        Sf = hydro.get_Sf()
        v = hydro.get_vel()
        for c in range(len(self._cs)):
            self._d90[c] = self._xss[c].calc_d90()
            self._ng[c] = self._xss[c].calc_ng()
            if do_gsv:
                gsv = self._xss[c].layers.calc_gsv(Sf[c], v[c], self._d90[c, 1])
                self._grain_shear_velocity[c] = gsv
            else:
                self._grain_shear_velocity[c] = 1
            self._sand_fraction[c] = self._xss[c].layers.calc_sand_fraction()
            self._dsm[c] = self._xss[c].layers.calc_dsm()
            self._Qb_jli[c] = self.calc_Qb_jli(c)

    def set_profile_arrays(self, wl: np.array):
        """Hydrodynamics model has sorted wl, this is to update all the profile
        arrays that rely on wl"""

        for c, w in zip(range(self._nc), wl):
            self._Bwet[c] = self._xss[c].calc_Bwet(w)
            for j, loc in enumerate((Loc.LEFT, Loc.CHANNEL, Loc.RIGHT)):
                self._area[c, j] = self.calc_area(c, w, loc)
                self._P[c, j] = self._xss[c].calc_P(w, loc)
                self._nf[c, j] = self._xss[c].calc_nf(w, loc)
            self._conveyance[c] = self._xss[c].calc_conveyance(
                self._area[c], self._P[c], self._nf[c], self._ng[c]
            )
            self._min_bed_level[c] = self._xss[c].calc_min_bed_level()
            self._mean_bed_level[c] = self._xss[c].calc_mean_bed_level()


class River(Channel):
    pass


class BraidedChannel(Channel):
    def _init_dt(self):
        raise NotImplementedError("BraidedChannel is unusable")
