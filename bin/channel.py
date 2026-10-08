import numpy as np
import pandas as pd
from gin import GrateConfig
from cross_section import CrossSection, Loc


class Channel:
    """Covers Flume and River.  Braided channel needs a subclass"""

    def __init__(self, cfg: GrateConfig):
        self._cfg = cfg
        self.max_dc = cfg.discretisation.max_dc
        self.poro = cfg.morphological.poro
        self.cs = self._chainpts()
        self.nc = len(self.cs)
        self.xss = self._get_interpolated_cross_sections()
        self._init_dt()

    def _init_dt(self):
        # start out at max dt which the cfg has set under dt
        self.dt = self.max_dt = self._cfg.max_dt

        # start out 10% of max rate
        self.cdt = self._cfg.simulation_time.cdt
        self.max_deta_over_dt = 0.1 * self.cdt / self.dt

    def _set_next_dt(self):
        self.max_deta_over_dt = max(1e-10, self.max_deta_over_dt)
        self.dt = min(self.cdt / self.max_deta_over_dt, self.max_dt)

    def get_dt(self):
        return self.dt

    def __str__(self):
        return f"Channel with cross sections:\n{'\n'.join(str(s) for s in self.xss)}"

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
        assert min(xss.keys()) <= self.cs[0], (
            f"Minimum chainage ({self.cs[0]}) isn't at least the minimum cross section chainage"
        )
        assert self.cs[-1] <= max(xss.keys()), (
            f"Maximum chainage ({self.cs[-1]}) is more than the maximum cross section chainage"
        )

        # some variables in CrossSections might need interpolated values
        CrossSection.resolve_interpolated_properties(xss)

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

    def _get_sediment_bc(self):
        """Return"""

    def area(self, c: int, wl: float, loc: Loc | None = None):
        """Return area of water between bed and wl"""
        return self.xss[c].area(wl, loc)

    def get_mean_bed_level(self, c: int):
        """Return mean bed level of profile at chainage c

        For a flume this is the bed_level, for other channels it is some sort
        of average of the cross-section profile
        """
        return self.xss[c].mean_bed_level

    def get_min_bed_level(self, c: int):
        """Return deepest part of the cross-section"""
        return self.xss[c].min_bed_level

    def Bwet(self, c: int, wl: float):
        """Water surface width"""
        return self.xss[c].Bwet(wl)

    def grain_stress(self, c: int, t: pd.Timestamp, hydro):
        return self.xss[c].grain_stress(hydro)

    def get_Qb_jli(self, c: int, hydro):
        """Transport rate for this chainage"""
        return self.xss[c].Qb_jli(hydro)

    def propogate_sediment(self, t: pd.Timestamp, hydro):

        # keep track of dy's to update max_deta_over_dt
        dys = []

        for c in range(1, self.nc):
            dc = self.cs[c] - self.cs[c - 1]

            # nbins x nlith rate of sediment coming in from boundary
            bdy_sediment_rate = sum(
                sb.value_at(t)
                for sb in self._cfg._processed_sediment_boundary
                if self.cs[c - 1] <= sb.ordinate < self.cs[c]
            )

            up_Qb_jli = self.xss[c - 1].Qb_jli(hydro) + bdy_sediment_rate
            my_Qb_jli = self.xss[c].Qb_jli(hydro)

            fact = self.dt / dc / (1 - self.poro) / self.xss[c].Bchan()
            dy = (up_Qb_jli.sum() - my_Qb_jli.sum()) * fact
            self.xss[c].aggrade_bed(dy)
            dys.append(dy)

            p = my_Qb_jli / my_Qb_jli.sum()
            df = (
                (up_Qb_jli - my_Qb_jli) * fact - self.xss[c].f_interface(dy > 0, p) * dy
            ) / self._cfg.morphological.la

            self.xss[c].update_alayer_proportions(df)

            # FIXME, change storage layer f_jli

        self.max_deta_over_dt = max(dys)
        self._set_next_dt()

    def conveyance(self, c: int, wl: float):
        return self.xss[c].conveyance(wl)

    # FIXME, do we need this?
    def R(self, c: int, wl: float):
        """Hydraulic radius A/P over entire xsection"""
        return self.xss[c].R(wl)

    def display_acfd(self):
        for xs in self.xss:
            print(xs.get_acfd())


class River(Channel):
    pass


class BraidedChannel(Channel):
    def _init_dt(self):
        raise NotImplementedError("BraidedChannel is unusable")
