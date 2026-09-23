import numpy as np
import pandas as pd
from gin import GrateConfig
from cross_section import CrossSection, Loc


class Channel:
    """Flume, river, braided channel"""

    def __init__(self, cfg: GrateConfig):
        self._cfg = cfg
        self.dc = cfg.discretisation.dc
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

    def beta(self):
        """Momentum correction factor"""
        return 1

    def __d90_del(self, c: int, loc: Loc):
        """90th percentile of the grain diameter.

        Referred to in Eq 8.4

        The grain diameter of the surface layer, and in the case of the
        floodplain use bank d90 from config file if specified else surface
        layer
        """
        return self.xss[c].d90(loc)

    def __ng_del(self, c: int, loc: Loc):
        """Grain roughness at given chainage"""
        return 0.044 * self.d90(c, loc) ** (1 / 6)

    def __nf_del(self, c: int, h: float, loc: Loc):
        """Form roughness"""
        return self.xss[c].nf(h, loc)

    def area(self, c: int, h: float, loc: Loc | None = None):
        """Return area of water between bed and h"""
        return self.xss[c].area(h, loc)

    def get_mean_bed_level(self, c: int):
        """Return mean bed level of profile at chainage c

        For a flume this is the bed_level, for other channels it is some sort
        of average of the cross-section profile
        """
        return self.xss[c].mean_bed_level

    def get_min_bed_level(self, c: int):
        """Return deepest part of the cross-section"""
        return self.xss[c].min_bed_level

    def S0(self, c: int):
        """Bed slope at c

        Bed slope is the slope between chainages of mean_bed_level
        """
        assert c < self.nc - 1, (
            f"Cannot calculate S0({c}), likely because this is the most downstream point"
        )
        return (self.get_mean_bed_level(c) - self.get_mean_bed_level(c + 1)) / self.dc

    def Bwet(self, c: int, h: float):
        """Water surface width"""
        return self.xss[c].Bwet(h)

    def grain_stress(self, c: int, t: pd.Timestamp, hydro):
        return self.xss[c].grain_stress(t, hydro)

    def get_Qb_jli(self, c: int, t: pd.Timestamp, hydro):
        """Transport rate for this chainage"""
        return self.xss[c].Qb_jli(t, hydro)

    def propogate_sediment(self, t: pd.Timestamp, hydro):

        # keep track of dy's to update max_deta_over_dt
        dys = []

        for c in range(1, self.nc):
            # nbins x nlith rate of sediment coming in from boundary
            bdy_sediment_rate = sum(
                sb.value_at(t)
                for sb in self._cfg._processed_sediment_boundary
                if self.cs[c - 1] <= sb.ordinate < self.cs[c]
            )

            up_Qb_jli = self.xss[c - 1].Qb_jli(t, hydro) + bdy_sediment_rate
            my_Qb_jli = self.xss[c].Qb_jli(t, hydro)

            fact = self.dt / self.dc / (1 - self.poro) / self.xss[c].Bchan()
            dy = (up_Qb_jli.sum() - my_Qb_jli.sum()) * fact
            self.xss[c].aggrade_bed(dy)
            dys.append(dy)

            # print(c, my_Qb_jli.sum())

            p = my_Qb_jli / my_Qb_jli.sum()
            df = (
                (up_Qb_jli - my_Qb_jli) * fact - self.xss[c].f_interface(dy > 0, p) * dy
            ) / self._cfg.morphological.la

            self.xss[c].update_alayer_proportions(df)

            # FIXME, change storage layer f_jli

        self.max_deta_over_dt = max(dys)
        self._set_next_dt()

    def conveyance(self, c: int, d: float):
        return self.xss[c].conveyance(d)

    def R(self, c: int, d: float):
        """Hydraulic radius A/P over entire xsection"""
        return self.xss[c].R(d)
