import math
import numpy as np
import pandas as pd
from functools import lru_cache

import utils
from gin import GrateConfig
from channel import Channel, Loc


class HydroDynamicModel:
    def __init__(self, cfg: GrateConfig, chan: Channel):
        self._cfg = cfg
        self._channel = chan
        self.cs = self._channel._chainpts()
        self.initialize(self._cfg.simulation_time.start)

        # cache these variables for current tempstep
        self._Sf_array = np.empty(len(self.cs))
        self._u_array = np.empty(len(self.cs))
        self._flow_array = np.empty(len(self.cs))

        # make sure the inflow boundary conditions know about (potentially) new chainage
        cfg._processed_inflow.rechainage(self.cs)

        # store bed slope in here each iteration
        # self.S0_array = []

    def initialize(self, t: pd.Timestamp):
        """Set water level (elevation above same datum cross-section profile y
        uses) to initial value.
        """
        # use the downstream for water level. 'normal' is a special case,
        # use wl_init
        tv = self._cfg._processed_downstream_boundary.value_at(t)
        # this is elevation
        if "normal" in tv:
            v = tv["normal"]["wl_init"]
        else:
            v = self.get_ds_wl(t)

        self.wl = np.full(len(self.cs), v, dtype=float)

        # get starting flow
        self._flow_array = self._cfg._processed_inflow.get_flows(t)

    def beta(self):
        """Momentum correction factor"""
        return self._cfg.morphological.beta

    def A(self, c: int, wl: float | None = None):
        """The area of the water at c chainage

        Parameters
        ----------
        c: int
            Chainage point along river in units of dc

        wl: float
            If None, then get the water level from self.wl, else calculate for this
            water level

        loc: Loc
            If None, get the total area, otherwise just in this Loc (left, main
            channel, or right bank0
        """
        if wl is None:
            wl = self.wl[c]
        return self._channel.area(c, wl, Loc.ALL)

    def get_u(self, c: int):
        """Previously calculated u"""
        return self._u_array[c]

    def get_flow(self, c: int):
        """Previously calculated flow"""
        return self._flow_array[c]

    def _u(self, c: int, wl: float | None = None):
        """The mean velocity, ie Q/A at current step"""
        if wl is None:
            wl = self.wl[c]
        return self._flow_array[c] / self.A(c, wl)

    def conveyance(self, c: int, wl: float | None = None):
        """K conveyance

        sum over left, main, right of
            A * R^(2/3) / (ng + nf)

        where
            A is the area
            R is A/P
            ng is grain roughness
            nf is form roughness
        """
        if wl is None:
            wl = self.wl[c]
        return self._channel.conveyance(c, wl)

    def get_Sf(self, c: int):
        """Previously calculated Sf"""
        return self._Sf_array[c]

    def _Sf(self, c: int, wl: float | None = None):
        """Return friction slope, ie. Q abs(Q) / K^2"""
        Q = self._flow_array[c]
        return Q * abs(Q) / self.conveyance(c, wl) ** 2

    # def _S0(self, c: int):
    #     """Bed slope at c"""
    #     return self._channel.S0(c)

    def Bwet(self, c: int, wl: float | None = None):
        """Water surface width"""
        if wl is None:
            wl = self.wl[c]
        return self._channel.Bwet(c, wl)

    def R(self, c: int, wl: float | None = None):
        """Hydraulic radius A/P over entire xsection"""
        if wl is None:
            wl = self.wl[c]
        return self._channel.R(c, wl)

    def update_water_level(self):
        raise NotImplementedError()

    def get_ds_wl(self, t: pd.Timestamp):
        """Return downstream water level of water."""
        lastchain = len(self.cs) - 1
        tv = self._cfg._processed_downstream_boundary.value_at(t)
        if "elevation" in tv:
            return tv["elevation"]
        if "depth" in tv:
            return tv["depth"] + self._channel.get_min_bed_level(lastchain)
        if "normal" in tv:
            slope = tv["normal"]["slope"]
            Q = self.Q(t, lastchain)

            def f(wl):
                K = self.conveyance(lastchain, wl)
                return Q - K * math.sqrt(slope)

            try:
                wl = utils.find_root(
                    f, self.wl[lastchain], self._channel.get_min_bed_level(lastchain)
                )
            except Exception as exp:
                raise RuntimeError("get_ds_wl failed to find root") from exp

            return wl

        raise ValueError(f"Unknown downstream boundary type: {tv['type']}")


class QuasiSteadyModel(HydroDynamicModel):
    # @lru_cache(maxsize=4000)
    # def Q(self, t: pd.Timestamp, c: int):
    #     """Return flow at point along river
    #
    #     Parameters
    #     ----------
    #     c: int
    #         Chainage point along river in dc units
    #
    #     t: pd.Timestamp
    #         Time
    #     """
    #     c = self.cs[c]  # convert index to chainage
    #     return sum(
    #         pi.value_at(t) for pi in self._cfg._processed_inflow if pi.ordinate <= c
    #     )

    def conservation_of_energy(self, c: int, wl: float):
        """Calculate equation 5.7, the change in energy between me and
        downstream

        f(h_i^*) =
            h_(i) + (β_i (u_i^* )^2)/2g
            - (h_(i+1) + (β_(i+1) u_(i+1)^2)/2g)
            + (S_0-S_f^*)Δx

        h_(i) is the water level (elevation above datum)
        S_0 is ZERO since using water level
        h_0 is water level at most upstream
        u is velocity of water.
        i is chain point index (c in this case)

        We should have already sorted at c+1
        """

        dc = self.cs[c + 1] - self.cs[c]
        sf = (self._Sf(c, wl) + self._Sf_array[c + 1]) / 2
        g = 9.8
        f = (
            wl
            # + (self.beta() * self.u(t, c, wl) ** 2 - self.beta() * self.u(t, c + 1) ** 2)
            # / (2 * g)
            + self.beta() * (self._u(c, wl) ** 2 - self._u_array[c + 1] ** 2) / (2 * g)
            - self.wl[c + 1]
            - sf * dc
        )
        return f

    def update_water_level(self, t: pd.Timestamp):
        """

        Parameters
        ----------
        t: pd.Timestamp
            Timestep
        """

        # get the most downstream water level
        self.wl[-1] = self.get_ds_wl(t)

        # current flow
        self._flow_array = self._cfg._processed_inflow.get_flows(t)

        # use wl[i+1] to calculate wl[i]
        for i in range(len(self.cs) - 2, -1, -1):
            self._Sf_array[i + 1] = self._Sf(i + 1)
            self._u_array[i + 1] = self._u(i + 1)

            @lru_cache(maxsize=1000)
            def f(wl):
                return self.conservation_of_energy(i, wl)

            init_wl = (
                self.wl[i + 1]
                - self._channel.get_min_bed_level(i + 1)
                + self._channel.get_min_bed_level(i)
            )

            # print(f"{t=} {i=} {init_wl=}")

            try:
                wl = utils.find_root(
                    f,
                    x0=init_wl,
                    lower=self._channel.get_min_bed_level(i),
                    max_growth=100.0,
                )

            except Exception as exp:
                wls = np.arange(
                    self._channel.get_min_bed_level(i) + 0.0001,
                    self._channel.get_min_bed_level(i) + 10.0,
                    0.0001,
                )
                f = np.array([f(wl) for wl in wls])
                Sfs = np.array([self._Sf(i, wl) for wl in wls])
                us = np.array([self._u(i, wl) for wl in wls])
                df = pd.DataFrame({"wl": wls, "f": f, "Sf": Sfs, "u": us})
                df = df[(-1 < df.f) & (df.f < 1)]
                df.to_csv("root_finding_failure_f_values.csv", index=False)
                self._channel.xss[i].df.to_csv("failured_profile.csv", index=False)
                print(f"Sf array = {self._Sf_array}")
                print(f"u array = {self._u_array}")
                raise RuntimeError(
                    f"find_root failed at cross-section {i} chainage={self.cs[i]}, f saved in root_finding_failure_f_values.csv"
                ) from exp

            self.wl[i] = wl
            self._Sf_array[i] = self._Sf(i)
            self._u_array[i] = self._u(i)

            # print(f"Finished {t=} {i=} {wl=}")


class DynamicWaveModel(HydroDynamicModel):
    def initialize(self, t: pd.Timestamp):
        raise NotImplementedError(f"DynamicWaveModel is unusable at time {t}")
