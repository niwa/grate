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

        # these are Sf and u for current timestep.  for caching
        self._Sf_array = np.empty(len(self.cs))
        self._u_array = np.empty(len(self.cs))

        # store bed slope in here each iteration
        self.S0_array = []

    def initialize(self, t: pd.Timestamp):
        """Set d to initial value."""
        # use the downstream for water level. 'normal' is a special case,
        # use wl_init
        tv = self._cfg._processed_downstream_boundary.value_at(t)
        # this is elevation, we want depth
        if "normal" in tv:
            v = tv["normal"]["wl_init"]
            v -= self._channel.get_min_bed_level(len(self.cs) - 1)
        else:
            v = self.get_ds_d(t)

        self.d = np.full(len(self.cs), v, dtype=float)

    def beta(self):
        """Momentum correction factor"""
        return self._cfg.morphological.beta

    def Q(self, t: pd.Timestamp, c: int):
        """Return flow at point along river

        Parameters
        ----------
        c: int
            Chainage index along river in units of dc

        t: pd.Timestamp
            Time
        """
        raise NotImplementedError(f"Q({t}, {c})")

    def A(self, c: int, d: float | None = None, loc: Loc | None = None):
        """The area of the water at c chainage

        Parameters
        ----------
        c: int
            Chainage point along river in units of dc

        d: float
            If None, then get the depth from self.d, else calculate for this
            depth

        loc: Loc
            If None, get the total area, otherwise just in this Loc (left, main
            channel, or right bank0
        """
        if d is None:
            d = self.d[c]
        return self._channel.area(c, d, loc)

    def get_u(self, c: int):
        """Previously calculated u"""
        return self._u_array[c]

    def _u(self, t: pd.Timestamp, c: int, d: float | None = None):
        """The mean velocity, ie Q/A"""
        if d is None:
            d = self.d[c]
        return self.Q(t, c) / self.A(c, d)

    def conveyance(self, c: int, d: float | None = None):
        """K conveyance

        sum over left, main, right of
            A * R^(2/3) / (ng + nf)

        where
            A is the area
            R is A/P
            ng is grain roughness
            nf is form roughness
        """
        if d is None:
            d = self.d[c]
        return self._channel.conveyance(c, d)

    def get_Sf(self, c: int):
        """Previously calculated Sf"""
        return self._Sf_array[c]

    def _Sf(self, t: pd.Timestamp, c: int, d: float | None = None):
        """Return friction slope, ie. Q abs(Q) / K^2"""
        Q = self.Q(t, c)
        return Q * abs(Q) / self.conveyance(c, d) ** 2

    def _S0(self, c: int):
        """Bed slope at c"""
        return self._channel.S0(c)

    def Bwet(self, c: int, d: float | None = None):
        """Water surface width"""
        if d is None:
            d = self.d[c]
        return self._channel.Bwet(c, d)

    def R(self, c: int, d: float | None = None):
        """Hydraulic radius A/P over entire xsection"""
        if d is None:
            d = self.d[c]
        return self._channel.R(c, d)

    def update_depth(self):
        raise NotImplementedError()

    def get_ds_d(self, t: pd.Timestamp):
        """Return downstream depth of water."""
        lastchain = len(self.cs) - 1
        tv = self._cfg._processed_downstream_boundary.value_at(t)
        if "elevation" in tv:
            return tv["elevation"] - self._channel.get_min_bed_level(lastchain)
        if "depth" in tv:
            return tv["depth"]
        if "normal" in tv:
            slope = tv["normal"]["slope"]
            Q = self.Q(t, lastchain)

            def f(d):
                K = self.conveyance(lastchain, d)
                return Q - K * math.sqrt(slope)

            try:
                d = utils.newton(f, self.d[lastchain])
            except Exception as exp:
                raise RuntimeError("Newton failed doing downstream normal") from exp

            return d

        raise ValueError(f"Unknown downstream boundary type: {tv['type']}")


class QuasiSteadyModel(HydroDynamicModel):
    @lru_cache(maxsize=400)
    def Q(self, t: pd.Timestamp, c: int):
        """Return flow at point along river

        Parameters
        ----------
        c: int
            Chainage point along river in dc units

        t: pd.Timestamp
            Time
        """
        c = self.cs[c]  # convert index to chainage
        return sum(
            pi.value_at(t) for pi in self._cfg._processed_inflow if pi.ordinate <= c
        )

    def conservation_of_energy(self, t: pd.Timestamp, c: int, d: float):
        """Calculate equation 5.7, the change in energy between me and
        downstream

        f(h_i^*) =
            h_(i) + (β_i (u_i^* )^2)/2g
            - (h_(i+1) + (β_(i+1) u_(i+1)^2)/2g)
            + (S_0-S_f^*)Δx

        h_0 is depth at most upstream
        u is velocity of water.
        i is chain point index (c in this case)

        We should have already sorted at c+1
        """

        dc = self.cs[c + 1] - self.cs[c]
        # sf = (self.Sf(t, c, d) + self.Sf(t, c + 1)) / 2
        sf = (self._Sf(t, c, d) + self._Sf_array[c + 1]) / 2
        g = 9.8
        f = (
            d
            # + (self.beta() * self.u(t, c, d) ** 2 - self.beta() * self.u(t, c + 1) ** 2)
            # / (2 * g)
            + self.beta()
            * (self._u(t, c, d) ** 2 - self._u_array[c + 1] ** 2)
            / (2 * g)
            - self.d[c + 1]
            - (self.S0_array[c] + sf) * dc
        )
        return f

    def update_depth(self, t: pd.Timestamp):
        """F

        Parameters
        ----------
        t: pd.Timestamp
            Timestep
        """

        # get the most downstream depth
        self.d[-1] = self.get_ds_d(t)

        # get all the bed slopes
        self.S0_array = [self._S0(c) for c in range(len(self.cs) - 1)]

        # use d[i+1] to calculate d[i]
        for i in range(len(self.cs) - 2, -1, -1):
            x0 = self.d[i + 1]
            self._Sf_array[i + 1] = self._Sf(t, i + 1)
            self._u_array[i + 1] = self._u(t, i + 1)

            @lru_cache(maxsize=1000)
            def f(d):
                return self.conservation_of_energy(t, i, d)

            try:
                d = utils.find_root(f, x0=x0)
            except Exception as exp:
                depths = np.arange(0.0001, x0 * 3, 0.0001)
                f = np.array([f(d) for d in depths])
                df = pd.DataFrame({"d": depths, "f": f})
                # df = df[(-1 < df.f) & (df.f < 1)]
                df.to_csv("root_finding_failure_f_values.csv", index=False)
                self._channel.xss[i].df.to_csv("failured_profile.csv", index=False)
                # run newton again recording values tried
                # d = utils.newton(f, self.d[i + 1], record=True)
                raise RuntimeError(
                    f"find_root failed at cross-section {i} chainage={self.cs[i]}, f saved in root_finding_failure_f_values.csv"
                ) from exp

            self.d[i] = d
            self._Sf_array[i] = self._Sf(t, i)
            self._u_array[i] = self._u(t, i)


class DynamicWaveModel(HydroDynamicModel):
    def initialize(self, t: pd.Timestamp):
        raise NotImplementedError(f"DynamicWaveModel is unusable at time {t}")
