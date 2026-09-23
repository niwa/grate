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
        self.dc = self._channel.dc
        self.cs = self._channel._chainpts()
        self.initialize(self._cfg.simulation_time.start)

    def initialize(self, t: pd.Timestamp):
        """Set depth to initial values."""
        raise NotImplementedError(f"initialize({t})")

    def beta(self):
        """Momentum correction factor"""
        return self._channel.beta()

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

    def u(self, t: pd.Timestamp, c: int, d: float | None = None):
        """The mean velocity, ie Q/A"""
        if d is None:
            d = self.d[c]
        return self.Q(t, c) / self.A(c, d)

    def ng(self, c: int, loc: Loc):
        """Grain roughness"""
        return self._channel.ng(c, loc)

    def nf(self, c: int, d: float, loc: Loc):
        """Form roughness"""
        return self._channel.nf(c, d, loc)

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

        K = 0
        for loc in (Loc.LEFT, Loc.CHANNEL, Loc.RIGHT):
            A = self.A(c, d, loc)
            P = self.P(c, d, loc)
            if P == 0:
                # no water in this part of channel
                continue
            R = A / P
            K += A * R ** (2 / 3) / (self.ng(c, loc) + self.nf(c, d, loc))

        assert K > 0, "No water"

        return K

    def Sf(self, t: pd.Timestamp, c: int, d: float | None = None):
        """Return friction slope, ie. Q abs(Q) / K^2"""
        Q = self.Q(t, c)
        return Q * abs(Q) / self.conveyance(c, d) ** 2

    def P(self, c: int, d: float | None = None, loc: Loc | None = None):
        """Wetted perimeter at chainage"""
        if d is None:
            d = self.d[c]
        return self._channel.P(c, d, loc)

    def S0(self, c: int):
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
        return self.A(c, d) / self.P(c, d)

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

            # d, info = scipy.optimize.newton(f, self.d[lastchain], full_output=True)
            # if not info.converged:
            #     raise ValueError(f"Can't solve downstream normal depth. {info=}")
            d = utils.newton(f, self.d[lastchain])
            return d

        raise ValueError(f"Unknown downstream boundary type: {tv['type']}")


class QuasiSteadyModel(HydroDynamicModel):
    def initialize(self, t: pd.Timestamp):
        """Set d to initial value."""
        # get the chain values from channel so we know lengths
        # FIXME, just starting 0.5m of water depth
        self.d = np.ones(len(self.cs)) / 2

        # if the downstream boundary condition is normal, grab the hinit which
        # is an elevation (need to subtract off
        tv = self._cfg._processed_downstream_boundary.value_at(t)
        if "normal" in tv:
            self.d[-1] = tv["normal"]["hinit"]
            self.d[-1] -= self._channel.get_min_bed_level(len(self.cs) - 1)

    @lru_cache(maxsize=100)
    def Q(self, t: pd.Timestamp, c: int):
        """Return flow at point along river

        Parameters
        ----------
        c: int
            Chainage point along river in dc units

        t: pd.Timestamp
            Time
        """
        c *= self.dc  # config inflow is in metres
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

        sf = (self.Sf(t, c, d) + self.Sf(t, c + 1)) / 2
        g = 9.8
        f = (
            d
            + (self.beta() * self.u(t, c, d) ** 2 - self.beta() * self.u(t, c + 1) ** 2)
            / (2 * g)
            - self.d[c + 1]
            + (self.S0(c) - sf) * self.dc
        )
        return f

    def dEdd(self, t: pd.Timestamp, c: int, d: float):
        """Calculate f prime equation 5.9"""

        sf = (self.Sf(t, c, d) + self.Sf(t, c + 1)) / 2
        g = 9.8
        B = self.Bwet(c, d)
        A = self.A(c, d)
        fprime = (
            1
            - (self.beta() * B * self.u(t, c, d) ** 2) / (g * A)
            + sf * (B + 0.667 / self.R(c, d)) * self.dc / A
        )
        return fprime

    def update_depth(self, t: pd.Timestamp):

        # get the most downstream depth
        self.d[-1] = self.get_ds_d(t)

        # use d[i+1] to calculate d[i]
        for i in range(len(self.cs) - 2, -1, -1):

            def f(d):
                return self.conservation_of_energy(t, i, d)

            def fprime(d):
                return self.dEdd(t, i, d)

            # d, info = scipy.optimize.newton(
            #     f,
            #     self.d[i],
            #     fprime=fprime,
            #     full_output=True,
            # )
            # if not info.converged:
            #     raise ValueError(f"Can't solve for new d at index {i}. {info=}")
            d = utils.newton(f, self.d[i], fprime=fprime)
            self.d[i] = d


class DynamicWaveModel(HydroDynamicModel):
    pass
