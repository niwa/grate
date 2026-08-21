import math
import numpy as np
import pandas as pd
import scipy
from gin import GrateConfig
from channel import Channel


class HydroDynamicModel:
    def __init__(self, cfg: GrateConfig, chan: Channel):
        self._cfg = cfg
        self._channel = chan
        self.dc = self._channel.dc
        self.cs = self._channel.chainpts
        self.initialize(self._cfg.SimulationTime.start)

    def initialize(self, t: pd.Timestamp):
        """Set h to initial values."""
        raise NotImplementedError(f"initialize({t})")

    def beta(self):
        """Momentum correction factor"""
        return self._channel.beta()

    def Q(self, t: pd.Timestamp, c: int):
        """Return flow at point along river

        Parameters
        ----------
        c: int
            Chainage point along river in units of dc

        t: pd.Timestamp
            Time
        """
        raise NotImplementedError(f"Q({t}, {c})")

    def A(self, c: int, h: float | None = None):
        """The area of the water at c chainage

        Parameters
        ----------
        c: int
            Chainage point along river in units of dc
        """
        if h is None:
            h = self.h[c]
        return self._channel.area(c, h)

    def u(self, t: pd.Timestamp, c: int, h: float | None = None):
        """The mean velocity, ie Q/A"""
        if h is None:
            h = self.h[c]
        return self.Q(t, c) / self.A(c, h)

    def ng(self, c: int):
        """Grain roughness"""
        return self._channel.ng(c)

    def nf(self, c: int, h: float):
        """Form roughness"""
        return self._channel.nf(c, h)

    def conveyance(self, c: int, h: float | None = None):
        """K conveyance

        A * R^(2/3) / (ng + nf)

        where
            A is the area
            R is A/P
            ng is grain roughness
            nf is form roughness
        """
        if h is None:
            h = self.h[c]
        A = self.A(c, h)
        P = self.P(c, h)
        R = A / P
        return A * R ** (2 / 3) / (self.ng(c) + self.nf(c, h))

    def Sf(self, t: pd.Timestamp, c: int, h: float | None = None):
        """Return friction slope, ie. Q abs(Q) / K^2"""
        Q = self.Q(t, c)
        return Q * abs(Q) / self.conveyance(c, h) ** 2

    def P(self, c: int, h: float | None = None):
        """Wetted perimeter at chainage"""
        if h is None:
            h = self.h[c]
        return self._channel.P(c, h)

    def S0(self, c: int):
        """Bed slope at c"""
        return self._channel.S0(c)

    def B(self, c: int, h: float | None = None):
        """Water surface width"""
        if h is None:
            h = self.h[c]
        return self._channel.B(c, h)

    def R(self, c: int, h: float | None = None):
        """Hydraulic radius A/P"""
        if h is None:
            h = self.h[c]
        return self.A(c, h) / self.P(c, h)

    def update_height(self):
        raise NotImplementedError()

    def get_ds_h(self, t: pd.Timestamp):
        """Return downstream heigh of water."""
        lastchain = len(self.cs) - 1
        tv = self._cfg._processed_downstream_boundary.value_at(t)
        if "elevation" in tv:
            return tv["value"] - self._channel.bed_level(lastchain)
        if "depth" in tv:
            return tv["value"]
        if "normal" in tv:
            slope = tv["normal"]["slope"]
            Q = self.Q(lastchain, t)

            def f(h):
                K = self.conveyance(lastchain, h)
                return Q - K * math.sqrt(slope)

            h, info = scipy.optimize.newton(f, self.h[lastchain], full_output=True)
            if not info.converged:
                raise ValueError(f"Can't solve downstream normal depth. {info=}")
            return h

        raise ValueError(f"Unknown downstream boundary type: {tv['type']}")


class QuasiSteadyModel(HydroDynamicModel):
    def initialize(self, t: pd.Timestamp):
        """Set h to initial value."""
        # get the chain values from channel so we know lengths
        # FIXME, just starting 1m of water depth
        self.h = np.ones(len(self.cs))

        # if the downstream boundary condition is normal, grab the hinit which
        # is an elevation (need to subtract off
        tv = self._cfg._processed_downstream_boundary.value_at(t)
        if "normal" in tv:
            self.h[-1] = tv["normal"]["hinit"]
            self.h[-1] -= self._channel.bed_level(len(self.cs) - 1)

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

    def f57(self, t: pd.Timestamp, c: int, h: float):
        """Calculate equation 5.7

        f(h_i^*) = h_(i) + (β_i (u_i^* )^2)/2g-(β_(i+1) u_(i+1)^2)/2g-h_(i+1)+(S_0-S_f^*)Δx

        h_0 is height at most upstream
        u is velocity of water.
        i is chain point index (c in this case)

        We should have already sorted at c+1

        """

        sf = (self.Sf(t, c, h) + self.Sf(t, c + 1)) / 2
        g = 9.8
        f = (
            h
            + (
                self.beta(c) * self.u(t, c, h) ** 2
                - self.beta(c + 1) * self.u(t, c + 1) ** 2
            )
            / (2 * g)
            - self.h[c + 1]
            + (self.S0(c) - sf) * self.dc
        )
        return f

    def fprime59(self, t: pd.Timestamp, c: int, h: float):
        """Calculate f prime equation 5.9"""

        sf = (self.Sf(t, c, h) + self.Sf(t, c + 1)) / 2
        g = 9.8
        B = self.B(c, h)
        A = self.A(c, h)
        fprime = (
            1
            - (self.beta(c) * B * self.u(t, c, h) ** 2) / (g * A)
            + sf * (B + 0.667 / self.R(c, h)) * self.dc / A
        )
        return fprime

    def update_height(self, t: pd.Timestamp):

        # get the most downstream height
        self.h[-1] = self.get_ds_h(t)

        # use h[i+1] to calculate h[i]
        for i in range(len(self.cs) - 2, 0, -1):

            def f(h):
                return self.f57(t, i, h)

            def fprime(h):
                return self.fprime59(t, i, h)

            h, info = scipy.optimize.newton(
                f,
                self.h[i],
                fprime=fprime,
                full_output=True,
            )
            if not info.converged:
                raise ValueError(f"Can't solve for new h at index {i}. {info=}")
            print(f"At {i=} was {self.h[i]} now {h}")
            self.h[i] = h


class DynamicWaveModel(HydroDynamicModel):
    pass
