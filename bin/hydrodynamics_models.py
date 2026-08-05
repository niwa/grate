import numpy as np
import pandas as pd
from gin import GrateConfig
from channel import Channel


class HydroDynamicModel:
    def __init__(self, cfg: GrateConfig, chan: Channel):
        self._cfg = cfg
        self._channel = chan

    def Q(self, x: float, t: pd.Timestamp):
        """Return flow at point along river

        Parameters
        ----------
        x: float
            Distance along river

        t: pd.Timestamp
            Time
        """
        raise NotImplementedError(f"Q({x}, {t})")

    def conveyance(i: int, cfg):
        """K conveyance

        A * R^(2/3) / (ng + nf)

        where
            A is the area
            R is A/P
            ng is grain roughness
            nf is form roughness
        """
        R = A(i) / P(i)
        return A(i) * R ** (2 / 3) / (ng(cfg) + nf(i, cfg))

    def Sf(self, x: float, t: pd.Timestamp):
        """Return friction slope, ie. Q abs(Q) / K^2"""
        Q = self.Q(x, t)
        return Q * abs(Q) / self.conveyance(x)


class QuasiSteadyModel(HydroDynamicModel):
    def Q(self, x: float, t: pd.Timestamp):
        """Return flow at point along river

        Parameters
        ----------
        x: float
            Distance along river

        t: pd.Timestamp
            Time
        """
        return sum(
            pi.value_at(t) for pi in self._cfg._processed_inflow if pi.ordinate <= x
        )


class DynamicWaveModel(HydroDynamicModel):
    pass


def beta(i: int):
    """Momentum correction factor"""
    # FIXME
    return 1


def Q(i: int):
    """The flow at i"""
    pass


def A(h: np.array, i: int):
    """The area of the water at i"""
    # FIXME, need the channel geom to get width at i
    pass


def vel(i: int):
    """The mean velocity, ie Q/A"""
    return Q(i) / A(i)


def ng(cfg: GrateConfig):
    """Grain roughness"""
    return 0.044 * cfg.bankd90 ** (1 / 6)


def nf(i: int, cfg: GrateConfig):
    """Form roughness

    nfc * sum (rk * pk) / pc

    where nfc is default form roughness of cross-section
    rk and pk are relative roughness factory wetted perimeter FIXME
    pc FIXME

    FIXME, needs changing for nonflume
    """
    return cfg.formrf


def P(i: int):
    """FIXME"""


def S0():
    """FIXME"""


def f57(h: np.array, i: int, cfg: GrateConfig):
    """Calculate

    f(h_i^*) = h_(i) + (β_i (u_i^* )^2)/2g-(β_(i+1) u_(i+1)^2)/2g-h_(i+1)+(S_0-S_f^*)Δx

    h_0 is height at most upstream

    """

    dx = cfg.simulation_time.max_dx
    sf = (Sf(i) + Sf(i + 1)) / 2
    g = 9.8
    f = (
        h[i]
        + (beta(i) * vel(i) ** 2 - beta(i + i) * vel(i + 1) ** 2) / 2 / g
        - h[i + 1]
        + (S0 - sf) * dx
    )
    return f


def fprime59(h: np.array, i: int, cfg: GrateConfig):
    dx = cfg.simulation_time.max_dx
    sf = (Sf(i) + Sf(i + 1)) / 2
    g = 9.8
    R = A(i) / P(i)
    fprime = (
        1
        - (beta(i) * vel(i) ** 2 * beta(i)) / g / A(i)
        + sf * (beta(i) + 0.667 * R) * dx / A(i)
    )
    return fprime


def update_height_at_i(h: np.array, i: int, cfg: GrateConfig):
    newh = h[i] - f57(h, i, cfg) / fprime59(h, i, cfg)
