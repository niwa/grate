import numpy as np
from gin import GrateConfig


class Channel:
    """Flume, river, braided channels

    FIXME: ideally this will return the cross-section info, slope at a point,
    roughness etc
    """

    def __init__(self, cfg: GrateConfig):
        self._cfg = cfg
        self.dc = self._cfg.discretisation.dc

    @property
    def chainpts(self):
        """Return chain points"""
        d = self._cfg.discretisation
        return np.arange(d.chainage_min, d.chainage_max + self.dc / 2, self.dc)

    def beta(self):
        """Momentum correction factor"""
        return 1

    def ng(self, c: int):
        """Grain roughness at given chainage"""
        return 0.044 * self.d90(c) ** (1 / 6)


class Flume(Channel):
    def d90(self, c: int):
        """90 percentile of grain diameter of surface layer."""
        _ = c
        # FIXME
        return self._cfg.bankd90

    def nf(self, c: int):
        """Form roughness"""
        _ = c
        return self._cfg.formrf


class River(Channel):
    def d90(self, c: int):
        """90 percentile of grain diameter of surface layer."""
        raise NotImplementedError(f"d90({c})")

    def nf(self, c: int):
        """Form roughness

        nfc * sum_k (r_k * p_k) / pc

        where nfc is default form roughness of cross-section
        rk and pk are relative roughness factory wetted perimeter FIXME
        pc FIXME

        """
        raise NotImplementedError(f"nf({c})")
