import numpy as np
from gin import GrateConfig


class Channel:
    """Flume, river, braided channel"""

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

    def d90(self, c: int):
        """90th percentile of the grain diameter.

        Referred to in Eq 8.4

        The grain diameter of the surface layer, and in the case of the
        floodplain use bank d90 from config file if specified else surface
        layer
        """
        raise NotImplementedError(f"d90({c})")

    def ng(self, c: int):
        """Grain roughness at given chainage"""
        return 0.044 * self.d90(c) ** (1 / 6)

    def area(self, c: int, h: float):
        """Return area of water between active layer surface and h"""
        raise NotImplementedError(f"area({c}, {h})")

    def nf(self, c: int):
        """Form roughness"""
        raise NotImplementedError(f"nf({c})")

    def P(self, c: int, h: float):
        """Wetted perimeter at chainage"""
        raise NotImplementedError(f"P({c}, {h})")

    def S0(self, c: int):
        """Bed slope at c

        Bed slope is the slope of the active layer surface
        """
        raise NotImplementedError(f"S0({c})")

    def B(self, c: int, h: float):
        """Water surface width"""
        raise NotImplementedError(f"B({c}, {h})")

    def active_layer_elevation(self, c: int):
        """Return active layer elevation"""
        raise NotImplementedError(f"active_layer_elevation({c})")


class Flume(Channel):
    def __init__(self, cfg: GrateConfig):
        super().__init__(self, cfg)
        num = self.chainpts()
        self.storage_layer_heights = np.full(num, self._cfg.morphological.layer)
        self.active_layer_heights = np.full(num, self._cfg.morphological.la)

    def area(self, c: int, h: float):
        """Return area of water between active layer surface and h

        Parameters:
        -----------
        h: float
            The height of the water.  FIXME.  maybe elevation.

        """
        raise NotImplementedError(f"area({c}, {h})")

    def d90(self, c: int):
        """90 percentile of grain diameter of surface layer."""
        _ = c
        # FIXME
        return self._cfg.bankd90

    def nf(self, c: int):
        """Form roughness"""
        _ = c
        return self._cfg.formrf

    def active_layer_elevation(self, c: int):
        """Return active layer elevation"""
        self.storage_layer_heights[c] + self.activate_layer_heights[c]


class River(Channel):
    def nf(self, c: int):
        """Form roughness

        nfc * sum_k (r_k * p_k) / pc

        where nfc is default form roughness of cross-section
        rk and pk are relative roughness factory wetted perimeter FIXME
        pc FIXME

        """
        raise NotImplementedError(f"nf({c})")
