import numpy as np
import warnings

from gin import GrateConfig
from cross_section import CrossSection


class StorageLayers:
    """Holds sprop across all chainages in (nc, nlayers, nbins, nlith) array

    It would be handy if [:, 0, :, :] was the top storage layer for transport
    and output.  However because we are adding and removing layers there would
    be a lot of copying to maintain that.  Rather, keep an array of indices to
    the top storage layer and treat the 4d array as a circular list in the
    nlayers dimension.  Ie suppose we have 3 layers called a, b, and c at a
    chainage (doesn't matter which one):
        [, a, :, :]
        [, b, :, :]
        [, c, :, :]
    and head = [...0,...] indicating layer a is at the top at this chainage.
    We add a layer, call it d, so b and c must be combined:
        [, a, :, :]
        [, b+c, :, :]
        [, c, :, :]
    and now the third row can be reused as the top layer
        [, a, :, :]
        [, b+c, :, :]
        [, d, :, :]
    and head is updated to [...2,...].  We read this dimension circularly thus
    have layers d, a, b+c.

    Similar process when deleting a layer, except we must update 'count' so we
    know that layer isn't in use.  Eg suppose we continue with the
    above, but delete layer d (since it is the top one), we have layers a and
    b+c left
        [, a, :, :]
        [, b+c, :, :]
        [, ?, :, :]
    with head back to [...0,...] and count [...2,...]

    """

    def __init__(self, cfg: GrateConfig, xss: list[CrossSection], space):
        """

        Parameters
        ----------
        cfg: GrateConfig
            To get the number of layers

        xss: list[CrossSection]
            For getting initial xprop

        space:  np.array
            Length nc
            How much space for storage layers per chainage
        """

        # initialise with sprop duplicated for each layer
        i = np.array([xs._sprop for xs in xss])  # (nc, nbins, nlith)
        self.nlayers = nlayers = cfg.morphological.num_slayers
        self._sprop = np.repeat(i[:, np.newaxis, :, :], nlayers, axis=1)

        self.max_thick = max_thick = cfg.morphological.slayer_max_thickness

        # use space[c] for the layers at chainage c.  each layer is at most max_thick,
        # last layer uses up all the remaining space down to bedrock_rl.
        # thickness[i, j] is the thickness of sprop[i, j, :, :]
        # (nc, nlayers)
        self._thickness = np.minimum(
            max_thick,
            np.maximum(space[:, None] - max_thick * np.arange(nlayers), 0.0),
        )
        self._thickness[:, -1] = np.maximum(
            space - self._thickness[:, :-1].sum(axis=1), 0.0
        )

        # number of storage layers at each chainage, (nc, )
        self._count = np.count_nonzero(self._thickness > 0, axis=1)

        # index to top storage layer (nc, )
        self._head = np.zeros(len(xss), dtype=np.intp)

        # print()
        # print(self)

    def get_top_sprop(self):
        nc = self._sprop.shape[0]
        # this necessitates a copy
        return self._sprop[np.arange(nc), self._head]

    def get_thickness(self):
        """Return thicknesses in logical top-to-bottom order, padding inactive layers with NaN."""
        out = np.full(self._thickness.shape, np.nan)
        for c, (head, count) in enumerate(zip(self._head, self._count)):
            idx = (head + np.arange(count)) % self.nlayers
            out[c, :count] = self._thickness[c, idx]
        return out

    def get_sprop(self):
        """Return sprop in logical top-to-bottom order, padding inactive layers with NaN."""
        out = np.full(self._sprop.shape, np.nan)
        for c, (head, count) in enumerate(zip(self._head, self._count)):
            idx = (head + np.arange(count)) % self.nlayers
            out[c, :count] = self._sprop[c, idx]
        return out

    def __str__(self):
        ret = ""
        for c in range(len(self._count)):
            ret += f"{self._count[c]:5d}"
        ret += "\n"
        for lay in range(self._thickness.shape[1]):
            for c in range(self._thickness.shape[0]):
                ret += f"{self._thickness[c, lay]:5.2f}"
            ret += "\n"
        for c in range(len(self._head)):
            ret += f"{self._head[c]:5d}"

        return ret

    def add_top_layer(self, c, new_thickness):
        """Insert a new empty top layer at chainage c."""

        # print(f"Making new storage layer at {c}")
        # print(self)

        old_head = self._head[c]
        new_head = (old_head - 1) % self.nlayers

        if self._count[c] == self.nlayers:
            # print(f"Combining bottom layers at {c}")
            # mix the bottom two layers to free a slot
            bottom = (old_head + self._count[c] - 1) % self.nlayers
            above_bottom = (old_head + self._count[c] - 2) % self.nlayers
            tup = self._thickness[c, above_bottom]
            tbot = self._thickness[c, bottom]

            if np.isinf(tbot):
                self._thickness[c, above_bottom] = np.inf
                self._sprop[c, above_bottom] = self._sprop[c, bottom].copy()
            else:
                total = tup + tbot
                assert total > 0
                self._sprop[c, above_bottom] = (
                    self._sprop[c, above_bottom] * tup + self._sprop[c, bottom] * tbot
                ) / total
                self._thickness[c, above_bottom] = total

            self._count[c] -= 1

        # slot immediately before the old head is available
        self._head[c] = new_head

        # new layer has undefined composition, so just leave it
        # self._sprop[c, new_head] = self._sprop[c, old_head].copy()
        self._thickness[c, new_head] = new_thickness
        self._count[c] += 1

        # print()
        # print(self)

    def update(self, f_int: np.array, dy: np.array):
        """Modify top layer, proportions and thickness

        Parameters
        ----------
        f_int: np.array
            Shape is (nc-1, nbins, nlith).

        dy: np.array
            Shape is (nc-1,)
        """

        for c in range(1, len(self._sprop)):
            change = dy[c - 1]

            if abs(change) > self.max_thick:
                warnings.warn(
                    f"dy[{c - 1}]={change} exceeds the allowed range "
                    f"(-{self.max_thick}, {self.max_thick}); capping it."
                )
                change = np.clip(change, -self.max_thick, self.max_thick)

            if change == 0:
                continue

            if change > 0:
                # aggradation: fill the existing top layer first
                i = self._head[c]
                thick = self._thickness[c, i]
                fill = min(change, max(self.max_thick - thick, 0.0))

                if fill > 0:
                    # FIXME
                    # print(
                    #     f"Aggrading with fill at {c}, fill is {fill}, thickness is {self._thickness[c, i]}"
                    # )
                    s = self._sprop[c, i]
                    s[:] = (s * thick + f_int[c - 1] * fill) / (thick + fill)
                    self._thickness[c, i] = thick + fill
                    change -= fill
                    # print(f"done thickening, now {self._thickness[c, i]}")

                # remaining aggradation forms new top layer
                if change > 0:
                    self.add_top_layer(c, change)
                    self._sprop[c, self._head[c]] = f_int[c - 1].copy()
                    # FIXME
                    # print(
                    #     f"Aggrading at {c}, made a new top layer with {change}, its thickness is {self._thickness[c, self._head[c]]}"
                    # )

            else:
                # degradation: remove material from the top
                remaining = -change

                while remaining > 0 and self._count[c] > 0:
                    i = self._head[c]
                    thick = self._thickness[c, i]

                    # can't degrade, we are done
                    if np.isinf(thick):
                        break

                    removed = min(remaining, thick)
                    new_thick = thick - removed
                    self._thickness[c, i] = new_thick
                    remaining -= removed

                    # more to go?
                    if new_thick <= 0:
                        self._thickness[c, i] = 0
                        self._head[c] = (i + 1) % self.nlayers
                        self._count[c] -= 1

        # print()
        # print(self)
