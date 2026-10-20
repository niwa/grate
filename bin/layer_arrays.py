"""Array-of-all-chainages version of layers.LayerStack.

Where LayerStack holds the grain-size state for ONE cross section, LayerArrays
holds it for ALL nc chainage points at once:

    acfd, scfd          (nc, nbins, nlith)   active / storage layer proportions
    rgsizes             (nbins,)             shared by every chainage
    sediment_densities  (nlith,)             shared by every chainage

and every calc_* method returns an (nc,) (or (nc, nbins, nlith)) array, so
Channel never has to loop over cross sections for sediment calculations.

The maths is a line-for-line translation of layers.LayerStack, the only change
is an extra leading chainage axis.
"""

import numpy as np
import scipy.special

from grainprofile import (
    get_grain_props,
    get_representative_grain_sizes,
    get_grain_size_bin_uppers,
)

KAPPA = 0.4  # Von Karman's constant
GRAVITY = 9.81
WATER_DENSITY = 1000
SAND_SIZE = 0.002  # sand grain size in metres


def interp_rows(xp, fp, x):
    """Piecewise-linear interpolation of whole rows.

    Like np.interp(x, xp, fp) but fp may have extra trailing dimensions, so a
    single call interpolates every (nbins, nlith) slab.  Unlike np.interp this
    refuses to extrapolate/clamp (Channel already asserts chainages lie inside
    the surveyed range), and it returns the surveyed row exactly (not
    a + 1.0*(b - a)) when x lands on a surveyed chainage.

    Parameters
    ----------
    xp: (m,) increasing, e.g. surveyed cross-section chainages
    fp: (m, ...) values at xp
    x:  (n,) targets, e.g. Channel._cs

    Returns
    -------
    (n, ...) array
    """
    xp = np.asarray(xp, dtype=float)
    fp = np.asarray(fp, dtype=float)
    x = np.asarray(x, dtype=float)
    if x.min() < xp[0] or x.max() > xp[-1]:
        raise ValueError("interp_rows: target outside the range of xp")

    j = np.clip(np.searchsorted(xp, x, side="right") - 1, 0, len(xp) - 2)
    f = (x - xp[j]) / (xp[j + 1] - xp[j])
    f = f.reshape((-1,) + (1,) * (fp.ndim - 1))

    out = fp[j] + f * (fp[j + 1] - fp[j])
    # exact hits on a surveyed chainage give the surveyed value, bit for bit
    out = np.where(f == 0, fp[j], out)
    out = np.where(f == 1, fp[j + 1], out)
    return out


def resolve_interp(xp, fp):
    """Fill rows of fp that are NaN ("Interp" in the config) from the nearest
    valid neighbours either side, linearly in xp.

    This reproduces CrossSection.resolve_interpolated_properties: because that
    function interpolates between the nearest *valid* neighbours, the result is
    just piecewise-linear through the valid rows.  Raises like it does if a
    NaN row has no valid row on one side.
    """
    xp = np.asarray(xp, dtype=float)
    fp = np.array(fp, dtype=float)  # copy
    bad = np.isnan(fp.reshape(len(fp), -1)).any(axis=1)
    if not bad.any():
        return fp
    good = ~bad
    if (
        not good.any()
        or (xp[bad] < xp[good].min()).any()
        or (xp[bad] > xp[good].max()).any()
    ):
        raise ValueError("Can't interp: no valid cross-sections either side")
    fp[bad] = interp_rows(xp[good], fp[good], xp[bad])
    return fp


class LayerArrays:
    def __init__(self, acfd, scfd, rgsizes, uppersizes, sed_dens, chi):
        self.acfd = np.asarray(acfd, dtype=float)  # (nc, nbins, nlith)
        self.scfd = np.asarray(scfd, dtype=float)  # (nc, nbins, nlith)
        self.rgsizes = np.asarray(rgsizes, dtype=float)  # (nbins,)
        self.uppersizes = np.asarray(uppersizes, dtype=float)  # (nbins,)
        self.sediment_densities = np.asarray(sed_dens, dtype=float)  # (nprofiles,)
        self.chi = chi
        self.nc, self.nbins, self.nlith = self.acfd.shape

        # upper grain size in "phi" units, used for percentiles
        self._phi = -np.log(self.uppersizes)

    @classmethod
    def from_config(cls, cfg, chainpts):
        """Build the layer state at every chainage point straight from the
        config.  Replaces LayerStack.__init__ for each surveyed profile plus the
        "Interp" resolution and LayerStack.interpolate for each chainage point.

        cfg:       GrateConfig (needs cross_sections.profiles,
                   grain_size_profiles, morphological.chi)
        chainpts:  (nc,) Channel._cs
        """
        gs = cfg.grain_size_profiles
        profiles = sorted(cfg.cross_sections.profiles, key=lambda xs: xs.chainage)
        nbins, nlith = gs.num_bins, gs.num_lith

        def props(attr):
            """(np, nbins, nlith), NaN rows where the group is "Interp" """
            rows = []
            for xs in profiles:
                g = getattr(xs, attr)
                if isinstance(g, str) and g == "Interp":
                    rows.append(np.full((nbins, nlith), np.nan))
                else:
                    rows.append(
                        get_grain_props(g - 1, gs.grain_size_cfds, gs.lithfractions)
                    )
            return np.stack(rows)

        return cls.from_profiles(
            np.array([xs.chainage for xs in profiles], dtype=float),
            props("active_layer_group"),
            props("storage_layer_group"),
            chainpts,
            get_representative_grain_sizes(gs.grain_size_cfds),
            get_grain_size_bin_uppers(gs.grain_size_cfds),
            np.array(gs.sediment_densities),
            cfg.morphological.chi,
        )

    @classmethod
    def from_profiles(
        cls,
        profile_chainage,
        acfd_p,
        scfd_p,
        chainpts,
        rgsizes,
        uppersizes,
        sediment_densities,
        chi,
    ):
        """Build straight from the surveyed profiles, no LayerStack objects.

        profile_chainage: (np,) increasing chainages of the surveyed profiles
        acfd_p, scfd_p:   (np, nbins, nlith); NaN rows mean "Interp"
        chainpts:         (nc,) Channel._cs
        """
        acfd_p = resolve_interp(profile_chainage, acfd_p)
        scfd_p = resolve_interp(profile_chainage, scfd_p)
        return cls(
            interp_rows(profile_chainage, acfd_p, chainpts),
            interp_rows(profile_chainage, scfd_p, chainpts),
            rgsizes,
            uppersizes,
            sediment_densities,
            chi,
        )

    # ----------------------------------------------------- grain statistics

    def _cumfreq(self):
        """(nc, nbins) cumulative proportion, summed over lith"""
        return np.cumsum(self.acfd.sum(axis=2), axis=1)

    def sand_fraction(self):
        """(nc,) proportion of active layer smaller than SAND_SIZE"""
        cf = self._cumfreq()
        xp = self._phi[::-1]  # np.interp wants x increasing
        # FIXME
        # print(f"x={-np.log(SAND_SIZE)} xp={xp} row={cf}")
        return np.array([np.interp(-np.log(SAND_SIZE), xp, row[::-1]) for row in cf])

    def percentile(self, x: float):
        """(nc,) grain size at percentile x (0..1).  Row-wise
        np.interp(x, cf, -log(rgsizes)); like np.interp this assumes each cf
        row is non-decreasing."""
        assert 0 <= x <= 1
        cf = self._cumfreq()
        phi = self._phi
        n = self.nbins

        j = (cf <= x).sum(axis=1) - 1  # last bin with cf <= x, -1 if none
        jc = np.clip(j, 0, n - 2)
        rows = np.arange(self.nc)
        c0, c1 = cf[rows, jc], cf[rows, jc + 1]
        p0, p1 = phi[jc], phi[jc + 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            mid = (p1 - p0) / (c1 - c0) * (x - c0) + p0
        # outside the table np.interp clamps to the end values
        out = np.where(j < 0, phi[0], np.where(x >= cf[:, -1], phi[-1], mid))
        return np.exp(-out)

    def dsm(self):
        return self.percentile(0.5)

    def d90(self):
        return self.percentile(0.9)

    # ------------------------------------------------------------ hydraulics

    def gsv(self, Sf, velocity, d90):
        """(nc,) grain shear velocity u*.  See LayerStack.calc_gsv for the
        derivation (Lambert W solution).  All arguments are (nc,) arrays."""
        ks = 2 * np.asarray(d90)
        a = KAPPA * np.asarray(velocity)
        b = 11 / GRAVITY / np.asarray(Sf) / ks
        x = a * np.sqrt(b) / 2
        bad = ~(x > 0)  # also catches nan
        if bad.any():
            raise ValueError(
                "ustar calc, arg to lambertw <= 0, possibly complex solutions, "
                f"at chainage indices {np.flatnonzero(bad)}"
            )
        return (a / 2 / scipy.special.lambertw(x)).real

    def qb_jli(self, gsv, sand_fraction, dsm):
        """Wilcock & Crowe (2003) eq 9.33 for every chainage at once.

        gsv, sand_fraction, dsm: (nc,)
        Returns (nc, nbins, nlith), per unit width (multiply by Bwet outside).
        """
        # print(f"\ngsv={gsv}\nFs={sand_fraction}\ndsm={dsm}")
        gsv = np.asarray(gsv)[:, None, None]  # (nc,1,1)
        Fs = np.asarray(sand_fraction)  # (nc,)
        dsm = np.asarray(dsm)  # (nc,)

        with np.errstate(over="raise"):
            phirm = 0.021 + 0.015 * np.exp(-20 * Fs)  # (nc,)

        s = self.sediment_densities / WATER_DENSITY  # (nlith,)
        tau_rm = (
            phirm[:, None] * (s - 1) * WATER_DENSITY * GRAVITY * dsm[:, None]
        )  # (nc, nlith)

        ratio = self.rgsizes[None, :] / dsm[:, None]  # (nc, nbins)
        b = 0.67 / (1 + np.exp(1.5 - ratio))  # (nc, nbins)
        tau_rj = tau_rm[:, None, :] * (ratio**b)[:, :, None]  # (nc, nbins, nlith)

        phi = (WATER_DENSITY * gsv**2) / tau_rj  # (nc, nbins, nlith)
        q = self.acfd * gsv**3 / (s - 1) / GRAVITY  # (nc, nbins, nlith)

        # np.where would evaluate both branches everywhere (nan/overflow
        # warnings), so fill through a mask like the original did
        w = np.empty_like(phi)
        mask = phi < 1.35
        w[mask] = 0.002 * phi[mask] ** 7.5
        w[~mask] = 14 * (1 - 0.894 / np.sqrt(phi[~mask])) ** 4.5
        return q * w

    # ------------------------------------------------------------- evolution

    def f_interface(self, aggrading, p, rows=slice(None)):
        """Interface distribution (how much moves INTO the active layer).

        aggrading: (n,) bool,  p: (n, nbins, nlith)
        rows selects which chainages those n are (default all)
        """
        aggrading = np.asarray(aggrading)[:, None, None]
        return np.where(
            aggrading,
            self.chi * self.acfd[rows] + (1 - self.chi) * p,
            self.scfd[rows],
        )

    def update_active_layer(self, dy, flux_diff, p, la, rows=slice(None)):
        """Vectorised version of the per-cross-section loop in
        Channel.propogate_sediment: acfd += (flux_diff - f_interface * dy) / la

        dy: (n,), flux_diff and p: (n, nbins, nlith), rows: which chainages
        """
        f_int = self.f_interface(dy > 0, p, rows)
        df = (flux_diff - f_int * dy[:, None, None]) / la
        self.add_to_acfd(df, rows)

    def add_to_acfd(self, df, rows=slice(None)):
        self.acfd[rows] += df  # in place, so views handed out stay in sync

        # FIXME carried over from LayerStack.add_to_acfd
        row_sums = self.acfd[rows].sum(axis=2)
        if not np.all((row_sums >= 0) & (row_sums <= 1)):
            print(self.acfd[rows])
