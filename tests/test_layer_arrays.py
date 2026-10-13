"""Checks that the array code reproduces the original per-cross-section code.

Run from the directory holding your modules (layers.py is kept only as the
reference implementation these tests compare against):

    python test_layer_arrays.py            # tests
    python test_layer_arrays.py --bench    # tests + old-vs-new timings

Everything is built from synthetic data (no yaml, no profile csvs).  gin needs
pydantic and static_interpolator isn't needed by any test, so if they can't be
imported they are replaced by empty stubs.  grainprofile is the real module.

The "old" route is the original code: LayerStack.__init__ per surveyed profile,
the "Interp" resolution that used to be in CrossSection.
resolve_interpolated_properties, the LayerStack.interpolate loop that used to
be in Channel._get_interpolated_cross_sections, and the per-chainage loops that
used to be in Channel.  Those loops are copied below as ref_*.
"""

import importlib
import sys
import time
import types
from types import SimpleNamespace

import numpy as np


def _stub(name, **attrs):
    try:
        importlib.import_module(name)
    except ImportError:
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        sys.modules[name] = m


_stub("static_interpolator", StaticInterpolator=None)
_stub("gin", GrateConfig=object, CrossSectionProfile=object)

from layers import LayerStack  # noqa: E402  (reference implementation)
from cross_section import CrossSection  # noqa: E402
from channel import Channel  # noqa: E402
from layer_arrays import LayerArrays, interp_rows  # noqa: E402

NBINS, NLITH = 8, 3
DENS = np.array([2650.0, 2700.0, 2400.0])
CHI = 0.5
close = lambda a, b, what="": np.testing.assert_allclose(  # noqa: E731
    a, b, rtol=1e-10, atol=1e-14, err_msg=what
)


# ------------------------------------------------------------------ fixtures


def make_cfg(seed, nc, nprof_groups=3):
    """A config-like object with the fields the layer code reads.

    Cross-section profiles sit every third of the way along a channel of nc
    points spaced 10 apart; some active/storage groups are "Interp".
    """
    rng = np.random.default_rng(seed)
    bounds = 0.0005 * 2.0 ** np.arange(NBINS + 1)  # 0.5mm .. 128mm
    cum = np.cumsum(rng.random((NBINS, nprof_groups)) + 0.05, axis=0)
    cum = np.vstack([np.zeros(nprof_groups), cum])
    cum = cum / cum[-1] * 100
    cfds = [[bounds[k], *cum[k]] for k in range(NBINS + 1)]
    lith = (
        rng.random((NBINS * NLITH, nprof_groups))
        * (rng.random((NBINS * NLITH, nprof_groups)) > 0.3)
        + 0.01
    )

    last = 10.0 * (nc - 1)
    chain = [0.0, 10.0 * (nc // 3), 10.0 * (2 * nc // 3), last]
    active = [1, "Interp", 2, 3]
    storage = [2, 1, "Interp", 3]
    profiles = [
        SimpleNamespace(chainage=c, active_layer_group=a, storage_layer_group=s)
        for c, a, s in zip(chain, active, storage)
    ]
    # shuffle: from_config must not depend on input order
    profiles = [profiles[i] for i in (2, 0, 3, 1)]

    return SimpleNamespace(
        max_dt=60.0,
        simulation_time=SimpleNamespace(cdt=0.5),
        cross_sections=SimpleNamespace(profiles=profiles),
        grain_size_profiles=SimpleNamespace(
            num_lith=NLITH,
            num_bins=NBINS,
            abrasion_coeffs=[0.1] * NLITH,
            sediment_densities=list(DENS),
            grain_size_cfds=cfds,
            lithfractions=lith.tolist(),
        ),
        morphological=SimpleNamespace(la=0.2, poro=0.3, chi=CHI),
        _processed_sediment_boundary=SimpleNamespace(
            value_at=(lambda arr: lambda t: arr)(1e-5 * rng.random((nc, NBINS, NLITH)))
        ),
    )


def old_route(cfg, cs):
    """ORIGINAL code: list of LayerStack at each chainage point"""
    profiles = sorted(cfg.cross_sections.profiles, key=lambda p: p.chainage)
    stacks = {p.chainage: LayerStack(p, cfg) for p in profiles}
    keys = sorted(stacks)

    def interp(cidx, prop):  # from CrossSection.resolve_interpolated_properties
        i = keys.index(cidx)
        left = right = None
        for c in reversed(keys[:i]):
            v = getattr(stacks[c], prop)
            if not (isinstance(v, str) and v == "Interp"):
                left = c
                break
        for c in keys[i + 1 :]:
            v = getattr(stacks[c], prop)
            if not (isinstance(v, str) and v == "Interp"):
                right = c
                break
        lv, rv = getattr(stacks[left], prop), getattr(stacks[right], prop)
        return lv + (cidx - left) / (right - left) * (rv - lv)

    for c in keys:
        for prop in ("_acfd", "_scfd"):
            v = getattr(stacks[c], prop)
            if isinstance(v, str) and v == "Interp":
                setattr(stacks[c], prop, interp(c, prop))

    out = []  # from Channel._get_interpolated_cross_sections
    for i, cpt in enumerate(cs):
        if cpt in stacks:
            out.append(stacks[cpt])
            continue
        for c0, c1 in zip(keys[:-1], keys[1:]):
            if c0 <= cpt <= c1:
                p0, p1 = stacks[c0], stacks[c1]
                break
        out.append(p0.interpolate(p1, (cpt - c0) / (c1 - c0), i))
    return out


def make_xs(rng, c, bankd90):
    """Geometry-only CrossSection (layers no longer live here)"""
    xs = object.__new__(CrossSection)
    xs.chainage = 10.0 * c
    xs.chainidx = c
    xs.bankd90 = bankd90
    n = 6
    xs.channel = np.column_stack(
        [np.cumsum(rng.random(n) + 0.5), rng.random(n), np.ones(n)]
    )
    xs.min_bed_level = xs.calc_min_bed_level()
    xs.mean_bed_level = xs.calc_mean_bed_level()
    return xs


def make_channel(seed, nc=30):
    """Channel with the state __init__ would give it, minus file loading, and
    the old-route LayerStacks for the same config to compare against"""
    rng = np.random.default_rng(seed)
    cfg = make_cfg(seed, nc)
    ch = object.__new__(Channel)
    ch._cfg = cfg
    ch.max_dc, ch.poro = 10.0, 0.3
    ch._cs = 10.0 * np.arange(nc)
    ch._nc = nc
    ch._xss = [
        make_xs(rng, c, None if c % 3 == 0 else float(rng.uniform(0.02, 0.1)))
        for c in range(nc)
    ]
    ch._init_layer_arrays()
    ch._init_dt()
    ch._Bchan = 20.0 + rng.random(nc)
    ch._Bwet = 5 + 10 * rng.random(nc)
    ch._min_bed_level = np.empty(nc)
    ch._mean_bed_level = np.empty(nc)
    ch._Qb_jli = np.empty((nc, NBINS, NLITH))
    ch._grain_shear_velocity = np.empty(nc)
    ch._sand_fraction = np.empty(nc)
    ch._dsm = np.empty(nc)
    ch._d90 = np.empty((nc, 3))
    ch._ng = np.empty((nc, 3))
    return ch, old_route(cfg, ch._cs)


class Hydro:
    def __init__(self, rng, nc):
        self.Sf = 1e-3 * (0.5 + rng.random(nc))
        self.vel = 1.0 + 2.0 * rng.random(nc)

    def get_Sf(self):
        return self.Sf

    def get_vel(self):
        return self.vel


# ----------------------------------- ORIGINAL per-chainage loops, for reference


def ref_d90(stack, bankd90):  # was CrossSection.calc_d90
    ch = stack.calc_d90()
    return (
        np.array([ch, ch, ch]) if bankd90 is None else np.array([bankd90, ch, bankd90])
    )


def ref_set_layer_arrays(ch, stacks, hydro, do_gsv=True):
    Sf, v = hydro.get_Sf(), hydro.get_vel()
    for c in range(ch._nc):
        xs = ch._xss[c]
        ch._d90[c] = ref_d90(stacks[c], xs.bankd90)
        ch._ng[c] = 0.044 * ch._d90[c] ** (1 / 6)
        if do_gsv:
            ch._grain_shear_velocity[c] = stacks[c].calc_gsv(Sf[c], v[c], ch._d90[c, 1])
        else:
            ch._grain_shear_velocity[c] = 1
        ch._sand_fraction[c] = stacks[c].calc_sand_fraction()
        ch._dsm[c] = stacks[c].calc_dsm()
        ch._min_bed_level[c] = xs.calc_min_bed_level()
        ch._mean_bed_level[c] = xs.calc_mean_bed_level()


def ref_set_transport_rates(ch, stacks):
    for c in range(ch._nc):
        ch._Qb_jli[c] = (
            stacks[c].qb_jli(
                ch._grain_shear_velocity[c], ch._sand_fraction[c], ch._dsm[c]
            )
            * ch._Bwet[c]
        )


def ref_propogate_sediment(ch, stacks, t, hydro):
    bdy = ch._cfg._processed_sediment_boundary.value_at(t)
    up = ch._Qb_jli[:-1] + bdy[:-1]
    my = ch._Qb_jli[1:]
    fact = 1.0 / np.diff(ch._cs) / (1 - ch.poro) / ch._Bchan[1:]
    dydt = fact * (up.sum(axis=(1, 2)) - my.sum(axis=(1, 2)))
    ch.max_dydt = max(np.abs(dydt))
    ch._set_next_dt()
    fact *= ch._dt
    dy = dydt * ch._dt
    p = my / my.sum(axis=(1, 2))[:, None, None]
    flux_diff = (up - my) * fact[:, None, None]
    for i, c in enumerate(range(1, ch._nc)):
        ch._xss[c].aggrade_bed(dy[i])
        df = (
            flux_diff[i] - stacks[c].f_interface(dy[i] > 0, p[i]) * dy[i]
        ) / ch._cfg.morphological.la
        stacks[c].add_to_acfd(df)
    ref_set_layer_arrays(ch, stacks, hydro)


# --------------------------------------------------------------------- tests


def test_interp_rows_matches_np_interp():
    rng = np.random.default_rng(0)
    xp = np.sort(rng.random(7)) * 100
    fp = rng.random((7, 4))
    x = np.concatenate([xp, rng.uniform(xp[0], xp[-1], 50)])
    got = interp_rows(xp, fp, x)
    want = np.stack([np.interp(x, xp, fp[:, k]) for k in range(4)], axis=1)
    close(got, want)
    np.testing.assert_array_equal(got[:7], fp)  # exact hits are bit-identical


def test_from_config_matches_original_construction_and_interpolation():
    for nc in (30, 61):
        cfg = make_cfg(2, nc)
        cs = 10.0 * np.arange(nc)
        stacks = old_route(cfg, cs)
        lay = LayerArrays.from_config(cfg, cs)

        close(lay.acfd, np.stack([s._acfd for s in stacks]), "acfd")
        close(lay.scfd, np.stack([s._scfd for s in stacks]), "scfd")
        close(lay.rgsizes, stacks[0].rgsizes)
        close(lay.sediment_densities, stacks[0].sediment_densities)
        assert lay.chi == stacks[0].chi

        # the "Interp" profiles really were exercised (not all rows equal etc.)
        assert not np.allclose(lay.acfd[1], lay.acfd[nc // 2])

        # chainages on a surveyed, non-Interp profile are copied exactly
        for c in (0, nc - 1):
            np.testing.assert_array_equal(lay.acfd[c], stacks[c]._acfd)


def _methods_vs_stacks(rgsizes_scale, rgsizes_start):
    """LayerArrays methods vs LayerStack methods, with awkward distributions"""
    rng = np.random.default_rng(1)
    rg = rgsizes_start * 2.0 ** np.arange(NBINS)
    stacks = []
    for c in range(24):
        a = rng.random((NBINS, NLITH)) + 0.05
        kind = c % 4
        if kind == 1:
            a[:3] = 0  # flat cumulative frequency at the start
        elif kind == 2:
            a[-2:] = 0  # ... at the end
        elif kind == 3:
            a[2:4] = 0  # ... in the middle
        s = object.__new__(LayerStack)
        s.chainage, s.chainidx, s.chi, s.nlith = 0.0, None, CHI, NLITH
        s.abrasion_coeffs, s.sediment_densities, s.rgsizes = (0.1,) * NLITH, DENS, rg
        s._acfd, s._scfd = a / a.sum(), a / a.sum()
        stacks.append(s)
    lay = LayerArrays.from_stacks(stacks, share=False)
    h = Hydro(rng, len(stacks))

    close(lay.sand_fraction(), [s.calc_sand_fraction() for s in stacks], "sand")
    close(lay.dsm(), [s.calc_dsm() for s in stacks], "dsm")
    close(lay.d90(), [s.calc_d90() for s in stacks], "d90")
    for x in (0.0, 0.1, 0.3, 0.5, 0.9, 1.0):
        close(
            lay.percentile(x),
            [s._LayerStack__grain_size_percentile(x) for s in stacks],
            f"percentile {x}",
        )
    d90 = lay.d90()
    gsv = lay.gsv(h.Sf, h.vel, d90)
    close(gsv, [s.calc_gsv(h.Sf[c], h.vel[c], d90[c]) for c, s in enumerate(stacks)])
    got = lay.qb_jli(gsv, lay.sand_fraction(), lay.dsm())
    want = np.stack(
        [
            s.qb_jli(gsv[c], s.calc_sand_fraction(), s.calc_dsm())
            for c, s in enumerate(stacks)
        ]
    )
    close(got, want, "qb_jli")
    assert (want > 0).any()  # not a vacuous comparison


def test_channel_new_vs_original_loops():
    new, _ = make_channel(7)
    old, stacks = make_channel(7)
    rng = np.random.default_rng(11)
    h = Hydro(rng, new._nc)

    # same start-up sequence the hydro model uses
    new.set_layer_arrays(h, do_gsv=False)
    ref_set_layer_arrays(old, stacks, h, do_gsv=False)
    new.set_layer_arrays(h)
    ref_set_layer_arrays(old, stacks, h)

    start_acfd = new._layers.acfd.copy()
    for step in range(8):
        new._Bwet[:] = old._Bwet[:] = 5 + 10 * rng.random(new._nc)
        h = Hydro(rng, new._nc)

        new.set_transport_rates()
        ref_set_transport_rates(old, stacks)
        close(new._Qb_jli, old._Qb_jli, f"Qb step {step}")

        new.propogate_sediment(step, h)
        ref_propogate_sediment(old, stacks, step, h)

        for name in (
            "_d90", "_ng", "_grain_shear_velocity", "_sand_fraction", "_dsm",
            "_min_bed_level", "_mean_bed_level",
        ):  # fmt: skip
            close(getattr(new, name), getattr(old, name), f"{name} step {step}")
        close(new._dt, old._dt)
        close(new.max_dydt, old.max_dydt)
        close(
            new._layers.acfd, np.stack([s._acfd for s in stacks]), f"acfd step {step}"
        )
        close(
            np.stack([xs.channel for xs in new._xss]),
            np.stack([xs.channel for xs in old._xss]),
        )
    assert not np.allclose(new._layers.acfd, start_acfd)  # state really evolved


# ------------------------------------------------------------------ benchmark


def bench(nc=1000, reps=5):
    new, _ = make_channel(7, nc=nc)
    old, stacks = make_channel(7, nc=nc)
    h = Hydro(np.random.default_rng(1), nc)
    new.set_layer_arrays(h)
    ref_set_layer_arrays(old, stacks, h)

    def timeit(f):
        best = 1e9
        for _ in range(reps):
            t0 = time.perf_counter()
            f()
            best = min(best, time.perf_counter() - t0)
        return best * 1e3

    print(f"\nnc={nc}, nbins={NBINS}, nlith={NLITH}   (best of {reps}, ms)")
    for label, fnew, fold in [
        ("transport rates", new.set_transport_rates, lambda: ref_set_transport_rates(old, stacks)),
        ("set_layer_arrays", lambda: new.set_layer_arrays(h), lambda: ref_set_layer_arrays(old, stacks, h)),
        ("propogate_sediment", lambda: new.propogate_sediment(0, h), lambda: ref_propogate_sediment(old, stacks, 0, h)),
    ]:  # fmt: skip
        tn, to = timeit(fnew), timeit(fold)
        print(f"  {label:20s} old {to:8.2f}   new {tn:8.2f}   x{to / tn:5.1f}")


if __name__ == "__main__":
    for t in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        t()
        print("ok  ", t.__name__)
    if "--bench" in sys.argv:
        bench()
