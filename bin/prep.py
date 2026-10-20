# import sys
import datetime as dt
import yaml
import argparse
import pathlib
import numpy as np

# import pandas as pd
from gin import GrateConfig
from channel import River, BraidedChannel
from hydrodynamics_models import QuasiSteadyModel, DynamicWaveModel
from cross_section import Loc


def prep_model(infile: pathlib.Path, odir: pathlib.Path):

    odir.mkdir(parents=True, exist_ok=True)

    # We could skip a little bit of initializing hydromodel and channel, but it
    # is easier and clearer just to do all this
    with open(infile) as f:
        print("Parsing config and boundary conditions...", end="", flush=True)
        cfg = GrateConfig.model_validate(yaml.safe_load(f))

    print("done\nCreating channel...", end="", flush=True)
    chan = {"flume": River, "river": River, "braided_channel": BraidedChannel}[
        cfg.model.channel_type
    ](cfg)

    print("done\nCreating hydromodel...", end="", flush=True)
    _ = {
        "quasi_ss": QuasiSteadyModel,
        "dynamic": DynamicWaveModel,
    }[cfg.model.hydro_model_type](cfg, chan)
    print("done", flush=True)

    # simplified config file
    kv = {}

    # inflows
    const, data = cfg._processed_inflow.get_df()
    if const:
        t = np.empty(0)
        v = data[:, 1]
    else:
        t = (
            data.index.astype("datetime64[ns]").astype("int64").to_numpy() / 1e9
        )  # fraction seconds since epoch
        v = data.to_numpy()
    fname = odir / "inflow.npz"
    print(f"Writing {fname}")
    np.savez(fname, t=t, v=v)

    # sediment
    times, data = cfg._processed_sediment_boundary.get_np()
    if times is None:
        t = np.empty(0)
        v = data
    else:
        t = (
            times.astype("datetime64[ns]").astype("int64").to_numpy() / 1e9
        )  # fraction seconds since epoch
        v = data
    fname = odir / "sediment.npz"
    print(f"Writing {fname}")
    np.savez(fname, t=t, v=v)

    # downstream bc
    d = cfg._processed_downstream_boundary
    match d.type:
        case "depth":
            kv["downstream"] = {"type": "depth", "value": d.value}
        case "normal":
            kv["downstream"] = {
                "type": "normal",
                "slope": d.slope,
                "wl_init": d.wl_init,
            }
        case "elevation":
            fname = "downstream.npz"
            kv["downstream"] = {"type": "elevation", "fname": str(fname)}
            fname = odir / "downstream.npz"
            print(f"Writing {fname}")
            np.savez(fname, t=np.empty(0), v=np.array([d.value]))
        case "ts":
            fname = "downstream.npz"
            kv["downstream"] = {"type": "elevation", "fname": str(fname)}
            fname = odir / "downstream.npz"
            print(f"Writing {fname}")
            t = d.value.index.astype("datetime64[ns]").astype("int64").to_numpy() / 1e9
            v = d.value.to_numpy()
            fname = odir / "downstream.npz"
            np.savez(fname, t=t, v=v)

    # per cross section profile stuff
    fname = odir / "profiles.npz"
    print(f"Writing {fname}")
    np.savez(
        fname,
        cs=chan._cs,  # (nc, )
        bchan=chan._Bchan,  # (nc, )
        min_bed_level=chan.get_min_bed_level(),  # (nc, )
        mean_bed_level=chan.get_mean_bed_level(),  # (nc, )
    )

    # interpolator tables for each cross-section
    for name in ("area", "P", "Bwet", "nf"):
        tabs = {}
        for i, xs in enumerate(chan._xss):
            for j, loc in enumerate((Loc.LEFT, Loc.CHANNEL, Loc.RIGHT)):
                ip = getattr(xs, f"_{name}_interpolator")[loc]
                if ip is not None:
                    tabs[f"{i}_{j}"] = ip.get_np()
        fname = odir / f"{name}_tabs.npz"
        print(f"Writing {fname}")
        np.savez(fname, **tabs)

    # print(chan._xss[44]._P_interpolator[Loc.CHANNEL].xs)
    # print(chan._xss[44]._P_interpolator[Loc.CHANNEL].fs)
    #
    # np.set_printoptions(precision=17)
    # xs = np.asarray(chan._xss[44]._P_interpolator[Loc.CHANNEL].xs)
    # fs = np.asarray(chan._xss[44]._P_interpolator[Loc.CHANNEL].fs)
    # print("xs:", xs)
    # print("fs:", fs)
    # x = 0.43
    # print("x:", repr(x))
    # print("np.interp:", np.interp(x, xs, fs))

    # layer grain props
    fname = odir / "grain.npz"
    print(f"Writing {fname}")
    np.savez(
        fname,
        acfd=chan._layers.acfd,  # (nc, nbins, nlith)
        scfd=chan._layers.scfd,  # (nc, nbins, nlith)
        rgsizes=chan._layers.rgsizes,  # (nbins, )
        uppersizes=chan._layers.uppersizes,  # (nbins, )
        sediment_densities=chan._layers.sediment_densities,  # (nprofiles,)
        bankd90=chan._bankd90,  # (nc, ) where nan use channel d90
    )

    kv["morphological"] = {
        "la": cfg.morphological.la,
        "poro": cfg.morphological.poro,
        "chi": cfg.morphological.chi,
        "beta": cfg.morphological.beta,
        "qthres": cfg.morphological.qthres,
        "qthres_dtmultiplier": cfg.morphological.qthres_dtmultiplier,
    }
    kv["simulation_time"] = {
        "start": cfg.simulation_time.start.replace(tzinfo=dt.timezone.utc).timestamp(),
        "end": cfg.simulation_time.end.replace(tzinfo=dt.timezone.utc).timestamp(),
        "max_dt": cfg.max_dt,
        "cdt": cfg.simulation_time.cdt,
    }
    kv["output"] = {
        "frequency": cfg.output.frequency,
        "variables": cfg.output.variables,
        "fname": str(cfg.output.fname),
    }
    with open(odir / "config.yaml", "w") as fh:
        yaml.dump(kv, fh, default_flow_style=False, sort_keys=False)


def main():
    # parse command line
    p = argparse.ArgumentParser(
        description="""Prep a sim, doing x-section interp and setting up profile interp tables""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("infile", type=pathlib.Path, help="Grate yaml input file")
    p.add_argument("dir", type=pathlib.Path, help="Directory to store outputs in")
    args = p.parse_args()
    prep_model(args.infile, args.dir)


if __name__ == "__main__":
    main()
