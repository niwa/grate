import yaml
import argparse
import pathlib
import numpy as np
import pandas as pd
import xarray as xr
from gin import GrateConfig
from channel import Channel
from hydrodynamics_models import QuasiSteadyModel


def run_model(infile: pathlib.Path, debugfile: pathlib.Path):
    with open(infile) as f:
        cfg = GrateConfig.model_validate(yaml.safe_load(f))
    chan = Channel(cfg)

    hmodel = QuasiSteadyModel(cfg, chan)

    start = cfg.simulation_time.start
    end = cfg.simulation_time.end
    t = start

    # store water height, water flow
    if debugfile:
        times = []
        heights = []
        flows = []

    while t <= end:
        print(f"At time = {t}")
        # print("Updating water height")
        hmodel.update_height(t)
        # print(f"heights are {hmodel.h}")
        # print("Propogating sediment")

        if debugfile:
            times.append(t)
            heights.append(hmodel.h.copy())
            Qs = [hmodel.Q(t, c) for c in chan.cs]
            flows.append(Qs)
            outds = xr.Dataset(
                data_vars={
                    "height": (
                        ("time", "cidx"),
                        np.asarray(heights),
                    ),
                    "flow": (("time", "cidx"), np.asarray(flows)),
                },
                coords={
                    "time": times,
                    "cross_section": np.arange(len(chan.cs)),
                },
            )
            outds.to_netcdf(debugfile, mode="w")

        chan.propogate_sediment(t, hmodel)
        t += pd.Timedelta(seconds=chan.get_dt())


def main():
    # parse command line
    p = argparse.ArgumentParser(
        description="""Run a simulation on given model""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("infile", type=pathlib.Path, help="Grate yaml input file")
    p.add_argument("--debugfile", type=pathlib.Path, help="Debug output .nc")
    args = p.parse_args()

    run_model(args.infile, args.debugfile)


if __name__ == "__main__":
    main()
