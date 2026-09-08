import yaml
import argparse
import pathlib
import pandas as pd
from gin import GrateConfig
from channel import Channel
from hydrodynamics_models import QuasiSteadyModel


def run_model(infile: pathlib.Path):
    with open(infile) as f:
        cfg = GrateConfig.model_validate(yaml.safe_load(f))
    chan = Channel(cfg)

    hmodel = QuasiSteadyModel(cfg, chan)

    start = cfg.simulation_time.start
    end = cfg.simulation_time.end
    t = start

    # store water height, water flow
    while t <= end:
        print(f"At time = {t}")
        print("Updating water height")
        hmodel.update_height(t)
        print(f"heights are {hmodel.h}")
        print("Propogating sediment")
        chan.propogate_sediment(t, hmodel)
        t += pd.Timedelta(seconds=chan.get_dt())


def main():
    # parse command line
    p = argparse.ArgumentParser(
        description="""Run a simulation on given model""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("infile", type=pathlib.Path, help="Grate yaml input file")
    args = p.parse_args()

    run_model(args.infile)


if __name__ == "__main__":
    main()
