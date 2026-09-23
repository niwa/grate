import yaml
import argparse
import pathlib
import pandas as pd
from gin import GrateConfig
from channel import Channel
from hydrodynamics_models import QuasiSteadyModel
from output import OutputH5


def run_model(infile: pathlib.Path):
    with open(infile) as f:
        cfg = GrateConfig.model_validate(yaml.safe_load(f))
    chan = Channel(cfg)
    hmodel = QuasiSteadyModel(cfg, chan)

    start = cfg.simulation_time.start
    end = cfg.simulation_time.end

    out = OutputH5(cfg, hmodel, chan)

    step = 0
    t = start
    dt = None
    while t <= end:
        print(f"At time = {t}")
        hmodel.update_depth(t)
        if step % cfg.output.frequency == 0:
            out.write_step(step, t)
        chan.propogate_sediment(t, hmodel)
        dt = pd.Timedelta(seconds=chan.get_dt())
        t += dt
        step += 1

    # write a final step even if not on the frequency
    if (step - 1) % cfg.output.frequency != 0:
        out.write_step(step - 1, t - dt)

    # for H5 don't need this since already been updating file
    out.write_final()


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
