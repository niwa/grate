import yaml
import argparse
import pathlib
import pandas as pd
import time
from gin import GrateConfig
from channel import Channel
from hydrodynamics_models import QuasiSteadyModel
from output import Output


def run_model(infile: pathlib.Path):

    with open(infile) as f:
        cfg = GrateConfig.model_validate(yaml.safe_load(f))
    chan = Channel(cfg)
    hmodel = QuasiSteadyModel(cfg, chan)
    out = Output(cfg, hmodel, chan)

    start = cfg.simulation_time.start
    end = cfg.simulation_time.end
    step = 0
    t = start
    dt = pd.Timedelta(seconds=chan.get_dt())
    run_start = time.perf_counter()

    while t <= end:
        hmodel.update_depth(t)
        chan.propogate_sediment(t, hmodel)
        steps_to_go = int((end - t) / dt)

        if step % cfg.output.frequency == 0:
            out.write_step(t)

            if step > 100:
                elapsed = time.perf_counter() - run_start
                seconds_per_step = elapsed / step
                seconds_left = steps_to_go * seconds_per_step
                finish = pd.Timestamp.now() + pd.Timedelta(seconds=seconds_left)

                print(
                    f"\rApproximate steps left... {steps_to_go:,} (at {finish:%H:%M:%S})    ",
                    end="",
                    flush=True,
                )

        step += 1
        dt = pd.Timedelta(seconds=chan.get_dt())
        t += dt

    print()

    # write a final step even if not on the frequency
    if (step - 1) % cfg.output.frequency != 0:
        out.write_step(t - dt)


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
