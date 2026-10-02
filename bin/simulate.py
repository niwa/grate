import sys
import yaml
import argparse
import pathlib
import pandas as pd
import time
from gin import GrateConfig
from channel import River, BraidedChannel
from hydrodynamics_models import QuasiSteadyModel, DynamicWaveModel
from output import Output


def run_model(infile: pathlib.Path):

    def cache_stats(caches):
        infos = [cache.cache_info() for cache in caches]
        hits = sum(x.hits for x in infos)
        misses = sum(x.misses for x in infos)
        return {
            "hit": hits / (hits + misses) if hits + misses > 0 else "NaN",
            "hits": hits,
            "misses": misses,
            "currsize": sum(x.currsize for x in infos),
        }

    def cache_info():
        q = cache_stats([hmodel.Q])
        w = cache_stats([xs._wetted_segments for xs in chan.xss])
        b = cache_stats([xs._Bwet_cached for xs in chan.xss])
        a = cache_stats([xs._area_cached for xs in chan.xss])
        c = cache_stats([xs._conveyance_cached for xs in chan.xss])
        ng = cache_stats([xs.ng for xs in chan.xss])

        return f"Q: {q}\nws {w}\nbw {b}\na  {a}\nc {c}\nng {ng}"

    with open(infile) as f:
        cfg = GrateConfig.model_validate(yaml.safe_load(f))

    print("Creating channel...", end="", flush=True)
    chan = {"flume": River, "river": River, "braided_channel": BraidedChannel}[
        cfg.model.channel_type
    ](cfg)

    print("done\nCreating hydromodel...", end="", flush=True)
    hmodel = {
        "quasi_ss": QuasiSteadyModel,
        "dynamic": DynamicWaveModel,
    }[cfg.model.hydro_model_type](cfg, chan)

    print("done", flush=True)
    out = Output(cfg, hmodel, chan)

    do_sediment_transport = True
    start = cfg.simulation_time.start
    end = cfg.simulation_time.end
    sim_total_seconds = (end - start).total_seconds()
    step = 0
    t = start
    dt = pd.Timedelta(seconds=chan.get_dt())
    run_start = time.perf_counter()

    while t <= end:
        try:
            hmodel.update_depth(t)
            if do_sediment_transport:
                chan.propogate_sediment(t, hmodel)
        except Exception:
            out.write_step(t)
            sys.stderr.write("Error occured, final state written to output file")
            raise

        if step > 1 and step % 10 == 0:
            elapsed = time.perf_counter() - run_start

            sim_elapsed = (t - start).total_seconds()
            seconds_left = elapsed * (sim_total_seconds / sim_elapsed - 1)

            finish = pd.Timestamp.now() + pd.Timedelta(seconds=seconds_left)

            print(
                f"\rProgress={int(100 * sim_elapsed / sim_total_seconds)}% dt={dt.total_seconds():.2f}s (est. finish at {finish.isoformat(timespec='seconds')})    ",
                end="",
                flush=True,
            )

            # print(
            #     f"\033[7F"
            #     f"\rProgress={int(100 * sim_elapsed / sim_total_seconds)}% dt={dt.total_seconds():.2f}s (est. finish at {finish.isoformat(timespec='seconds')})\n",
            #     f"Cache:\n{cache_info()}",
            #     end="",
            #     flush=True,
            # )

        if step % cfg.output.frequency == 0:
            out.write_step(t)

        step += 1
        dt = pd.Timedelta(seconds=chan.get_dt())

        # if above qthres increase dt and don't do sediment transport
        if hmodel.Q(t, len(hmodel.cs) - 1) < cfg.morphological.qthres:
            dt = pd.Timedelta(
                seconds=cfg.simulation_time.max_dt_qs
                * cfg.morphological.qthres_dtmultiplier
            )
            do_sediment_transport = False
        else:
            do_sediment_transport = True

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
