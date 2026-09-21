#!/usr/bin/env python

# Compilation mode, support OS-specific options
# nuitka-project: --mode=standalone
# nuitka-project: --include-data-dir=etc=etc

import argparse
import yaml
import pathlib
import updates
from convert_gin import parse_gin
from gin import GrateConfig
from simulate import run_model
from output import combine_netcdfs
from movie import make_movie

# parse command line
p = argparse.ArgumentParser(
    description="""
Convert gin grate models to yaml and run them
""",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
sub = p.add_subparsers(dest="command")
sub.add_parser("versions", help="Display versions")
sub.add_parser("update", help="Update to latest version")
convert = sub.add_parser(
    "convert",
    help="Convert old gin file to yaml",
    description="""
Convert gin input file into yaml.

Files references by gin file are also processed, in
particular the xsectfile .dat file
""",
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
convert.add_argument("gin", type=pathlib.Path, help="Input gin file")
convert.add_argument("yaml", type=pathlib.Path, help="Output yaml file")

validate = sub.add_parser(
    "validate",
    help="Validate yaml model",
    description="""
Check format of yaml model.

Files referenced by gin file are not currently checked
""",
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
validate.add_argument("yaml", type=pathlib.Path, help="Input yaml file")

runmode = sub.add_parser("run", help="Run yaml model")
runmode.add_argument("yaml", type=pathlib.Path, help="Input yaml file")

combinemode = sub.add_parser("combine", help="Combine single timestep netcdfs")
combinemode.add_argument("idir", type=pathlib.Path, help="Input directory")
combinemode.add_argument("fname", type=pathlib.Path, help="Output file.nc")

moviemode = sub.add_parser("movie", help="Make movie from netcdf of a variable")
moviemode.add_argument("infile", type=pathlib.Path, help="Input file.nc")
moviemode.add_argument("var", help="Variable to plot, eg depth ")
moviemode.add_argument("--xdim", default="chainage", help="x-axis dim, eg chainage")
moviemode.add_argument(
    "--sel",
    action="append",
    default=[],
    metavar="DIM=INDEX",
    help="Select an index for a dimension, eg --selection rgsize=10",
)
moviemode.add_argument(
    "--dur",
    type=float,
    default=0.5,
    help="Seconds to display each frame (default: 0.5)",
)
moviemode.add_argument("outfile", type=pathlib.Path, help="Output filename, eg out.mp4")

args = p.parse_args()

match args.command:
    case "versions":
        cver = updates.get_prog_version() or "unknown"
        ghver = updates.get_github_version() or "unknown"
        ivers = updates.get_installable_versions() or []
        print(f"Your version = {cver}\nGit hub version = {ghver}")
        print(f"Installable versions = {','.join(v['version'] for v in ivers)}")

    case "update":
        updates.possibly_update()

    case "convert":
        print(f"Parsing {args.gin}")
        conf = parse_gin(args.gin)
        with open(args.yaml, "w") as fh:
            yaml.dump(conf, fh, default_flow_style=False, sort_keys=False)
        print(f"Written to {args.yaml}")

    case "validate":
        with open(args.yaml) as f:
            cfg = GrateConfig.model_validate(yaml.safe_load(f))

    case "run":
        updates.version_check()
        print(f"Running {args.yaml}")
        run_model(args.yaml)

    case "combine":
        updates.version_check()
        ds = combine_netcdfs(args.idir)
        print(f"Writing {args.fname}")
        ds.to_netcdf(args.fname)

    case "movie":
        updates.version_check()
        sels = {}
        for kv in args.sel:
            try:
                dim, index = kv.split("=", 1)
                sels[dim] = int(index)
            except ValueError:
                moviemode.error(f"Invalid selection {kv}; expected DIM=INT")
        make_movie(args.infile, args.var, args.xdim, sels, args.dur, args.outfile)
        print(f"Movie written to {args.outfile}")

    case _:
        p.print_help()
