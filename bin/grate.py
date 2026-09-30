#!/usr/bin/env python

# Compilation mode, support OS-specific options
# nuitka-project: --mode=standalone
# nuitka-project: --include-data-dir=etc=etc
# pandas/numpy can pull in matplotlib and scipy, don't want the bloat
# nuitka-project: --nofollow-import-to=matplotlib
# nuitka-project: --nofollow-import-to=scipy
# xarray needs an engine
# nuitka-project: --include-package=h5netcdf
# nuitka-project: --include-package=h5py
# nuitka-project: --nofollow-import-to=h5netcdf.tests,pytest,unittest

import argparse
import yaml
import pathlib
import updates
from convert_gin import parse_gin
from gin import GrateConfig
from simulate import run_model
from utils import positive_float
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

moviemode = sub.add_parser("movie", help="Make movie for xarray NetCDF variable")
moviemode.add_argument("inf", type=pathlib.Path, help="Input NetCDF")
moviemode.add_argument("var", help="Variable to plot, eg depth")
moviemode.add_argument("--xdim", default="chainage", help="x-axis dim, eg chainage")
moviemode.add_argument(
    "--sel",
    action="append",
    default=[],
    metavar="<dim=<idx>",
    help="Select index for a dim, eg --sel rgsize=10",
)
moviemode.add_argument("outf", type=pathlib.Path, help="Output filename, eg out.gif")
moviemode.add_argument("-p", type=positive_float, default=0.1, help="Period (def: 0.1)")
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

    case "movie":
        try:
            sels = {dim: int(idx) for kv in args.sel for dim, idx in [kv.split("=", 1)]}
        except ValueError:
            p.error("Invalid selection; expected DIM=INT")
        make_movie(args.inf, args.var, args.xdim, sels, args.p, args.outf)
        print(f"Movie written to {args.outf}")

    case _:
        p.print_help()
