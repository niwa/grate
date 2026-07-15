#!/usr/bin/env python

# Compilation mode, support OS-specific options
# nuitka-project: --mode=standalone
# nuitka-project: --include-data-dir=etc=etc

import sys
import argparse
import pathlib
import utils

# parse command line
p = argparse.ArgumentParser(
    description="""
Run grate
FIXME
""",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
p.add_argument(
    "gin",
    type=pathlib.Path,
    help="Grate input yaml file",
)
args = p.parse_args()

print(f"Got {args.gin} input file")

utils.testit()

with open(utils.resolved_path("etc/datafile.txt")) as fh:
    print(fh.read())
