#!/usr/bin/env python

# Compilation mode, support OS-specific options
# nuitka-project: --mode=standalone
# nuitka-project: --include-data-dir=etc=etc

import sys
import argparse
import pathlib
import updates
from packaging import version 

# parse command line
p = argparse.ArgumentParser(
    description="""
Run grate FIXME
""",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
p.add_argument("--version", action="store_true", help="Display versions")
p.add_argument("--update", action="store_true", help="Update to latest")
p.add_argument(
    "gin",
    type=pathlib.Path,
    nargs="?",
    help="Grate input yaml file",
)

args = p.parse_args()

if args.version:
    cver = updates.get_prog_version() or "unknown"
    ghver = updates.get_github_version() or "unknown"
    ivers = updates.get_installable_versions() or []
    print(f"Your version = {cver}\nGit hub version = {ghver}")
    print(f"Installable versions = {','.join(v['version'] for v in ivers)}")
    sys.exit(0)

if args.update:
    updates.possibly_update()
    sys.exit(0)

if args.gin is None:
    p.error("Specify <gin> yaml file")

# tell the user is they are out of date
cver = updates.get_prog_version()
ivers = updates.get_installable_versions()
if cver and ivers and version.parse(cver) < version.parse(ivers[0]["version"]):
    print(f"WARNING: ver {cver} is old, run with --update to get {ivers[0]['version']}")


print(f"Should do something with {args.gin}")
