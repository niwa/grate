# GRATE

This software is based on the old Delphi version described in
Walsh, J. (2014). GRATE v3.50 (Gravel Routing and Textural Evolution) User Manual / Technical Description. NIWA Software Publication.

Currently there is only a command line version of the software.

## Installation

Releases are provided [here](https://github.com/niwa/grate/releases).  Download
the latest `grate-x.y.z.exe`, run it without admin privileges, and ignore any Microsoft Windows defender
warnings.  The installer will place a binary (grate.exe) in your user
directory, and update your path.  Open a terminal and run `grate --help`

## Development environment

1. Install Python 3.13 or newer.

2. Make a virtual environment and install packages
    ```
    python -m venv <SOMEWHERE>
    . <SOMEWHERE>/bin/activate
    pip install --upgrade pip
    pip install -r requirements.pip
    ```

3. Run using `python bin/grate.py <input file>`


## Installer

### Linux

1. `sudo apt install build-essential patchelf ccache`

2. Make a standalone `grate.dist/` directory using `python -m nuitka bin/grate.py`

3. Make sure `grate.dist/` is on your path

### Windows

1. Install

    a. Visual studio community 2026 with Desktop development with C++ workload.
    b. Inno Setup 7

2. Make a standalone `grate.dist/` directory using `python -m nuitka bin/grate.py`

3. Run inno on inno.iss, installer is in `Output/`



