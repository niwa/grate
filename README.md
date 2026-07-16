# GRATE

This software is based on the old Delphi version described in
Walsh, J. (2014). GRATE v3.50 (Gravel Routing and Textural Evolution) User Manual / Technical Description. NIWA Software Publication.

Currently there is only a command line version of the software.

## Installation

Releases are provided [here](https://github.com/niwa/grate/releases).  Download the
latest, run the installer and ignore any Microsoft Windows defender warnings.  The installer will
place a binary (grate.exe) in your user directory, and update your path.  Open
a terminal and run `grate.exe`

## Development environment

### Linux

1. Install Python 3.13 or newer.

2. `sudo apt install build-essential patchelf ccache`

3. Make a virtual environment and install packages
    ```
    python -m venv /var/tmp/venvs/grate
    . /var/tmp/venvs/grate/bin/activate
    pip install --upgrade pip
    pip install -r requirements.pip
    ```

4. Run using `python bin/grate.py <input file>`

5. Or make a standalone `grate.dist/` directory using `python -m nuitka bin/grate.py`

### Windows

1. Install
    a. Python 3.13 or newer.
    b. Visual studio community 2026 with Desktop development with C++ workload.
    c. Inno Setup 7

2. Make a virtual environment and install packages
    ```
    python -m venv ~/Documents/venvs/grate
    . ~/Documents/venvs/grate/Scripts/activate
    python -m pip install --upgrade pip
    pip install -r requirements.pip
    ```
3. Run using `python bin/grate.py <input file>`

4. Make a standalone `grate.dist/` directory using `python -m nuitka bin/grate.py`


