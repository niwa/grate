import os
import pathlib
import requests
import platformdirs
from packaging import version

import utils


def parse_version_txt(contents):
    """Return None or version string

    Parameters
    ----------
    contents: str
        The contents of version.txt file

    Returns
    -------
    None or str
        On any error return None, else return a string version like 1.2.3
    """
    try:
        return [
            ln.split("Version: ", 1)[1].strip()
            for ln in contents.splitlines()
            if ln.startswith("Version: ")
        ][0]
    except Exception:
        return None


def get_prog_version():
    """Return version of this running program or None"""

    try:
        with open(utils.resolved_path("etc/version.txt")) as fh:
            return parse_version_txt(fh.read())
    except Exception:
        return None


def get_github_version():
    url = "https://raw.githubusercontent.com/niwa/grate/main/etc/version.txt"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return parse_version_txt(r.text)
    except Exception:
        return None


def get_installable_versions():
    """Return a sorted list of [{'version': version, 'name': name, 'url': url}]"""

    url = "https://api.github.com/repos/niwa/grate/releases"
    headers = {"Accept": "application/vnd.github+json"}

    try:
        r = requests.get(url, headers=headers)
        r.raise_for_status()
        releases = []
        for release in r.json():
            releases.append(
                {
                    "version": release["tag_name"],
                    "name": release["assets"][0]["name"],
                    "url": release["assets"][0]["browser_download_url"],
                }
            )
    except Exception as exp:
        print(exp)
        return []

    return sorted(releases, key=lambda i: version.Version(i["version"]), reverse=True)


def version_check():
    cver = get_prog_version()
    ivers = get_installable_versions()
    if cver and ivers and version.parse(cver) < version.parse(ivers[0]["version"]):
        print(f"WARNING: ver {cver} is old, run update to get {ivers[0]['version']}")


def possibly_update():
    cver = get_prog_version()
    ivers = get_installable_versions()
    if not (cver and ivers):
        print("Cannot determine installable versions")
        return

    if version.parse(cver) >= version.parse(ivers[0]["version"]):
        print(f"{cver} is already latest")
        return

    # update to ivers[0]
    r = input(f"Install version {ivers[0]['version']}? [y/N]: ").strip().lower()

    if r != "y":
        return

    # Start download
    try:
        installer = download_version(ivers[0])
    except Exception as e:
        print(f"Download failed: {e}")
        return

    os.execl(installer, installer)


def download_version(vnu: dict):
    """Download installer and return its path.

    Parameters
    ----------
    vnu: dict
        {'version', 'name', 'url'}

    Returns
    -------
    pathlib.Path
        Path to downloaded installer
    """

    ddir = pathlib.Path(platformdirs.user_downloads_dir())
    installer = ddir / vnu["name"]

    with requests.get(vnu["url"], stream=True) as r:
        r.raise_for_status()

        with open(installer, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    return installer
