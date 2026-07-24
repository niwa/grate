import sys
import argparse
import pathlib
import yaml
import re
import lark
import pandas as pd
import datetime as dt
import utils

# a ! === section that starts with this, gets mapped to value
SECTION_MAPPINGS = {
    "GRATE": "header",
    "Input File Version": "header",
    "Run Identifier": "header",
    "Model Type": "model",
    "Simulation Time": "simulation_time",
    "HD Parameters": "hd_params",
    "Morphological Parameters": "morphological",
    "Discretisation Parameters": "discretisation",
    "Bed Layer Setup": "bed_layer",
    "Active Layer Setup": "active_layer",
    "Active Width Option": "morphological",
    "CROSS - SECTIONS": "cross_sections",
    "Inflow Boundary Conditions": "inflow_boundary",
    "Sediment Inflow Boundary Conditions": "sediment_boundary",
    "Grain-Size Profiles": "grain_size_profiles",
    "Print Options": "print",
    "Display Options": "display",
    "Downstream Water Level Boundary": "downstream_boundary",
    "Hydraulic calibration data": "hydraulic_calibration",
    "Sediment Extraction": "sediment_extraction",
}

# these sections are surrounded by ! ---
OPTIONAL_SECTION_MAPPINGS = {
    "Sediment Ripping Event": "sediment_ripping",
    "Bed Ripping Event": "sediment_ripping",
}

GIN_VALUE_MAPPINGS = {
    ("model", "type"): {
        "1": "flume",
        "2": "river",
        "4": "braided_channel",
    }
}
SECTIONS_TO_IGNORE = ["display", "hydraulic_calibration", "bed_layer", "active_layer"]
KEYS_TO_IGNORE = ["VERSID"]


def ykey(key: str) -> str:
    """Map gin key names to yaml key names"""
    return {
        "MODELTYPE": "type",
        "TS": "start",
        "TE": "end",
        "NO_CYCLES": "num_cycles",
    }.get(key, key.lower())


def yval(section: str, key: str, val: str):
    """Map gin value names to yaml value names"""
    if section == "simulation_time" and key in ("start", "end"):
        return dt.datetime.strptime(val, "%Y%m%d %H%M%S")
    if val in GIN_VALUE_MAPPINGS.setdefault((section, key), {}):
        val = GIN_VALUE_MAPPINGS[(section, key)][val]
    return utils.try_to_num(val)


def parse_keyval_line(line):
    """Return key, val from a key = val with optional comments"""
    line = line.split("!", 1)[0].strip()
    key, value = None, None
    if line and line.count("=") == 1:
        key, value = (s.strip() for s in line.split("=", 1))

    if key in KEYS_TO_IGNORE:
        return None, None

    return (key, value)


def parse_section_line(section, line, kv):
    """Add stuff to kv depending on what section we are in"""
    if section in SECTIONS_TO_IGNORE:
        return
    match section:
        case "inflow_boundary":
            kv.setdefault("inflow_boundary", [])
            parse_inflow_boundary_line(line, kv["inflow_boundary"])
        case "downstream_boundary":
            parse_downstream_boundary_line(line, kv)
        case "sediment_boundary":
            kv.setdefault("sediment_boundary", [])
            parse_sediment_boundary_line(line, kv["sediment_boundary"])
        case "sediment_extraction":
            kv.setdefault("sediment_extraction", [])
            parse_sediment_extraction_line(line, kv["sediment_extraction"])
        case "sediment_ripping":
            kv.setdefault("sediment_ripping", [])
            parse_sediment_ripping_line(line, kv["sediment_ripping"])
        case "grain_size_profiles":
            kv.setdefault("grain_size_profiles", "")
            if not re.match(r"^\s*!", line):
                kv["grain_size_profiles"] += line  # parse this later
        case _:
            key, value = parse_keyval_line(line)
            if key is None and value is None:
                return

            key = ykey(key)
            kv.setdefault(section, {})[key] = yval(section, key, value)


def parse_inflow_boundary_line(line, kv):
    ma = re.match(r"(\d+)\s+(C|TS)\s+(.*)", line, re.I)
    if not ma:
        return
    if ma.group(2).lower() == "c":
        kv.append(
            {
                "ordinate": int(ma.group(1)),
                "type": "const",
                "value": float(ma.group(3)),
            }
        )
    else:
        kv.append(
            {
                "ordinate": int(ma.group(1)),
                "type": "ts",
                "value": ma.group(3),
            }
        )


def parse_downstream_boundary_line(line, kv):
    ma = re.match(r"\s*(C|D|TS)\s+(\S+)|^\s*(N)\s+(\S+)\s+(\S+)", line, re.I)
    if not ma:
        return
    if ma.group(1) and ma.group(1).lower() == "c":
        kv["downstream_boundary"] = {"type": "elevation", "value": float(ma.group(2))}
    elif ma.group(1) and ma.group(1).lower() == "d":
        kv["downstream_boundary"] = {"type": "depth", "value": float(ma.group(2))}
    elif ma.group(1) and ma.group(1).lower() == "ts":
        kv["downstream_boundary"] = {
            "type": "elevation_timeseries",
            "value": ma.group(2),
        }
    else:
        kv["downstream_boundary"] = {
            "type": "normal",
            "slope": float(ma.group(4)),
            "hinit": float(ma.group(5)),
        }


def parse_sediment_boundary_line(line, kv):
    ma = re.match(r"(\d+)\s+RC", line, re.I)
    if ma:
        kv.append({"ordinate": int(ma.group(1)), "type": "rc"})
        return
    ma = re.match(r"(\d+)\s+C\s+(\d+)\s+(\S+)", line, re.I)
    if ma:
        kv.append(
            {
                "ordinate": int(ma.group(1)),
                "type": "const",
                "group": int(ma.group(2)),
                "value": float(ma.group(3)),
            }
        )
        return
    ma = re.match(r"(\d+)\s+TS\s+(\d+)\s+(\S+)\s+(\S+)", line, re.I)
    if ma:
        kv.append(
            {
                "ordinate": int(ma.group(1)),
                "type": "ts",
                "group": int(ma.group(2)),
                "scale": float(ma.group(3)),
                "fname": ma.group(4),
            }
        )


def parse_sediment_extraction_line(line, kv):
    ma = re.match(r"\s*(\d+\.*\d+)\s+C\s+(\S+)\s+(\S+)", line, re.I)
    if ma:
        kv.append(
            {
                "ordinate": float(ma.group(1)),
                "type": "const",
                "rate": float(ma.group(2)),
                "proportion": float(ma.group(3)),
            }
        )
        return
    ma = re.match(r"\s*(\d+\.*\d+)\s+TS\s+(\S+)", line, re.I)
    if ma:
        kv.append(
            {
                "ordinate": float(ma.group(1)),
                "type": "ts",
                "fname": str(pathlib.Path(ma.group(2))),
            }
        )


def parse_sediment_ripping_line(line, kv):
    ma = re.match(r"\s*(\d+\.*\d+)\s+(\S+)", line, re.I)
    if ma:
        kv.append(
            {"ordinate": float(ma.group(1)), "fname": str(pathlib.Path(ma.group(2)))}
        )


class XsectTransformer(lark.Transformer):
    def start(self, items):
        return {
            "header": items[0],
            "cross_sections": items[1:],
        }

    def header(self, items):
        r = {
            "nsect": int(items[0]),
            "formrf": float(items[2]),
        }
        if len(items) >= 5:
            r.update({"wallrf": float(items[4])})
        return r

    def cross_section(self, items):
        result = {}
        for item in items:
            result.update(item)
        return result

    def topoid(self, items):
        return {"topoid": str(items[0])}

    def river_name(self, items):
        return {"river_name": str(items[0])}

    def chainage(self, items):
        return {"chainage": float(items[0])}

    def property(self, items):
        return items[0]

    def formrf(self, items):
        return {"formrf": float(items[1])}

    def bankd90(self, items):
        return {"bankd90": float(items[1])}

    def storage(self, items):
        return {
            "active_layer_group": int(items[1]),
            "storage_layer_group": int(items[2]),
        }

    def bedrock(self, items):
        return {"bedrock_rl": float(items[1])}

    def lsf(self, items):
        return {"lsf": float(items[1])}

    def qsfact(self, items):
        return {"qsfact": float(items[1])}

    def profile(self, items):
        return {"profile": [t for t in items if isinstance(t, tuple)]}

    def profilerow(self, items):
        return tuple(map(float, items[:-1]))


def parse_allmodels_xsectfile(ifile, ofile: str, kv):
    """Parse flume (type 1), river (type 2) and braided channel (type 4) ifile updating kv and writing a bunch of
    csv files with prefix ofile with new profile info"""

    par = lark.Lark.open(
        utils.resolved_path("etc/gin_model_grammars/braided_channel.ebnf"),
        parser="lalr",
    )
    tree = par.parse(
        "".join(
            n
            for n in open(ifile, encoding="utf-8-sig")
            if n.strip() and not n.strip().startswith("!")
        )
    )

    cfg = XsectTransformer().transform(tree)

    # check number of sections match
    assert cfg["header"]["nsect"] == len(cfg["cross_sections"]), (
        f"{ifile} has {len(cfg['cross_sections'])} PROFILES which doesn't match NSECT = {cfg['header']['nsect']}"
    )

    kv["cross_sections"] = {k: v for k, v in cfg["header"].items() if k != "nsect"}
    kv["cross_sections"]["profiles"] = []

    for xs in cfg["cross_sections"]:
        pfile = f"{ofile}_{xs['chainage']}.csv"
        p = {
            "chainage": xs["chainage"],
            "topoid": xs["topoid"],
            "river_name": xs["river_name"],
        }
        for k in (
            "formrf",
            "bankd90",
            "active_layer_group",
            "storage_layer_group",
            "bedrock_rl",
            "lsf",
            "qsfact",
        ):
            if xs.get(k, None):
                p[k] = xs[k]

        p["profile"] = pfile
        kv["cross_sections"]["profiles"].append(p)

        rows = []
        for pt in xs["profile"]:
            row = {"x": pt[0], "y": pt[1]}
            if len(pt) > 2:
                row["relrf"] = pt[2]
            if len(pt) > 3:
                row["ob"] = pt[3]
            rows.append(row)
        df = pd.DataFrame(rows)
        df.to_csv(pfile, index=False)


class GrainSizeProfilesTransformer(lark.Transformer):
    def start(self, items):
        return {
            "ngrp": items[0],
            "ngsz": items[1],
            "nlith": items[2],
            "abrasion": items[3],
            "sediment": items[4],
            "datarows": items[5],
        }

    def ngrain(self, items):
        return int(items[0])

    def nclasses(self, items):
        return int(items[0])

    def nlith(self, items):
        return int(items[0])

    def abrasion(self, items):
        return [float(x) for x in items[:-1]]

    def sediment(self, items):
        return [float(x) for x in items[:-1]]

    def datarows(self, items):
        return [[float(x) for x in row[:-1]] for row in items]

    def nums(self, items):
        return list(items)


def parse_grain_size_profiles(kv):
    """Parse grain size profiles section"""

    par = lark.Lark.open(utils.resolved_path("etc/grain_size_profiles.ebnf"))
    tree = par.parse(kv["grain_size_profiles"])
    cfg = GrainSizeProfilesTransformer().transform(tree)

    ngrp = cfg["ngrp"]
    ngsz = cfg["ngsz"]
    nlith = cfg["nlith"]

    # there should be ngsz+1 grain size rows, and nzgs*nlith lith rows
    assert ngsz * (nlith + 1) + 1 == len(cfg["datarows"]), (
        f"NGSZ={ngsz}, NLITH={nlith} but not {ngsz + 1} + {ngsz * nlith} grain size distribution and lith rows"
    )

    grows = cfg["datarows"][: ngsz + 1]
    lrows = cfg["datarows"][ngsz + 1 :]

    assert all([len(g) == ngrp + 1 for g in grows]), (
        "Grain distribution rows didn't have {ngrp+1} elements"
    )
    assert all([len(g) == ngrp + 1 for g in lrows]), (
        "Lithology rows didn't have {ngrp+1} elements"
    )

    kv["grain_size_profiles"] = {}
    kv["grain_size_profiles"]["num_profiles"] = ngrp
    kv["grain_size_profiles"]["num_bins"] = ngsz
    kv["grain_size_profiles"]["num_lith"] = nlith
    kv["grain_size_profiles"]["grain_size_cfds"] = grows
    if nlith > 1:
        kv["grain_size_profiles"]["lithfractions"] = lrows


def adhoc_fixes(lines: list[str]) -> list[str]:
    """Any weird broken things that can't be generalized"""
    newlines = []
    for e in lines:
        # comment character mistakenly typed as ' in a lot of examples
        if ma := re.search(r"NCALPTS\s*=\s*(\S+)\s*'\s*Number of points in eac", e):
            e = f"NCALPTS = {ma.group(1)}"
        newlines.append(e)

    return newlines


def replace_section_headers(lines: list[str], mappings: dict, sep: str) -> list[str]:
    """Replace gin section headers with standardised names.

    Parameters
    ----------
    lines: list
        The lines from .gin file

    mappings: dict
        Gin section names to our header names

    sep: =|-
        What surrounds section headers, either ! ====== or ! -----
    """

    result = []
    i = 0
    while i < len(lines):
        line = lines[i]

        # Look for separator line, ! ======== or ! --------
        s = rf"!\s*{re.escape(sep)}{{6,}}\s*\n"
        if (
            re.fullmatch(s, line)
            and i + 2 < len(lines)
            and lines[i + 1].startswith("!")
        ):
            title = lines[i + 1][1:].strip()

            for old, new in mappings.items():
                if title.startswith(old):
                    # skip serparator, title and following comments
                    i += 2
                    while i < len(lines) and lines[i].startswith("!"):
                        if re.fullmatch(s, lines[i]):
                            i += 1
                            break
                        i += 1
                    result.extend([f"! {'=' * 70}\n", f"! {new}\n", f"! {'=' * 70}\n"])
                    break
            else:
                result.append(line)
                i += 1
            continue

        result.append(line)
        i += 1

    return result


def parse_gin(fname: pathlib.Path) -> dict:
    """Return new config given old gin file.

    Parameters
    ----------
    fname: pathlib.Path
        Input gin file

    """
    state = None
    section = None

    # the optional section headers are surrounded by ! ---, so easier to clean
    # this up first
    with open(fname, "r", encoding="utf-8-sig") as fh:
        lines = fh.readlines()

    lines = adhoc_fixes(lines)
    lines = replace_section_headers(lines, SECTION_MAPPINGS, sep="=")
    lines = replace_section_headers(lines, OPTIONAL_SECTION_MAPPINGS, sep="-")

    # for el in lines:
    #     print(el)

    kv = {}
    for line in lines:
        match state:
            case None if line.startswith("! ======"):
                state = "expect section name"

            case "expect section name" if line.startswith("! "):
                section = line[2:].strip()
                state = "expect header end"

            case "expect header end" if line.startswith("! ========="):
                state = "reading section"

            case "reading section":
                if line.startswith("! ======"):
                    state = "expect section name"
                    continue
                parse_section_line(section, line, kv)

    # parse grain sizes
    assert len(kv.setdefault("grain_size_profiles", "")) > 0, (
        f"No grain size profiles defined in {fname}"
    )
    parse_grain_size_profiles(kv)

    # parse xsect if known model type
    assert "cross_sections" in kv and "xsectfile" in kv["cross_sections"]
    ifile = fname.parent / pathlib.Path(kv["cross_sections"]["xsectfile"])
    ofile = str(ifile.with_suffix(""))
    print(f"Converting {ifile} to {ofile}")
    match kv["model"]["type"]:
        case "flume" | "river" | "braided_channel":
            parse_allmodels_xsectfile(ifile, ofile, kv)
        case _:
            sys.stderr.write(
                "Can only handle flume, river and braided_channel model, left cross sections file alone\n"
            )

    return kv


def main():
    # parse command line
    p = argparse.ArgumentParser(
        description="""Convert an old gin file to new yaml format""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("gin", type=pathlib.Path, help="Grate input gin file")
    p.add_argument("gout", type=pathlib.Path, help="Grate yaml output")
    args = p.parse_args()

    print(f"Parsing {args.gin}")
    conf = parse_gin(args.gin)
    with open(args.gout, "w") as fh:
        yaml.dump(conf, fh, default_flow_style=False, sort_keys=False)
    print(f"Written to {args.gout}")


if __name__ == "__main__":
    main()
