import argparse
import pathlib
import subprocess
import tempfile
import matplotlib.pyplot as plt
import xarray as xr


def _prepare_movie_data(ds: xr.Dataset, var: str, xdim: str, sels: dict[str, int]):
    """Return data to plot as a list of tuples and x values

    This allows use to plot many variables

    Returns
    -------
    tuple
        A list and x values.
        The list consists of (variable name, dataarray) tuples
        The x values are what to plot on x axis
        Eg
            [("depth", dataarray)], x dataarray

    """

    """special case if var is elev.  plot min_bed_level and that plus depth"""
    if var == "elev":
        vnames = ["min_bed_level", "depth"]
    else:
        vnames = [var]

    if not all(v in ds for v in vnames):
        raise ValueError(f"{vnames} in dataset, we have {list(ds.data_vars)}")

    data = {}

    for v in vnames:
        da = ds[v]

        if "time" not in da.dims:
            raise ValueError(f"Missing time dim, we have {da.dims}")

        if xdim not in da.dims:
            raise ValueError(f"Missing {xdim!r}, we have {da.dims}")

        for dim, index in sels.items():
            if dim not in da.dims:
                raise ValueError(f"Missing {dim!r}, we have {da.dims}")
            da = da.isel({dim: index})

        if set(da.dims) != {"time", xdim}:
            raise ValueError(f"Too many dims: {da.dims}, expect 'time' & {xdim!r}")

        data[v] = da.transpose("time", xdim)

    # get the x values
    # first = data[vnames[0]]
    # if xdim in first.coords:
    x = data[vnames[0]][xdim].values
    # else:
    #     x = np.arange(first.sizes[xdim])

    """special case if var is elev.  plot min_bed_level and that plus depth"""
    if var == "elev":
        plot_data = [
            ("min_bed_level", data["min_bed_level"]),
            ("elevation", data["min_bed_level"] + data["depth"]),
        ]
    else:
        plot_data = [(var, data[var])]

    return plot_data, x


def make_movie(
    infile: pathlib.Path,
    var: str,
    xdim: str,
    sels: dict[str, int],
    dur: float,
    outfile: pathlib.Path,
):

    ds = xr.open_dataset(infile, engine="h5netcdf")

    plot_data, x = _prepare_movie_data(ds, var, xdim, sels)

    ntime = plot_data[0][1].sizes["time"]

    ymin = min(float(da.min()) for _, da in plot_data)
    ymax = max(float(da.max()) for _, da in plot_data)

    if ymin == ymax:
        padding = 1.0
    else:
        padding = (ymax - ymin) * 0.05
    ymin -= padding
    ymax += padding

    outfile.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="grate_movie_") as tmpdir:
        tmpdir = pathlib.Path(tmpdir)

        for i in range(ntime):
            t = plot_data[0][1]["time"].isel(time=i).values

            fig, ax = plt.subplots(figsize=(10, 6))

            for label, da in plot_data:
                ax.plot(x, da.isel(time=i).values, label=label)

            ax.set_xlabel(xdim)
            ax.set_ylabel(var)
            ax.set_ylim(ymin, ymax)
            ax.set_title(f"{var}   {t}")
            ax.grid(True)

            if len(plot_data) > 1:
                ax.legend()

            frame = tmpdir / f"frame_{i:06d}.png"

            fig.tight_layout()
            fig.savefig(frame, dpi=120)
            plt.close(fig)

            print(f"\rFrame {i + 1}/{ntime}", end="", flush=True)

        print()

        cmd = [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            str(1 / dur),
            "-i",
            str(tmpdir / "frame_%06d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(outfile),
        ]

        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)


def main():
    parser = argparse.ArgumentParser(
        description="Make movie for xarray NetCDF variable"
    )
    parser.add_argument("netcdf", type=pathlib.Path, help="Input NetCDF")
    parser.add_argument("var", help="Variable to plot, eg depth ")
    parser.add_argument("--xdim", default="chainage", help="x-axis dim, eg chainage")
    parser.add_argument(
        "--sel",
        action="append",
        default=[],
        metavar="DIM=INDEX",
        help="Select an index for a dimension, eg --sel rgsize=10",
    )
    parser.add_argument("movie", type=pathlib.Path, help="Output filename, eg out.mp4")
    parser.add_argument(
        "--dur",
        type=float,
        default=0.5,
        help="Seconds to display each frame (default: 0.5)",
    )

    args = parser.parse_args()

    sels = {}
    for kv in args.sel:
        try:
            dim, index = kv.split("=", 1)
            sels[dim] = int(index)
        except ValueError:
            parser.error(f"Invalid selection {kv}; expected DIM=INT")

    if args.dur <= 0:
        parser.error("--dur must be greater than zero")

    make_movie(
        args.netcdf,
        args.var,
        args.xdim,
        sels,
        args.dur,
        args.movie,
    )
    print(f"Movie written to {args.movie}")


if __name__ == "__main__":
    main()
