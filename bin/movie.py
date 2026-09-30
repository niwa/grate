import argparse
import pathlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import xarray as xr
from utils import positive_float


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

    ymin = min(float(da.min()) for _, da in plot_data)
    ymax = max(float(da.max()) for _, da in plot_data)
    padding = (ymax - ymin) * 0.05 if ymin != ymax else 1
    ymin -= padding
    ymax += padding

    fig, ax = plt.subplots(figsize=(10, 6))

    # make the empty lines, one per var in plot_data
    lines = [ax.plot([], [], label=label)[0] for label, _ in plot_data]

    ax.set_xlabel(xdim)
    ax.set_ylabel(var)
    ax.set_xlim(float(x.min()), float(x.max()))
    ax.set_ylim(ymin, ymax)
    ax.grid(True)

    if len(plot_data) > 1:
        ax.legend()

    def update(i):
        t = plot_data[0][1]["time"].isel(time=i).values

        # put each var in for ith frame
        for line, (_, da) in zip(lines, plot_data):
            line.set_data(x, da.isel(time=i).values)

        ax.set_title(f"{var}   {t}")
        print(f"\rFrame {i + 1}/{steps}", end="", flush=True)
        return lines

    steps = plot_data[0][1].sizes["time"]
    ani = FuncAnimation(fig, update, frames=steps, interval=dur * 1000, blit=True)

    outfile.parent.mkdir(parents=True, exist_ok=True)
    ani.save(outfile, writer=PillowWriter(fps=1 / dur))

    print()
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Make movie for xarray NetCDF variable")
    p.add_argument("inf", type=pathlib.Path, help="Input NetCDF")
    p.add_argument("var", help="Variable to plot, eg depth ")
    p.add_argument("--xdim", default="chainage", help="x-axis dim, eg chainage")
    p.add_argument(
        "--sel",
        action="append",
        default=[],
        metavar="<dim>=<idx>",
        help="Select an index for a dimension, eg --sel rgsize=10",
    )
    p.add_argument("outf", type=pathlib.Path, help="Output filename, eg out.mp4")
    p.add_argument("-p", type=positive_float, default=0.1, help="Period (def: 0.1)")

    args = p.parse_args()

    try:
        sels = {dim: int(idx) for kv in args.sel for dim, idx in [kv.split("=", 1)]}
    except ValueError:
        p.error("Invalid selection; expected DIM=INT")

    make_movie(args.inf, args.var, args.xdim, sels, args.p, args.outf)
    print(f"Movie written to {args.outf}")


if __name__ == "__main__":
    main()
