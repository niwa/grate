#!/usr/bin/env python3

import argparse
import pathlib
import subprocess
import tempfile

import matplotlib.pyplot as plt
import xarray as xr


def make_movie(
    netcdf_file: pathlib.Path,
    variable: str,
    movie_file: pathlib.Path,
    duration: float,
):
    ds = xr.open_dataset(netcdf_file)

    if variable not in ds:
        raise ValueError(
            f"Variable {variable!r} not found in {netcdf_file}. "
            f"Available variables: {list(ds.data_vars)}"
        )

    data = ds[variable]

    if "time" not in data.dims:
        raise ValueError(
            f"Variable {variable!r} does not have a 'time' dimension. "
            f"Dimensions are: {data.dims}"
        )

    # Everything other than time is plotted against cidx.
    other_dims = [dim for dim in data.dims if dim != "time"]

    if len(other_dims) != 1:
        raise ValueError(
            f"Expected {variable!r} to have dimensions "
            f"(time, cidx), but got {data.dims}"
        )

    cidx_dim = other_dims[0]

    # Use the actual cidx coordinate if it exists, otherwise use
    # integer indices.
    if cidx_dim in data.coords:
        cidx = data[cidx_dim].values
    else:
        cidx = range(data.sizes[cidx_dim])

    movie_file.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="grate_movie_") as tmpdir:
        tmpdir = pathlib.Path(tmpdir)

        print(f"Writing frames to {tmpdir}")

        # Keep the y-axis fixed across all frames so that changes in
        # the water height are visually meaningful.
        ymin = float(data.min())
        ymax = float(data.max())

        # Give the plot a little padding.
        if ymin == ymax:
            padding = 1.0
        else:
            padding = (ymax - ymin) * 0.05

        ymin -= padding
        ymax += padding

        for i in range(data.sizes["time"]):
            t = data["time"].isel(time=i).values
            values = data.isel(time=i).values

            fig, ax = plt.subplots(figsize=(10, 6))

            ax.plot(cidx, values)

            ax.set_xlabel("cidx")
            ax.set_ylabel(variable)
            ax.set_ylim(ymin, ymax)
            ax.set_title(f"{variable}   {t}")

            ax.grid(True)

            frame = tmpdir / f"frame_{i:06d}.png"

            fig.tight_layout()
            fig.savefig(frame, dpi=120)
            plt.close(fig)

            print(f"\rFrame {i + 1}/{data.sizes['time']}", end="", flush=True)

        print()

        # duration = seconds/frame
        fps = 1.0 / duration

        cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(tmpdir / "frame_%06d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(movie_file),
        ]

        print("Running ffmpeg...")
        subprocess.run(cmd, check=True)

    ds.close()

    print(f"Movie written to {movie_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Make a movie from an xarray NetCDF variable."
    )

    parser.add_argument(
        "netcdf",
        type=pathlib.Path,
        help="Input NetCDF file",
    )

    parser.add_argument(
        "variable",
        help="Variable to plot, e.g. height or flow",
    )

    parser.add_argument(
        "movie",
        type=pathlib.Path,
        help="Output movie filename, e.g. height.mp4",
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=0.2,
        help="Seconds to display each frame (default: 0.2)",
    )

    args = parser.parse_args()

    if args.duration <= 0:
        parser.error("--duration must be greater than zero")

    make_movie(
        args.netcdf,
        args.variable,
        args.movie,
        args.duration,
    )


if __name__ == "__main__":
    main()
