import shutil
import pathlib
import datetime as dt
import numpy as np
import pandas as pd
import xarray as xr
import tempfile
import h5py
from gin import GrateConfig
from channel import Channel
from hydrodynamics_models import HydroDynamicModel
from grainprofile import get_representative_grain_sizes


def combine_netcdfs(idir: pathlib.Path) -> xr.Dataset:
    """Combine netcdfs (they should have a single time) into one netcdf

    Parameters
    ----------
    idir: pathlib.Path
        Inside this directory should be a bunch of netcdf files, each one with
        a single time value.  When the files are sorted they time should be
        increasing.  Easiest way to do this is name the files 000.nc 001.nc etc

    Returns
    -------
    xr.Dataset
        A dataset containing the netcdf files combined over time dimension
    """

    files = sorted(idir.glob("*.nc"))
    datasets = [xr.open_dataset(f, engine="h5netcdf") for f in files]
    try:
        ds = xr.concat(datasets, dim="time")
        time0 = pd.Timestamp(ds.time.values[0])
        ds.time.encoding.update(
            {
                "units": f"seconds since {time0:%Y-%m-%d}",
                "dtype": "int64",
            }
        )
    finally:
        for d in datasets:
            d.close()
    return ds


class Output:
    def __init__(self, cfg: GrateConfig, hmodel: HydroDynamicModel, chan: Channel):
        self._hmodel = hmodel
        self._cs = chan.cs
        self._cidx = list(range(len(self._cs)))
        self._chan = chan
        self._rgsizes = get_representative_grain_sizes(
            cfg.grain_size_profiles.grain_size_cfds
        )
        self._nlith = cfg.grain_size_profiles.num_lith
        self._outfile = cfg.output.fname

        # use a temp dir if none given
        if cfg.output.idir:
            self.idir = cfg.output.idir
            shutil.rmtree(self.idir, ignore_errors=True)
            self.idir.mkdir(parents=True, exist_ok=True)
        else:
            self._tmpdir = tempfile.TemporaryDirectory()
            self.idir = pathlib.Path(self._tmpdir.name)

        data_funs = {
            "depth": self._get_depth,
            "velocity": self._get_velocity,
            "grain_stress": self._get_grain_stress,
            "total_transport_rate": self._get_total_transport_rate,
            "transport_rate": self._get_transport_rate,
            "mean_bed_level": self._get_mean_bed_level,
            "min_bed_level": self._get_min_bed_level,
        }

        # map user variable name to method for producing that DataArray
        self._v2np = {v: data_funs[v] for v in cfg.output.variables}
        self._v2da = {v: getattr(self, f"get_{v}") for v in cfg.output.variables}

    def _get_depth(self):
        return self._hmodel.d.copy()

    def get_depth(self):
        return xr.DataArray(
            self._get_depth(),
            dims=("chainage",),
            coords={"chainage": self._cs},
            name="depth",
        )

    def _get_velocity(self):
        return np.array([self._hmodel.u(self.time, c) for c in self._cidx])

    def get_velocity(self):
        return xr.DataArray(
            self._get_velocity(),
            dims=("chainage",),
            coords={"chainage": self._cs},
            name="velocity",
        )

    def _get_grain_stress(self):
        return np.array(
            [self._chan.grain_stress(c, self.time, self._hmodel) for c in self._cidx]
        )

    def get_grain_stress(self):
        return xr.DataArray(
            self._get_grain_stress(),
            dims=("chainage",),
            coords={"chainage": self._cs},
            name="grain_stress",
        )

    def _get_total_transport_rate(self):
        return np.array(
            [
                self._chan.get_Qb_jli(c, self.time, self._hmodel).sum()
                for c in self._cidx
            ]
        )

    def get_total_transport_rate(self):
        return xr.DataArray(
            self._get_total_transport_rate(),
            dims=("chainage",),
            coords={"chainage": self._cs},
            name="total_transport_rate",
        )

    def _get_transport_rate(self):
        return np.array(
            [self._chan.get_Qb_jli(c, self.time, self._hmodel) for c in self._cidx]
        )

    def get_transport_rate(self):
        return xr.DataArray(
            self._get_transport_rate(),
            dims=("chainage", "rel_grain_sizes", "lith"),
            coords={
                "chainage": self._cs,
                "rel_grain_sizes": self._rgsizes,
                "lith": list(range(self._nlith)),
            },
            name="transport_rate",
        )

    def _get_mean_bed_level(self):
        return np.array([self._chan.get_mean_bed_level(c) for c in self._cidx])

    def get_mean_bed_level(self):
        return xr.DataArray(
            self._get_mean_bed_level(),
            dims=("chainage",),
            coords={"chainage": self._cs},
            name="mean_bed_level",
        )

    def _get_min_bed_level(self):
        return np.array([self._chan.get_min_bed_level(c) for c in self._cidx])

    def get_min_bed_level(self):
        return xr.DataArray(
            self._get_min_bed_level(),
            dims=("chainage",),
            coords={"chainage": self._cs},
            name="min_bed_level",
        )

    def write_step(self, step: int, t: dt.datetime):
        """Possibly write output for given step"""
        self.time = t
        data_vars = {v: fun().expand_dims(time=[t]) for v, fun in self._v2da.items()}
        outds = xr.Dataset(data_vars=data_vars)
        outfile = self.idir / f"{step:010d}.nc"
        outds.to_netcdf(outfile, mode="w", engine="h5netcdf")

    def write_final(self):
        """Combine steps into one file"""
        ds = combine_netcdfs(self.idir)
        ds.to_netcdf(self._outfile, engine="h5netcdf")


class OutputH5(Output):
    """Output directly to a single NetCDF file using h5py."""

    def __init__(self, cfg: GrateConfig, hmodel: HydroDynamicModel, chan: Channel):
        super().__init__(cfg, hmodel, chan)
        self._initialised = False

    def write_step(self, step: int, t: dt.datetime):
        """Append output for this timestep directly to the NetCDF file."""
        self.time = t

        # Use xarray to make first file
        if not self._initialised:
            ds = xr.Dataset(
                data_vars={
                    v: fun().expand_dims(time=[t]) for v, fun in self._v2da.items()
                }
            )
            ds.to_netcdf(
                self._outfile, mode="w", engine="h5netcdf", unlimited_dims=["time"]
            )
            self._initialised = True
            return

        with h5py.File(self._outfile, "r+") as f:
            n = f["time"].shape[0]
            f["time"].resize((n + 1,))
            f["time"][n] = np.datetime64(t)
            for v, fun in self._v2np.items():
                dset = f[v]
                dset.resize((n + 1,) + dset.shape[1:])
                dset[n, ...] = fun()

    def write_final(self):
        pass
