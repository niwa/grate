import shutil
import pathlib
import datetime as dt
import numpy as np
import pandas as pd
import xarray as xr
import tempfile
from gin import GrateConfig
from channel import Channel
from hydrodynamics_models import HydroDynamicModel
from grainprofile import get_representative_grain_sizes


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

        # map user variable name to method for producing that DataArray
        v2fun = {
            "depth": self.get_depth,
            "velocity": self.get_velocity,
            "grain_stress": self.get_grain_stress,
            "total_transport_rate": self.get_total_transport_rate,
            "transport_rate": self.get_transport_rate,
            "mean_bed_level": self.get_mean_bed_level,
            "min_bed_level": self.get_min_bed_level,
        }
        self._v2fun = {v: v2fun[v] for v in cfg.output.variables}

    def get_depth(self):
        return xr.DataArray(
            self._hmodel.d.copy(),
            dims=("chainage",),
            coords={"chainage": self._cs},
            name="depth",
        )

    def get_velocity(self):
        return xr.DataArray(
            np.array([self._hmodel.u(self.time, c) for c in self._cidx]),
            dims=("chainage",),
            coords={"chainage": self._cs},
            name="velocity",
        )

    def get_grain_stress(self):
        return xr.DataArray(
            np.array(
                [
                    self._chan.grain_stress(c, self.time, self._hmodel)
                    for c in self._cidx
                ]
            ),
            dims=("chainage",),
            coords={"chainage": self._cs},
            name="velocity",
        )

    def get_total_transport_rate(self):
        return xr.DataArray(
            np.array(
                [
                    self._chan.get_Qb_jli(c, self.time, self._hmodel).sum()
                    for c in self._cidx
                ]
            ),
            dims=("chainage",),
            coords={"chainage": self._cs},
            name="total_transport_rate",
        )

    def get_transport_rate(self):
        return xr.DataArray(
            np.array(
                [self._chan.get_Qb_jli(c, self.time, self._hmodel) for c in self._cidx]
            ),
            dims=("chainage", "rel_grain_sizes", "lith"),
            coords={
                "chainage": self._cs,
                "rel_grain_sizes": self._rgsizes,
                "lith": list(range(self._nlith)),
            },
            name="transport_rate",
        )

    def get_mean_bed_level(self):
        return xr.DataArray(
            np.array([self._chan.get_mean_bed_level(c) for c in self._cidx]),
            dims=("chainage",),
            coords={"chainage": self._cs},
            name="mean_bed_level",
        )

    def get_min_bed_level(self):
        return xr.DataArray(
            np.array([self._chan.get_min_bed_level(c) for c in self._cidx]),
            dims=("chainage",),
            coords={"chainage": self._cs},
            name="min_bed_level",
        )

    def write_step(self, step: int, t: dt.datetime):
        """Possibly write output for given step"""
        self.time = t
        data_vars = {v: fun().expand_dims(time=[t]) for v, fun in self._v2fun.items()}
        outds = xr.Dataset(data_vars=data_vars)
        outfile = self.idir / f"{step:010d}.nc"
        outds.to_netcdf(outfile, mode="w")

    def write_final(self):
        """Combine steps into one file"""
        files = sorted(self.idir.glob("*.nc"))
        datasets = [xr.open_dataset(f) for f in files]
        try:
            ds = xr.concat(datasets, dim="time")
            time0 = pd.Timestamp(ds.time.values[0])
            ds.time.encoding.update(
                {
                    "units": f"seconds since {time0:%Y-%m-%d}",
                    "dtype": "int64",
                }
            )
            ds.to_netcdf(self._outfile)
        finally:
            for d in datasets:
                d.close()
