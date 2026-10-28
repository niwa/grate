import datetime as dt
import numpy as np
import xarray as xr
import h5py
from gin import GrateConfig
from channel import Channel
from hydrodynamics_models import HydroDynamicModel
from grainprofile import get_representative_grain_sizes

WATER_DENSITY = 1000


class Output:
    """Output directly to a single NetCDF file using h5py."""

    def __init__(self, cfg: GrateConfig, hmodel: HydroDynamicModel, chan: Channel):
        self._hmodel = hmodel
        self._cs = chan._cs
        self._cidx = list(range(len(self._cs)))
        self._chan = chan
        self._rgsizes = get_representative_grain_sizes(
            cfg.grain_size_profiles.grain_size_cfds
        )
        self._nlith = cfg.grain_size_profiles.num_lith
        self._nlayers = 1 + cfg.morphological.num_slayers
        self._outfile = cfg.output.fname

        data_funs = {
            "water_level": self._get_water_level,
            "flow": self._get_flow,
            "dt": self._get_dt,
            "timestep": self._get_timestep,
            "cum_bed_level_change": self._get_cum_bed_level_change,
            "cum_transport": self._get_cum_transport,
            "cum_total_transport": self._get_cum_total_transport,
            "cum_flow_vol": self._get_cum_flow_vol,
            "velocity": self._get_velocity,
            "grain_stress": self._get_grain_stress,
            "total_transport_rate": self._get_total_transport_rate,
            "transport_rate": self._get_transport_rate,
            "mean_bed_level": self._get_mean_bed_level,
            "min_bed_level": self._get_min_bed_level,
            "layer_thickness": self._get_layer_thickness,
            "layer_composition": self._get_layer_composition,
            "bedrock": self._get_bedrock,
        }

        # map user variable name to method for producing that DataArray
        self._v2np = {v: data_funs[v] for v in cfg.output.variables}
        self._v2da = {v: getattr(self, f"get_{v}") for v in cfg.output.variables}

        # if we want cumulative bed level change, get the initial bed level now
        if "cum_bed_level_change" in self._v2np:
            self._initial_bed_level = self._get_min_bed_level().copy()

        # first step is written with xarray
        self._initialised = False
        self._epoch = None

    def _get_water_level(self):
        return self._hmodel.get_wl()

    def get_water_level(self):
        return xr.DataArray(
            self._get_water_level(),
            dims=("chainage",),
            coords={"chainage": self._cs},
            name="water_level",
        )

    def _get_flow(self):
        return self._hmodel.get_flow()

    def get_flow(self):
        return xr.DataArray(
            self._get_flow(),
            dims=("chainage",),
            coords={"chainage": self._cs},
            name="flow",
        )

    def _get_dt(self):
        return self._chan._dt

    def get_dt(self):
        return xr.DataArray(
            self._get_dt(),
            name="dt",
        )

    def _get_cum_bed_level_change(self):
        return self._get_min_bed_level() - self._initial_bed_level

    def get_cum_bed_level_change(self):
        return xr.DataArray(
            self._get_cum_bed_level_change(),
            dims=("chainage",),
            coords={"chainage": self._cs},
            name="cum_bed_level_change",
        )

    def _get_timestep(self):
        return np.uint32(self.step)

    def get_timestep(self):
        return xr.DataArray(self._get_timestep(), name="timestep")

    def _get_velocity(self):
        return self._hmodel.get_vel()

    def get_velocity(self):
        return xr.DataArray(
            self._get_velocity(),
            dims=("chainage",),
            coords={"chainage": self._cs},
            name="velocity",
        )

    def _get_grain_stress(self):
        r = self._chan.get_grain_shear_velocity()
        return WATER_DENSITY * r**2

    def get_grain_stress(self):
        return xr.DataArray(
            self._get_grain_stress(),
            dims=("chainage",),
            coords={"chainage": self._cs},
            name="grain_stress",
        )

    def _get_cum_flow_vol(self):
        return self._hmodel.get_cum_flow_vol()

    def get_cum_flow_vol(self):
        return xr.DataArray(
            self._get_cum_flow_vol(),
            dims=("chainage",),
            coords={
                "chainage": self._cs,
            },
            name="cum_flow_vol",
        )

    def _get_cum_transport(self):
        return self._chan.get_cum_transport()

    def get_cum_transport(self):
        return xr.DataArray(
            self._get_cum_transport(),
            dims=("chainage", "rel_grain_sizes", "lith"),
            coords={
                "chainage": self._cs,
                "rel_grain_sizes": self._rgsizes,
                "lith": np.arange(self._nlith, dtype=np.uint32),
            },
            name="cum_transport",
        )

    def _get_cum_total_transport(self):
        return self._chan.get_cum_transport().sum(axis=(1, 2))

    def get_cum_total_transport(self):
        return xr.DataArray(
            self._get_cum_total_transport(),
            dims=("chainage",),
            coords={
                "chainage": self._cs,
            },
            name="cum_total_transport",
        )

    def _get_total_transport_rate(self):
        return self._chan.get_Qb_jli().sum(axis=(1, 2))

    def get_total_transport_rate(self):
        return xr.DataArray(
            self._get_total_transport_rate(),
            dims=("chainage",),
            coords={"chainage": self._cs},
            name="total_transport_rate",
        )

    def _get_transport_rate(self):
        return self._chan.get_Qb_jli()

    def get_transport_rate(self):
        return xr.DataArray(
            self._get_transport_rate(),
            dims=("chainage", "rel_grain_sizes", "lith"),
            coords={
                "chainage": self._cs,
                "rel_grain_sizes": self._rgsizes,
                "lith": np.arange(self._nlith, dtype=np.uint32),
            },
            name="transport_rate",
        )

    def _get_mean_bed_level(self):
        return self._chan.get_mean_bed_level()

    def get_mean_bed_level(self):
        return xr.DataArray(
            self._get_mean_bed_level(),
            dims=("chainage",),
            coords={"chainage": self._cs},
            name="mean_bed_level",
        )

    def _get_min_bed_level(self):
        return self._chan.get_min_bed_level()

    def get_min_bed_level(self):
        return xr.DataArray(
            self._get_min_bed_level(),
            dims=("chainage",),
            coords={"chainage": self._cs},
            name="min_bed_level",
        )

    def _get_layer_thickness(self):
        return self._chan.get_layer_thickness()

    def get_layer_thickness(self):
        return xr.DataArray(
            self._get_layer_thickness(),
            dims=("chainage", "layer"),
            coords={
                "chainage": self._cs,
                "layer": np.arange(self._nlayers, dtype=np.uint32),
            },
            name="layer_thickness",
        )

    def _get_layer_composition(self):
        return self._chan.get_layer_composition()

    def get_layer_composition(self):
        return xr.DataArray(
            self._get_layer_composition(),
            dims=("chainage", "layer", "rel_grain_sizes", "lith"),
            coords={
                "chainage": self._cs,
                "layer": np.arange(self._nlayers, dtype=np.uint32),
                "rel_grain_sizes": self._rgsizes,
                "lith": np.arange(self._nlith, dtype=np.uint32),
            },
            name="layer_composition",
        )

    def _get_bedrock(self):
        return self._chan.get_bedrock()

    def get_bedrock(self):
        return xr.DataArray(
            self._get_bedrock(),
            dims=("chainage",),
            coords={"chainage": self._cs},
            name="bedrock",
        )

    def write_step(self, step: int, t: dt.datetime):
        """Append output for this timestep to netcdf file."""
        self.time = t
        self.step = step

        # Use xarray to make first file
        if not self._initialised:
            self._epoch = np.datetime64(t)
            ds = xr.Dataset(
                data_vars={
                    v: fun().expand_dims(time=[t]) for v, fun in self._v2da.items()
                }
            )
            ds["time"].encoding = {
                "units": f"seconds since {self._epoch}",
                "calendar": "proleptic_gregorian",
                "dtype": "int64",
            }
            ds.to_netcdf(
                self._outfile, mode="w", engine="h5netcdf", unlimited_dims=["time"]
            )
            self._initialised = True
            return

        with h5py.File(self._outfile, "r+") as f:
            n = f["time"].shape[0]
            f["time"].resize((n + 1,))
            # f["time"][n] = np.datetime64(t)
            f["time"][n] = int(
                (np.datetime64(t) - self._epoch) / np.timedelta64(1, "s")
            )

            for v, fun in self._v2np.items():
                dset = f[v]
                dset.resize((n + 1,) + dset.shape[1:])
                dset[n, ...] = fun()
