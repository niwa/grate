import math
import datetime as dt
import numpy as np
import pandas as pd
import pathlib
import typing
import pydantic as p
from dataclasses import dataclass
from grainprofile import get_grain_props


class GrateBase(p.BaseModel):
    model_config = p.ConfigDict(extra="forbid")


class Header(GrateBase):
    runid: str


class Model(GrateBase):
    channel_type: typing.Literal["flume", "river", "braided_channel"]
    hydro_model_type: typing.Literal["quasi_ss", "dynamic"]


class SimulationTime(GrateBase):
    start: dt.datetime
    end: dt.datetime
    num_cycles: p.StrictInt
    max_dt_qs: p.StrictFloat
    max_dt_fd: p.StrictFloat
    cdt: p.StrictFloat
    max_dq_over_dt: p.StrictFloat | None = None


class HDParams(GrateBase):
    fd_toler: p.StrictFloat
    fd_itermax: p.StrictFloat
    fd_fr_min: p.StrictFloat
    fd_fr_max: p.StrictFloat


class Morphological(GrateBase):
    layer: p.StrictFloat
    la: p.StrictFloat
    nbs: p.StrictInt
    poro: p.StrictFloat
    alpha_s: p.StrictFloat | None = None
    neqal: p.StrictInt = 0.0
    dk: p.StrictFloat = 2.0
    chi: p.StrictFloat = 0.7
    qthres: p.StrictFloat = 0.0
    qthres_dtmultiplier: p.StrictFloat = 20.0
    beta: p.StrictFloat = 1.0


class Discretisation(GrateBase):
    theta: p.StrictFloat
    theta_s: p.StrictFloat
    psi_s: p.StrictFloat
    chainage_min: p.StrictFloat
    chainage_max: p.StrictFloat
    max_dc: p.StrictFloat


class CrossSectionProfile(GrateBase):
    chainage: p.StrictFloat
    topoid: str
    river_name: str
    formrf: typing.Literal["Interp"] | p.StrictFloat | None = (
        None  # override default roughness, can * by relrf in csv
    )
    bankd90: typing.Literal["Interp"] | p.StrictFloat | None = None
    active_layer_group: (
        typing.Literal["Interp"] | typing.Annotated[p.StrictInt, p.Field(ge=1)]
    )
    storage_layer_group: (
        typing.Literal["Interp"] | typing.Annotated[p.StrictInt, p.Field(ge=1)]
    )
    bedrock_rl: typing.Literal["Interp"] | p.StrictFloat | None = None
    qsfact: typing.Literal["Interp"] | p.StrictFloat | None = None
    lsf: typing.Literal["Interp"] | p.StrictFloat | None = None
    profile: pathlib.Path


class CrossSections(GrateBase):
    formrf: p.StrictFloat  # default form roughness
    # wallrf: p.StrictFloat | None = None  # vertical wall roughness for Flume
    profiles: list[CrossSectionProfile]


class InflowBoundaryTS(GrateBase):
    type: typing.Literal["ts"]
    ordinate: p.StrictFloat
    value: pathlib.Path


class InflowBoundaryConst(GrateBase):
    type: typing.Literal["const"]
    ordinate: p.StrictFloat
    value: p.StrictFloat


InflowBoundary = typing.Annotated[
    typing.Union[InflowBoundaryTS, InflowBoundaryConst],
    p.Field(discriminator="type"),
]


class RuntimeInflowBoundary:
    """Combine all the inflow boundary conditions into one datastructure.

    If all the inflow BCs are constant, then we can just store the cumsum of
    the flows downstream.  Otherwise we store a dataframe with time index and a
    column for each chainage (some columns might be constant).  This is
    converted to cumulative downstream flow so we can quickly get the inflow at all chainages.

    The rows are trimmed so we don't store data outside the simulation time
    domain.

    After the list of chainages is known (the model may do interpolation) this
    class allows the chainages to be resampled (using constant flat resampling
    which is what flow should be resampled as)
    """

    def __init__(self, ibs: list, start, end):
        """Take a list of inflowboundaries and convert to list or dataframe"""

        if not ibs:
            raise ValueError("At least one inflow boundary is required")

        if len({ib.ordinate for ib in ibs}) != len(ibs):
            raise ValueError("Inflow boundary ordinates must be unique")

        # if all inflows are constant, we just store a simple 2d array for flow
        self.__const = False

        if all(isinstance(ib, InflowBoundaryConst) for ib in ibs):
            self.__const = True
            # first column is chainage, second is the values
            vals = np.array([(ib.ordinate, ib.value) for ib in ibs])
            vals = vals[np.argsort(vals[:, 0])]
            vals[:, 1] = vals[:, 1].cumsum()
            self.__flows = vals
            return

        # so at least one of the inflows is a timeseries, put everything into
        # dataframe
        #
        # first gather into a list of series since the time indices might not
        # match
        seris = []
        for ib in ibs:
            if ib.type == "ts":
                try:
                    seri = pd.read_csv(ib.value, index_col=0, parse_dates=True)[
                        "flow"
                    ].sort_index()
                    if start < seri.index[0] or seri.index[-1] < end:
                        raise ValueError(
                            f"{ib.value} does not cover the simulation period "
                            f"{start} to {end}; "
                            f"timeseries covers {seri.index[0]} to {seri.index[-1]}"
                        )
                except Exception as exp:
                    raise ValueError(f"Could not parse {ib.value}: {exp}")
                # only bother with storing the time domain necessary
                i0 = seri.index.searchsorted(start, side="left")
                i1 = seri.index.searchsorted(end, side="right")
                seri = seri.iloc[max(0, i0 - 1) : min(len(seri), i1 + 1)]
                seri = seri.rename(ib.ordinate)
                seris.append(seri)

        df = pd.concat(seris, axis=1, sort=True)

        # some seri might have had different time index
        df = df.interpolate(method="time")

        # add in the constant bcs
        for ib in ibs:
            if ib.type == "const":
                df[ib.ordinate] = ib.value

        # finally sort the columns so we can cumsum
        df = df.sort_index(axis=1)
        df = df.cumsum(axis=1)

        self.__flows = df

    def get_flows(self, t: pd.Timestamp):
        """Return the downstream flows at current chainages."""
        if self.__const:
            return self.__flows[:, 1]

        if t in self.__flows.index:
            return self.__flows.loc[t].to_numpy()

        # have to interpolate
        pos = self.__flows.index.searchsorted(t)

        # bounds check
        if pos == 0:
            return self.__flows.iloc[0].to_numpy()
        if pos == len(self.__flows):
            return self.__flows.iloc[-1].to_numpy()

        t0 = self.__flows.index[pos - 1]
        t1 = self.__flows.index[pos]
        v0 = self.__flows.iloc[pos - 1].to_numpy()
        v1 = self.__flows.iloc[pos].to_numpy()
        fraction = (t - t0) / (t1 - t0)
        return v0 + fraction * (v1 - v0)

    def rechainage(self, cs):
        """Update the flows because new chainage"""

        cs = np.sort(np.asarray(cs))

        if self.__const:
            idx = np.searchsorted(self.__flows[:, 0], cs, side="right") - 1
            values = np.zeros(len(cs), dtype=self.__flows.dtype)
            mask = idx >= 0
            values[mask] = self.__flows[idx[mask], 1]
            self.__flows = np.column_stack((cs, values))
            return

        old_cs = self.__flows.columns.to_numpy()
        idx = np.searchsorted(old_cs, cs, side="right") - 1

        values = np.zeros((len(self.__flows), len(cs)))
        mask = idx >= 0
        values[:, mask] = self.__flows.to_numpy()[:, idx[mask]]

        self.__flows = pd.DataFrame(
            values,
            index=self.__flows.index,
            columns=cs,
        )


@dataclass
class RuntimeDownstreamBoundary:
    type: str
    value: float | pd.Series | None = None
    slope: float | None = None
    wl_init: float | None = None

    def value_at(self, t: pd.Timestamp) -> dict:
        if self.type in ("elevation", "depth"):
            return {self.type: self.value}

        if self.type == "normal":
            return {self.type: {"slope": self.slope, "wl_init": self.wl_init}}

        s = self.value
        if t in s.index:
            return {"elevation": float(s.loc[t])}

        # have to interpolate
        pos = s.index.searchsorted(t)

        # bounds check
        if pos == 0:
            return {"elevation": float(s.iloc[0])}
        if pos == len(s):
            return {"elevation": float(s.iloc[-1])}

        t0 = s.index[pos - 1]
        t1 = s.index[pos]
        v0 = s.iloc[pos - 1]
        v1 = s.iloc[pos]
        fraction = (t - t0) / (t1 - t0)
        return {"elevation": float(v0 + fraction * (v1 - v0))}


class DownstreamBoundaryTS(GrateBase):
    type: typing.Literal["ts"]
    value: pathlib.Path


class DownstreamBoundaryConst(GrateBase):
    type: typing.Literal["elevation"]
    value: p.StrictFloat


class DownstreamBoundaryDepth(GrateBase):
    type: typing.Literal["depth"]
    value: p.StrictFloat


class DownstreamBoundaryNorm(GrateBase):
    type: typing.Literal["normal"]
    slope: p.StrictFloat
    wl_init: p.StrictFloat


DownstreamBoundary = typing.Annotated[
    typing.Union[
        DownstreamBoundaryDepth,
        DownstreamBoundaryNorm,
        DownstreamBoundaryTS,
        DownstreamBoundaryConst,
    ],
    p.Field(discriminator="type"),
]


class SedimentBoundaryConst(GrateBase):
    type: typing.Literal["const"]
    ordinate: p.StrictFloat
    group: p.StrictInt
    value: p.StrictFloat


class SedimentBoundaryTS(GrateBase):
    type: typing.Literal["ts"]
    ordinate: p.StrictFloat
    group: p.StrictInt
    scale: p.StrictFloat
    value: pathlib.Path


# class SedimentBoundaryRC(GrateBase):
#     type: typing.Literal["rc"]
#     ordinate: p.StrictFloat


SedimentBoundary = typing.Annotated[
    # typing.Union[SedimentBoundaryRC, SedimentBoundaryTS, SedimentBoundaryConst],
    typing.Union[SedimentBoundaryTS, SedimentBoundaryConst],
    p.Field(discriminator="type"),
]


@dataclass
class RuntimeSedimentBoundary:
    ordinate: float
    type: str
    nbins: int
    nlith: int
    value: np.ndarray | pd.DataFrame

    def unravel(self, x):
        return np.asarray(x).reshape(self.nbins, self.nlith)

    def value_at(self, t: pd.Timestamp) -> np.ndarray:
        """Return nbins x nlith volume/s sediment transport"""

        if self.type == "const":
            return self.value

        s = self.value
        if t in s.index:
            return self.unravel(s.loc[t])

        # have to interpolate
        pos = s.index.searchsorted(t)

        # bounds check
        if pos == 0:
            return self.unravel(s.iloc[0])
        if pos == len(s):
            return self.unravel(s.iloc[-1])

        t0 = s.index[pos - 1]
        t1 = s.index[pos]
        v0 = s.iloc[pos - 1]
        v1 = s.iloc[pos]
        fraction = (t - t0) / (t1 - t0)
        return self.unravel(v0 + fraction * (v1 - v0))


class SedimentExtraction(GrateBase):
    ordinate: p.StrictFloat
    type: str
    value: pathlib.Path


class SedimentRipping(GrateBase):
    ordinate: p.StrictFloat
    value: pathlib.Path


class GrainSizeProfiles(GrateBase):
    num_profiles: p.StrictInt
    num_bins: p.StrictInt
    num_lith: p.StrictInt
    abrasion_coeffs: list[p.StrictFloat]
    sediment_densities: list[p.StrictFloat]
    grain_size_cfds: list[list[p.StrictFloat]]
    lithfractions: list[list[p.StrictFloat]] | None = None

    @p.model_validator(mode="after")
    def post_validate(self):
        if self.num_lith == 1 and not self.lithfractions:
            self.lithfractions = [
                [100.0] * self.num_profiles for _ in range(self.num_bins)
            ]
        return self


class OutputOptions(GrateBase):
    frequency: p.StrictInt
    fname: pathlib.Path
    variables: (
        typing.Literal["all"]
        | list[
            typing.Literal[
                "water_level",
                "velocity",
                "grain_stress",
                "total_transport_rate",
                "transport_rate",
                "mean_bed_level",
                "min_bed_level",
            ]
        ]
    ) = p.Field(default_factory=list)


class GrateConfig(GrateBase):
    header: Header
    model: Model
    simulation_time: SimulationTime
    hd_params: HDParams = None
    morphological: Morphological
    discretisation: Discretisation
    cross_sections: CrossSections

    inflow_boundary: list[InflowBoundary]
    _processed_inflow: RuntimeInflowBoundary
    downstream_boundary: DownstreamBoundary
    _processed_downstream_boundary: RuntimeDownstreamBoundary = p.PrivateAttr(
        default=None
    )
    sediment_boundary: list[SedimentBoundary]
    _processed_sediment_boundary: list[RuntimeSedimentBoundary] = p.PrivateAttr(
        default_factory=list
    )

    sediment_extraction: list[SedimentExtraction] = []
    sediment_ripping: list[SedimentRipping] = []

    grain_size_profiles: GrainSizeProfiles

    output: OutputOptions

    # start out with maximum dt
    @p.computed_field
    @property
    def max_dt(self) -> float:
        if self.model.hydro_model_type == "quasi_ss":
            return self.simulation_time.max_dt_qs
        return self.simulation_time.max_dt_fd

    @p.model_validator(mode="after")
    def post_validate(self):
        self._load_output_variables()
        self._check_discretisation()
        self._check_cross_sections()
        self._check_grain_size()
        self._processed_inflow = RuntimeInflowBoundary(
            self.inflow_boundary, self.simulation_time.start, self.simulation_time.end
        )
        self._load_downstream_boundary()
        self._load_sediment_boundary_timeseries()
        self._check_sediment_boundary()
        self._check_unit_changes()
        return self

    def _load_output_variables(self):
        if self.output.variables == "all":
            self.output.variables = [
                "water_level",
                "velocity",
                "grain_stress",
                "total_transport_rate",
                "transport_rate",
                "mean_bed_level",
                "min_bed_level",
            ]

    def _check_unit_changes(self):
        # old gin was in tons and mm, we go to m, so check some ranges
        if not all(
            1_000 < d < 10_000 for d in self.grain_size_profiles.sediment_densities
        ):
            raise ValueError("Sediment densities not in 1_000 to 10_000")

        for row in self.grain_size_profiles.grain_size_cfds:
            if row[0] < 0.00004 or row[0] > 2:
                raise ValueError(f"Grain size profile {row[0]} not in [0.00004, 2]")

    def _check_discretisation(self):
        if self.discretisation.chainage_min >= self.discretisation.chainage_max:
            raise ValueError("chainage_min must be less than chainage_max")

    def _check_cross_sections(self):
        # if self.model.channel_type == "flume" and self.cross_sections.wallrf is None:
        #     raise ValueError("cross_sections.wallrf is required for flume models")
        # elif (
        #     self.model.channel_type != "flume"
        #     and self.cross_sections.wallrf is not None
        # ):
        #     raise ValueError("cross_sections.wallrf is only valid for flume models")

        nprof = self.grain_size_profiles.num_profiles
        for cs in self.cross_sections.profiles:
            if cs.active_layer_group != "Interp" and cs.active_layer_group > nprof:
                raise ValueError(
                    f"cross_sections.active_layer_group ({cs.active_layer_group}) must be <= number of grain size profiles ({nprof})"
                )
            if cs.storage_layer_group != "Interp" and cs.storage_layer_group > nprof:
                raise ValueError(
                    f"cross_sections.storage_layer_group ({cs.storage_layer_group}) must be <= number of grain size profiles ({nprof})"
                )

    def _check_grain_size(self):
        nprof = self.grain_size_profiles.num_profiles
        nbins = self.grain_size_profiles.num_bins
        nlith = self.grain_size_profiles.num_lith
        if nbins + 1 != len(self.grain_size_profiles.grain_size_cfds):
            raise ValueError(
                f"grain_size_profiles: {nbins=} but number of lines is {len(self.grain_size_profiles.grain_size_cfds)}"
            )
        for row in self.grain_size_profiles.grain_size_cfds:
            if nprof + 1 != len(row):
                raise ValueError(
                    f"grain_size_profiles: {nprof=} but number of columns is {len(row)}"
                )
        if nbins * nlith != len(self.grain_size_profiles.lithfractions):
            raise ValueError(
                f"grain_size_profiles: {nbins=} {nlith=} but number of lines is {len(self.grain_size_profiles.lithfractions)}"
            )
        for row in self.grain_size_profiles.lithfractions:
            if nprof != len(row):
                raise ValueError(
                    f"grain_size_profiles: {nprof=} but number of columns is {len(row)}.  NB lith table shouldnt have leading 1s"
                )
        if len(self.grain_size_profiles.abrasion_coeffs) != nlith:
            raise ValueError(
                f"grain_size_profiles: {len(self.grain_size_profiles.abrasion_coeffs)=} != {nlith=}"
            )
        if len(self.grain_size_profiles.sediment_densities) != nlith:
            raise ValueError(
                f"grain_size_profiles: {len(self.grain_size_profiles.sediment_densities)=} != {nlith=}"
            )

    def _chk_csv_covers_timeperiod(self, msg: str, seri):
        """check covers simulation period"""

        start = self.simulation_time.start
        end = self.simulation_time.end
        if start < seri.index[0] or seri.index[-1] < end:
            raise ValueError(
                f"{msg} does not cover the simulation period "
                f"{start} to {end}; "
                f"timeseries covers {seri.index[0]} to {seri.index[-1]}"
            )

    """
    def _load_inflow_timeseries(self):
        self._processed_inflow.clear()
        for boundary in self.inflow_boundary:
            val = boundary.value
            if boundary.type == "ts":
                try:
                    val = pd.read_csv(val, index_col=0, parse_dates=True)[
                        "flow"
                    ].sort_index()
                    self._chk_csv_covers_timeperiod(
                        f"Inflow ts {boundary.value} at {boundary.ordinate}", val
                    )
                except Exception as exp:
                    raise ValueError(f"Could not parse {boundary.value}: {exp}")
            self._processed_inflow.append(
                RuntimeInflowBoundary(
                    ordinate=boundary.ordinate,
                    type=boundary.type,
                    value=val,
                )
            )
    """

    def _load_downstream_boundary(self):
        b = self.downstream_boundary
        if b.type == "normal":
            self._processed_downstream_boundary = RuntimeDownstreamBoundary(
                type=b.type, slope=b.slope, wl_init=b.wl_init
            )
        else:
            val = b.value
            if b.type == "ts":
                try:
                    val = pd.read_csv(val, index_col=0, parse_dates=True)[
                        "water_level"
                    ].sort_index()
                    self._chk_csv_covers_timeperiod(f"Downstream ts {b.value}", val)
                except Exception as exp:
                    raise ValueError(f"Could not parse {b.value}: {exp}")
            self._processed_downstream_boundary = RuntimeDownstreamBoundary(
                type=b.type,
                value=val,
            )

    def _load_sediment_boundary_timeseries(self):
        self._processed_sediment_boundary.clear()
        gs = self.grain_size_profiles
        nbins = gs.num_bins
        nlith = gs.num_lith
        densities = gs.sediment_densities

        for boundary in self.sediment_boundary:
            # matrix nbins x nlith
            jliprops = get_grain_props(
                boundary.group - 1, gs.grain_size_cfds, gs.lithfractions
            )

            # sum the columns to get weights
            rho = np.average(densities, weights=jliprops.sum(axis=0))

            # val is kg/s
            val = boundary.value
            if boundary.type == "ts":
                try:
                    val = pd.read_csv(val, index_col=0, parse_dates=True)[
                        "unknown_0_FIXME"
                    ].sort_index()
                    self._chk_csv_covers_timeperiod(
                        f"Sediment ts {boundary.value} at {boundary.ordinate}", val
                    )
                except Exception as exp:
                    raise ValueError(f"Could not parse {boundary.value}: {exp}")
                val *= boundary.scale
                # make each row be val * jliprops unravelled
                values = val.to_numpy()[:, None] * jliprops.ravel()[None, :]
                val = pd.DataFrame(values, index=val.index)
            else:
                val *= jliprops

            # divide by density to get val in volume/s
            val /= rho

            self._processed_sediment_boundary.append(
                RuntimeSedimentBoundary(
                    ordinate=boundary.ordinate,
                    type=boundary.type,
                    nbins=nbins,
                    nlith=nlith,
                    value=val,
                )
            )

    def _check_sediment_boundary(self):
        cm = self.discretisation.chainage_min
        if any(sb.ordinate == cm for sb in self.sediment_boundary):
            return
        raise ValueError(
            f"Must be atleast one sediment_boundary condition with ordinate {cm}"
        )
