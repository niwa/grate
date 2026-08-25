import math
import datetime as dt
import pandas as pd
import pathlib
import typing
import pydantic as p
from dataclasses import dataclass


class GrateBase(p.BaseModel):
    model_config = p.ConfigDict(extra="forbid")


class Header(GrateBase):
    runid: str


class Model(GrateBase):
    type: typing.Literal["flume", "river", "braided_channel"]


class SimulationTime(GrateBase):
    start: dt.datetime
    end: dt.datetime
    num_cycles: p.StrictInt
    max_dt_qs: p.StrictFloat
    max_dt_fd: p.StrictFloat
    cdt: p.StrictFloat
    max_dq_over_dt: p.StrictFloat


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
    alpha_s: p.StrictFloat
    neqal: p.StrictInt
    dk: p.StrictFloat
    chi: p.StrictFloat = 0.7
    qthres: p.StrictFloat
    refgsz: p.StrictFloat
    refnode: p.StrictInt | None = None
    awopt: p.StrictInt | None = None


class Discretisation(GrateBase):
    theta: p.StrictFloat
    theta_s: p.StrictFloat
    psi_s: p.StrictFloat
    chainage_min: p.StrictFloat
    chainage_max: p.StrictFloat
    max_dc: p.StrictFloat

    @p.computed_field
    @property
    def dc(self) -> float:
        length = self.chainage_max - self.chainage_min
        num = math.ceil(length / self.max_dc)
        return length / num


class CrossSectionProfile(GrateBase):
    chainage: p.StrictFloat
    topoid: str
    river_name: str
    formrf: p.StrictFloat | None = (
        None  # override default roughness, can * by relrf in csv
    )
    bankd90: p.StrictFloat | None = None
    active_layer_group: p.StrictInt | None = None
    storage_layer_group: p.StrictInt | None = None
    bedrock_rl: p.StrictFloat | None = None
    qsfact: p.StrictFloat | None = None
    lsf: p.StrictFloat | None = None
    profile: pathlib.Path


class CrossSections(GrateBase):
    formrf: p.StrictFloat  # default form roughness
    wallrf: p.StrictFloat | None = None  # vertical wall roughness for Flume
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


@dataclass
class RuntimeInflowBoundary:
    ordinate: float
    type: str
    value: float | pd.Series

    def value_at(self, t: pd.Timestamp) -> float:
        if self.type == "const":
            return self.value

        s = self.value
        if t in s.index:
            return float(s.loc[t])

        # have to interpolate
        pos = s.index.searchsorted(t)

        # bounds check
        if pos == 0:
            return float(s.iloc[0])
        if pos == len(s):
            return float(s.iloc[-1])

        t0 = s.index[pos - 1]
        t1 = s.index[pos]
        v0 = s.iloc[pos - 1]
        v1 = s.iloc[pos]
        fraction = (t - t0) / (t1 - t0)
        return float(v0 + fraction * (v1 - v0))


@dataclass
class RuntimeDownstreamBoundary:
    ordinate: float
    type: str
    value: float | pd.Series

    def value_at(self, t: pd.Timestamp) -> dict:
        if self.type in ("elevation", "depth"):
            return {self.type: self.value}

        if self.type == "normal":
            return {self.type: {"slope": self.slope, "hinit": self.hinit}}

        s = self.value
        if t in s.index:
            return float(s.loc[t])

        # have to interpolate
        pos = s.index.searchsorted(t)

        # bounds check
        if pos == 0:
            return float(s.iloc[0])
        if pos == len(s):
            return float(s.iloc[-1])

        t0 = s.index[pos - 1]
        t1 = s.index[pos]
        v0 = s.iloc[pos - 1]
        v1 = s.iloc[pos]
        fraction = (t - t0) / (t1 - t0)
        return {"elevation": float(v0 + fraction * (v1 - v0))}


class DownstreamBoundaryTS(GrateBase):
    type: typing.Literal["elevation_timeseries"]
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
    hinit: p.StrictFloat


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
    fname: pathlib.Path


class SedimentBoundaryRC(GrateBase):
    type: typing.Literal["rc"]
    ordinate: p.StrictFloat


SedimentBoundary = typing.Annotated[
    typing.Union[SedimentBoundaryRC, SedimentBoundaryTS, SedimentBoundaryConst],
    p.Field(discriminator="type"),
]


class SedimentExtraction(GrateBase):
    ordinate: p.StrictFloat
    type: str
    fname: pathlib.Path


class SedimentRipping(GrateBase):
    ordinate: p.StrictFloat
    fname: pathlib.Path


class GrainSizeProfiles(GrateBase):
    num_profiles: p.StrictInt
    num_bins: p.StrictInt
    num_lith: p.StrictInt
    abrasion_coeffs: list[p.StrictFloat]
    sediment_densities: list[p.StrictFloat]
    grain_size_cfds: list[list[p.StrictFloat]]
    lithfractions: list[list[p.StrictFloat]] | None = []


class PrintOptions(GrateBase):
    nprtf: p.StrictInt
    outxsparms: p.StrictInt


class GrateConfig(GrateBase):
    header: Header
    model: Model
    simulation_time: SimulationTime
    hd_params: HDParams = None
    morphological: Morphological
    discretisation: Discretisation
    cross_sections: CrossSections

    inflow_boundary: list[InflowBoundary]
    _processed_inflow: list[RuntimeInflowBoundary] = p.PrivateAttr(default_factory=list)
    downstream_boundary: DownstreamBoundary
    _processed_downstream_boundary: RuntimeDownstreamBoundary = p.PrivateAttr(
        default=None
    )
    sediment_boundary: list[SedimentBoundary]

    sediment_extraction: list[SedimentExtraction] = []
    sediment_ripping: list[SedimentRipping] = []

    grain_size_profiles: GrainSizeProfiles

    print: PrintOptions

    @p.model_validator(mode="after")
    def post_validate(self):
        self._check_discretisation()
        self._check_cross_sections()
        self._check_grain_size()
        self._load_inflow_timeseries()
        self._load_downstream_boundary()
        return self

    def _check_discretisation(self):
        if self.discretisation.chainage_min >= self.discretisation.chainage_max:
            raise ValueError("chainage_min must be less than chainage_max")

    def _check_cross_sections(self):
        if self.model.type == "flume" and self.cross_sections.wallrf is None:
            raise ValueError("cross_sections.wallrf is required for flume models")
        elif self.model.type != "flume" and self.cross_sections.wallrf is not None:
            raise ValueError("cross_sections.wallrf is only valid for flume models")

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
        if nlith > 1 and nbins * nlith != len(self.grain_size_profiles.lithfractions):
            raise ValueError(
                f"grain_size_profiles: {nbins=} {nlith=} but number of lines is {len(self.grain_size_profiles.lithfractions)}"
            )
        for row in self.grain_size_profiles.lithfractions:
            if nprof + 1 != len(row):
                raise ValueError(
                    f"grain_size_profiles: {nprof=} but number of columns is {len(row)}"
                )

    def _load_inflow_timeseries(self):
        self._processed_inflow.clear()
        for boundary in self.inflow_boundary:
            val = boundary.value
            if boundary.type == "ts":
                val = pd.read_csv(val, index_col=0, parse_dates=True)[
                    "flow"
                ].sort_index()
            self._processed_inflow.append(
                RuntimeInflowBoundary(
                    ordinate=boundary.ordinate,
                    type=boundary.type,
                    value=val,
                )
            )

    def _load_downstream_boundary(self):
        b = self.downstream_boundary
        val = b.value
        if b.type == "ts":
            val = pd.read_csv(val, index_col=0, parse_dates=True)["flow"].sort_index()
        self._processed_downstream_boundary = RuntimeDownstreamBoundary(
            ordinate=b.ordinate,
            type=b.type,
            value=val,
        )
