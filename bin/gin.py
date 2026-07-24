import datetime as dt
import pathlib
import typing
import pydantic as p


class GrateBase(p.BaseModel):
    model_config = p.ConfigDict(extra="forbid")


class Header(GrateBase):
    runid: str


class Model(GrateBase):
    type: str


class SimulationTime(GrateBase):
    start: dt.datetime
    end: dt.datetime
    num_cycles: p.StrictInt
    max_dt_qs: p.StrictFloat
    max_dt_fd: p.StrictFloat
    max_dx: p.StrictFloat
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


class CrossSectionProfiles(GrateBase):
    chainage: p.StrictFloat
    topoid: str
    river_name: str
    formrf: p.StrictFloat | None = None
    bankd90: p.StrictFloat | None = None
    active_layer_group: p.StrictInt | None = None
    storage_layer_group: p.StrictInt | None = None
    bedrock_rl: p.StrictFloat | None = None
    qsfact: p.StrictFloat | None = None
    lsf: p.StrictFloat | None = None
    profile: pathlib.Path


class CrossSections(GrateBase):
    formrf: p.StrictFloat
    wallrf: p.StrictFloat | None = None
    profiles: list[CrossSectionProfiles]


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
    downstream_boundary: DownstreamBoundary
    sediment_boundary: list[SedimentBoundary]

    sediment_extraction: list[SedimentExtraction] = []
    sediment_ripping: list[SedimentRipping] = []

    grain_size_profiles: GrainSizeProfiles

    print: PrintOptions

    @p.model_validator(mode="after")
    def check_cross_sections(self):
        if self.model.type == "flume" and self.cross_sections.wallrf is None:
            raise ValueError("cross_sections.wallrf is required for flume models")
        elif self.model.type != "flume" and self.cross_sections.wallrf is not None:
            raise ValueError("cross_sections.wallrf is only valid for flume models")
        return self

    @p.model_validator(mode="after")
    def check_grain_size(self):
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

        return self
