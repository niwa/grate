use std::path::Path;

use anyhow::{anyhow, Result};
use chrono::DateTime;
use ndarray::{Array1, Array2, Array3, Array4, Axis};
use netcdf::{VariableMut, FileMut};

use crate::channel::Channel;
use crate::config::Config;
use crate::quasi_steady_state::QuasiSteadyState;

const WATER_DENSITY: f64 = 1000.0;
const TIME_BLOCK: usize = 64;

// ---------------------------------------------------------------------------
// Variables
// ---------------------------------------------------------------------------

/// Output variables. strum gives us `"water_level".parse::<Variable>()` and
/// `<&'static str>::from(variable)` using the snake_case names.
#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::EnumString, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
pub enum Variable {
    WaterLevel,
    Flow,
    Dt,
    Timestep,
    CumBedLevelChange,
    CumTransport,
    CumTotalTransport,
    CumFlowVol,
    Velocity,
    GrainStress,
    TotalTransportRate,
    TransportRate,
    MeanBedLevel,
    MinBedLevel,
    LayerThickness,
    LayerComposition,
    Bedrock,
}

impl Variable {
    pub fn name(self) -> &'static str {
        self.into()
    }

    /// Dimensions of this variable in the file.
    fn dims(self) -> &'static [&'static str] {
        use Variable::*;
        match self {
            Dt | Timestep => &["time"],
            CumTransport | TransportRate => &["time", "chainage", "rel_grain_sizes", "lith"],
            LayerThickness => &["time", "chainage", "layer"],
            LayerComposition => &["time", "chainage", "layer", "rel_grain_sizes", "lith"],
            _ => &["time", "chainage"],
        }
    }

    pub fn parse(s: &str) -> Result<Variable> {
        s.parse::<Variable>()
            .map_err(|e| anyhow!("unknown output variable '{s}': {e}"))
    }

    /// Time-blocked chunk shape: `block` steps of time, everything else whole.
    pub fn chunks(&self, nc: usize, nlayer: usize, nbins: usize, nlith: usize) -> Vec<usize> {
        self.dims()
            .iter()
            .map(|&d| match d {
                "time" => TIME_BLOCK,
                "chainage" => nc,
                "layer" => nlayer + 1,      // for the active + storage
                "rel_grain_sizes" => nbins,
                "lith" => nlith,
                _ => unreachable!("unknown dimension {d}"),
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Small write helpers: write one timestep (index n) of an array
// ---------------------------------------------------------------------------

fn put1(var: &mut VariableMut<'_>, a: &Array1<f64>, n: usize) -> Result<()> {
    let a = a.as_standard_layout(); // no copy if already contiguous
    var.put_values(a.as_slice().unwrap(), (n..n + 1, ..))?;
    Ok(())
}

fn put2(var: &mut VariableMut<'_>, a: &Array2<f64>, n: usize) -> Result<()> {
    let a = a.as_standard_layout();
    var.put_values(a.as_slice().unwrap(), (n..n + 1, .., ..))?;
    Ok(())
}

fn put3(var: &mut VariableMut<'_>, a: &Array3<f64>, n: usize) -> Result<()> {
    let a = a.as_standard_layout();
    var.put_values(a.as_slice().unwrap(), (n..n + 1, .., .., ..))?;
    Ok(())
}

fn put4(var: &mut VariableMut<'_>, a: &Array4<f64>, n: usize) -> Result<()> {
    let a = a.as_standard_layout();
    var.put_values(a.as_slice().unwrap(), (n..n + 1, .., .., .., ..))?;
    Ok(())
}

/// (nc, nbins, nlith) -> (nc), summing over bins and lithologies.
fn total_over_grains(a: &Array3<f64>) -> Array1<f64> {
    a.axis_iter(Axis(0)).map(|s| s.sum()).collect()
}

// ---------------------------------------------------------------------------
// File creation
// ---------------------------------------------------------------------------

fn create_file(fname: &Path, epoch: usize, cs: &Array1<f64>, rgsizes: &Array1<f64>, nlayer: usize, nlith: usize, vars: &[Variable]) -> Result<FileMut> {
    let mut file = netcdf::create(fname)?;

    file.add_unlimited_dimension("time")?;
    file.add_dimension("chainage", cs.len())?;
    file.add_dimension("layer", nlayer + 1)?;       // + 1 for active
    file.add_dimension("rel_grain_sizes", rgsizes.len())?;
    file.add_dimension("lith", nlith)?;

    let epoch = DateTime::from_timestamp(epoch as i64, 0)
        .ok_or_else(|| anyhow!("Invalid start time"))?
        .naive_utc();
    let units = format!("seconds since {}", epoch.format("%Y-%m-%d %H:%M:%S"));

    // time: f64 seconds since epoch
    {
        let mut tv = file.add_variable::<f64>("time", &["time"])?;
        tv.set_chunking(&[TIME_BLOCK])?;
        tv.put_attribute("units", units)?;
        tv.put_attribute("calendar", "proleptic_gregorian")?;
    }

    // coordinate variables (same name as their dimension), written once
    {
        let mut v = file.add_variable::<f64>("chainage", &["chainage"])?;
        v.put_values(cs.as_slice().unwrap(), ..)?;
    }
    {
        let mut v = file.add_variable::<u32>("layer", &["layer"])?;
        v.put_values(&(0..=nlayer as u32).collect::<Vec<_>>(), ..)?;        // + 1 for active
    }
    {
        let mut v = file.add_variable::<f64>("rel_grain_sizes", &["rel_grain_sizes"])?;
        v.put_values(rgsizes.as_slice().unwrap(), ..)?;
    }
    {
        let mut v = file.add_variable::<u32>("lith", &["lith"])?;
        v.put_values(&(0..nlith as u32).collect::<Vec<_>>(), ..)?;
    }

    // data variables
    for v in vars {
        let chunks = v.chunks(cs.len(), nlayer, rgsizes.len(), nlith);

        let mut var = match v {
            // timestep is an integer
            Variable::Timestep => file.add_variable::<u32>(v.name(), v.dims()),
            _ => file.add_variable::<f64>(v.name(), v.dims())
        }?;

        var.set_chunking(&chunks)?;
        // var.set_compression(2, true)?;
    }

    Ok(file)
}

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

pub struct Output {
    /// whole seconds; times are written as offsets from this, so the `units`
    /// string and the stored values always agree
    epoch: usize,
    variables: Vec<Variable>,

    // only set if cum_bed_level_change was requested
    initial_bed_level: Option<Array1<f64>>,

    file: FileMut,
    n_written: usize,
}

impl Output {

    pub fn new(cfg: &Config, chan: &Channel) -> Result<Self> {

        let mut variables: Vec<Variable> = Vec::new();
        for s in &cfg.output.variables {
            let v = Variable::parse(s)?;
            if !variables.contains(&v) {
                variables.push(v);
            }
        }

        // if we want cumulative bed level change, get the initial bed level now
        let initial_bed_level = variables
            .contains(&Variable::CumBedLevelChange)
            .then(|| chan.get_min_bed_level().clone());

        // FIXME, check this is integer
        let epoch: usize = cfg.simulation_time.start.floor() as usize;

        let file = create_file(
            &cfg.output.fname,
            epoch,
            &cfg.cs,
            &cfg.rgsizes,
            chan.nlayer,
            cfg.sediment_densities.len(),
            &variables,
        )?;

        Ok(Self {
            epoch: epoch,
            variables,
            initial_bed_level,
            file,
            n_written: 0,
        })
    }

    /// Append output for this timestep to the netcdf file.
    pub fn write_step(
        &mut self,
        step: usize,
        t: f64,
        hmodel: &QuasiSteadyState,
        chan: &Channel,
    ) -> Result<()> {
        let n = self.n_written;

        self.file
            .variable_mut("time")
            .ok_or_else(|| anyhow!("no time variable"))?
            .put_values(&[t - self.epoch as f64], n..n + 1)?;

        // disjoint field borrows: file (mut), variables, initial_bed_level
        for &v in &self.variables {
            let mut var = self.file.variable_mut(v.name()).ok_or_else(|| anyhow!("no variable {}", v.name()))?;

            match v {
                Variable::Dt => var.put_values(&[chan.get_dt()], n..n + 1)?,
                Variable::Timestep => var.put_values(&[step as u32], n..n + 1)?,

                Variable::WaterLevel => put1(&mut var, hmodel.get_wl(), n)?,
                Variable::Flow => put1(&mut var, hmodel.get_flow(), n)?,
                Variable::CumFlowVol => put1(&mut var, hmodel.get_cum_flow_vol(), n)?,
                Variable::Velocity => put1(&mut var, hmodel.get_vel(), n)?,

                Variable::MeanBedLevel => put1(&mut var, chan.get_mean_bed_level(), n)?,
                Variable::MinBedLevel => put1(&mut var, chan.get_min_bed_level(), n)?,
                Variable::CumBedLevelChange => {
                    let init = self
                        .initial_bed_level
                        .as_ref()
                        .expect("initial bed level set in new()");
                    put1(&mut var, &(chan.get_min_bed_level() - init), n)?
                }
                Variable::GrainStress => {
                    let stress = chan.get_gsv().mapv(|u| WATER_DENSITY * u * u);
                    put1(&mut var, &stress, n)?
                }

                Variable::CumTransport => put3(&mut var, chan.get_cum_transport(), n)?,
                Variable::TransportRate => put3(&mut var, chan.get_trans_rate_jli(), n)?,
                Variable::CumTotalTransport => {
                    put1(&mut var, &total_over_grains(chan.get_cum_transport()), n)?
                }
                Variable::TotalTransportRate => {
                    put1(&mut var, &total_over_grains(chan.get_trans_rate_jli()), n)?
                }
                Variable::LayerThickness => {
                    put2(&mut var, &chan.get_layer_thickness(), n)?
                }
                Variable::LayerComposition => {
                    put4(&mut var, &chan.get_layer_composition(), n)?
                }
                Variable::Bedrock => put1(&mut var, chan.get_bedrock(), n)?,
            }
        }

        self.n_written += 1;
        Ok(())
    }

    /// Flush to disk. Consider calling this every N steps on long runs.
    pub fn sync(&mut self) -> Result<()> {
        self.file.sync()?;
        Ok(())
    }

}
