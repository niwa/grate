use anyhow::{Context, Result};
use ndarray::{Array1, Ix1};

use crate::channel::Channel;
use crate::config::Config;
use crate::downstream::Downstream;
use crate::interp_array::InterpArray;
use crate::utils;

const G: f64 = 9.8;

pub struct QuasiSteadyState {
    beta: f64,
    nc: usize,
    cs: Array1<f64>,                // (nc, )
    sf: Array1<f64>,                // (nc, )
    vel: Array1<f64>,               // (nc, )
    flow: Array1<f64>,              // (nc, )
    cum_flow_vol: Array1<f64>,      // (nc, )
    wl: Array1<f64>,                // (nc, )
    downstream: Downstream,
    inflow: InterpArray,
}

impl QuasiSteadyState {
    pub fn new(cfg: &mut Config) -> Result<Self> {
        let nc: usize = cfg.nc;
        let model = Self {
            beta: cfg.morphological.beta,
            nc: cfg.nc,
            cs: cfg.cs.clone(),
            sf: Array1::ones(nc),
            vel: Array1::ones(nc),
            flow: Array1::ones(nc),
            cum_flow_vol: Array1::zeros(nc),
            wl: Array1::ones(nc),
            downstream: cfg.downstream.take().unwrap(),
            inflow: cfg.inflow.take().unwrap(),
        };

        // model.initialize(cfg.simulation_time.start)?;
        Ok(model)
    }

    pub fn initialize(&mut self, t: f64, chan: &Channel) -> Result<()> {
        // For a normal-flow boundary, start from its specified initial level.
        // Otherwise, calculate the downstream level from the boundary.
        self.wl[self.nc - 1] = self.downstream.wl_init.unwrap_or(self.calc_ds_wl(t, chan)?);

        let bed = chan.get_min_bed_level();
        for c in (0..self.nc - 1).rev() {
            self.wl[c] = self.wl[c + 1] - bed[c + 1] + bed[c];
        }

        self.flow = self.inflow.value_at(t).into_dimensionality::<Ix1>()?;
        self.update_water_level(t, chan)?;

        // set back to zero for proper counting
        self.cum_flow_vol = Array1::<f64>::zeros(self.nc);
        Ok(())
    }

    fn calc_ds_wl(&self, t: f64, chan: &Channel) -> Result<f64> {
        let c = self.nc - 1;
        let bed = chan.get_min_bed_level()[c];

        if let Some(depth) = self.downstream.depth {
            return Ok(depth + bed);
        }

        if let Some(slope) = self.downstream.slope {
            let q = self.flow[c];
            let f = |wl| q - chan.calc_conveyance(c, wl) * slope.sqrt();

            return utils::find_root(f, self.wl[c], bed)
                .context("Failed to find downstream water level for normal flow");
        }

        return Ok(self.downstream.value_at(t));
    }

    pub fn get_vel(&self) -> &Array1<f64> {
        &self.vel
    }

    pub fn get_flow(&self) -> &Array1<f64> {
        &self.flow
    }

    pub fn get_cum_flow_vol(&self) -> &Array1<f64> {
        &self.cum_flow_vol
    }

    pub fn get_sf(&self) -> &Array1<f64> {
        &self.sf
    }

    pub fn get_wl(&self) -> &Array1<f64> {
        &self.wl
    }

    fn calc_vel(&self, c: usize, wl: f64, chan: &Channel) -> f64 {
        self.flow[c] / chan.calc_area(c, wl).sum()
    }

    fn calc_sf(&self, c: usize, wl: f64, chan: &Channel) -> f64 {
        let q = self.flow[c];
        q * q.abs() / chan.calc_conveyance(c, wl).powi(2)
    }

    fn conservation_of_energy(&self, c: usize, wl: f64, chan: &Channel) -> f64 {
        let dc = self.cs[c + 1] - self.cs[c];
        let sf = (self.calc_sf(c, wl, chan) + self.sf[c + 1]) / 2.0;
        let velocity_head =
            self.beta * (self.calc_vel(c, wl, chan).powi(2) - self.vel[c + 1].powi(2))
                / (2.0 * G);

        wl + velocity_head - self.wl[c + 1] - sf * dc
    }

    pub fn update_water_level(&mut self, t: f64, chan: &Channel) -> Result<()> {

        // FIXME
        // println!("Update water level at t={t}");

        self.flow = self.inflow.value_at(t).into_dimensionality::<Ix1>()?;
        self.wl[self.nc - 1] = self.calc_ds_wl(t, chan)?;

        // FIXME, really?
        let bed = chan.get_min_bed_level().clone();

        // println!("Water level is {:?}", self.wl);
        // println!("bed is {:?}", bed);
        // for i in 0..self.nc {
        //     println!("i={i} Depth is {}", self.wl[i] - bed[i]);
        // }


        for i in (0..self.nc - 1).rev() {
            self.sf[i + 1] = self.calc_sf(i + 1, self.wl[i + 1], chan);
            self.vel[i + 1] = self.calc_vel(i + 1, self.wl[i + 1], chan);

            let init_wl = self.wl[i + 1] - bed[i + 1] + bed[i];

            let f = |wl| self.conservation_of_energy(i, wl, chan);
            let wl = match utils::find_root(f, init_wl, bed[i]) {
                Ok(wl) => wl,
                Err(e) => {
                    let mut wtr = csv::Writer::from_path("root_finding_failure_f_values.csv")?;
                    wtr.write_record(["wl", "f", "Sf", "u"])?;
                    let mut wl = bed[i] + 1e-7;
                    while wl < bed[i] + 10.0 {
                        let fv = f(wl);
                        if -1.0 < fv && fv < 1.0 {
                            let sf = self.calc_sf(i, wl, chan);
                            let u = self.calc_vel(i, wl, chan);
                            wtr.serialize((wl, fv, sf, u))?;
                        }
                        wl += 0.0001;
                    }
                    wtr.flush()?;
                    return Err(e).with_context(|| {
                        format!("Failed to find wl at x-section {i}, chainage={}", self.cs[i])
                    });
                }
            };

            self.wl[i] = wl;
            self.sf[i] = self.calc_sf(i, wl, chan);
            self.vel[i] = self.calc_vel(i, wl, chan);
        }
        // FIXME
        // println!("Water level at t={t}\nwl={:?}\nSf={:?}", self.wl, self.sf);
        // let wlfoo = 0.43172180567252344;
        // println!("wl={wlfoo}\nK={}\nA={}\nP={}\nnf={}", chan.calc_conveyance(44, wlfoo), chan.calc_area(44, wlfoo), chan.calc_p(44, wlfoo), chan.calc_nf(44, wlfoo));
        // let wlfoo = 0.43;
        // println!("wl={wlfoo}\nK={}\nA={}\nP={}\nnf={}", chan.calc_conveyance(44, wlfoo), chan.calc_area(44, wlfoo), chan.calc_p(44, wlfoo), chan.calc_nf(44, wlfoo));


        self.cum_flow_vol += &(&self.flow * chan.get_dt());

        // Cache channel quantities that depend on the updated water levels.
        // self.channel.set_profile_arrays(&self.wl);

        Ok(())
    }
}
