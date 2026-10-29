use anyhow::Result;
use lambert_w::lambert_w0;
use ndarray::{concatenate, ArrayView1, Zip, Axis, Array1, Array2, Array3, Array4, Ix3};
use crate::config::{Config, CrossSectionTab};
use crate::interp_array::InterpArray;
use crate::storage_layers::StorageLayers;

const SAND_SIZE: f64 = 0.002;       // sand grain size in metres
const GRAVITY: f64 = 9.81;
const KAPPA: f64 = 0.4;             // Von Karman's constant
const WATER_DENSITY: f64 = 1000.0;

pub struct Channel {
    nc: usize,
    pub nlayer: usize,

    // time stepping
    dt: f64,
    pub max_dt: f64,
    cdt: f64,
    max_dydt: f64,

    sand_size: f64,                     // sand grain size in metres in phi units
    alayer_max_thick: f64,              // self._cfg.morphological.alayar_max_thick
    chi: f64,                           // self._cfg.morphological.chi
    sediment_densities: Array1<f64>,    // (nlith, )
    min_bed_level: Array1<f64>,         // (nc, )
    mean_bed_level: Array1<f64>,        // (nc, )
    bedrock: Array1<f64>,               // (nc, )
    fact: Array1<f64>,                  // (nc,), 1 / (cs[i + 1] - cs[i]) / (1 - poro) / bchan[i + 1]

    sediment: InterpArray,
    ng: Array2<f64>,                    // (nc, 3)

    // grain info
    rgsizes: Array1<f64>,               // (nbins, )
    phi: Array1<f64>,                   // (nbins, ) grain upper size in phi units, used for percentiles
    sand_fraction: Array1<f64>,         // (nc, ) how much sand in this chainage
    dsm: Array1<f64>,                   // (nc, ) median of grain size in channel
    d90_channel: Array1<f64>,           // (nc, ) 90 percentile grain size in channel
    d90: Array2<f64>,                   // (nc, 3 ) 90 percentile grain size left/channel/right
    bankd90: Array1<f64>,               // (nc, ) 90 percentile grain size on bank (if nan use channel)
    gsv: Array1<f64>,                   // (nc, ) 
    aprop: Array3<f64>,                 // (nc, nbins, nlith) FIXME
                                        
    // transport arrays
    trans_rate_jli: Array3<f64>,        // (nc, nbins, nlith)
    cum_transport: Array3<f64>,         // (nc, nbins, nlith)

    athickness: Array1<f64>,            // (nc, )
    storage_layers: StorageLayers,

    // caches for output, not sure we need these
    // bwet: Array1<f64>,               // (nc, )  this is needed for transport
    // area: Array2<f64>,               // (nc, 3)
    // p: Array2<f64>,                  // (nc, 3)
    // nf: Array2<f64>,                 // (nc, 3)

    // these are for calculating conveyance
    area_tab: Vec<CrossSectionTab>,
    p_tab: Vec<CrossSectionTab>,
    bwet_tab: Vec<CrossSectionTab>,
    nf_tab: Vec<CrossSectionTab>
}

impl Channel {
    pub fn new(cfg: &mut Config) -> Self {
        let xstabs = cfg.xstabs.take().unwrap();
        let fact: Array1<f64> = Array1::from_iter(
            (0..cfg.nc).map(|i| {
                let dc = if i > 0 {
                    cfg.cs[i] - cfg.cs[i - 1]
                } else {
                    cfg.cs[1] - cfg.cs[0]
                };
                1.0 / dc / (1.0 - cfg.morphological.poro) / cfg.bchan[i]
            })
        );
        
        let sl = StorageLayers::new(
            cfg.morphological.slayer_max_thick,
            cfg.sprop.clone(),
            cfg.sthickness.clone(),
        );

        let mut chan = Self {
            nc: cfg.nc,
            nlayer: cfg.nlayer,

            // time stepping
            dt: cfg.simulation_time.max_dt,
            max_dt: cfg.simulation_time.max_dt,
            cdt: cfg.simulation_time.cdt,
            max_dydt: 0.1 * cfg.simulation_time.cdt / cfg.simulation_time.max_dt,

            sand_size: -SAND_SIZE.ln(),
            alayer_max_thick: cfg.morphological.alayer_max_thick,
            chi: cfg.morphological.chi,
            sediment_densities: cfg.sediment_densities.clone(),
            min_bed_level: cfg.min_bed_level.clone(),
            mean_bed_level: cfg.mean_bed_level.clone(),
            bedrock: cfg.bedrock.clone(),
            fact: fact,

            sediment: cfg.sediment.take().unwrap(),
            ng: Array2::<f64>::zeros((cfg.nc, 3)),              // (nc, 3)
                
            rgsizes: cfg.rgsizes.clone(),
            phi: cfg.uppersizes.map(|x| -x.ln()),
            sand_fraction: Array1::<f64>::zeros(cfg.nc),
            dsm: Array1::<f64>::ones(cfg.nc),
            d90: Array2::<f64>::ones((cfg.nc, 3)),
            d90_channel: Array1::<f64>::ones(cfg.nc),
            bankd90: cfg.bankd90.clone(),
            gsv: Array1::<f64>::ones(cfg.nc),
            aprop: cfg.aprop.clone(),

            // transport arrays
            trans_rate_jli: Array3::<f64>::zeros(cfg.aprop.raw_dim()),
            cum_transport: Array3::<f64>::zeros(cfg.aprop.raw_dim()),
    
            athickness: cfg.athickness.clone(),
            storage_layers: sl,

            // bwet: Array1::<f64>::zeros(cfg.nc),
            // area: Array2::<f64>::zeros((cfg.nc, 3)),
            // p: Array2::<f64>::zeros((cfg.nc, 3)),
            // nf: Array2::<f64>::zeros((cfg.nc, 3)),

            area_tab: xstabs.area,
            p_tab: xstabs.p,
            bwet_tab: xstabs.bwet,
            nf_tab: xstabs.nf,
        };

        // sets stuff including ng which is needed for conveyance, no gsv though, but that will
        // set set when hydromodel runs and has some Sf and velocity.
        chan.set_grain_arrays();
        chan
    }

    /// Allow simulation to set massive dt if flow below qthres
    pub fn set_dt(&mut self, dt: f64) {
        self.dt = dt;
    }

    pub fn get_min_bed_level(&self) -> &Array1<f64> {
        &self.min_bed_level
    }

    pub fn get_mean_bed_level(&self) -> &Array1<f64> {
        &self.mean_bed_level
    }

    pub fn get_dt(&self) -> f64 {
        self.dt
    }

    pub fn get_cum_transport(&self) -> &Array3<f64> {
        &self.cum_transport
    }

    pub fn get_gsv(&self) -> &Array1<f64> {
        &self.gsv
    }

    pub fn get_trans_rate_jli(&self) -> &Array3<f64> {
        &self.trans_rate_jli
    }

    ///  Return thickness of active and storage for chainages
    ///
    ///  (nc, 1 + num_slayers).  The first layer is active layer, then top
    ///  storage etc.  Any layer
    ///  can have zero thickness, and if the wl is close to the bedrock the
    ///  lower storage layers often do have zero thickness.  The lowest real
    ///  storage layer can have inf thickness.  Storage layers that don't
    ///  exist have NaN thickness
    pub fn get_layer_thickness(&self) -> Array2<f64> {
        concatenate(
            Axis(1),
            &[
                self.athickness.view().insert_axis(Axis(1)),
                self.storage_layers.get_thickness().view(),
            ],
        ).unwrap()
    }

    /// Return active and storage layer proportions
    ///
    /// (nc, 1 + num_slayers, nbin, nlith)
    /// The first layer is the active layer, then storage layers from the
    /// top down.  Storage layers that don't exist are filled with nan.
    pub fn get_layer_composition(&self) -> Array4<f64> {
        concatenate(
            Axis(1),
            &[
                self.aprop.view().insert_axis(Axis(1)),
                self.storage_layers.get_sprop().view()
            ],
        ).unwrap()
    }

    pub fn get_bedrock(&self) -> &Array1<f64> {
        &self.bedrock
    }

    pub fn calc_area(&self, c: usize, wl: f64) -> Array1<f64> {
        let tab = &self.area_tab[c];
        let d = wl - self.min_bed_level[c];
        Array1::from_vec(vec![
            tab.0.as_ref().map_or(0.0, |v| v.value_at(wl).into_dimensionality::<ndarray::Ix0>().unwrap()[()]),
            tab.1.as_ref().map_or(0.0, |v| v.value_at(d).into_dimensionality::<ndarray::Ix0>().unwrap()[()]),
            tab.2.as_ref().map_or(0.0, |v| v.value_at(wl).into_dimensionality::<ndarray::Ix0>().unwrap()[()]),
        ])
    }

    fn calc_bwet(&self, c: usize, wl: f64) -> f64 {
        let tab = &self.bwet_tab[c];
        let d = wl - self.min_bed_level[c];
        return 
            tab.0.as_ref().map_or(0.0, |v| v.value_at(wl).into_dimensionality::<ndarray::Ix0>().unwrap()[()]) +
            tab.1.as_ref().map_or(0.0, |v| v.value_at(d).into_dimensionality::<ndarray::Ix0>().unwrap()[()]) +
            tab.2.as_ref().map_or(0.0, |v| v.value_at(wl).into_dimensionality::<ndarray::Ix0>().unwrap()[()]);
    }

    fn calc_p(&self, c: usize, wl: f64) -> Array1<f64> {
        let tab = &self.p_tab[c];
        let d = wl - self.min_bed_level[c];
        Array1::from_vec(vec![
            tab.0.as_ref().map_or(0.0, |v| v.value_at(wl).into_dimensionality::<ndarray::Ix0>().unwrap()[()]),
            tab.1.as_ref().map_or(0.0, |v| v.value_at(d).into_dimensionality::<ndarray::Ix0>().unwrap()[()]),
            tab.2.as_ref().map_or(0.0, |v| v.value_at(wl).into_dimensionality::<ndarray::Ix0>().unwrap()[()]),
        ])
    }

    fn calc_nf(&self, c: usize, wl: f64) -> Array1<f64> {
        let tab = &self.nf_tab[c];
        let d = wl - self.min_bed_level[c];
        Array1::from_vec(vec![
            tab.0.as_ref().map_or(0.0, |v| v.value_at(wl).into_dimensionality::<ndarray::Ix0>().unwrap()[()]),
            tab.1.as_ref().map_or(0.0, |v| v.value_at(d).into_dimensionality::<ndarray::Ix0>().unwrap()[()]),
            tab.2.as_ref().map_or(0.0, |v| v.value_at(wl).into_dimensionality::<ndarray::Ix0>().unwrap()[()]),
        ])
    }

    pub fn calc_conveyance(&self, c: usize, wl: f64) -> f64 {
        let a = self.calc_area(c, wl);
        let p = self.calc_p(c, wl);
        let nf = self.calc_nf(c, wl);

        // FIXME
        // println!("c={} wl={} a={:?}, p={:?}, nf={:?}", c, wl, a, p, nf);

        let k: f64 = (0..3)
            .filter(|&i| p[i] != 0.0)
            .map(|i| a[i] * (a[i] / p[i]).powf(2.0 / 3.0) / (self.ng[(c, i)] + nf[i]))
            .sum();

        assert!(k > 0.0, "No water");
        k
    }

    /// Set sand_fraction/dsm/d90/ng which depend on aprop
    fn set_grain_arrays(&mut self) {

        // sum over nlith axis, to get (nc, nbins), then cumsum down nbins axis
        let cumf: Array2<f64> = cumf_lith(&self.aprop);

        // sand fraction for each chainage (nc, )
        cumf_smaller_than(self.sand_size, &self.phi, &cumf, &mut self.sand_fraction);

        // median grain size in channel
        percentile(0.5, &cumf, &self.phi, &mut self.dsm);

        // 90 percentile grain size in channel
        percentile(0.9, &cumf, &self.phi, &mut self.d90_channel);

        // combine the channel d90 and bankd90 into our d90
        for c in 0..self.nc {
            let bank = if self.bankd90[c].is_nan() {
                self.d90_channel[c]
            } else {
                self.bankd90[c]
            };

            self.d90[[c, 0]] = bank;
            self.d90[[c, 1]] = self.d90_channel[c];
            self.d90[[c, 2]] = bank;
        }

        Zip::from(&mut self.ng).and(&self.d90)
            .for_each(|ng, &d| *ng = 0.044 * d.powf(1.0 / 6.0));

    }

    /// Update ustar, grain shear velocity which depends on Sf/vel from hydro
    pub fn update_gsv(&mut self, sf: &Array1<f64>, velocity: &Array1<f64>) -> Result<()> {
        for i in 0..self.nc {
            let ks = 2.0 * self.d90[[i, 1]];       // channel
            let a = KAPPA * velocity[i];
            let b = 11.0 / GRAVITY / sf[i] / ks;
            let x = a * b.sqrt() / 2.0;

            if !(x > 0.0) {
                anyhow::bail!("ustar calc, arg <= 0, possibly complex solutions, at c={i}");
            }

            self.gsv[i] = a / 2.0 / lambert_w0(x);
        }
        Ok(())
    }

    /// Set Qb_jli (called trans_rate_jli)
    /// Needs waterlevel for bwet which we aren't storing
    pub fn set_transport_rates(&mut self, wl: &Array1<f64>) {

        // update trans_rate_jli with qb_jli
        self.qb_jli();

        // FIXME
        // let text = self.trans_rate_jli
        //     .iter()
        //     .map(|x| format!("{:.16e}", x))
        //     .collect::<Vec<_>>()
        //     .join("\n");
        // std::fs::write("/tmp/rust_array.txt", text).unwrap();
        //
        // std::process::exit(0);

        // now multiply by bwet
        for c in 0..self.nc {
            let bwet = self.calc_bwet(c, wl[c]);
            self.trans_rate_jli
                .index_axis_mut(Axis(0), c)
                .mapv_inplace(|q| q * bwet);
        }

        // let text = self.trans_rate_jli
        //     .iter()
        //     .map(|x| format!("{:.16e}", x))
        //     .collect::<Vec<_>>()
        //     .join("\n");
        // std::fs::write("/tmp/rust_array.txt", text).unwrap();
        // std::process::exit(0);

        // the cumulative transport rate
        self.cum_transport.scaled_add(self.dt, &self.trans_rate_jli);
        // FIXME
        // println!("Updating cum trans {}", self.cum_transport.sum());
    }

    // Wilcock & Crowe (2003) eq 9.33 for every chainage at once.
    //
    // Volumetric bed-load transport rate per unit width: m^3/s / m
    // Updates trans_rate_jli
    pub fn qb_jli(&mut self) {
        let nc = self.nc;
        let nbins = self.rgsizes.len();
        let nlith = self.sediment_densities.len();

        let s = &self.sediment_densities / WATER_DENSITY;   // (nlith, )

        // FIXME
        // println!("\ngsv={:?}\nFs={:?}\ndsm={:?}", self.gsv, self.sand_fraction, self.dsm);
        // println!("\nbedrock={:?}", self.bedrock);


        for c in 0..nc {
            let phirm = 0.021 + 0.015 * (-20.0 * self.sand_fraction[c]).exp();

            for j in 0..nbins {
                let ratio = self.rgsizes[j] / self.dsm[c];
                let b = 0.67 / (1.0 + (1.5 - ratio).exp());

                for l in 0..nlith {
                    let tau_rm = phirm * (s[l] - 1.0) * WATER_DENSITY * GRAVITY * self.dsm[c];
                    let tau_rj = tau_rm * ratio.powf(b);
                    let phi = WATER_DENSITY * self.gsv[c].powi(2) / tau_rj;

                    let q = self.aprop[[c, j, l]] * (self.athickness[[c]] / self.alayer_max_thick) * self.gsv[c].powi(3) / (s[l] - 1.0) / GRAVITY;
                    let w = if phi < 1.35 {
                        0.002 * phi.powf(7.5)
                    } else {
                        14.0 * (1.0 - 0.894 / phi.sqrt()).powf(4.5)
                    };

                    self.trans_rate_jli[[c, j, l]] = q * w;
                }
            }
        }

    }

    pub fn propagate_sediment(&mut self, t: f64) {

        // Sediment rates at all chainages: (nc, nbins, nlith)
        let bdy_sed_rate = self.sediment.value_at(t).into_dimensionality::<Ix3>().expect("sediment boundary rate must be 3D (nc, nbins, nlith)");

        let nc = self.nc;

        // println!("Nc is {nc}");

        // Calculate bed change rate at each downstream chainage
        //
        // NB: call this dy even though for this loop it is actually dy/dt
        // since that is what we need to figure out the next dt, but don't want
        // to allocate another array
        //
        // FIXME, there must be a way to optimize, we are caculating things twice
        let mut dy = Array1::<f64>::zeros(nc);
        for c in 0..nc {
            let mut up_sum = bdy_sed_rate.index_axis(ndarray::Axis(0), c).sum();
            if c > 0 {
                up_sum += self.trans_rate_jli.index_axis(ndarray::Axis(0), c-1).sum()
            }
            let my_sum = self.trans_rate_jli.index_axis(ndarray::Axis(0), c).sum();
            dy[c] = (up_sum - my_sum) * self.fact[c];
        }

        // update dt
        self.max_dydt = dy.iter().map(|v| v.abs()).fold(1e-10_f64, f64::max);
        self.dt = self.max_dt.min(self.cdt / self.max_dydt);

        // update dy (which is dy/dt) to actual dt
        dy *= self.dt;
        // now dy really is dy
        
        // aggrade, update bed levels
        for c in 0..nc {
            self.min_bed_level[c] += dy[c];
            self.mean_bed_level[c] += dy[c];
        }

        for c in 0..nc {

            // (nbins, nlith)
            let mut up_jli = bdy_sed_rate.index_axis(ndarray::Axis(0), c).to_owned();
            if c > 0 {
                up_jli += &self.trans_rate_jli.index_axis(ndarray::Axis(0), c-1);
            }

            // (nbins, nlith)
            let my_jli  = self.trans_rate_jli.index_axis(Axis(0), c);
            let my_sum = my_jli.sum();

            let fact = self.fact[c] * self.dt;
            let dy_i = dy[c];
            let athick = self.athickness[c];

            // split dy_i into the change in active layer thickness, and the remainder into the
            // shift in active layer (which modifies storage layer)
            let change_alayer: f64 = if dy_i > 0.0 {
                (self.alayer_max_thick - athick).max(0.0).min(dy_i)
            } else {
                -(athick.max(0.0).min(-dy_i))
            };
            let shift_alayer:f64 = dy_i - change_alayer;
            let new_athick = athick + change_alayer;

            if dy_i > 0.0 {
                // update storage using current aprop
                let aprop = self.aprop.index_axis(Axis(0), c);
                self.storage_layers.aggrade_at(c, &aprop, &my_jli, my_sum, self.chi, shift_alayer);

                // update aprop, doesn't need sprop
                let mut aprop = self.aprop.index_axis_mut(Axis(0), c);
                if new_athick > 0.0 {
                    Zip::from(&mut aprop)
                        .and(&up_jli)
                        .and(&my_jli)
                        .for_each(|a, &up, &my| {
                            let fi = {
                                if my_sum == 0.0 {
                                    *a  // FIXME, check this
                                } else {
                                    self.chi * *a + (1.0 - self.chi) * (my / my_sum)
                                }
                            };
                            let dy_jli = (up - my) * fact;
                            *a = (*a * athick + dy_jli - fi * shift_alayer) / new_athick;
                        });
                }

            } else if dy_i < 0.0 {
                let sprop = self.storage_layers.get_top_sprop_at(c);

                // update aprop
                let mut aprop = self.aprop.index_axis_mut(Axis(0), c);
                if new_athick > 0.0 {
                    Zip::from(&mut aprop)
                        .and(&up_jli)
                        .and(&my_jli)
                        .and(&sprop)
                        .for_each(|a, &up, &my, &s| {
                            let dy_jli = (up - my) * fact;
                            *a = (*a * athick + dy_jli - s * shift_alayer) / new_athick;
                        });
                }

                // update storage
                self.storage_layers.degrade_at(c, shift_alayer);
            }

            self.athickness[c] = new_athick;


        }
        
        // let text = self.aprop
        //     .iter()
        //     .map(|x| format!("{:.16e}", x))
        //     .collect::<Vec<_>>()
        //     .join("\n");
        // std::fs::write("/tmp/rust_array.txt", text).unwrap();


        

        // after grain deets
        self.set_grain_arrays();
    }

}

/// Sum over 3d axis (the lithologies) to get (nc, nbins) so the proportion of grains in each bin for
/// each chainage.  Then cumsum over the proportion of grains
fn cumf_lith(arr: &Array3<f64>) -> Array2<f64> {
    let mut cumf = arr.sum_axis(Axis(2));
    cumf.accumulate_axis_inplace(Axis(1), |&prev, curr| *curr += prev);
    return cumf;
}

/// Find x in xs and interpolate between two columns of arr for every row.
/// xs may be sorted ascending or descending.
fn cumf_smaller_than(x: f64, xs: &Array1<f64>, arr: &Array2<f64>, out: &mut Array1<f64>) {
    let n = xs.len();
    let asc = xs[0] < xs[n - 1];

    if (asc && x <= xs[0]) || (!asc && x >= xs[0]) {
        for c in 0..arr.nrows() {
            out[c] = arr[[c, 0]];
        }
        return;
    }

    if (asc && x >= xs[n - 1]) || (!asc && x <= xs[n - 1]) {
        for c in 0..arr.nrows() {
            out[c] = arr[[c, n - 1]];
        }
        return;
    }

    let pos = partition_point(xs.view(), x, asc);
    let fraction = (x - xs[pos - 1]) / (xs[pos] - xs[pos - 1]);

    for c in 0..arr.nrows() {
        let v0 = arr[[c, pos - 1]];
        let v1 = arr[[c, pos]];
        out[c] = v0 + (v1 - v0) * fraction;
    }
}

/// Find where x (0..1) lies in each row of cf (which bin) and return the interpolated grain size in
/// exp(-ys)
///
/// Equivalent to doing np.interp(x, cf, -log(rgsizes)) for each row in cf
fn percentile(x: f64, cf: &Array2<f64>, ys: &Array1<f64>, out: &mut Array1<f64>)
{
    assert!(0.0 <= x && x <= 1.0);
    let n = ys.len();

    // go by chainages
    for (c, row) in cf.rows().into_iter().enumerate() {
        let p = if x >= row[n - 1] {
            ys[n - 1]
        } else if x <= row[0] {
            ys[0]
        } else {
            // j is index of first value > x, minus 1
            let j = partition_point(row, x, true) - 1;

            let c0 = row[j];
            let c1 = row[j + 1];
            let p0 = ys[j];
            let p1 = ys[j + 1];

            p0 + (p1 - p0) / (c1 - c0) * (x - c0)
        };

        out[c] = (-p).exp();
    }
}

/// First value > x in sorted row, or len(row) if there isn't one
/// If asc assume row is ascending sorted so need <= compare, otherwise
/// >= compare
fn partition_point(row: ArrayView1<f64>, x: f64, asc: bool) -> usize {
    let (mut lo, mut hi) = (0, row.len());

    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if (asc && row[mid] <= x) || (!asc && row[mid] >= x) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }

    lo
}


#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_cumf_lith() {
        let a = array![
            [
                [1.0, 2.0, 3.0, 1.0],
                [4.0, 5.0, 6.0, 1.5],
            ],
            [
                [11.0, 12.0, 3.0, 0.6],
                [14.0, 15.0, 6.0, 0.4],
            ]
        ];
        let a = cumf_lith(&a);
        let r = array![
            [7.0, 23.5],
            [26.6, 62.0]
        ];
        assert_eq!(a, r);
    }

    #[test]
    fn test_sand_size() {
        let arr = array![
            [7.0, 9.0, 11.0],
            [26.0, 30.0, 40.0]
        ];
        let xs = array![1.0, 2.0, 4.0];
        let mut out = array![0.0, 0.0];
        cumf_smaller_than(1.5, &xs, &arr, &mut out);
        assert_eq!(out, array![8.0, 28.0]);

        cumf_smaller_than(0.5, &xs, &arr, &mut out);
        assert_eq!(out, array![7.0, 26.0]);

        cumf_smaller_than(1.0, &xs, &arr, &mut out);
        assert_eq!(out, array![7.0, 26.0]);

        cumf_smaller_than(3.0, &xs, &arr, &mut out);
        assert_eq!(out, array![10.0, 35.0]);

        cumf_smaller_than(5.0, &xs, &arr, &mut out);
        assert_eq!(out, array![11.0, 40.0]);

        // try descending
        let arr = array![
            [11.0, 9.0, 7.0],
            [40.0, 30.0, 26.0]
        ];
        let xs = array![4.0, 2.0, 1.0];
        cumf_smaller_than(3.0, &xs, &arr, &mut out);
        assert_eq!(out, array![10.0, 35.0]);

        cumf_smaller_than(5.0, &xs, &arr, &mut out);
        assert_eq!(out, array![11.0, 40.0]);
    }

    #[test]
    fn test_percentile() {
        let arr = array![
            [0.1, 0.5, 0.9, 1.0],
            [0.0, 0.8, 0.9, 0.95]
        ];
        let mut out = array![0.0, 0.0];
        let ys = array![1.0, 2.0, 3.0, 6.0];
        percentile(0.5, &arr, &ys, &mut out);
        assert_eq!(out, array![(-2.0_f64).exp(), (-1.625_f64).exp()]);

        percentile(0.0, &arr, &ys, &mut out);
        assert_eq!(out, array![(-1.0_f64).exp(), (-1.0_f64).exp()]);

        percentile(1.0, &arr, &ys, &mut out);
        assert_eq!(out, array![(-6.0_f64).exp(), (-6.0_f64).exp()]);

    }

    #[test]
    fn test_partition_point() {
        let a = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let p = partition_point(a.view(), -1.0, true);
        assert_eq!(p, 0);
        let p = partition_point(a.view(), 0.0, true);
        assert_eq!(p, 0);
        let p = partition_point(a.view(), 0.5, true);
        assert_eq!(p, 0);
        let p = partition_point(a.view(), 1.0, true);
        assert_eq!(p, 1);
        let p = partition_point(a.view(), 2.0, true);
        assert_eq!(p, 2);
        let p = partition_point(a.view(), 3.0, true);
        assert_eq!(p, 3);
        let p = partition_point(a.view(), 4.0, true);
        assert_eq!(p, 4);
        let p = partition_point(a.view(), 5.0, true);
        assert_eq!(p, 5);
        let p = partition_point(a.view(), 6.0, true);
        assert_eq!(p, 5);
        let a = array![5.0, 4.0, 3.0, 2.0, 1.0];
        let p = partition_point(a.view(), 2.0, false);
        assert_eq!(p, 4);
    }
}

