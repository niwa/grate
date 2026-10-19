use anyhow::Result;
use lambert_w::lambert_w0;
use ndarray::{ArrayView1, Zip, Axis, Array1, Array2, Array3, Ix3};
use crate::config::{Config, CrossSectionTab};
use crate::interp_array::InterpArray;

const SAND_SIZE: f64 = 0.002;       // sand grain size in metres
const GRAVITY: f64 = 9.81;
const KAPPA: f64 = 0.4;             // Von Karman's constant
const WATER_DENSITY: f64 = 1000.0;

pub struct Channel {
    nc: usize,
    cs: Array1<f64>,                    // (nc, )

    // time stepping
    dt: f64,
    pub max_dt: f64,
    cdt: f64,
    max_dydt: f64,

    poro: f64,                          // porosity
    sand_size: f64,                     // sand grain size in metres in phi units
    morphla: f64,                       // self._cfg.morphological.la
    chi: f64,                           // self._cfg.morphological.chi
    sediment_densities: Array1<f64>,    // (nlith, )
    min_bed_level: Array1<f64>,         // (nc, )
    mean_bed_level: Array1<f64>,        // (nc, )
    fact: Array1<f64>,                  // (nc-1,), cache 1 / (cs[i + 1] - cs[i]) / (1 - poro) / bchan[i + 1]

    sediment: InterpArray,
    ng: Array2<f64>,                    // (nc, 3)

    // grain info
    rgsizes: Array1<f64>,               // (nbins, )
    phi: Array1<f64>,                   // (bbins, ) grain size in phi units, used for percentiles
    sand_fraction: Array1<f64>,         // (nc, ) how much sand in this chainage
    dsm: Array1<f64>,                   // (nc, ) median of grain size in channel
    d90_channel: Array1<f64>,           // (nc, ) 90 percentile grain size in channel
    d90: Array2<f64>,                   // (nc, 3 ) 90 percentile grain size left/channel/right
    bankd90: Array1<f64>,               // (nc, ) 90 percentile grain size on bank (if nan use channel)
    gsv: Array1<f64>,                   // (nc, ) 
    acfd: Array3<f64>,                  // (nc, nbins, nlith)
    scfd: Array3<f64>,                  // (nc, nbins, nlith)
                                        
    // transport arrays
    trans_rate_jli: Array3<f64>,        // (nc, nbins, nlith)
    cum_transport: Array3<f64>,         // (nc, nbins, nlith)
                                        
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
            (0..cfg.nc - 1).map(|i| {
                1.0 / (cfg.cs[i + 1] - cfg.cs[i]) / (1.0 - cfg.morphological.poro) / cfg.bchan[i + 1]
            })
        );

        let mut chan = Self {
            nc: cfg.nc,
            cs: cfg.cs.clone(),

            // time stepping
            dt: cfg.simulation_time.max_dt,
            max_dt: cfg.simulation_time.max_dt,
            cdt: cfg.simulation_time.cdt,
            max_dydt: 0.1 * cfg.simulation_time.cdt / cfg.simulation_time.max_dt,

            poro: cfg.morphological.poro,
            sand_size: -SAND_SIZE.ln(),
            morphla: cfg.morphological.la,
            chi: cfg.morphological.chi,
            sediment_densities: cfg.sediment_densities.clone(),
            min_bed_level: cfg.min_bed_level.clone(),
            mean_bed_level: cfg.mean_bed_level.clone(),
            fact: fact,

            sediment: cfg.sediment.take().unwrap(),
            ng: Array2::<f64>::zeros((cfg.nc, 3)),              // (nc, 3)
                
            rgsizes: cfg.rgsizes.clone(),
            phi: cfg.rgsizes.map(|x| -x.ln()),
            sand_fraction: Array1::<f64>::zeros(cfg.nc),
            dsm: Array1::<f64>::ones(cfg.nc),
            d90: Array2::<f64>::ones((cfg.nc, 3)),
            d90_channel: Array1::<f64>::ones(cfg.nc),
            bankd90: cfg.bankd90.clone(),
            gsv: Array1::<f64>::ones(cfg.nc),
            acfd: cfg.acfd.clone(),
            scfd: cfg.scfd.clone(),

            // transport arrays
            trans_rate_jli: Array3::<f64>::zeros(cfg.acfd.raw_dim()),
            cum_transport: Array3::<f64>::zeros(cfg.acfd.raw_dim()),

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

    pub fn calc_area(&self, c: usize, wl: f64) -> Array1<f64> {
        let tab = &self.area_tab[c];
        let d = wl - self.min_bed_level[c];
        Array1::from_vec(vec![
            tab.0.as_ref().map_or(0.0, |v| v.value_at(d).into_dimensionality::<ndarray::Ix0>().unwrap()[()]),
            tab.1.as_ref().map_or(0.0, |v| v.value_at(d).into_dimensionality::<ndarray::Ix0>().unwrap()[()]),
            tab.2.as_ref().map_or(0.0, |v| v.value_at(d).into_dimensionality::<ndarray::Ix0>().unwrap()[()]),
        ])
    }

    pub fn calc_bwet(&self, c: usize, wl: f64) -> f64 {
        let tab = &self.bwet_tab[c];
        return 
            tab.0.as_ref().map_or(0.0, |v| v.value_at(wl).into_dimensionality::<ndarray::Ix0>().unwrap()[()]) +
            tab.1.as_ref().map_or(0.0, |v| v.value_at(wl).into_dimensionality::<ndarray::Ix0>().unwrap()[()]) +
            tab.2.as_ref().map_or(0.0, |v| v.value_at(wl).into_dimensionality::<ndarray::Ix0>().unwrap()[()]);
    }

    fn calc_p(&self, c: usize, wl: f64) -> Array1<f64> {
        let tab = &self.p_tab[c];
        Array1::from_vec(vec![
            tab.0.as_ref().map_or(0.0, |v| v.value_at(wl).into_dimensionality::<ndarray::Ix0>().unwrap()[()]),
            tab.1.as_ref().map_or(0.0, |v| v.value_at(wl).into_dimensionality::<ndarray::Ix0>().unwrap()[()]),
            tab.2.as_ref().map_or(0.0, |v| v.value_at(wl).into_dimensionality::<ndarray::Ix0>().unwrap()[()]),
        ])
    }

    fn calc_nf(&self, c: usize, wl: f64) -> Array1<f64> {
        let tab = &self.nf_tab[c];
        Array1::from_vec(vec![
            tab.0.as_ref().map_or(0.0, |v| v.value_at(wl).into_dimensionality::<ndarray::Ix0>().unwrap()[()]),
            tab.1.as_ref().map_or(0.0, |v| v.value_at(wl).into_dimensionality::<ndarray::Ix0>().unwrap()[()]),
            tab.2.as_ref().map_or(0.0, |v| v.value_at(wl).into_dimensionality::<ndarray::Ix0>().unwrap()[()]),
        ])
    }

    pub fn calc_conveyance(&self, c: usize, wl: f64) -> f64 {
        let a = self.calc_area(c, wl);
        let p = self.calc_p(c, wl);
        let nf = self.calc_nf(c, wl);

        // println!("c={} wl={} a={:?}, p={:?}, nf={:?}", c, wl, a, p, nf);

        let k: f64 = (0..3)
            .filter(|&i| p[i] != 0.0)
            .map(|i| a[i] * (a[i] / p[i]).powf(2.0 / 3.0) / (self.ng[(c, i)] + nf[i]))
            .sum();

        assert!(k > 0.0, "No water");
        k
    }

    /// Set sand_fraction/dsm/d90/ng which depend on acfd
    fn set_grain_arrays(&mut self) {

        // sum over nlith axis, to get (nc, nbins), then cumsum down nbins axis
        let cumf: Array2<f64> = cumf_lith(&self.acfd);

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

        // now multiply by bwet
        for c in 0..self.nc {
            let bwet = self.calc_bwet(c, wl[c]);
            self.trans_rate_jli
                .index_axis_mut(Axis(0), c)
                .mapv_inplace(|q| q * bwet);
        }

        // the cumulative transport rate
        self.cum_transport.scaled_add(self.dt, &self.trans_rate_jli);
    }

    // Wilcock & Crowe (2003) eq 9.33 for every chainage at once.
    //
    // Updates trans_rate_jli
    pub fn qb_jli(&mut self) {
        let nc = self.nc;
        let nbins = self.rgsizes.len();
        let nlith = self.sediment_densities.len();

        let s = &self.sediment_densities / WATER_DENSITY;   // (nlith, )

        for c in 0..nc {
            let phirm = 0.021 + 0.015 * (-20.0 * self.sand_fraction[c]).exp();

            for j in 0..nbins {
                let ratio = self.rgsizes[j] / self.dsm[c];
                let b = 0.67 / (1.0 + (1.5 - ratio).exp());

                for l in 0..nlith {
                    let tau_rm = phirm * (s[l] - 1.0) * WATER_DENSITY * GRAVITY * self.dsm[c];
                    let tau_rj = tau_rm * ratio.powf(b);
                    let phi = WATER_DENSITY * self.gsv[c].powi(2) / tau_rj;

                    let q = self.acfd[[c, j, l]] * self.gsv[c].powi(3) / (s[l] - 1.0) / GRAVITY;
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

        // Calculate bed change rate at each downstream chainage
        //
        // NB: call this dy even though for this loop it is actually dy/dt
        // since that is what we need to figure out the next dt, but don't want
        // to allocate another array
        //
        let mut dy = Array1::<f64>::zeros(nc - 1); // dy[0] is change at chainage 1
        for i in 0..nc - 1 {
            let up_sum = self.trans_rate_jli.index_axis(ndarray::Axis(0), i).sum() + bdy_sed_rate.index_axis(ndarray::Axis(0), i).sum();
            let my_sum = self.trans_rate_jli.index_axis(ndarray::Axis(0), i + 1).sum();
            dy[i] = (up_sum - my_sum) * self.fact[i];
        }

        // update dt
        self.max_dydt = dy.iter().map(|v| v.abs()).fold(1e-10_f64, f64::max);
        self.dt = self.max_dt.min(self.cdt / self.max_dydt);

        // update dy (which is dy/dt) to actual dt
        dy *= self.dt;
        // now dy really is dy
        
        // aggrade, update bed levels
        for i in 0..nc - 1 {
            self.min_bed_level[i+1] += dy[i];
            self.mean_bed_level[i+1] += dy[i];
        }
        
        for i in 0..nc - 1 {
            let my_sum = self.trans_rate_jli.index_axis(ndarray::Axis(0), i + 1).sum();

            // Bed has aggraded, but composition cannot change
            if my_sum == 0.0 {
                eprintln!("WARNING: at {i}, total_trans_rate=0, did bed aggregation, but skipping composition change");
                continue;
            }

            let fact = self.fact[i] * self.dt;
            let dy_i = dy[i];
            let aggrading = dy_i > 0.0;
            let (chi, morphla) = (self.chi, self.morphla);

            let up_jli  = self.trans_rate_jli.index_axis(Axis(0), i);
            let bdy_jli = bdy_sed_rate.index_axis(Axis(0), i);
            let my_jli  = self.trans_rate_jli.index_axis(Axis(0), i + 1);
            let scfd  = self.scfd.index_axis(Axis(0), i + 1);
            let mut acfd = self.acfd.index_axis_mut(Axis(0), i + 1);

            Zip::from(&mut acfd)
                .and(&up_jli)
                .and(&bdy_jli)
                .and(&my_jli)
                .and(&scfd)
                .for_each(|a, &up, &bdy, &my, &s| {
                    let f_int = if aggrading {
                        chi * *a + (1.0 - chi) * (my / my_sum)
                    } else {
                        s
                    };
                    let flux_diff = (up + bdy - my) * fact;
                    *a += (flux_diff - f_int * dy_i) / morphla;
                });


        }

        // FIXME, change storage layer f_jli
        
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

// fn cumf_smaller_than(x: f64, xs: &Array1<f64>, arr: &Array2<f64>) -> Array1<f64> {
//     let pos = xs.iter().position(|&v| v >= x);
//
//     let Some(pos) = pos else {
//         return arr.column(xs.len() - 1).to_owned();
//     };
//
//     if pos == 0 {
//         return arr.column(0).to_owned();
//     }
//
//     if xs[pos] == x {
//         return arr.column(pos).to_owned();
//     }
//
//     let fraction = (x - xs[pos - 1]) / (xs[pos] - xs[pos - 1]);
//     &arr.column(pos - 1) + &((&arr.column(pos) - &arr.column(pos - 1)) * fraction)
// }

/// Find x in xs and interpolate between two columns of 2d array for every row in arr
fn cumf_smaller_than(x: f64, xs: &Array1<f64>, arr: &Array2<f64>, out: &mut Array1<f64>) {
    let n = xs.len();

    if x <= xs[0] {
        for c in 0..arr.nrows() {
            out[c] = arr[[c, 0]];
        }
        return;
    } else if x >= xs[n-1] {
        for c in 0..arr.nrows() {
            out[c] = arr[[c, n-1]];
        }
        return;
    }

    let pos = partition_point(xs.view(), x);
    let fraction = (x - xs[pos-1]) / (xs[pos] - xs[pos-1]);
    for c in 0..arr.nrows() {
        let v0 = arr[[c, pos-1]];
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
            let j = partition_point(row, x) - 1;

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
fn partition_point(row: ArrayView1<f64>, x: f64) -> usize {
    let (mut lo, mut hi) = (0, row.len());

    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if row[mid] <= x {
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
        let p = partition_point(a.view(), -1.0);
        assert_eq!(p, 0);
        let p = partition_point(a.view(), 0.0);
        assert_eq!(p, 0);
        let p = partition_point(a.view(), 0.5);
        assert_eq!(p, 0);
        let p = partition_point(a.view(), 1.0);
        assert_eq!(p, 1);
        let p = partition_point(a.view(), 2.0);
        assert_eq!(p, 2);
        let p = partition_point(a.view(), 3.0);
        assert_eq!(p, 3);
        let p = partition_point(a.view(), 4.0);
        assert_eq!(p, 4);
        let p = partition_point(a.view(), 5.0);
        assert_eq!(p, 5);
        let p = partition_point(a.view(), 6.0);
        assert_eq!(p, 5);
    }
}

