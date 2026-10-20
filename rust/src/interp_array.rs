use ndarray::{Array1, ArrayD, Axis};
use ndarray_npy::NpzReader;
use anyhow::{Context, Result};
use std::fs::File;
use std::path::PathBuf;

pub struct InterpArray {
    zero_below: bool,       // if value_at t < t.min(), return 0, otherwise v[0]
    t: Vec<f64>,
    v: ArrayD<f64>,
}

impl InterpArray {

    pub fn from_arrays(t: Array1<f64>, v: ArrayD<f64>, zero_below: bool) -> Self {
        assert!(!v.is_empty(), "Array 'v' must not be empty");
        assert!(v.ndim() >= 1,"Array 'v' must have at least 1D");
        assert!(
            t.is_empty() || t.len() == v.shape()[0],
            "Array 't' must be empty or have the same length as the first dimension of 'v'"
        );
        assert!(
            t.iter().all(|x| x.is_finite()),
            "Array 't' must contain only finite values"
        );
        assert!(
            t.windows(2).into_iter().all(|w| w[0] < w[1]),
            "Coordinates must be strictly increasing"
        );

        Self {
            zero_below,
            t: t.to_vec(),
            v,
        }
    }

    pub fn new(fname: &PathBuf, zero_below: bool) -> Result<Self> {
        let file = File::open(fname).with_context(|| format!("Failed to open {}", fname.display()))?;
        let mut npz = NpzReader::new(file)?;
        let t: Array1<f64> = npz.by_name("t")?;
        let v: ArrayD<f64> = npz.by_name("v")?;

        Ok(Self::from_arrays(t, v, zero_below))
    }

    pub fn value_at(&self, t: f64) -> ArrayD<f64> {

        // constant
        if self.t.is_empty() {
            return self.v.clone();
        }

        assert!(t.is_finite(), "Requested timestamp must be finite");

        let pos = self.t.partition_point(|&i| i < t);

        if pos < self.t.len() && self.t[pos] == t {
            // Exact match
            return self.v.index_axis(Axis(0), pos).to_owned();
        } else if pos == 0 {
            // t is below the first element
            if self.zero_below {
                return ArrayD::zeros(self.v.index_axis(Axis(0), 0).raw_dim());
            } else {
                return self.v.index_axis(Axis(0), pos).to_owned();
            }
        } else if pos == self.t.len() {
            return self.v.index_axis(Axis(0), pos - 1).to_owned();
        } 

        // linear interp
        let t0 = self.t[pos - 1];
        let t1 = self.t[pos];
        let fraction = (t - t0) / (t1 - t0);

        let v0 = self.v.index_axis(Axis(0), pos - 1);
        let v1 = self.v.index_axis(Axis(0), pos);

        &v0 + &((&v1 - &v0) * fraction)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Ix0, Ix1, Ix2};

    #[test]
    fn test_value_at_dimensions() {
        // 1D v -> scalar result (0D)
        let t = array![0.0, 10.0];
        let v = array![2.0, 6.0].into_dyn();
        let a = InterpArray::from_arrays(t, v, false);
        assert_eq!(a.value_at(5.0).into_dimensionality::<Ix0>().unwrap()[()], 4.0);

        // 1D v -> scalar result (0D)
        let t = array![0.0, 10.0];
        let v = array![2.0, 6.0].into_dyn();
        let a = InterpArray::from_arrays(t, v, true);
        assert_eq!(a.value_at(-1.0).into_dimensionality::<Ix0>().unwrap()[()], 0.0);
        assert_eq!(a.value_at(11.0).into_dimensionality::<Ix0>().unwrap()[()], 6.0);

        // 1D v -> scalar result (0D)
        let t = array![0.0, 10.0];
        let v = array![2.0, 6.0].into_dyn();
        let a = InterpArray::from_arrays(t, v, false);
        assert_eq!(a.value_at(-1.0).into_dimensionality::<Ix0>().unwrap()[()], 2.0);

        // 2D v -> 1D result
        let t = array![0.0, 10.0];
        let v = array![[1.0, 2.0], [3.0, 4.0]].into_dyn();
        let a = InterpArray::from_arrays(t, v, false);
        assert_eq!(
            a.value_at(5.0).into_dimensionality::<Ix1>().unwrap(),
            array![2.0, 3.0]
        );
        assert_eq!(
            a.value_at(-5.0).into_dimensionality::<Ix1>().unwrap(),
            array![1.0, 2.0]
        );

        let t = array![0.0, 10.0];
        let v = array![[1.0, 2.0], [3.0, 4.0]].into_dyn();
        let a = InterpArray::from_arrays(t, v, true);
        assert_eq!(
            a.value_at(-5.0).into_dimensionality::<Ix1>().unwrap(),
            array![0.0, 0.0]
        );
        assert_eq!(
            a.value_at(15.0).into_dimensionality::<Ix1>().unwrap(),
            array![3.0, 4.0]
        );

        // 3D v -> 2D result
        let t = array![0.0, 10.0];
        let v = array![
            [[1.0, 2.0], [3.0, 4.0]],
            [[3.0, 4.0], [5.0, 6.0]]
        ].into_dyn();
        let a = InterpArray::from_arrays(t, v, false);
        assert_eq!(
            a.value_at(5.0).into_dimensionality::<Ix2>().unwrap(),
            array![[2.0, 3.0], [4.0, 5.0]]
        );
    }
}

