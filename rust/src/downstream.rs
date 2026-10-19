use anyhow::Result;
use crate::interp_array::InterpArray;
use std::path::PathBuf;
use saphyr::YamlOwned;

pub struct Downstream {
    pub depth: Option<f64>,                
    pub slope: Option<f64>,
    pub wl_init: Option<f64>,
    pub elevation: Option<InterpArray>,
}

impl Downstream {
    pub fn new(config: &YamlOwned) -> Result<Self> {

        let get = |key: &str| config.as_mapping_get(key).unwrap();
        let get_f64 = |key: &str| get(key).as_floating_point().unwrap();
        let kind = get("type").as_str().unwrap().to_string();

        let mut ds = Self {
            depth: None,
            slope: None,
            wl_init: None,
            elevation: None,
        };

        match kind.as_str() {
            "depth" => ds.depth = Some(get_f64("value")),
            "normal" => {
                ds.slope = Some(get_f64("slope"));
                ds.wl_init = Some(get_f64("wl_init"));
            },
            "elevation" => {
                let fname = PathBuf::from(get("fname").as_str().unwrap());
                ds.elevation = Some(InterpArray::new(&fname)?);
            },
            _ => panic!("Unknown downstream boundary type '{kind}'"),
        }

        Ok(ds)
    }

    pub fn value_at(&self, t: f64) -> f64 {
        // FIXME, should panic if elevation is None???
        let values = self.elevation.as_ref().unwrap().value_at(t);
        assert_eq!(values.len(), 1, "Downstream elevation data must contain exactly one value");
        values.iter().next().copied().unwrap()
    }
}


