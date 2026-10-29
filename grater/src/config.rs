use std::path::PathBuf;
use std::fs::File;

use anyhow::{Context, Result};
use ndarray_npy::NpzReader;
use saphyr::YamlOwned;
use ndarray::{Array1, Array2, Array3, Array4};

use crate::downstream::Downstream;
use crate::utils::read_config;
use crate::interp_array::InterpArray;

pub struct Config {
    pub morphological: Morphological,
    pub simulation_time: SimulationTime,
    pub output: Output,
    pub cs: Array1<f64>,                            // (nc,)
    pub nc: usize,
    pub nlayer: usize,
    pub bchan: Array1<f64>,                         // (nc,)
    pub min_bed_level: Array1<f64>,                 // (nc,)
    pub mean_bed_level: Array1<f64>,                // (nc,)
    pub bedrock: Array1<f64>,                       // (nc,)
    pub bankd90: Array1<f64>,                       // (nc,)
    pub rgsizes: Array1<f64>,                       // (nbins,)
    pub uppersizes: Array1<f64>,                    // (nbins,)
    pub sediment_densities: Array1<f64>,            // (nlith,)
    pub sediment: Option<InterpArray>,              // time, (nc, nbins, nlith), option so channel can take
    pub inflow: Option<InterpArray>,                // time, (nc, ), flow, option so hmodel can take
    pub downstream: Option<Downstream>,             // gives downstream flow, option so hmodel can take
    pub aprop: Array3<f64>,                         // (nc, nbins, nlith)
    pub sprop: Array4<f64>,                         // (nc, nlayers, nbins, nlith)
    pub athickness: Array1<f64>,                    // (nc, )
    pub sthickness: Array2<f64>,                    // (nc, nlayers)
    pub xstabs: Option<CrossSectionTabs>,           // (nc, ) four tables for each chainage, option so channel take
}

pub type CrossSectionTab = (
    Option<InterpArray>,
    Option<InterpArray>,
    Option<InterpArray>,
);

pub struct CrossSectionTabs {
    pub area: Vec<CrossSectionTab>,
    pub p: Vec<CrossSectionTab>,
    pub bwet: Vec<CrossSectionTab>,
    pub nf: Vec<CrossSectionTab>,
}

pub struct Morphological {
    pub alayer_max_thick: f64,
    pub slayer_max_thick: f64,
    pub beta: f64,
    pub chi: f64,
    pub poro: f64,
    pub qthres: f64,
    pub qthres_dtmultiplier: f64,
}

pub struct SimulationTime {
    pub start: f64,
    pub end: f64,
    pub max_dt: f64,
    pub cdt: f64,
}

pub struct Output {
    pub frequency: usize,
    pub fname: PathBuf,
    pub variables: Vec<String>,
}

fn read_tabs(dir: &PathBuf, fname: &str, nc: usize) -> Result<Vec<CrossSectionTab>> {
    let fname = config_path(dir, fname);
    let file = File::open(&fname).with_context(|| format!("Failed to open {}", fname.display()))?;
    let mut npz = NpzReader::new(file)?;
    let arrs = npz.names()?;

    let mut tabs = Vec::with_capacity(nc);

    for i in 0..nc {
        let mut read = |j| -> Result<Option<InterpArray>> {
            let name = format!("{i}_{j}");
            if !arrs.contains(&name) {
                return Ok(None);
            }

            let a: Array2<f64> = npz.by_name::<ndarray::OwnedRepr<f64>, ndarray::Ix2>(&name)?;

            let t = a.column(0).to_owned();
            let v = a.column(1).to_owned().into_dyn();

            Ok(Some(InterpArray::from_arrays(t, v, true)))
        };
        tabs.push((read(0)?, read(1)?, read(2)?));
    }

    Ok(tabs)
}

impl Config {
    pub fn load(dir: &PathBuf) -> Result<Self> {
        let yaml = read_config(dir)?;

        let fname = config_path(dir, "profiles.npz");
        let file = File::open(&fname).with_context(|| format!("Failed to open {}", fname.display()))?;
        let mut profiles = NpzReader::new(file)?;
        let cs = read_array1(&mut profiles, "cs");
        let nc: usize = cs.len();
        
        let fname = config_path(dir, "grain.npz");
        let file = File::open(&fname).with_context(|| format!("Failed to open {}", fname.display()))?;
        let mut grain = NpzReader::new(file)?;
        
        let morph = get(&yaml, "morphological");
        let st = get(&yaml, "simulation_time");
        let oput = get(&yaml, "output");

        let tabs = CrossSectionTabs {
            area: read_tabs(dir, "area_tabs.npz", nc)?,
            p: read_tabs(dir, "P_tabs.npz", nc)?,
            bwet: read_tabs(dir, "Bwet_tabs.npz", nc)?,
            nf: read_tabs(dir, "nf_tabs.npz", nc)?,
        };

        let sprop = read_array4(&mut grain, "sprop");

        Ok(
            Self {
                morphological: Morphological {
                    alayer_max_thick: number(morph, "alayer_max_thick"),
                    slayer_max_thick: number(morph, "slayer_max_thick"),
                    beta: number(morph, "beta"),
                    chi: number(morph, "chi"),
                    poro: number(morph, "poro"),
                    qthres: number(morph, "qthres"),
                    qthres_dtmultiplier: number(morph, "qthres_dtmultiplier"),
                },
                simulation_time: SimulationTime {
                    start: number(st, "start"),
                    end: number(st, "end"),
                    max_dt: number(st, "max_dt"),
                    cdt: number(st, "cdt"),
                },
                output: Output {
                    frequency: integer(oput, "frequency") as usize,
                    fname: path(oput, "fname"),
                    variables: sequence(oput, "variables"),
                },
                cs: cs,
                nc,
                nlayer: sprop.shape()[1],
                bchan: read_array1(&mut profiles, "bchan"),
                min_bed_level: read_array1(&mut profiles, "min_bed_level"),
                mean_bed_level: read_array1(&mut profiles, "mean_bed_level"),
                bedrock: read_array1(&mut profiles, "bedrock"),
                bankd90: read_array1(&mut grain, "bankd90"),
                rgsizes: read_array1(&mut grain, "rgsizes"),
                uppersizes: read_array1(&mut grain, "uppersizes"),
                sediment_densities: read_array1(&mut grain, "sediment_densities"),
                sediment: Some(InterpArray::new(&config_path(dir, "sediment.npz"), false)?),
                inflow: Some(InterpArray::new(&config_path(dir, "inflow.npz"), false)?),
                downstream: Some(Downstream::new(get(&yaml, "downstream"))?),
                aprop: read_array3(&mut grain, "aprop"),
                sprop: sprop,
                athickness: read_array1(&mut grain, "athickness"),
                sthickness: read_array2(&mut grain, "sthickness"),
                xstabs: Some(tabs),
            }
        )
    }
}

fn get<'a>(node: &'a YamlOwned, key: &str) -> &'a YamlOwned {
    node.as_mapping_get(key)
        .unwrap_or_else(|| panic!("Missing config key '{key}'"))
}

fn string(node: &YamlOwned, key: &str) -> String {
    get(node, key).as_str().unwrap().to_string()
}

fn path(node: &YamlOwned, key: &str) -> PathBuf {
    PathBuf::from(string(node, key))
}

fn number(node: &YamlOwned, key: &str) -> f64 {
    get(node, key).as_floating_point().unwrap()
}

fn integer(node: &YamlOwned, key: &str) -> i64 {
    get(node, key).as_integer().unwrap()
}

fn sequence(node: &YamlOwned, key: &str) -> Vec<String> {
    get(node, key).as_sequence().unwrap().iter().map(|v| v.as_str().unwrap().to_string()).collect()
}

fn config_path(dir: &PathBuf, fname: &str) -> PathBuf {
    dir.join(fname)
}

fn read_array1(npz: &mut NpzReader<File>, name: &str) -> Array1<f64> {
    npz.by_name(name)
        .unwrap_or_else(|e| panic!("Failed to read '{name}': {e}"))
}

fn read_array2(npz: &mut NpzReader<File>, name: &str) -> Array2<f64> {
    npz.by_name(name)
        .unwrap_or_else(|e| panic!("Failed to read '{name}': {e}"))
}

fn read_array3(npz: &mut NpzReader<File>, name: &str) -> Array3<f64> {
    npz.by_name(name)
        .unwrap_or_else(|e| panic!("Failed to read '{name}': {e}"))
}

fn read_array4(npz: &mut NpzReader<File>, name: &str) -> Array4<f64> {
    npz.by_name(name)
        .unwrap_or_else(|e| panic!("Failed to read '{name}': {e}"))
}

