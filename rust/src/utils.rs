use std::fs;
use std::path::PathBuf;

use anyhow::{bail, Result};
use saphyr::{LoadableYamlNode, ScalarOwned, YamlOwned};
use rootseek::brent;

const MAXITS: u32 = 100;
const MAX_GROWTH: f64 = 100.0;

fn prefix_npz_paths(node: &mut YamlOwned, dir: &PathBuf) {
    match node {
        YamlOwned::Mapping(mapping) => {
            for value in mapping.values_mut() {
                prefix_npz_paths(value, dir);
            }
        }

        YamlOwned::Sequence(sequence) => {
            for value in sequence {
                prefix_npz_paths(value, dir);
            }
        }

        YamlOwned::Value(ScalarOwned::String(s))
        | YamlOwned::Representation(s, _, _) => {
            if s.ends_with(".npz") {
                *s = dir.join(&*s).to_string_lossy().into_owned();
            }
        }

        YamlOwned::Tagged(_, value) => {
            prefix_npz_paths(value, dir);
        }

        _ => {}
    }
}

pub fn read_config(dir: &PathBuf) -> Result<YamlOwned> {
    let path = dir.join("config.yaml");
    let contents = fs::read_to_string(&path)?;
    let mut docs = YamlOwned::load_from_str(&contents)?;
    if docs.is_empty() {
        bail!("{} contains no YAML document", path.display());
    }
    let mut config = docs.remove(0);
    prefix_npz_paths(&mut config, dir);
    Ok(config)
}


fn get_bracket<F>(f: F, x0: f64, lower: f64, steps: u32) -> Result<Option<(f64, f64)>>
where
    F: Fn(f64) -> f64,
{

    if x0 <= lower {
        bail!("x0 must be > {lower}, got {x0}");
    }

    let fx0 = f(x0);
    if fx0 == 0.0 {
        return Ok(Some((x0, x0)));
    }

    let dx = (x0 - lower) / steps as f64;
    let mut x_prev = x0;
    let mut f_prev = fx0;

    // Search left
    let mut x = x0 - dx;
    while x > lower {
        let fx = f(x);
        if fx == 0.0 {
            return Ok(Some((x, x)));
        }
        if (fx > 0.0) != (f_prev > 0.0) {
            return Ok(Some((x, x_prev)));
        }
        x_prev = x;
        f_prev = fx;
        x -= dx;
    }

    // Search right
    x_prev = x0;
    f_prev = fx0;
    let x_max = x0 + (x0 - lower);
    let mut x = x0 + dx;
    while x < x_max {
        let fx = f(x);

        if fx == 0.0 {
            return Ok(Some((x, x)));
        }
        if (fx > 0.0) != (f_prev > 0.0) {
            return Ok(Some((x_prev, x)));
        }

        x_prev = x;
        f_prev = fx;
        x += dx;
    }

    Ok(None)
}

fn get_bracket_for_monotonic<F>(f: F, x0: f64, lower: f64, upper: f64, maxits: u32) -> Result<Option<(f64, f64)>>
where
    F: Fn(f64) -> f64,
{
    if x0 <= lower {
        bail!("x0 must be > {lower}, got {x0}");
    }

    let fx0 = f(x0);
    if fx0 == 0.0 {
        return Ok(Some((x0, x0)));
    }

    // Determine whether f is increasing or decreasing.
    let dx = 0.1 * (x0 - lower);
    let fx = f(x0 + dx);
    if fx == fx0 {
        bail!("Cannot determine f' at x0={x0} because f is flat");
    }

    let increasing = fx > fx0;
    let search_left = (fx0 > 0.0) == increasing;

    // Try a quick estimate using the derivative
    let fprime = (fx - fx0) / dx;
    let x = x0 - 2.0 * fx0 / fprime;

    if x > lower {
        let fx = f(x);
        if (fx <= 0.0) == (fx0 > 0.0) {
            return Ok(Some(if search_left { (x, x0) } else { (x0, x) }));
        }
    }

    // Fall back to repeated halving or doubling
    let mut x = x0;

    for _ in 0..maxits {
        x = if search_left {
            (x + lower) / 2.0
        } else {
            lower + 2.0 * (x - lower)
        };

        if x <= lower || x > upper {
            return Ok(None);
        }

        let fx = f(x);
        if (fx <= 0.0) == (fx0 > 0.0) {
            return Ok(Some(if search_left { (x, x0) } else { (x0, x) }));
        }
    }

    Ok(None)
}

pub fn find_root<F>(f: F, x0: f64, lower: f64) -> Result<f64>
where
    F: Fn(f64) -> f64,
{
    let bracket = get_bracket_for_monotonic(&f, x0, lower, lower+MAX_GROWTH, MAXITS)?;
    let bracket = match bracket {
        Some(b) => b,
        None => {
            eprintln!("WARNING: can't find bracket assuming monotonic f at {x0}");
            match get_bracket(&f, x0, lower, 1000)? {
                Some(b) => {
                    eprintln!("Got a bracket using the exhaustive approach");
                    b
                }
                None => bail!("Cannot find bracket for root finding at {x0}"),
            }
        }
    };

    let (a, b) = bracket;

    if a == b {
        return Ok(a);
    }

    let root = brent(f, a, b, 1e-12, MAXITS)?;
    Ok(root.x)
}


