use clap::Parser;
use std::path::PathBuf;
use std::time::Instant;
use std::io::Write;
use anyhow::Result;
use config::Config;
use quasi_steady_state::QuasiSteadyState;
use channel::Channel;
use output::Output;

mod storage_layers;
mod output;
mod channel;
mod quasi_steady_state;
mod config;
mod utils;
mod interp_array;
mod downstream;

#[derive(Parser)]
#[command(name = "grater")]
struct Cli {
    /// Folder containing config.yaml
    idir: PathBuf,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let mut cfg = Config::load(&cli.idir)?;

    // make the channel, this gets everything set (including ng) but no gsv (since that needs
    // Sf/vel).  having ng is necessary for conveyance
    let mut channel = Channel::new(&mut cfg);

    // make and init hydro, so Sf/vel set
    let mut hmodel = QuasiSteadyState::new(&mut cfg)?;
    hmodel.initialize(cfg.simulation_time.start, &channel)?;

    // using Sf/vel we can get gsv, this allows transport
    channel.update_gsv(hmodel.get_sf(), hmodel.get_vel())?;

    let mut out = Output::new(&cfg, &channel)?;

    let mut do_sediment_transport: bool;
    let start: f64 = cfg.simulation_time.start;
    let end: f64 = cfg.simulation_time.end;
    let sim_total_seconds: f64 = end - start;
    let mut step: usize = 0;
    let mut t: f64 = start;
    let mut dt: f64 = channel.get_dt();
    let run_start = Instant::now();

    while t <= end {
        hmodel.update_water_level(t, &channel)?;

        // if above qthres increase dt and don't do sediment transport
        if hmodel.get_flow()[cfg.nc-1] < cfg.morphological.qthres {
            channel.set_dt(channel.max_dt * cfg.morphological.qthres_dtmultiplier);
            do_sediment_transport = false;
        } else {
            do_sediment_transport = true;
        }

        if do_sediment_transport {
            // set the gsv using Sf/vel from hydro, which allows transport
            channel.update_gsv(hmodel.get_sf(), hmodel.get_vel())?;
            channel.set_transport_rates(hmodel.get_wl());
        }
        if step % cfg.output.frequency == 0 {
           out.write_step(step, t, &hmodel, &channel)?;
        }
        if do_sediment_transport {
            // this updates bed_level and acfd, and sets d90 etc, but not gsv
            channel.propagate_sediment(t);
        }

        if step > 1 && step % 10 == 0 {
            let elapsed = run_start.elapsed().as_secs_f64();
            let sim_elapsed: f64 = t - start;
            let seconds_left: f64 = elapsed * (sim_total_seconds / sim_elapsed - 1.0);
            let finish = chrono::Local::now() + chrono::Duration::milliseconds((seconds_left * 1000.0) as i64);

            print!(
                "\rProgress={}%, dt={:.2}s st={} (est. finish at {})    ",
                (100.0 * sim_elapsed / sim_total_seconds) as u32,
                dt,
                if do_sediment_transport {"on "} else {"off"},
                finish.format("%Y-%m-%dT%H:%M:%S"),
            );
            std::io::stdout().flush()?;
        }

        step += 1;
        dt = channel.get_dt();
        t += dt;


    }

    println!();

    // write a final step even if not on the frequency
    if (step - 1) % cfg.output.frequency != 0 {
        out.write_step(step - 1, t - dt, &hmodel, &channel)?;
    }
    
    out.sync()?;

    Ok(())
}
