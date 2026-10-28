//! Read-only PDH sampler: default is THIS probe process, not another Mimic PID.
use anyhow::{Result, bail};
use clap::Parser;
use engine::gpu::GpuMeter;
use std::time::{Duration, Instant};

#[derive(Parser)]
#[command(
    about = "Print per-process busiest 3D/Compute engine contribution; no GPU workload is created"
)]
struct Args {
    /// Explicit target process. By default only this gpu_probe process is sampled.
    #[arg(long)]
    pid: Option<u32>,
    #[arg(long, default_value_t = 5.0)]
    seconds: f64,
    #[arg(long, default_value_t = 250, value_parser = clap::value_parser!(u64).range(200..=10000))]
    interval_ms: u64,
}
fn main() -> Result<()> {
    let args = Args::parse();
    if !args.seconds.is_finite() || args.seconds <= 0.0 || args.seconds > 3600.0 {
        bail!("--seconds must be in (0,3600]");
    }
    let target = args.pid.unwrap_or_else(std::process::id);
    println!(
        "target_pid={target}; scope=process contribution to busiest 3D/Compute physical engine; this program creates no GPU workload"
    );
    let Some(mut meter) = (if args.pid.is_some() {
        GpuMeter::for_pid(target)
    } else {
        GpuMeter::new()
    }) else {
        println!(
            "GPU telemetry unavailable (PDH not supported/accessible); no NVML fallback installed"
        );
        return Ok(());
    };
    let begin = Instant::now();
    let mut first_snapshot = true;
    while begin.elapsed().as_secs_f64() < args.seconds {
        std::thread::sleep(Duration::from_millis(args.interval_ms));
        match meter.sample() {
            Some(value) => {
                let snapshot = meter.latest().expect("sample published a snapshot");
                println!(
                    "t={:.3}s gpu={:.3}% 3D={:.3}% Compute={:.3}% engines={}",
                    begin.elapsed().as_secs_f64(),
                    value * 100.0,
                    snapshot.three_d * 100.0,
                    snapshot.compute * 100.0,
                    snapshot.engines.len()
                );
                for item in &snapshot.engines {
                    if item.utilization > 0.0 || first_snapshot {
                        println!("  {} {:.3}%", item.instance, item.utilization * 100.0);
                    }
                }
                first_snapshot = false;
            }
            None => println!(
                "t={:.3}s unavailable/warming-up",
                begin.elapsed().as_secs_f64()
            ),
        }
    }
    Ok(())
}
