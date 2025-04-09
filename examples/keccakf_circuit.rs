use pyroscope::PyroscopeAgent;
use pyroscope_pprofrs::{pprof_backend, PprofConfig};

use std::vec;

use anyhow::Result;
use binius_circuits::builder::{types::U, ConstraintSystemBuilder};
use binius_core::{constraint_system, fiat_shamir::HasherChallenger, tower::CanonicalTowerFamily};
use binius_hal::make_portable_backend;
use binius_hash::groestl::{Groestl256, Groestl256ByteCompression};
use binius_utils::{checked_arithmetics::log2_ceil_usize, rayon::adjust_thread_pool};
use bytesize::ByteSize;
use clap::{value_parser, Parser};
use tracing_profile::init_tracing;

#[derive(Debug, Parser)]
struct Args {
    /// The number of permutations to verify.
    #[arg(short, long, default_value_t = 128, value_parser = value_parser!(u32).range(1 << 3..))]
    n_permutations: u32,
    /// The negative binary logarithm of the Reed–Solomon code rate.
    #[arg(long, default_value_t = 1, value_parser = value_parser!(u32).range(1..))]
    log_inv_rate: u32,
}

fn main() -> Result<()> {
    const SECURITY_BITS: usize = 100;

    // 👇 Start Pyroscope agent
    let agent = PyroscopeAgent::builder(
        "https://profiles-prod-001.grafana.net".to_string(),       // Pyroscope URL
        "rust-app.keccakf_circuit".to_string(),                    // App name
    )
    .basic_auth(
        "1049539".to_string(),                                     // Grafana Cloud user ID
        "CHANGE".to_string(), // Grafana API token
    )
    .backend(pprof_backend(PprofConfig::new().sample_rate(100)))  // Sample rate
    .tags([("example", "keccakf_4")].to_vec())                      // Custom tags
    .build()?;                                                   // Start the profiler

    // 👇 Set up the rest of your benchmark as usual
    adjust_thread_pool()
        .as_ref()
        .expect("failed to init thread pool");

    let args = Args::parse();

    let _guard = init_tracing().expect("failed to initialize tracing");

    println!("Verifying {} Keccak-f permutations", args.n_permutations);

    let log_n_permutations = log2_ceil_usize(args.n_permutations as usize);
    let allocator = bumpalo::Bump::new();
    let mut builder = ConstraintSystemBuilder::new_with_witness(&allocator);
    let log_size = log_n_permutations;

    let agent_running = agent.start().unwrap();

    let trace_gen_scope = tracing::info_span!("generating trace").entered();

    let input_witness = vec![];
    let _state_out =
        binius_circuits::keccakf::keccakf(&mut builder, &Some(input_witness), log_size)?;
    drop(trace_gen_scope);


    let witness = builder
        .take_witness()
        .expect("builder created with witness");
    let constraint_system = builder.build()?;

    let backend = make_portable_backend();

    let proof = constraint_system::prove::<
        U,
        CanonicalTowerFamily,
        Groestl256,
        Groestl256ByteCompression,
        HasherChallenger<Groestl256>,
        _,
    >(
        &constraint_system,
        args.log_inv_rate as usize,
        SECURITY_BITS,
        &[],
        witness,
        &backend,
    )?;

    println!("Proof size: {}", ByteSize::b(proof.get_proof_size() as u64));

    constraint_system::verify::<
        U,
        CanonicalTowerFamily,
        Groestl256,
        Groestl256ByteCompression,
        HasherChallenger<Groestl256>,
    >(
        &constraint_system.no_base_constraints(),
        args.log_inv_rate as usize,
        SECURITY_BITS,
        &[],
        proof,
    )?;

    // ⏳ Give Pyroscope agent time to flush data before exiting
    std::thread::sleep(std::time::Duration::from_secs(10));

    Ok(())
}
