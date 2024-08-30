// Copyright 2024 Ulvetanna Inc.

#![feature(step_trait)]

use anyhow::Result;
use binius_backend_provider::make_best_backend;
use binius_core::{
	challenger::new_hasher_challenger,
	composition::{empty_mix_composition, index_composition},
	oracle::{BatchId, CompositePolyOracle, MultilinearOracleSet, OracleId, ShiftVariant},
	poly_commit::{tensor_pcs, PolyCommitScheme},
	polynomial::{CompositionPoly, MultilinearComposite},
	protocols::{
		abstract_sumcheck::standard_switchover_heuristic,
		greedy_evalcheck::{self, GreedyEvalcheckProof, GreedyEvalcheckProveOutput},
		zerocheck::{self, ZerocheckBatchProof, ZerocheckBatchProveOutput, ZerocheckClaim},
	},
	witness::MultilinearExtensionIndex,
};
use binius_field::{
	as_packed_field::{PackScalar, PackedType},
	underlier::{UnderlierType, WithUnderlier},
	BinaryField, BinaryField128b, BinaryField16b, BinaryField1b, ExtensionField, Field,
	PackedBinaryField128x1b, PackedField, PackedFieldIndexable, TowerField,
};
use binius_hal::ComputationBackend;
use binius_hash::GroestlHasher;
use binius_macros::{composition_poly, IterOracles};
use binius_math::{EvaluationDomainFactory, IsomorphicEvaluationDomainFactory};
use binius_utils::{
	examples::get_log_trace_size, rayon::adjust_thread_pool, tracing::init_tracing,
};
use bytemuck::{must_cast_slice_mut, Pod};
use p3_challenger::{CanObserve, CanSample, CanSampleBits};
use rand::{thread_rng, Rng};
use rayon::prelude::*;
use std::{fmt::Debug, iter};
use tracing::{debug, info, instrument};

// mod field_types is a selector of different sets of types which provide
// equivalent functionality but may differ significantly in performance.
#[cfg(feature = "aes-tower")]
mod field_types {
	pub type Field = binius_field::AESTowerField128b;
	pub type DomainField = binius_field::AESTowerField8b;
	pub type DomainFieldWithStep = binius_field::AESTowerField8b;
}

#[cfg(not(feature = "aes-tower"))]
mod field_types {
	pub type Field = binius_field::BinaryField128bPolyval;
	pub type DomainField = binius_field::BinaryField128bPolyval;
	pub type DomainFieldWithStep = binius_field::BinaryField128b;
}

#[instrument(skip_all, level = "debug")]
fn generate_trace<U, FW>(
	log_size: usize,
	trace: &U32AddOracle,
) -> Result<MultilinearExtensionIndex<'static, U, FW>>
where
	U: UnderlierType + PackScalar<BinaryField1b> + PackScalar<FW> + Pod,
	FW: BinaryField,
{
	assert!(log_size >= <PackedType<U, BinaryField1b>>::LOG_WIDTH);
	let len = 1 << (log_size - <PackedType<U, BinaryField1b>>::LOG_WIDTH);
	let build_trace_column = || vec![U::default(); len].into_boxed_slice();

	let mut x_in = build_trace_column();
	let mut y_in = build_trace_column();
	let mut z_out = build_trace_column();
	let mut c_out = build_trace_column();
	let mut c_in = build_trace_column();

	// Fill the trace
	(
		must_cast_slice_mut::<_, u32>(&mut x_in),
		must_cast_slice_mut::<_, u32>(&mut y_in),
		must_cast_slice_mut::<_, u32>(&mut z_out),
		must_cast_slice_mut::<_, u32>(&mut c_out),
		must_cast_slice_mut::<_, u32>(&mut c_in),
	)
		.into_par_iter()
		.for_each_init(thread_rng, |rng, (x, y, z, cout, cin)| {
			*x = rng.gen();
			*y = rng.gen();
			let carry;
			(*z, carry) = (*x).overflowing_add(*y);
			*cin = (*x) ^ (*y) ^ (*z);
			*cout = *cin >> 1;
			if carry {
				*cout |= 1 << 31;
			}
		});

	let index = MultilinearExtensionIndex::new().update_owned::<BinaryField1b, _>(iter::zip(
		[trace.x_in, trace.y_in, trace.z_out, trace.c_out, trace.c_in],
		[x_in, y_in, z_out, c_out, c_in],
	))?;
	Ok(index)
}

#[derive(IterOracles)]
struct U32AddOracle {
	x_in: OracleId,
	y_in: OracleId,
	z_out: OracleId,
	c_out: OracleId,
	c_in: OracleId,

	batch_id: BatchId,
}

impl U32AddOracle {
	pub fn new<F: TowerField>(oracles: &mut MultilinearOracleSet<F>, n_vars: usize) -> Self {
		let batch_id = oracles.add_committed_batch(n_vars, BinaryField1b::TOWER_LEVEL);
		let [x_in, y_in, z_out, c_out] = oracles
			.add_named("add_oracles")
			.committed_multiple(batch_id);
		let c_in = oracles
			.add_named("cin")
			.shifted(c_out, 1, 5, ShiftVariant::LogicalLeft)
			.unwrap();
		Self {
			x_in,
			y_in,
			z_out,
			c_out,
			c_in,
			batch_id,
		}
	}

	pub fn mixed_constraints<P: PackedField<Scalar: TowerField>>(
		&self,
		challenge: P::Scalar,
	) -> Result<impl CompositionPoly<P> + Clone> {
		let all_columns = &self.iter_oracles().collect::<Vec<_>>();
		let mix = empty_mix_composition(all_columns.len(), challenge);
		let mix = mix.include([index_composition(
			all_columns,
			[self.x_in, self.y_in, self.c_in, self.z_out],
			composition_poly!([x, y, cin, z] = x + y + cin - z),
		)?])?;
		let mix = mix.include([index_composition(
			all_columns,
			[self.x_in, self.y_in, self.c_in, self.c_out],
			composition_poly!([x, y, cin, cout] = (x + cin) * (y + cin) + cin - cout),
		)?])?;
		Ok(mix)
	}
}

#[instrument(skip_all, level = "debug")]
#[allow(clippy::too_many_arguments)]
fn prove<U, F, FW, DomainField, PCS, CH, Backend>(
	log_size: usize,
	oracles: &mut MultilinearOracleSet<F>,
	pcs: &PCS,
	trace: &U32AddOracle,
	mut witness: MultilinearExtensionIndex<U, FW>,
	mut challenger: CH,
	domain_factory: impl EvaluationDomainFactory<DomainField>,
	backend: Backend,
) -> Result<Proof<F, PCS::Commitment, PCS::Proof>>
where
	U: UnderlierType + PackScalar<BinaryField1b> + PackScalar<FW> + PackScalar<DomainField>,
	PackedType<U, FW>: PackedFieldIndexable,
	F: TowerField + From<FW>,
	FW: TowerField + From<F> + ExtensionField<DomainField>,
	DomainField: TowerField,
	PCS: PolyCommitScheme<PackedType<U, BinaryField1b>, F, Error: Debug, Proof: 'static>,
	CH: CanObserve<F> + CanObserve<PCS::Commitment> + CanSample<F> + CanSampleBits<usize>,
	Backend: ComputationBackend,
{
	assert_eq!(pcs.n_vars(), log_size);

	// Round 1
	let trace_commit_polys = oracles
		.committed_oracle_ids(trace.batch_id)
		.map(|oracle_id| witness.get::<BinaryField1b>(oracle_id))
		.collect::<Result<Vec<_>, _>>()?;
	let (trace_comm, trace_committed) = pcs.commit(&trace_commit_polys)?;
	challenger.observe(trace_comm.clone());

	// Zerocheck mixing
	let mixing_challenge = challenger.sample();

	let mix_composition_verifier = trace.mixed_constraints(mixing_challenge)?;
	let mix_composition_prover = trace.mixed_constraints(FW::from(mixing_challenge))?;

	let zerocheck_column_oracles = trace.iter_oracles().map(|id| oracles.oracle(id)).collect();
	let zerocheck_claim = ZerocheckClaim {
		poly: CompositePolyOracle::new(
			log_size,
			zerocheck_column_oracles,
			mix_composition_verifier,
		)?,
	};

	let zerocheck_witness = MultilinearComposite::new(
		log_size,
		mix_composition_prover,
		trace
			.iter_oracles()
			.map(|oracle_id| witness.get_multilin_poly(oracle_id))
			.collect::<Result<_, _>>()?,
	)?;

	// zerocheck::prove is instrumented
	let switchover_fn = standard_switchover_heuristic(-2);

	let ZerocheckBatchProveOutput {
		evalcheck_claims,
		proof: zerocheck_proof,
	} = zerocheck::batch_prove(
		[(zerocheck_claim, zerocheck_witness)],
		domain_factory.clone(),
		switchover_fn,
		mixing_challenge,
		&mut challenger,
		backend.clone(),
	)?;

	// Prove evaluation claims
	let GreedyEvalcheckProveOutput {
		same_query_claims,
		proof: evalcheck_proof,
	} = greedy_evalcheck::prove::<_, PackedType<U, FW>, _, _, _>(
		oracles,
		&mut witness,
		evalcheck_claims,
		switchover_fn,
		&mut challenger,
		domain_factory,
		backend.clone(),
	)?;

	assert_eq!(same_query_claims.len(), 1);
	let (batch_id, same_query_claim) = same_query_claims
		.into_iter()
		.next()
		.expect("length is asserted to be 1");
	assert_eq!(batch_id, trace.batch_id);

	let trace_commit_polys = oracles
		.committed_oracle_ids(trace.batch_id)
		.map(|oracle_id| witness.get::<BinaryField1b>(oracle_id))
		.collect::<Result<Vec<_>, _>>()?;
	// Prove commitment openings
	let trace_open_proof = pcs.prove_evaluation(
		&mut challenger,
		&trace_committed,
		&trace_commit_polys,
		&same_query_claim.eval_point,
		backend,
	)?;

	Ok(Proof {
		trace_comm,
		zerocheck_proof,
		evalcheck_proof,
		trace_open_proof,
	})
}

struct Proof<F: Field, PCSComm, PCSProof> {
	trace_comm: PCSComm,
	zerocheck_proof: ZerocheckBatchProof<F>,
	evalcheck_proof: GreedyEvalcheckProof<F>,
	trace_open_proof: PCSProof,
}

#[allow(clippy::too_many_arguments)]
#[instrument(skip_all, level = "debug")]
fn verify<P, F, PCS, CH, Backend>(
	log_size: usize,
	oracles: &mut MultilinearOracleSet<F>,
	oracle: &U32AddOracle,
	pcs: &PCS,
	mut challenger: CH,
	proof: Proof<F, PCS::Commitment, PCS::Proof>,
	backend: Backend,
) -> Result<()>
where
	P: PackedField<Scalar = BinaryField1b> + Pod,
	F: TowerField,
	PCS: PolyCommitScheme<P, F, Error: Debug, Proof: 'static>,
	CH: CanObserve<F> + CanObserve<PCS::Commitment> + CanSample<F> + CanSampleBits<usize>,
	Backend: ComputationBackend,
{
	let Proof {
		trace_comm,
		zerocheck_proof,
		evalcheck_proof,
		trace_open_proof,
	} = proof;

	// Round 1
	challenger.observe(trace_comm.clone());

	// Zerocheck mixing
	let mixing_challenge = challenger.sample();
	let mix_composition = oracle.mixed_constraints(mixing_challenge)?;

	// Zerocheck
	let zerocheck_column_oracles = oracle.iter_oracles().map(|id| oracles.oracle(id)).collect();
	let zerocheck_claim = ZerocheckClaim {
		poly: CompositePolyOracle::new(log_size, zerocheck_column_oracles, mix_composition)?,
	};

	let evalcheck_claims =
		zerocheck::batch_verify([zerocheck_claim], zerocheck_proof, &mut challenger)?;

	// Evalcheck
	let same_query_claims =
		greedy_evalcheck::verify(oracles, evalcheck_claims, evalcheck_proof, &mut challenger)?;

	assert_eq!(same_query_claims.len(), 1);
	let (batch_id, same_query_claim) = same_query_claims
		.into_iter()
		.next()
		.expect("length is asserted to be 1");
	assert_eq!(batch_id, oracle.batch_id);

	pcs.verify_evaluation(
		&mut challenger,
		&trace_comm,
		&same_query_claim.eval_point,
		trace_open_proof,
		&same_query_claim.evals,
		backend,
	)?;

	Ok(())
}

fn main() {
	const SECURITY_BITS: usize = 100;

	adjust_thread_pool()
		.as_ref()
		.expect("failed to init thread pool");

	init_tracing().expect("failed to initialize tracing");

	// Values below 14 are rejected by `find_proof_size_optimal_pcs()`.
	let log_size = get_log_trace_size().unwrap_or(14);
	let log_inv_rate = 1;
	let backend = make_best_backend();

	type U = <PackedBinaryField128x1b as WithUnderlier>::Underlier;

	let mut oracles = MultilinearOracleSet::new();
	let trace = U32AddOracle::new(&mut oracles, log_size);

	let trace_batch = oracles.committed_batch(trace.batch_id);

	debug!(num_bits = 1 << log_size, num_u32s = 1 << (log_size - 5), "U32 Addition");

	let pcs = tensor_pcs::find_proof_size_optimal_pcs::<
		<PackedBinaryField128x1b as WithUnderlier>::Underlier,
		BinaryField1b,
		BinaryField16b,
		BinaryField16b,
		BinaryField128b,
	>(SECURITY_BITS, trace_batch.n_vars, trace_batch.n_polys, log_inv_rate, false)
	.unwrap();

	let witness = generate_trace::<U, field_types::Field>(log_size, &trace).unwrap();

	let challenger = new_hasher_challenger::<_, GroestlHasher<_>>();
	let domain_factory =
		IsomorphicEvaluationDomainFactory::<field_types::DomainFieldWithStep>::default();

	info!("Proving");
	let proof = prove::<_, BinaryField128b, field_types::Field, field_types::DomainField, _, _, _>(
		log_size,
		&mut oracles,
		&pcs,
		&trace,
		witness,
		challenger.clone(),
		domain_factory,
		backend.clone(),
	)
	.unwrap();

	info!("Verifying");
	verify(log_size, &mut oracles.clone(), &trace, &pcs, challenger.clone(), proof, backend)
		.unwrap();
}
