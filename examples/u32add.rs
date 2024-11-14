// Copyright 2024 Irreducible Inc.

#![feature(step_trait)]

use anyhow::Result;
use binius_core::{
	fiat_shamir::HasherChallenger,
	oracle::{BatchId, ConstraintSetBuilder, MultilinearOracleSet, OracleId, ShiftVariant},
	poly_commit::{tensor_pcs, PolyCommitScheme},
	protocols::{
		greedy_evalcheck::{self, GreedyEvalcheckProof, GreedyEvalcheckProveOutput},
		sumcheck::{self, standard_switchover_heuristic, Proof as ZerocheckProof},
	},
	transcript::{AdviceReader, AdviceWriter, CanRead, CanWrite, TranscriptWriter},
	witness::MultilinearExtensionIndex,
};
use binius_field::{
	as_packed_field::{PackScalar, PackedType},
	underlier::{UnderlierType, WithUnderlier},
	BinaryField, BinaryField128b, BinaryField16b, BinaryField1b, ExtensionField, Field,
	PackedBinaryField128x1b, PackedField, PackedFieldIndexable, TowerField,
};
use binius_hal::{make_portable_backend, ComputationBackend};
use binius_macros::{composition_poly, IterOracles};
use binius_math::{EvaluationDomainFactory, IsomorphicEvaluationDomainFactory};
use binius_utils::{
	examples::get_log_trace_size, rayon::adjust_thread_pool, tracing::init_tracing,
};
use bytemuck::{must_cast_slice_mut, Pod};
use groestl_crypto::Groestl256;
use p3_challenger::{CanObserve, CanSample, CanSampleBits};
use rand::{thread_rng, Rng};
use rayon::prelude::*;
use std::{fmt::Debug, iter};
use tracing::{debug, info, instrument};

#[cfg(feature = "fp-tower")]
mod field_types {
	use binius_field::{BinaryField128b, BinaryField8b};
	pub type FW = BinaryField128b;
	pub type FBase = BinaryField8b;
	pub type DomainFieldWithStep = BinaryField8b;
	pub type FDomain = BinaryField8b;
}

#[cfg(all(feature = "aes-tower", not(feature = "fp-tower")))]
mod field_types {
	use binius_field::{AESTowerField128b, AESTowerField8b};
	pub type FW = AESTowerField128b;
	pub type FBase = AESTowerField8b;
	pub type DomainFieldWithStep = AESTowerField8b;
	pub type FDomain = AESTowerField8b;
}

#[cfg(all(not(feature = "fp-tower"), not(feature = "aes-tower")))]
mod field_types {
	use binius_field::{BinaryField128b, BinaryField128bPolyval};
	pub type FW = BinaryField128bPolyval;
	pub type FBase = BinaryField128bPolyval;
	pub type DomainFieldWithStep = BinaryField128b;
	pub type FDomain = BinaryField128bPolyval;
}

#[instrument(skip_all, level = "debug")]
fn generate_trace<U, F>(
	log_size: usize,
	trace: &U32AddOracle,
) -> Result<MultilinearExtensionIndex<'static, U, F>>
where
	U: UnderlierType + PackScalar<BinaryField1b> + PackScalar<F> + Pod,
	F: BinaryField,
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

	let mut index = MultilinearExtensionIndex::new();
	index.set_owned::<BinaryField1b, _>(iter::zip(
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

	pub fn mixed_constraints<P: PackedField<Scalar: TowerField>>(&self) -> ConstraintSetBuilder<P> {
		let mut builder = ConstraintSetBuilder::new();

		builder.add_zerocheck(
			[self.x_in, self.y_in, self.c_in, self.z_out],
			composition_poly!([x, y, cin, z] = x + y + cin - z),
		);
		builder.add_zerocheck(
			[self.x_in, self.y_in, self.c_in, self.c_out],
			composition_poly!([x, y, cin, cout] = (x + cin) * (y + cin) + cin - cout),
		);
		builder
	}
}

#[instrument(skip_all, level = "debug")]
#[allow(clippy::too_many_arguments)]
fn prove<U, F, FBase, DomainField, FEPCS, PCS, Transcript, Backend>(
	log_size: usize,
	oracles: &mut MultilinearOracleSet<F>,
	pcs: &PCS,
	trace: &U32AddOracle,
	mut witness: MultilinearExtensionIndex<U, F>,
	mut transcript: Transcript,
	advice: &mut AdviceWriter,
	domain_factory: impl EvaluationDomainFactory<DomainField>,
	backend: Backend,
) -> Result<Proof<F, PCS::Commitment, PCS::Proof>>
where
	U: UnderlierType
		+ PackScalar<BinaryField1b>
		+ PackScalar<F>
		+ PackScalar<DomainField>
		+ PackScalar<FBase>,
	PackedType<U, F>: PackedFieldIndexable,
	FEPCS: TowerField + From<F>,
	F: TowerField + ExtensionField<FBase> + From<FEPCS> + ExtensionField<DomainField>,
	FBase: TowerField + ExtensionField<DomainField>,
	DomainField: TowerField,
	PCS: PolyCommitScheme<PackedType<U, BinaryField1b>, FEPCS, Error: Debug, Proof: 'static>,
	Transcript: CanSample<F>
		+ CanObserve<F>
		+ CanObserve<FEPCS>
		+ CanObserve<PCS::Commitment>
		+ CanSample<FEPCS>
		+ CanSampleBits<usize>
		+ CanWrite,
	Backend: ComputationBackend,
{
	assert_eq!(pcs.n_vars(), log_size);

	// Round 1
	let trace_commit_polys = oracles
		.committed_oracle_ids(trace.batch_id)
		.map(|oracle_id| witness.get::<BinaryField1b>(oracle_id))
		.collect::<Result<Vec<_>, _>>()?;
	let (trace_comm, trace_committed) = pcs.commit(&trace_commit_polys)?;
	transcript.observe(trace_comm.clone());

	// Zerocheck
	let zerocheck_challenges = transcript.sample_vec(log_size);

	let switchover_fn = standard_switchover_heuristic(-2);

	let constraint_set = trace.mixed_constraints().build_one(oracles)?;
	let constraint_set_base = trace.mixed_constraints().build_one(oracles)?;

	let (zerocheck_claim, meta) = sumcheck::constraint_set_zerocheck_claim(constraint_set.clone())?;

	let prover = sumcheck::prove::constraint_set_zerocheck_prover::<_, FBase, _, _, _>(
		constraint_set_base,
		constraint_set,
		&witness,
		domain_factory.clone(),
		switchover_fn,
		zerocheck_challenges.as_slice(),
		&backend,
	)?
	.into_regular_zerocheck()?;

	let (sumcheck_output, zerocheck_proof) =
		sumcheck::prove::batch_prove(vec![prover], &mut transcript)?;

	let zerocheck_output = sumcheck::zerocheck::verify_sumcheck_outputs(
		&[zerocheck_claim],
		&zerocheck_challenges,
		sumcheck_output,
	)?;

	let evalcheck_multilinear_claims =
		sumcheck::make_eval_claims(oracles, [meta], zerocheck_output)?;

	// Prove evaluation claims
	let GreedyEvalcheckProveOutput {
		same_query_claims,
		proof: evalcheck_proof,
	} = greedy_evalcheck::prove::<U, F, _, _, _>(
		oracles,
		&mut witness,
		evalcheck_multilinear_claims,
		switchover_fn,
		&mut transcript,
		advice,
		domain_factory,
		&backend,
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

	let eval_point: Vec<FEPCS> = same_query_claim
		.eval_point
		.into_iter()
		.map(|x| x.into())
		.collect();
	let trace_open_proof = pcs.prove_evaluation(
		&mut transcript,
		&trace_committed,
		&trace_commit_polys,
		&eval_point,
		&backend,
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
	zerocheck_proof: ZerocheckProof<F>,
	evalcheck_proof: GreedyEvalcheckProof<F>,
	trace_open_proof: PCSProof,
}

impl<F: Field, PCSComm, PCSProof> Proof<F, PCSComm, PCSProof> {
	fn isomorphic<F2: Field + From<F>>(self) -> Proof<F2, PCSComm, PCSProof> {
		Proof {
			trace_comm: self.trace_comm,
			zerocheck_proof: self.zerocheck_proof.isomorphic(),
			evalcheck_proof: self.evalcheck_proof.isomorphic(),
			trace_open_proof: self.trace_open_proof,
		}
	}
}

#[allow(clippy::too_many_arguments)]
#[instrument(skip_all, level = "debug")]
fn verify<P, F, PCS, Transcript, Backend>(
	log_size: usize,
	oracles: &mut MultilinearOracleSet<F>,
	oracle: &U32AddOracle,
	pcs: &PCS,
	mut transcript: Transcript,
	advice: &mut AdviceReader,
	proof: Proof<F, PCS::Commitment, PCS::Proof>,
	backend: Backend,
) -> Result<()>
where
	P: PackedField<Scalar = BinaryField1b>,
	F: TowerField,
	PCS: PolyCommitScheme<P, F, Error: Debug, Proof: 'static>,
	Transcript:
		CanObserve<F> + CanObserve<PCS::Commitment> + CanSample<F> + CanSampleBits<usize> + CanRead,
	Backend: ComputationBackend,
{
	let Proof {
		trace_comm,
		zerocheck_proof,
		evalcheck_proof,
		trace_open_proof,
	} = proof;

	// Round 1
	transcript.observe(trace_comm.clone());

	// Zerocheck
	let zerocheck_challenges = transcript.sample_vec(log_size);

	let constraint_set = oracle.mixed_constraints::<F>().build_one(oracles)?;

	let (zerocheck_claim, meta) = sumcheck::constraint_set_zerocheck_claim(constraint_set)?;
	let zerocheck_claims = [zerocheck_claim];

	let sumcheck_claims = sumcheck::zerocheck::reduce_to_sumchecks(&zerocheck_claims)?;

	let sumcheck_output =
		sumcheck::batch_verify(&sumcheck_claims, zerocheck_proof, &mut transcript)?;

	let zerocheck_output = sumcheck::zerocheck::verify_sumcheck_outputs(
		&zerocheck_claims,
		&zerocheck_challenges,
		sumcheck_output,
	)?;

	let evalcheck_multilinear_claims =
		sumcheck::make_eval_claims(oracles, [meta], zerocheck_output)?;

	// Evalcheck
	let same_query_claims = greedy_evalcheck::verify(
		oracles,
		evalcheck_multilinear_claims,
		evalcheck_proof,
		&mut transcript,
		advice,
	)?;

	assert_eq!(same_query_claims.len(), 1);
	let (batch_id, same_query_claim) = same_query_claims
		.into_iter()
		.next()
		.expect("length is asserted to be 1");
	assert_eq!(batch_id, oracle.batch_id);

	pcs.verify_evaluation(
		&mut transcript,
		&trace_comm,
		&same_query_claim.eval_point,
		trace_open_proof,
		&same_query_claim.evals,
		&backend,
	)?;

	Ok(())
}

fn main() {
	const SECURITY_BITS: usize = 100;

	adjust_thread_pool()
		.as_ref()
		.expect("failed to init thread pool");

	let _guard = init_tracing().expect("failed to initialize tracing");

	// Values below 14 are rejected by `find_proof_size_optimal_pcs()`.
	let log_size = get_log_trace_size().unwrap_or(14);
	let log_inv_rate = 1;
	let backend = make_portable_backend();

	type U = <PackedBinaryField128x1b as WithUnderlier>::Underlier;

	let mut prover_oracles = MultilinearOracleSet::new();
	let prover_trace = U32AddOracle::new::<field_types::FW>(&mut prover_oracles, log_size);

	let trace_batch = prover_oracles.committed_batch(prover_trace.batch_id);

	debug!(num_bits = 1 << log_size, num_u32s = 1 << (log_size - 5), "U32 Addition");

	let pcs = tensor_pcs::find_proof_size_optimal_pcs::<
		<PackedBinaryField128x1b as WithUnderlier>::Underlier,
		BinaryField1b,
		BinaryField16b,
		BinaryField16b,
		BinaryField128b,
	>(SECURITY_BITS, trace_batch.n_vars, trace_batch.n_polys, log_inv_rate, false)
	.unwrap();

	let witness = generate_trace::<U, field_types::FW>(log_size, &prover_trace).unwrap();

	let mut prover_context = binius_core::transcript::Proof {
		transcript: TranscriptWriter::<HasherChallenger<Groestl256>>::default(),
		advice: AdviceWriter::default(),
	};
	let domain_factory =
		IsomorphicEvaluationDomainFactory::<field_types::DomainFieldWithStep>::default();

	info!("Proving");
	let proof = prove::<
		_,
		field_types::FW,
		field_types::FBase,
		field_types::FDomain,
		BinaryField128b,
		_,
		_,
		_,
	>(
		log_size,
		&mut prover_oracles,
		&pcs,
		&prover_trace,
		witness,
		&mut prover_context.transcript,
		&mut prover_context.advice,
		domain_factory,
		&backend,
	)
	.unwrap();

	let mut verifier_context = prover_context.into_verifier();
	let mut verifier_oracles = MultilinearOracleSet::new();
	let verifier_trace = U32AddOracle::new::<BinaryField128b>(&mut verifier_oracles, log_size);

	info!("Verifying");
	verify(
		log_size,
		&mut verifier_oracles.clone(),
		&verifier_trace,
		&pcs,
		&mut verifier_context.transcript,
		&mut verifier_context.advice,
		proof.isomorphic(),
		backend,
	)
	.unwrap();

	verifier_context.finalize().unwrap();
}
