// Copyright 2024 Ulvetanna Inc.

use anyhow::Result;
use binius_core::{
	challenger::{new_hasher_challenger, IsomorphicChallenger},
	oracle::{
		BatchId, CommittedId, ConstraintSet, ConstraintSetBuilder, MultilinearOracleSet, OracleId,
	},
	poly_commit::{tensor_pcs, PolyCommitScheme},
	protocols::{
		abstract_sumcheck::standard_switchover_heuristic,
		greedy_evalcheck::{self, GreedyEvalcheckProof, GreedyEvalcheckProveOutput},
		sumcheck_v2::{self, Proof as ZerocheckProof},
		test_utils::conflate_multilinear_evalchecks,
	},
	witness::MultilinearExtensionIndex,
};
use binius_field::{
	as_packed_field::{PackScalar, PackedType},
	underlier::{UnderlierType, WithUnderlier},
	BinaryField, BinaryField128b, BinaryField128bPolyval, BinaryField16b, BinaryField1b,
	PackedBinaryField128x1b, PackedField, PackedFieldIndexable, TowerField,
};
use binius_hal::ComputationBackend;
use binius_hash::GroestlHasher;
use binius_macros::{composition_poly, IterOracles};
use binius_math::{EvaluationDomainFactory, IsomorphicEvaluationDomainFactory};
use binius_utils::{
	examples::get_log_trace_size, rayon::adjust_thread_pool, tracing::init_tracing,
};
use bytemuck::{must_cast, must_cast_mut, Pod};
use p3_challenger::{CanObserve, CanSample, CanSampleBits};
use rand::thread_rng;
use rayon::prelude::*;
use std::{fmt::Debug, iter};
use tracing::instrument;

composition_poly!(BitwiseAndConstraint[a, b, c] = a * b - c);

#[allow(clippy::too_many_arguments)]
#[instrument(skip_all, level = "debug")]
fn prove<U, PCS, CH, Backend>(
	pcs: &PCS,
	oracles: &mut MultilinearOracleSet<BinaryField128b>,
	trace: &TraceOracle,
	constraint_set: ConstraintSet<PackedType<U, BinaryField128bPolyval>>,
	mut witness: MultilinearExtensionIndex<U, BinaryField128bPolyval>,
	mut challenger: CH,
	domain_factory: impl EvaluationDomainFactory<BinaryField128bPolyval>,
	backend: Backend,
) -> Result<Proof<PCS::Commitment, PCS::Proof>>
where
	U: UnderlierType + PackScalar<BinaryField1b> + PackScalar<BinaryField128bPolyval>,
	PackedType<U, BinaryField128bPolyval>: PackedFieldIndexable,
	PCS: PolyCommitScheme<PackedType<U, BinaryField1b>, BinaryField128b>,
	PCS::Error: Debug,
	PCS::Proof: 'static,
	CH: CanObserve<BinaryField128b>
		+ CanObserve<PCS::Commitment>
		+ CanSample<BinaryField128b>
		+ CanSampleBits<usize>
		+ Clone,
	Backend: ComputationBackend,
{
	let log_size = trace.log_size;
	let commit_span = tracing::debug_span!("commit").entered();
	assert_eq!(pcs.n_vars(), log_size);

	// Round 1
	let commit_polys = oracles
		.committed_oracle_ids(trace.batch_id)
		.map(|oracle_id| witness.get::<BinaryField1b>(oracle_id))
		.collect::<Result<Vec<_>, _>>()?;
	let (abc_comm, abc_committed) = pcs.commit(&commit_polys).unwrap();
	challenger.observe(abc_comm.clone());

	drop(commit_span);

	// Challenger over the isomorphic POLYVAL field
	let mut iso_challenger =
		IsomorphicChallenger::<_, _, BinaryField128bPolyval>::new(&mut challenger);

	let zerocheck_challenges = iso_challenger.sample_vec(log_size);

	// Round 2
	tracing::debug!("Proving zerocheck");
	let switchover_fn = standard_switchover_heuristic(-2);

	let (zerocheck_claim, meta) =
		sumcheck_v2::constraint_set_zerocheck_claim(constraint_set.clone(), oracles)?;

	let prover =
		sumcheck_v2::prove::constraint_set_zerocheck_prover::<_, BinaryField128bPolyval, _, _, _>(
			constraint_set.clone(),
			constraint_set,
			&witness,
			domain_factory.clone(),
			switchover_fn,
			zerocheck_challenges.as_slice(),
			backend.clone(),
		)?;

	let (sumcheck_output, zerocheck_proof) =
		sumcheck_v2::prove::batch_prove(vec![prover], &mut iso_challenger)?;

	let zerocheck_output = sumcheck_v2::verify_sumcheck_outputs(
		&[zerocheck_claim],
		&zerocheck_challenges,
		sumcheck_output,
	)?;

	let evalcheck_multilinear_claims =
		sumcheck_v2::make_eval_claims(oracles, [meta], zerocheck_output.isomorphic())?;

	let evalcheck_claims = conflate_multilinear_evalchecks(evalcheck_multilinear_claims)?;

	// Prove evaluation claims
	let GreedyEvalcheckProveOutput {
		same_query_claims,
		proof: evalcheck_proof,
	} = greedy_evalcheck::prove::<_, PackedType<U, BinaryField128bPolyval>, _, _, _>(
		oracles,
		&mut witness,
		evalcheck_claims,
		switchover_fn,
		&mut challenger,
		domain_factory,
		backend.clone(),
	)?;

	assert_eq!(same_query_claims.len(), 1);
	let (_, same_query_pcs_claim) = same_query_claims
		.into_iter()
		.next()
		.expect("length is asserted to be 1");

	let commit_polys = oracles
		.committed_oracle_ids(trace.batch_id)
		.map(|oracle_id| witness.get::<BinaryField1b>(oracle_id))
		.collect::<Result<Vec<_>, _>>()?;

	// Prove commitment openings
	let abc_eval_proof = pcs.prove_evaluation(
		&mut challenger,
		&abc_committed,
		&commit_polys,
		&same_query_pcs_claim.eval_point,
		backend,
	)?;

	Ok(Proof {
		abc_comm,
		abc_eval_proof,
		zerocheck_proof: zerocheck_proof.isomorphic(),
		evalcheck_proof,
	})
}

struct Proof<C, P> {
	abc_comm: C,
	abc_eval_proof: P,
	zerocheck_proof: ZerocheckProof<BinaryField128b>,
	evalcheck_proof: GreedyEvalcheckProof<BinaryField128b>,
}

#[instrument(skip_all, level = "debug")]
fn verify<PCS, CH, Backend>(
	log_size: usize,
	pcs: &PCS,
	trace: &mut MultilinearOracleSet<BinaryField128b>,
	constraint_set: ConstraintSet<BinaryField128b>,
	proof: Proof<PCS::Commitment, PCS::Proof>,
	mut challenger: CH,
	backend: Backend,
) -> Result<()>
where
	PCS: PolyCommitScheme<PackedBinaryField128x1b, BinaryField128b>,
	PCS::Error: Debug,
	PCS::Proof: 'static,
	CH: CanObserve<BinaryField128b>
		+ CanObserve<PCS::Commitment>
		+ CanSample<BinaryField128b>
		+ CanSampleBits<usize>,
	Backend: ComputationBackend,
{
	assert_eq!(pcs.n_vars(), log_size);

	let Proof {
		abc_comm,
		abc_eval_proof,
		zerocheck_proof,
		evalcheck_proof,
	} = proof;

	// Observe the trace commitments
	challenger.observe(abc_comm.clone());

	// Run zerocheck protocol
	let zerocheck_challenges = challenger.sample_vec(log_size);

	let (zerocheck_claim, meta) =
		sumcheck_v2::constraint_set_zerocheck_claim(constraint_set, trace)?;
	let zerocheck_claims = [zerocheck_claim];

	let sumcheck_claims = sumcheck_v2::reduce_to_sumchecks(&zerocheck_claims)?;

	let sumcheck_output =
		sumcheck_v2::batch_verify(&sumcheck_claims, zerocheck_proof, &mut challenger)?;

	let zerocheck_output = sumcheck_v2::verify_sumcheck_outputs(
		&zerocheck_claims,
		&zerocheck_challenges,
		sumcheck_output,
	)?;

	let evalcheck_multilinear_claims =
		sumcheck_v2::make_eval_claims(trace, [meta], zerocheck_output)?;

	let evalcheck_claims = conflate_multilinear_evalchecks(evalcheck_multilinear_claims)?;

	// Verify evaluation claims
	let same_query_claims =
		greedy_evalcheck::verify(trace, evalcheck_claims, evalcheck_proof, &mut challenger)?;

	assert_eq!(same_query_claims.len(), 1);
	let (_, same_query_pcs_claim) = same_query_claims
		.into_iter()
		.next()
		.expect("length is asserted to be 1");

	// Verify commitment openings
	pcs.verify_evaluation(
		&mut challenger,
		&abc_comm,
		&same_query_pcs_claim.eval_point,
		abc_eval_proof,
		&same_query_pcs_claim.evals,
		backend,
	)?;

	Ok(())
}

#[derive(Debug, IterOracles)]
struct TraceOracle {
	log_size: usize,
	batch_id: BatchId,

	a_in: OracleId,
	b_in: OracleId,
	c_out: OracleId,
}

impl TraceOracle {
	pub fn new<F: TowerField>(oracles: &mut MultilinearOracleSet<F>, log_size: usize) -> Self {
		let batch_id = oracles.add_committed_batch(log_size, BinaryField1b::TOWER_LEVEL);
		let [a_in, b_in, c_out] = oracles.add_committed_multiple(batch_id);
		Self {
			log_size,
			batch_id,

			a_in,
			b_in,
			c_out,
		}
	}
}

#[instrument(skip_all, level = "debug")]
fn generate_trace<U, FW>(
	log_size: usize,
	trace_oracle: &TraceOracle,
) -> Result<MultilinearExtensionIndex<'static, U, FW>>
where
	U: UnderlierType + PackScalar<BinaryField1b> + PackScalar<FW> + Pod,
	FW: BinaryField,
{
	assert!(log_size >= <PackedType<U, BinaryField1b>>::LOG_WIDTH);
	let len = 1 << (log_size - <PackedType<U, BinaryField1b>>::LOG_WIDTH);
	let mut a_in = vec![U::default(); len];
	let mut b_in = vec![U::default(); len];
	let mut c_out = vec![U::default(); len];

	a_in.par_iter_mut()
		.zip(b_in.par_iter_mut())
		.zip(c_out.par_iter_mut())
		.for_each_init(thread_rng, |rng, ((a_i, b_i), c_i)| {
			*a_i = U::random(&mut *rng);
			*b_i = U::random(&mut *rng);
			let a_i_uint128 = must_cast::<_, u128>(*a_i);
			let b_i_uint128 = must_cast::<_, u128>(*b_i);
			let c_i_uint128 = must_cast_mut::<_, u128>(c_i);
			*c_i_uint128 = a_i_uint128 & b_i_uint128;
		});

	MultilinearExtensionIndex::new()
		.update_owned::<BinaryField1b, _>(iter::zip(
			[trace_oracle.a_in, trace_oracle.b_in, trace_oracle.c_out],
			[a_in, b_in, c_out],
		))
		.map_err(Into::into)
}

fn make_constraints<P: PackedField, F: TowerField>(
	trace_oracle: &MultilinearOracleSet<F>,
) -> ConstraintSet<P> {
	let a_in_oracle = trace_oracle.committed_oracle_id(CommittedId {
		batch_id: 0,
		index: 0,
	});
	let b_in_oracle = trace_oracle.committed_oracle_id(CommittedId {
		batch_id: 0,
		index: 1,
	});
	let c_out_oracle = trace_oracle.committed_oracle_id(CommittedId {
		batch_id: 0,
		index: 2,
	});

	let mut builder = ConstraintSetBuilder::new();
	builder.add_zerocheck([a_in_oracle, b_in_oracle, c_out_oracle], BitwiseAndConstraint);
	builder.build()
}

fn main() {
	adjust_thread_pool()
		.as_ref()
		.expect("failed to init thread pool");

	let _guard = init_tracing().expect("failed to initialize tracing");

	const SECURITY_BITS: usize = 100;

	let log_size = get_log_trace_size().unwrap_or(20);
	let log_inv_rate = 1;

	type U = <PackedBinaryField128x1b as WithUnderlier>::Underlier;

	let mut oracles = MultilinearOracleSet::new();
	let trace_oracle = TraceOracle::new(&mut oracles, log_size);
	let batch = oracles.committed_batch(trace_oracle.batch_id);

	// Set up the public parameters
	let pcs = tensor_pcs::find_proof_size_optimal_pcs::<
		U,
		BinaryField1b,
		BinaryField16b,
		BinaryField16b,
		BinaryField128b,
	>(SECURITY_BITS, batch.n_vars, batch.n_polys, log_inv_rate, false)
	.unwrap();

	tracing::debug!(
		"Using BlockTensorPCS with log_rows = {}, log_cols = {}, proof_size = {}",
		pcs.log_rows(),
		pcs.log_cols(),
		pcs.proof_size(3),
	);

	let prover_constraints = make_constraints::<PackedType<U, BinaryField128bPolyval>, _>(&oracles);
	let verifier_constraints = make_constraints::<BinaryField128b, _>(&oracles);

	let witness = generate_trace::<U, BinaryField128bPolyval>(log_size, &trace_oracle).unwrap();
	let challenger = new_hasher_challenger::<_, GroestlHasher<_>>();
	let domain_factory = IsomorphicEvaluationDomainFactory::<BinaryField128b>::default();
	let backend = binius_hal::make_portable_backend();

	let proof = prove(
		&pcs,
		&mut oracles.clone(),
		&trace_oracle,
		prover_constraints,
		witness,
		challenger.clone(),
		domain_factory,
		backend.clone(),
	)
	.unwrap();

	verify(
		log_size,
		&pcs,
		&mut oracles.clone(),
		verifier_constraints,
		proof,
		challenger.clone(),
		backend,
	)
	.unwrap();
}
