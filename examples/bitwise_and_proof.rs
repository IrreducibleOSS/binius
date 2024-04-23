use binius_core::{
	challenger::HashChallenger,
	oracle::{CommittedBatchSpec, CommittedId, CompositePolyOracle, MultilinearOracleSet},
	poly_commit::{tensor_pcs, PolyCommitScheme},
	polynomial::{
		CompositionPoly, Error as PolynomialError, EvaluationDomain, MultilinearComposite,
		MultilinearExtension,
	},
	protocols::{
		evalcheck::{
			prove as prove_evalcheck, verify as verify_evalcheck, BatchCommittedEvalClaims,
			EvalcheckProof,
		},
		sumcheck::{SumcheckProof, SumcheckProveOutput},
		test_utils::{full_prove_with_switchover, full_verify},
		zerocheck::{
			prove as prove_zerocheck, verify as verify_zerocheck, ZerocheckClaim, ZerocheckProof,
			ZerocheckProveOutput,
		},
	},
	witness::MultilinearWitnessIndex,
};
use binius_field::{
	BinaryField128b, BinaryField128bPolyval, ExtensionField, PackedBinaryField128x1b,
	PackedBinaryField1x128b, PackedBinaryField8x16b, PackedField, TowerField,
};
use binius_hash::GroestlHasher;
use bytemuck::{must_cast, must_cast_mut};
use p3_challenger::{CanObserve, CanSample, CanSampleBits};
use rand::thread_rng;
use rayon::prelude::*;
use std::{env, fmt::Debug};
use tracing_profile::{CsvLayer, PrintTreeLayer};
use tracing_subscriber::prelude::*;

#[derive(Clone, Debug)]
struct BitwiseAndConstraint;

impl BitwiseAndConstraint {
	fn evaluate<P: PackedField>(query: &[P]) -> Result<P, PolynomialError> {
		if query.len() != 3 {
			return Err(PolynomialError::IncorrectQuerySize { expected: 3 });
		}
		let a = query[0];
		let b = query[1];
		let c = query[2];
		Ok(a * b - c)
	}
}

impl<P: PackedField> CompositionPoly<P> for BitwiseAndConstraint {
	fn n_vars(&self) -> usize {
		3
	}

	fn degree(&self) -> usize {
		2
	}

	fn evaluate(&self, query: &[P::Scalar]) -> Result<P::Scalar, PolynomialError> {
		Self::evaluate(query)
	}

	fn evaluate_packed(&self, query: &[P]) -> Result<P, PolynomialError> {
		Self::evaluate(query)
	}

	fn binary_tower_level(&self) -> usize {
		0
	}
}

fn prove<PCS, CH>(
	log_size: usize,
	pcs: &PCS,
	trace: &mut MultilinearOracleSet<BinaryField128b>,
	constraints: &[CompositePolyOracle<BinaryField128b>],
	witness: &TraceWitness<PackedBinaryField128x1b>,
	mut challenger: CH,
) -> Proof<PCS::Commitment, PCS::Proof>
where
	PCS: PolyCommitScheme<PackedBinaryField128x1b, BinaryField128b>,
	PCS::Error: Debug,
	PCS::Proof: 'static,
	CH: CanObserve<BinaryField128b>
		+ CanObserve<PCS::Commitment>
		+ CanSample<BinaryField128b>
		+ CanSampleBits<usize>,
{
	let span = tracing::debug_span!("commit");
	let commit_scope = span.enter();

	assert_eq!(pcs.n_vars(), log_size);

	let mut witness_index = witness.to_index::<_, BinaryField128bPolyval>(trace);

	assert_eq!(constraints.len(), 1);
	let constraint = constraints[0].clone();

	// Round 1
	let (abc_comm, abc_committed) = pcs
		.commit(&[
			witness.a_in.to_ref(),
			witness.b_in.to_ref(),
			witness.c_out.to_ref(),
		])
		.unwrap();
	challenger.observe(abc_comm.clone());

	let trace_batch = trace.committed_batch(0);
	let mut batch_committed_eval_claims = BatchCommittedEvalClaims::new(&[trace_batch]);
	drop(commit_scope);

	// Round 2
	let zerocheck_challenge = challenger.sample_vec(log_size - 1);
	let zerocheck_witness = MultilinearComposite::<BinaryField128bPolyval, _, _>::new(
		log_size,
		BitwiseAndConstraint,
		vec![
			witness.a_in.to_ref().specialize_arc_dyn(),
			witness.b_in.to_ref().specialize_arc_dyn(),
			witness.c_out.to_ref().specialize_arc_dyn(),
		],
	)
	.unwrap();

	// prove_zerocheck is instrumented
	let zerocheck_claim = ZerocheckClaim { poly: constraint };
	let ZerocheckProveOutput {
		sumcheck_claim,
		sumcheck_witness,
		zerocheck_proof,
	} = prove_zerocheck(&zerocheck_claim, zerocheck_witness, zerocheck_challenge).unwrap();

	let sumcheck_domain = EvaluationDomain::<BinaryField128bPolyval>::new_isomorphic::<
		BinaryField128b,
	>(sumcheck_claim.poly.max_individual_degree() + 1)
	.unwrap();

	// TODO: Improve the logic to commit the optimal switchover.
	let switchover = log_size / 2;

	// full_prove_with_switchover is instrumented
	tracing::debug!("Proving sumcheck");
	let (_, output) = full_prove_with_switchover(
		&sumcheck_claim,
		sumcheck_witness,
		&sumcheck_domain,
		&mut challenger,
		switchover,
	);

	let SumcheckProveOutput {
		evalcheck_claim,
		sumcheck_proof,
	} = output;

	// prove_evalcheck is instrumented
	let mut new_sumchecks = Vec::new();
	let evalcheck_proof = prove_evalcheck(
		trace,
		&mut witness_index,
		evalcheck_claim,
		&mut batch_committed_eval_claims,
		&mut new_sumchecks,
	)
	.unwrap();

	// try_extract_same_query_pcs_claim is instrumented
	assert!(new_sumchecks.is_empty());
	assert_eq!(batch_committed_eval_claims.n_batches(), 1);
	let same_query_pcs_claim = batch_committed_eval_claims
		.try_extract_same_query_pcs_claim(0)
		.unwrap()
		.unwrap();

	// all implementaions of prove_evaluation are instrumented
	let abc_eval_proof = pcs
		.prove_evaluation(
			&mut challenger,
			&abc_committed,
			&[
				witness.a_in.to_ref(),
				witness.b_in.to_ref(),
				witness.c_out.to_ref(),
			],
			&same_query_pcs_claim.eval_point,
		)
		.unwrap();

	Proof {
		abc_comm,
		abc_eval_proof,
		zerocheck_proof,
		sumcheck_proof,
		evalcheck_proof,
	}
}

struct Proof<C, P> {
	abc_comm: C,
	abc_eval_proof: P,
	zerocheck_proof: ZerocheckProof,
	sumcheck_proof: SumcheckProof<BinaryField128b>,
	evalcheck_proof: EvalcheckProof<BinaryField128b>,
}

fn verify<PCS, CH>(
	log_size: usize,
	pcs: &PCS,
	trace: &mut MultilinearOracleSet<BinaryField128b>,
	constraints: &[CompositePolyOracle<BinaryField128b>],
	proof: Proof<PCS::Commitment, PCS::Proof>,
	mut challenger: CH,
) where
	PCS: PolyCommitScheme<PackedBinaryField128x1b, BinaryField128b>,
	PCS::Error: Debug,
	PCS::Proof: 'static,
	CH: CanObserve<BinaryField128b>
		+ CanObserve<PCS::Commitment>
		+ CanSample<BinaryField128b>
		+ CanSampleBits<usize>,
{
	assert_eq!(pcs.n_vars(), log_size);

	assert_eq!(constraints.len(), 1);
	let constraint = constraints[0].clone();

	let Proof {
		abc_comm,
		abc_eval_proof,
		zerocheck_proof,
		sumcheck_proof,
		evalcheck_proof,
	} = proof;

	// Observe the trace commitments
	challenger.observe(abc_comm.clone());

	let zerocheck_challenge = challenger.sample_vec(log_size - 1);

	// Run zerocheck protocol
	let zerocheck_claim = ZerocheckClaim { poly: constraint };
	let sumcheck_claim =
		verify_zerocheck(&zerocheck_claim, zerocheck_proof, zerocheck_challenge).unwrap();

	// Run sumcheck protocol
	let (_, evalcheck_claim) = full_verify(&sumcheck_claim, sumcheck_proof, &mut challenger);

	let trace_batch = trace.committed_batch(0);

	// Verify commitment openings
	let mut new_sumcheck_claims = Vec::new();
	let mut batch_committed_eval_claims = BatchCommittedEvalClaims::new(&[trace_batch]);
	verify_evalcheck(
		trace,
		evalcheck_claim,
		evalcheck_proof,
		&mut batch_committed_eval_claims,
		&mut new_sumcheck_claims,
	)
	.unwrap();

	assert!(new_sumcheck_claims.is_empty());
	assert_eq!(batch_committed_eval_claims.n_batches(), 1);
	let same_query_pcs_claim = batch_committed_eval_claims
		.try_extract_same_query_pcs_claim(0)
		.unwrap()
		.unwrap();

	pcs.verify_evaluation(
		&mut challenger,
		&abc_comm,
		&same_query_pcs_claim.eval_point,
		abc_eval_proof,
		&same_query_pcs_claim.evals,
	)
	.unwrap();
}

#[derive(Debug)]
struct TraceWitness<'a, P: PackedField> {
	a_in: MultilinearExtension<'a, P>,
	b_in: MultilinearExtension<'a, P>,
	c_out: MultilinearExtension<'a, P>,
}

impl<'a, P: PackedField> TraceWitness<'a, P> {
	fn to_index<F, PE>(&self, trace: &MultilinearOracleSet<F>) -> MultilinearWitnessIndex<PE>
	where
		F: TowerField + ExtensionField<P::Scalar>,
		PE: PackedField,
		PE::Scalar: ExtensionField<P::Scalar>,
	{
		let mut index = MultilinearWitnessIndex::new();

		index.set(
			trace.committed_oracle_id(CommittedId {
				batch_id: 0,
				index: 0,
			}),
			self.a_in.to_ref().specialize_arc_dyn(),
		);
		index.set(
			trace.committed_oracle_id(CommittedId {
				batch_id: 0,
				index: 1,
			}),
			self.b_in.to_ref().specialize_arc_dyn(),
		);
		index.set(
			trace.committed_oracle_id(CommittedId {
				batch_id: 0,
				index: 2,
			}),
			self.c_out.to_ref().specialize_arc_dyn(),
		);

		index
	}
}

fn generate_trace(log_size: usize) -> TraceWitness<'static, PackedBinaryField128x1b> {
	let len = (1 << log_size) >> PackedBinaryField128x1b::LOG_WIDTH;
	let mut a_in_vals = vec![PackedBinaryField128x1b::default(); len];
	let mut b_in_vals = vec![PackedBinaryField128x1b::default(); len];
	let mut c_out_vals = vec![PackedBinaryField128x1b::default(); len];
	a_in_vals
		.par_iter_mut()
		.zip(b_in_vals.par_iter_mut())
		.zip(c_out_vals.par_iter_mut())
		.for_each_init(thread_rng, |rng, ((a_i, b_i), c_i)| {
			*a_i = PackedBinaryField128x1b::random(&mut *rng);
			*b_i = PackedBinaryField128x1b::random(&mut *rng);
			let a_i_uint128 = must_cast::<_, u128>(*a_i);
			let b_i_uint128 = must_cast::<_, u128>(*b_i);
			let c_i_uint128 = must_cast_mut::<_, u128>(c_i);
			*c_i_uint128 = a_i_uint128 & b_i_uint128;
		});

	let a_in = MultilinearExtension::from_values(a_in_vals).unwrap();
	let b_in = MultilinearExtension::from_values(b_in_vals).unwrap();
	let c_out = MultilinearExtension::from_values(c_out_vals).unwrap();

	TraceWitness { a_in, b_in, c_out }
}

fn make_constraints<F: TowerField>(
	log_size: usize,
	trace_oracle: &MultilinearOracleSet<F>,
) -> Vec<CompositePolyOracle<F>> {
	let a_in_oracle = trace_oracle.committed_oracle(CommittedId {
		batch_id: 0,
		index: 0,
	});
	let b_in_oracle = trace_oracle.committed_oracle(CommittedId {
		batch_id: 0,
		index: 1,
	});
	let c_out_oracle = trace_oracle.committed_oracle(CommittedId {
		batch_id: 0,
		index: 2,
	});

	let constraint = CompositePolyOracle::new(
		log_size,
		vec![a_in_oracle, b_in_oracle, c_out_oracle],
		BitwiseAndConstraint,
	)
	.unwrap();

	vec![constraint]
}

fn main() {
	if let Ok(csv_path) = env::var("PROFILE_CSV_FILE") {
		let _ = tracing_subscriber::registry()
			.with(CsvLayer::new(csv_path))
			.with(tracing_subscriber::fmt::layer())
			.try_init();
	} else if env::var("PROFILE_PRINT_TREE").is_ok() {
		let _ = tracing_subscriber::registry()
			.with(PrintTreeLayer::new())
			.try_init();
	} else {
		let _ = tracing_subscriber::registry()
			.with(tracing_subscriber::fmt::layer())
			.try_init();
	};

	const SECURITY_BITS: usize = 100;

	let log_size = 24;
	let log_inv_rate = 1;

	// Set up the public parameters
	let pcs = tensor_pcs::find_proof_size_optimal_pcs::<
		_,
		PackedBinaryField128x1b,
		_,
		PackedBinaryField8x16b,
		_,
		PackedBinaryField8x16b,
		_,
		PackedBinaryField1x128b,
	>(SECURITY_BITS, log_size, 3, log_inv_rate, false)
	.unwrap();

	tracing::debug!(
		"Using BlockTensorPCS with log_rows = {}, log_cols = {}, proof_size = {}",
		pcs.log_rows(),
		pcs.log_cols(),
		pcs.proof_size(3),
	);

	let mut trace_oracle = MultilinearOracleSet::new();

	trace_oracle.add_committed_batch(CommittedBatchSpec {
		round_id: 1,
		n_vars: log_size,
		n_polys: 3,
		tower_level: 0,
	});

	let constraints = make_constraints(log_size, &trace_oracle);

	tracing::info!("Generating the trace");
	let trace_span = tracing::debug_span!("generate_trace").entered();
	let witness = generate_trace(log_size);
	drop(trace_span);

	let challenger = <HashChallenger<_, GroestlHasher<_>>>::new();

	tracing::info!("Proving");
	let prove_span = tracing::debug_span!(
		"prove",
		log_rows = pcs.log_rows(),
		log_cols = pcs.log_cols(),
		proof_size = pcs.proof_size(3),
		log_size = log_size,
		n_vars = pcs.n_vars(),
	)
	.entered();
	let proof = prove(
		log_size,
		&pcs,
		&mut trace_oracle.clone(),
		&constraints,
		&witness,
		challenger.clone(),
	);
	drop(prove_span);

	tracing::info!("Verifying");
	let verify_span = tracing::debug_span!(
		"verify",
		log_rows = pcs.log_rows(),
		log_cols = pcs.log_cols(),
		proof_size = pcs.proof_size(3),
		log_size = log_size,
		n_vars = pcs.n_vars(),
	)
	.entered();
	verify(log_size, &pcs, &mut trace_oracle.clone(), &constraints, proof, challenger.clone());
	drop(verify_span);
}
