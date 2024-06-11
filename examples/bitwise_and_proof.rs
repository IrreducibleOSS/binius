use anyhow::Result;
use binius_core::{
	challenger::HashChallenger,
	oracle::{CommittedBatchSpec, CommittedId, CompositePolyOracle, MultilinearOracleSet},
	poly_commit::{tensor_pcs, PolyCommitScheme},
	polynomial::{
		EvaluationDomainFactory, IsomorphicEvaluationDomainFactory, MultilinearComposite,
	},
	protocols::{
		greedy_evalcheck::{self, GreedyEvalcheckProof, GreedyEvalcheckProveOutput},
		zerocheck::{self, ZerocheckClaim, ZerocheckProof, ZerocheckProveOutput},
	},
	witness::MultilinearWitnessIndex,
};
use binius_field::{
	BinaryField128b, BinaryField128bPolyval, ExtensionField, PackedBinaryField128x1b,
	PackedBinaryField1x128b, PackedBinaryField8x16b, PackedField, TowerField,
};
use binius_hash::GroestlHasher;
use binius_macros::{composition_poly, IterPolys};
use binius_utils::{
	examples::get_log_trace_size, rayon::adjust_thread_pool, tracing::init_tracing,
};
use bytemuck::{must_cast, must_cast_mut};
use p3_challenger::{CanObserve, CanSample, CanSampleBits};
use rand::thread_rng;
use rayon::prelude::*;
use std::fmt::Debug;
use tracing::instrument;

composition_poly!(BitwiseAndConstraint[a, b, c] = a * b - c);

#[instrument(skip_all)]
fn prove<PCS, CH>(
	log_size: usize,
	pcs: &PCS,
	trace: &mut MultilinearOracleSet<BinaryField128b>,
	constraints: &[CompositePolyOracle<BinaryField128b>],
	witness: &TraceWitness<PackedBinaryField128x1b>,
	mut challenger: CH,
	domain_factory: impl EvaluationDomainFactory<BinaryField128bPolyval>,
) -> Result<Proof<PCS::Commitment, PCS::Proof>>
where
	PCS: PolyCommitScheme<PackedBinaryField128x1b, BinaryField128b>,
	PCS::Error: Debug,
	PCS::Proof: 'static,
	CH: CanObserve<BinaryField128b>
		+ CanObserve<PCS::Commitment>
		+ CanSample<BinaryField128b>
		+ CanSampleBits<usize>,
{
	let commit_span = tracing::debug_span!("commit").entered();
	assert_eq!(pcs.n_vars(), log_size);

	let mut witness_index = witness.to_index::<_, BinaryField128bPolyval>(trace);

	assert_eq!(constraints.len(), 1);
	let constraint = constraints[0].clone();

	// Round 1
	let (abc_comm, abc_committed) = pcs
		.commit(&witness.iter_polys().collect::<Vec<_>>())
		.unwrap();
	challenger.observe(abc_comm.clone());

	drop(commit_span);

	// Round 2

	tracing::debug!("Proving zerocheck");
	let zerocheck_witness = MultilinearComposite::<BinaryField128bPolyval, _, _>::new(
		log_size,
		BitwiseAndConstraint,
		witness
			.iter_polys()
			.map(|mle| mle.specialize_arc_dyn())
			.collect::<Vec<_>>(),
	)?;
	let zerocheck_claim = ZerocheckClaim { poly: constraint };

	// zerocheck::prove is instrumented
	let zerocheck_domain =
		domain_factory.create(zerocheck_claim.poly.max_individual_degree() + 1)?;
	let switchover_fn = |extension_degree| match extension_degree {
		128 => 5,
		_ => 1,
	};

	let ZerocheckProveOutput {
		evalcheck_claim,
		zerocheck_proof,
	} = zerocheck::prove::<BinaryField128b, BinaryField128bPolyval, BinaryField128bPolyval, _, _>(
		&zerocheck_claim,
		zerocheck_witness,
		&zerocheck_domain,
		&mut challenger,
		switchover_fn,
	)?;

	// Prove evaluation claims
	let GreedyEvalcheckProveOutput {
		same_query_claims,
		proof: evalcheck_proof,
	} = greedy_evalcheck::prove::<_, _, BinaryField128bPolyval, _>(
		trace,
		&mut witness_index,
		[evalcheck_claim],
		switchover_fn,
		&mut challenger,
		domain_factory,
	)?;

	assert_eq!(same_query_claims.len(), 1);
	let (_, same_query_pcs_claim) = same_query_claims
		.into_iter()
		.next()
		.expect("length is asserted to be 1");

	// Prove commitment openings
	let abc_eval_proof = pcs.prove_evaluation(
		&mut challenger,
		&abc_committed,
		&witness.iter_polys().collect::<Vec<_>>(),
		&same_query_pcs_claim.eval_point,
	)?;

	Ok(Proof {
		abc_comm,
		abc_eval_proof,
		zerocheck_proof,
		evalcheck_proof,
	})
}

struct Proof<C, P> {
	abc_comm: C,
	abc_eval_proof: P,
	zerocheck_proof: ZerocheckProof<BinaryField128b>,
	evalcheck_proof: GreedyEvalcheckProof<BinaryField128b>,
}

#[instrument(skip_all)]
fn verify<PCS, CH>(
	log_size: usize,
	pcs: &PCS,
	trace: &mut MultilinearOracleSet<BinaryField128b>,
	constraints: &[CompositePolyOracle<BinaryField128b>],
	proof: Proof<PCS::Commitment, PCS::Proof>,
	mut challenger: CH,
) -> Result<()>
where
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
		evalcheck_proof,
	} = proof;

	// Observe the trace commitments
	challenger.observe(abc_comm.clone());

	// Run zerocheck protocol
	let zerocheck_claim = ZerocheckClaim { poly: constraint };
	let evalcheck_claim = zerocheck::verify(&zerocheck_claim, zerocheck_proof, &mut challenger)?;

	// Verify evaluation claims
	let same_query_claims =
		greedy_evalcheck::verify(trace, [evalcheck_claim], evalcheck_proof, &mut challenger)?;

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
	)?;

	Ok(())
}

#[derive(Debug, IterPolys)]
struct TraceWitness<P: PackedField> {
	a_in: Vec<P>,
	b_in: Vec<P>,
	c_out: Vec<P>,
}

impl<P: PackedField> TraceWitness<P> {
	fn to_index<F, PE>(&self, trace: &MultilinearOracleSet<F>) -> MultilinearWitnessIndex<PE>
	where
		F: TowerField + ExtensionField<P::Scalar>,
		PE: PackedField,
		PE::Scalar: ExtensionField<P::Scalar>,
	{
		let mut index = MultilinearWitnessIndex::new();
		for (i, witness) in self.iter_polys().enumerate() {
			let id = CommittedId {
				index: i,
				batch_id: 0,
			};
			index.set(trace.committed_oracle_id(id), witness.specialize_arc_dyn());
		}
		index
	}
}

#[instrument(skip_all)]
fn generate_trace(log_size: usize) -> TraceWitness<PackedBinaryField128x1b> {
	let len = (1 << log_size) >> PackedBinaryField128x1b::LOG_WIDTH;
	let mut a_in = vec![PackedBinaryField128x1b::default(); len];
	let mut b_in = vec![PackedBinaryField128x1b::default(); len];
	let mut c_out = vec![PackedBinaryField128x1b::default(); len];
	a_in.par_iter_mut()
		.zip(b_in.par_iter_mut())
		.zip(c_out.par_iter_mut())
		.for_each_init(thread_rng, |rng, ((a_i, b_i), c_i)| {
			*a_i = PackedBinaryField128x1b::random(&mut *rng);
			*b_i = PackedBinaryField128x1b::random(&mut *rng);
			let a_i_uint128 = must_cast::<_, u128>(*a_i);
			let b_i_uint128 = must_cast::<_, u128>(*b_i);
			let c_i_uint128 = must_cast_mut::<_, u128>(c_i);
			*c_i_uint128 = a_i_uint128 & b_i_uint128;
		});
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
	adjust_thread_pool()
		.as_ref()
		.expect("failed to init thread pool");

	init_tracing();

	const SECURITY_BITS: usize = 100;

	let log_size = get_log_trace_size().unwrap_or(20);
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
		n_vars: log_size,
		n_polys: 3,
		tower_level: 0,
	});

	let constraints = make_constraints(log_size, &trace_oracle);

	tracing::info!("Generating the trace");
	let witness = generate_trace(log_size);
	let challenger = <HashChallenger<_, GroestlHasher<_>>>::new();
	let domain_factory = IsomorphicEvaluationDomainFactory::<BinaryField128b>::default();

	tracing::info!("Proving");
	let proof = prove(
		log_size,
		&pcs,
		&mut trace_oracle.clone(),
		&constraints,
		&witness,
		challenger.clone(),
		domain_factory,
	)
	.unwrap();

	tracing::info!("Verifying");
	verify(log_size, &pcs, &mut trace_oracle.clone(), &constraints, proof, challenger.clone())
		.unwrap();
}
