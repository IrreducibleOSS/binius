use binius::{
	challenger::HashChallenger,
	field::{
		BinaryField128b, PackedBinaryField128x1b, PackedBinaryField1x128b, PackedBinaryField8x16b,
		PackedField,
	},
	hash::GroestlHasher,
	iopoly::{CompositePolyOracle, MultilinearPolyOracle, MultivariatePolyOracle},
	poly_commit::{BlockTensorPCS, PolyCommitScheme},
	polynomial::{
		CompositionPoly, Error as PolynomialError, EvaluationDomain, MultilinearComposite,
		MultilinearExtension, MultilinearPoly,
	},
	protocols::{
		evalcheck::{
			evalcheck::{BatchCommittedEvalClaims, EvalcheckProof},
			prove::prove as prove_evalcheck,
			verify::verify as verify_evalcheck,
		},
		sumcheck::{SumcheckProof, SumcheckProveOutput},
		test_utils::{full_prove_with_switchover, full_verify},
		zerocheck::{
			prove::prove as prove_zerocheck,
			verify::verify as verify_zerocheck,
			zerocheck::{ZerocheckClaim, ZerocheckProof, ZerocheckProveOutput},
		},
	},
	reed_solomon::reed_solomon::ReedSolomonCode,
};
use bytemuck::{must_cast, must_cast_mut};
use p3_challenger::{CanObserve, CanSample, CanSampleBits};
use rand::thread_rng;
use rayon::prelude::*;
use std::{fmt::Debug, sync::Arc};

#[derive(Debug)]
struct BitwiseAndConstraint;

impl CompositionPoly<BinaryField128b> for BitwiseAndConstraint {
	fn n_vars(&self) -> usize {
		3
	}

	fn degree(&self) -> usize {
		2
	}

	fn evaluate(&self, query: &[BinaryField128b]) -> Result<BinaryField128b, PolynomialError> {
		self.evaluate_packed(query)
	}

	fn evaluate_packed(
		&self,
		query: &[BinaryField128b],
	) -> Result<BinaryField128b, PolynomialError> {
		if query.len() != 3 {
			return Err(PolynomialError::IncorrectQuerySize { expected: 3 });
		}
		let a = query[0];
		let b = query[1];
		let c = query[2];
		Ok(a * b - c)
	}
}

fn prove<PCS, CH>(
	log_size: usize,
	pcs: &PCS,
	a_in: MultilinearExtension<PackedBinaryField128x1b>,
	b_in: MultilinearExtension<PackedBinaryField128x1b>,
	c_out: MultilinearExtension<PackedBinaryField128x1b>,
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
	assert_eq!(pcs.n_vars(), log_size);
	assert_eq!(a_in.n_vars(), log_size);
	assert_eq!(b_in.n_vars(), log_size);
	assert_eq!(c_out.n_vars(), log_size);

	let a_in_oracle = MultilinearPolyOracle::Committed {
		id: 0,
		n_vars: log_size,
		tower_level: 0,
	};
	let b_in_oracle = MultilinearPolyOracle::Committed {
		id: 1,
		n_vars: log_size,
		tower_level: 0,
	};
	let c_out_oracle = MultilinearPolyOracle::Committed {
		id: 2,
		n_vars: log_size,
		tower_level: 0,
	};

	let constraint = MultivariatePolyOracle::Composite(
		CompositePolyOracle::new(
			log_size,
			vec![a_in_oracle, b_in_oracle, c_out_oracle],
			Arc::new(BitwiseAndConstraint),
		)
		.unwrap(),
	);

	// Round 1
	let (abc_comm, abc_committed) = pcs.commit(&[&a_in, &b_in, &c_out]).unwrap();
	challenger.observe(abc_comm.clone());

	let mut batch_committed_eval_claims = BatchCommittedEvalClaims::new(&[[0, 1, 2]]);

	// Round 2
	let zerocheck_challenge = challenger.sample_vec(log_size);

	let zerocheck_witness = MultilinearComposite::new(
		log_size,
		Arc::new(BitwiseAndConstraint),
		vec![
			Arc::new(a_in.borrow_copy()) as Arc<dyn MultilinearPoly<BinaryField128b> + Send + Sync>,
			Arc::new(b_in.borrow_copy()) as Arc<dyn MultilinearPoly<BinaryField128b> + Send + Sync>,
			Arc::new(c_out.borrow_copy())
				as Arc<dyn MultilinearPoly<BinaryField128b> + Send + Sync>,
		],
	)
	.unwrap();

	let zerocheck_claim = ZerocheckClaim { poly: constraint };
	let ZerocheckProveOutput {
		sumcheck_claim,
		sumcheck_witness,
		zerocheck_proof,
	} = prove_zerocheck(zerocheck_witness, &zerocheck_claim, zerocheck_challenge).unwrap();

	let sumcheck_domain =
		EvaluationDomain::new(sumcheck_claim.poly.max_individual_degree() + 1).unwrap();

	// TODO: Improve the logic to commit the optimal switchover.
	let switchover = log_size / 2;

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
		evalcheck_witness,
		sumcheck_proof,
	} = output;

	let evalcheck_proof =
		prove_evalcheck(evalcheck_witness, evalcheck_claim, &mut batch_committed_eval_claims)
			.unwrap();

	assert_eq!(batch_committed_eval_claims.nbatches(), 1);
	let same_query_pcs_claim = batch_committed_eval_claims
		.get_same_query_pcs_claim(0)
		.unwrap()
		.unwrap();

	let abc_eval_proof = pcs
		.prove_evaluation(
			&mut challenger,
			&abc_committed,
			&[&a_in, &b_in, &c_out],
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

	let a_in_oracle = MultilinearPolyOracle::Committed {
		id: 0,
		n_vars: log_size,
		tower_level: 0,
	};
	let b_in_oracle = MultilinearPolyOracle::Committed {
		id: 1,
		n_vars: log_size,
		tower_level: 0,
	};
	let c_out_oracle = MultilinearPolyOracle::Committed {
		id: 2,
		n_vars: log_size,
		tower_level: 0,
	};

	let constraint = MultivariatePolyOracle::Composite(
		CompositePolyOracle::new(
			log_size,
			vec![a_in_oracle, b_in_oracle, c_out_oracle],
			Arc::new(BitwiseAndConstraint),
		)
		.unwrap(),
	);

	let Proof {
		abc_comm,
		abc_eval_proof,
		zerocheck_proof,
		sumcheck_proof,
		evalcheck_proof,
	} = proof;

	// Observe the trace commitments
	challenger.observe(abc_comm.clone());

	let zerocheck_challenge = challenger.sample_vec(log_size);

	// Run zerocheck protocol
	let zerocheck_claim = ZerocheckClaim { poly: constraint };
	let sumcheck_claim =
		verify_zerocheck(&zerocheck_claim, zerocheck_proof, zerocheck_challenge).unwrap();

	// Run sumcheck protocol
	let sumcheck_domain =
		EvaluationDomain::new(sumcheck_claim.poly.max_individual_degree() + 1).unwrap();

	let (_, evalcheck_claim) =
		full_verify(&sumcheck_claim, sumcheck_proof, &sumcheck_domain, &mut challenger);

	// Verify commitment openings
	let mut batch_committed_eval_claims = BatchCommittedEvalClaims::new(&[[0, 1, 2]]);
	verify_evalcheck(evalcheck_claim, evalcheck_proof, &mut batch_committed_eval_claims).unwrap();

	assert_eq!(batch_committed_eval_claims.nbatches(), 1);
	let same_query_pcs_claim = batch_committed_eval_claims
		.get_same_query_pcs_claim(0)
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

fn main() {
	tracing_subscriber::fmt::init();

	let log_size = 20;

	// Set up the public parameters
	let rs_code = ReedSolomonCode::<PackedBinaryField8x16b>::new(8, 1, 64).unwrap();
	let pcs =
		BlockTensorPCS::<_, _, PackedBinaryField1x128b, _, _, _>::new_using_groestl_merkle_tree(
			8, rs_code,
		)
		.unwrap();

	tracing::info!("Generating the trace");

	let len = (1 << log_size) / PackedBinaryField128x1b::WIDTH;
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

	let challenger = <HashChallenger<_, GroestlHasher<_>>>::new();

	tracing::info!("Proving");
	let proof = prove(log_size, &pcs, a_in, b_in, c_out, challenger.clone());

	tracing::info!("Verifying");
	verify(log_size, &pcs, proof, challenger.clone());
}
