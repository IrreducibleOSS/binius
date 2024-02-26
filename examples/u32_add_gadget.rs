use binius::{
	challenger::HashChallenger,
	field::{
		BinaryField128b, PackedBinaryField128x1b, PackedBinaryField1x128b, PackedBinaryField8x16b,
		PackedField, TowerField,
	},
	hash::GroestlHasher,
	oracle::{
		CommittedBatch, CompositePolyOracle, MultilinearPolyOracle, MultivariatePolyOracle,
		Projected, ProjectionVariant, ShiftVariant, Shifted, TransparentPolyOracle,
	},
	poly_commit::{tensor_pcs, PolyCommitScheme},
	polynomial::{
		multilinear_query::MultilinearQuery,
		transparent::{eq_ind::EqIndPartialEval, shift_ind::LogicalRightShiftIndPartialEval},
		CompositionPoly, Error as PolynomialError, EvaluationDomain, MultilinearExtension,
		MultilinearPoly,
	},
	protocols::{
		evalcheck::{
			prove as prove_evalcheck, verify as verify_evalcheck, BatchCommittedEvalClaims,
			EvalcheckProof, SameQueryPcsClaim, ShiftedEvalClaim,
		},
		sumcheck::{
			mix_claims as mix_sumcheck_claims, mix_witnesses as mix_sumcheck_witness,
			SumcheckClaim, SumcheckProof, SumcheckProveOutput,
		},
		test_utils::{full_prove_with_switchover, full_verify},
		zerocheck::{
			mix_claims as mix_zerocheck_claims, mix_witness as mix_zerocheck_witness,
			prove as prove_zerocheck, verify as verify_zerocheck, ZerocheckClaim, ZerocheckProof,
			ZerocheckProveOutput,
		},
	},
};
use bytemuck::{must_cast_mut, must_cast_ref};
use getset::Getters;
use p3_challenger::{CanObserve, CanSample, CanSampleBits};
use rand::thread_rng;
use rayon::prelude::*;
use std::{fmt::Debug, sync::Arc};

// TRACE INFORMATION
const NUM_TRACE_COLUMNS: usize = 5;
const X_COL_IDX: usize = 0;
const Y_COL_IDX: usize = 1;
const Z_COL_IDX: usize = 2;
const C_OUT_COL_IDX: usize = 3;
const C_IN_COL_IDX: usize = 4;

// SHIFT INFORMATION
const LOG_BLOCK_SIZE: usize = 5; //u32 add, so 2^{BLOCK_SIZE} = 32

// CONSTRAINTS
#[derive(Debug)]
struct AddConstraint;

impl CompositionPoly<BinaryField128b> for AddConstraint {
	fn n_vars(&self) -> usize {
		NUM_TRACE_COLUMNS
	}

	fn degree(&self) -> usize {
		1
	}

	fn evaluate(&self, query: &[BinaryField128b]) -> Result<BinaryField128b, PolynomialError> {
		self.evaluate_packed(query)
	}

	fn evaluate_packed(
		&self,
		query: &[BinaryField128b],
	) -> Result<BinaryField128b, PolynomialError> {
		if query.len() != NUM_TRACE_COLUMNS {
			return Err(PolynomialError::IncorrectQuerySize {
				expected: NUM_TRACE_COLUMNS,
			});
		}
		let x = query[X_COL_IDX];
		let y = query[Y_COL_IDX];
		let z = query[Z_COL_IDX];
		let c_in = query[C_IN_COL_IDX];
		Ok(x + y - z + c_in)
	}

	fn binary_tower_level(&self) -> usize {
		0
	}
}

#[derive(Debug)]
struct CarryConstraint;

impl CompositionPoly<BinaryField128b> for CarryConstraint {
	fn n_vars(&self) -> usize {
		NUM_TRACE_COLUMNS
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
		if query.len() != NUM_TRACE_COLUMNS {
			return Err(PolynomialError::IncorrectQuerySize {
				expected: NUM_TRACE_COLUMNS,
			});
		}
		let x = query[X_COL_IDX];
		let y = query[Y_COL_IDX];
		let c_in = query[C_IN_COL_IDX];
		let c_out = query[C_OUT_COL_IDX];
		Ok(x * y + x * c_in + y * c_in - c_out)
	}

	fn binary_tower_level(&self) -> usize {
		0
	}
}

#[derive(Debug, Copy, Clone)]
struct TwofoldProductComposition;
impl<F: TowerField> CompositionPoly<F> for TwofoldProductComposition {
	fn n_vars(&self) -> usize {
		2
	}

	fn degree(&self) -> usize {
		2
	}

	fn evaluate(&self, query: &[F]) -> Result<F, PolynomialError> {
		self.evaluate_packed(query)
	}

	fn evaluate_packed(&self, query: &[F]) -> Result<F, PolynomialError> {
		if query.len() != 2 {
			return Err(PolynomialError::IncorrectQuerySize { expected: 2 });
		}
		Ok(query[0] * query[1])
	}

	fn binary_tower_level(&self) -> usize {
		0
	}
}

#[derive(Debug, Getters)]
struct ProverTrace {
	#[get = "pub"]
	x: MultilinearExtension<'static, PackedBinaryField128x1b>,
	#[get = "pub"]
	y: MultilinearExtension<'static, PackedBinaryField128x1b>,
	#[get = "pub"]
	z: MultilinearExtension<'static, PackedBinaryField128x1b>,
	#[get = "pub"]
	c_in: MultilinearExtension<'static, PackedBinaryField128x1b>,
	#[get = "pub"]
	c_out: MultilinearExtension<'static, PackedBinaryField128x1b>,
}

impl ProverTrace {
	fn get_all_columns(&self) -> Vec<Arc<dyn MultilinearPoly<BinaryField128b> + Send + Sync>> {
		vec![
			Arc::new(self.x.clone()) as Arc<dyn MultilinearPoly<BinaryField128b> + Send + Sync>,
			Arc::new(self.y.clone()) as _,
			Arc::new(self.z.clone()) as _,
			Arc::new(self.c_out.clone()) as _,
			Arc::new(self.c_in.clone()) as _,
		]
	}

	fn get_committed_columns_vec(
		&self,
	) -> Vec<&MultilinearExtension<'static, PackedBinaryField128x1b>> {
		vec![&self.x, &self.y, &self.z, &self.c_out]
	}
}

#[derive(Debug, Clone)]
struct VerifierTrace {
	trace_batch: CommittedBatch,
	x_oracle: MultilinearPolyOracle<BinaryField128b>,
	y_oracle: MultilinearPolyOracle<BinaryField128b>,
	z_oracle: MultilinearPolyOracle<BinaryField128b>,
	c_in_oracle: MultilinearPolyOracle<BinaryField128b>,
	c_out_oracle: MultilinearPolyOracle<BinaryField128b>,
}

impl VerifierTrace {
	fn into_all_column_oracles(self) -> Vec<MultilinearPolyOracle<BinaryField128b>> {
		vec![
			self.x_oracle,
			self.y_oracle,
			self.z_oracle,
			self.c_out_oracle,
			self.c_in_oracle,
		]
	}

	fn into_committed_column_oracles(self) -> Vec<MultilinearPolyOracle<BinaryField128b>> {
		vec![
			self.x_oracle,
			self.y_oracle,
			self.z_oracle,
			self.c_out_oracle,
		]
	}
}

fn project_to_last_vars(
	oracles: Vec<MultilinearPolyOracle<BinaryField128b>>,
	partial_eval_point: &[BinaryField128b],
) -> Vec<MultilinearPolyOracle<BinaryField128b>> {
	oracles
		.into_iter()
		.map(|oracle| {
			let projected =
				Projected::new(oracle, partial_eval_point.to_vec(), ProjectionVariant::LastVars)
					.unwrap();
			MultilinearPolyOracle::Projected(projected)
		})
		.collect()
}

// Makes the batched zerocheck claim for gate constraints
fn make_batched_zerocheck_claim<CH>(
	log_size: usize,
	vtrace: VerifierTrace,
	challenger: CH,
) -> ZerocheckClaim<BinaryField128b>
where
	CH: CanObserve<BinaryField128b> + CanSample<BinaryField128b>,
{
	// Addition Constraint PolyOracle
	let add_constraint = CompositePolyOracle::new(
		log_size,
		vtrace.clone().into_all_column_oracles(),
		Arc::new(AddConstraint),
	)
	.expect("Failed to create AddConstraint PolyOracle");

	let add_zero_claim = ZerocheckClaim {
		poly: MultivariatePolyOracle::Composite(add_constraint),
	};

	// Carry Constraint PolyOracle
	let carry_constraint = CompositePolyOracle::new(
		log_size,
		vtrace.clone().into_all_column_oracles(),
		Arc::new(CarryConstraint),
	)
	.expect("Failed to create CarryConstraint PolyOracle");
	let carry_zero_claim = ZerocheckClaim {
		poly: MultivariatePolyOracle::Composite(carry_constraint),
	};

	let zerocheck_claims = [add_zero_claim, carry_zero_claim];

	let batched_claim = mix_zerocheck_claims(
		log_size,
		vtrace.clone().into_all_column_oracles(),
		zerocheck_claims.iter(),
		challenger,
	)
	.expect("Failed to mix zerocheck claims");

	batched_claim
}

// Each reduced evalcheck claim on a multilinear can be thought of as a sumcheck claim
// with the appropriate transparent polynomial multiplied by the multilinear.
// This function makes each of these sumcheck claims and then mixes them into one.
fn make_second_sumcheck_mixed_claim<CH>(
	shifted_eval_claim: ShiftedEvalClaim<BinaryField128b>,
	pcs_eval_claim: SameQueryPcsClaim<BinaryField128b>,
	eval_point: Vec<BinaryField128b>,
	verifier_trace: VerifierTrace,
	challenger: CH,
) -> SumcheckClaim<BinaryField128b>
where
	CH: CanObserve<BinaryField128b> + CanSample<BinaryField128b>,
{
	// Make all the required multilinears
	let b = shifted_eval_claim.shifted.block_size();
	debug_assert_eq!(b, LOG_BLOCK_SIZE);
	debug_assert_eq!(eval_point, shifted_eval_claim.eval_point);
	debug_assert_eq!(eval_point, pcs_eval_claim.eval_point);
	let offset = shifted_eval_claim.shifted.shift_offset();

	let q_lo = &eval_point[..b];
	let q_hi = &eval_point[b..];

	let committed_col_oracles = verifier_trace.into_committed_column_oracles();
	let committed_multilins = project_to_last_vars(committed_col_oracles, q_hi);
	debug_assert_eq!(committed_multilins.len(), pcs_eval_claim.evals.len());
	let c_lo = committed_multilins[3].clone();

	let eq_qlo = EqIndPartialEval::new(b, q_lo.to_vec()).unwrap();
	let transparent = TransparentPolyOracle::new(Arc::new(eq_qlo), BinaryField128b::TOWER_LEVEL);
	let eq_lo = MultilinearPolyOracle::Transparent(transparent);

	let shift_ind_qlo = LogicalRightShiftIndPartialEval::new(b, offset, q_lo.to_vec()).unwrap();
	let transparent =
		TransparentPolyOracle::new(Arc::new(shift_ind_qlo), BinaryField128b::TOWER_LEVEL);
	let shift_lo = MultilinearPolyOracle::Transparent(transparent);

	let mut all_multilinears = committed_multilins.clone();
	all_multilinears.push(eq_lo.clone());
	all_multilinears.push(shift_lo.clone());

	// Make all the sumcheck claims
	let mut all_sumcheck_claims = committed_multilins
		.iter()
		.zip(pcs_eval_claim.evals.iter())
		.map(|(comm_multilin, &eval)| {
			let poly = MultivariatePolyOracle::Composite(
				CompositePolyOracle::new(
					b,
					vec![comm_multilin.clone(), eq_lo.clone()],
					Arc::new(TwofoldProductComposition),
				)
				.unwrap(),
			);
			SumcheckClaim { poly, sum: eval }
		})
		.collect::<Vec<_>>();
	all_sumcheck_claims.push(SumcheckClaim {
		poly: MultivariatePolyOracle::Composite(
			CompositePolyOracle::new(b, vec![c_lo, shift_lo], Arc::new(TwofoldProductComposition))
				.unwrap(),
		),
		sum: shifted_eval_claim.eval,
	});

	// Mix all the sumcheck claims
	let mixed_sumcheck_claim =
		mix_sumcheck_claims(b, all_multilinears, all_sumcheck_claims.iter(), challenger)
			.expect("Failed to mix sumcheck claims");
	mixed_sumcheck_claim
}

fn prove<PCS, CH>(
	log_size: usize,
	pcs: &PCS,
	prover_trace: ProverTrace,
	verifier_trace: VerifierTrace,
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
	let committed_cols = prover_trace.get_committed_columns_vec();
	debug_assert_eq!(pcs.n_vars(), log_size);

	// Round 1
	let (trace_comm, trace_committed) = pcs.commit(&committed_cols).unwrap();
	let mut batch_committed_eval_claims =
		BatchCommittedEvalClaims::new(&[verifier_trace.trace_batch.clone()]);

	challenger.observe(trace_comm.clone());

	// Round 2

	// Run zerocheck protocol on batched zerocheck claim/witness
	let zerocheck_claim =
		make_batched_zerocheck_claim(log_size, verifier_trace.clone(), &mut challenger);
	let multilinear_extensions = prover_trace.get_all_columns().clone();
	let zerocheck_witness = mix_zerocheck_witness(zerocheck_claim.clone(), multilinear_extensions)
		.expect("Failed to mix zerocheck witness");

	// sanity check that zerocheck claim is true
	for index in 0..(1 << log_size) {
		debug_assert_eq!(
			zerocheck_witness.evaluate_on_hypercube(index).unwrap(),
			BinaryField128b::new(0)
		);
	}

	let zerocheck_challenge = challenger.sample_vec(log_size);
	let ZerocheckProveOutput {
		sumcheck_claim,
		sumcheck_witness,
		zerocheck_proof,
	} = prove_zerocheck(zerocheck_witness, &zerocheck_claim, zerocheck_challenge).unwrap();

	let sumcheck_domain =
		EvaluationDomain::new(sumcheck_claim.poly.max_individual_degree() + 1).unwrap();
	// TODO: Improve the logic to commit the optimal switchover.
	let switchover = log_size / 2;

	// Run sumcheck protocol
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

	// Prove evalcheck
	tracing::debug!("Proving evalcheck");
	let eval_point = evalcheck_claim.eval_point.clone();
	debug_assert!(evalcheck_claim.is_random_point);
	let mut shifted_eval_claims = Vec::new();
	let mut packed_eval_claims = Vec::new();
	let evalcheck_proof = prove_evalcheck(
		&evalcheck_witness,
		evalcheck_claim,
		&mut batch_committed_eval_claims,
		&mut shifted_eval_claims,
		&mut packed_eval_claims,
	)
	.unwrap();
	debug_assert_eq!(shifted_eval_claims.len(), 1);
	debug_assert_eq!(packed_eval_claims.len(), 0);

	// Get all claimed evaluations
	let shifted_eval_claim = shifted_eval_claims.pop().unwrap();
	let mut all_claimed_evals = batch_committed_eval_claims
		.try_extract_same_query_pcs_claim(0)
		.unwrap()
		.unwrap()
		.evals;
	all_claimed_evals.push(shifted_eval_claim.eval);

	// Run sumcheck protocol again
	// make second sumcheck witness and claim
	let b = shifted_eval_claim.shifted.block_size();
	debug_assert_eq!(b, LOG_BLOCK_SIZE);
	let offset = shifted_eval_claim.shifted.shift_offset();
	let same_query_pcs_claim = batch_committed_eval_claims
		.try_extract_same_query_pcs_claim(0)
		.unwrap()
		.unwrap();
	let second_sumcheck_claim = make_second_sumcheck_mixed_claim(
		shifted_eval_claim,
		same_query_pcs_claim,
		eval_point.clone(),
		verifier_trace.clone(),
		&mut challenger,
	);

	let q_lo = eval_point[0..b].to_vec();
	let q_hi = eval_point[b..].to_vec();
	let eq_qlo = EqIndPartialEval::new(b, q_lo.clone())
		.unwrap()
		.multilinear_extension()
		.unwrap();
	let shift_ind_qlo = LogicalRightShiftIndPartialEval::new(b, offset, q_lo.clone())
		.unwrap()
		.multilinear_extension()
		.unwrap();

	let q_hi_query = MultilinearQuery::with_full_query(&q_hi).unwrap();
	let xlo = prover_trace.x().evaluate_partial_high(&q_hi_query).unwrap();
	let ylo = prover_trace.y().evaluate_partial_high(&q_hi_query).unwrap();
	let zlo = prover_trace.z().evaluate_partial_high(&q_hi_query).unwrap();
	let clo = prover_trace
		.c_out()
		.evaluate_partial_high(&q_hi_query)
		.unwrap();

	fn into_dyn<P: PackedField + Debug>(
		inp: MultilinearExtension<'static, P>,
	) -> Arc<dyn MultilinearPoly<P> + Send + Sync> {
		Arc::new(inp) as Arc<dyn MultilinearPoly<P> + Send + Sync>
	}

	let arc_xlo = into_dyn(xlo);
	let arc_ylo = into_dyn(ylo);
	let arc_zlo = into_dyn(zlo);
	let arc_clo = into_dyn(clo);
	let arc_eq_qlo = into_dyn(eq_qlo);
	let arc_shift_ind_qlo = into_dyn(shift_ind_qlo);

	let all_multilinear_extensions = vec![
		arc_xlo,
		arc_ylo,
		arc_zlo,
		arc_clo,
		arc_eq_qlo,
		arc_shift_ind_qlo,
	];
	type ProverPoly = dyn MultilinearPoly<BinaryField128b> + Send + Sync;
	let second_sumcheck_witness = mix_sumcheck_witness::<_, ProverPoly, Arc<ProverPoly>>(
		second_sumcheck_claim.clone(),
		all_multilinear_extensions,
	)
	.expect("Failed to mix second sumcheck witness");

	// second sumcheck domain
	let second_sumcheck_domain =
		EvaluationDomain::new(second_sumcheck_witness.degree() + 1).unwrap();
	let second_switchover = 2; // 5 vars now, 5/2 = 2
	tracing::debug!("Proving second sumcheck");
	let (_, second_sumcheck_prove_output) = full_prove_with_switchover(
		&second_sumcheck_claim,
		second_sumcheck_witness,
		&second_sumcheck_domain,
		&mut challenger,
		second_switchover,
	);

	let SumcheckProveOutput {
		evalcheck_claim: second_evalcheck_claim,
		evalcheck_witness: second_evalcheck_witness,
		sumcheck_proof: second_sumcheck_proof,
	} = second_sumcheck_prove_output;
	// do second evalcheck
	tracing::debug!("Proving second evalcheck");
	let mut batch_committed_eval_claims =
		BatchCommittedEvalClaims::new(&[verifier_trace.trace_batch.clone()]);
	debug_assert!(second_evalcheck_claim.is_random_point);
	let second_evalcheck_proof = prove_evalcheck(
		&second_evalcheck_witness,
		second_evalcheck_claim,
		&mut batch_committed_eval_claims,
		&mut shifted_eval_claims,
		&mut packed_eval_claims,
	)
	.unwrap();
	debug_assert_eq!(shifted_eval_claims.len(), 0);
	debug_assert_eq!(packed_eval_claims.len(), 0);
	debug_assert_eq!(batch_committed_eval_claims.n_batches(), 1);
	let same_query_pcs_claim = batch_committed_eval_claims
		.try_extract_same_query_pcs_claim(0)
		.unwrap()
		.unwrap();

	tracing::debug!("Proving xyzc PCS eval");
	let trace_eval_proof = pcs
		.prove_evaluation(
			&mut challenger,
			&trace_committed,
			&committed_cols,
			&same_query_pcs_claim.eval_point,
		)
		.unwrap();

	Proof {
		trace_comm,
		trace_eval_proof,
		zerocheck_proof,
		sumcheck_proof,
		evalcheck_proof,
		second_sumcheck_proof,
		second_evalcheck_proof,
	}
}

struct Proof<C, P> {
	trace_comm: C,
	trace_eval_proof: P,
	zerocheck_proof: ZerocheckProof,
	sumcheck_proof: SumcheckProof<BinaryField128b>,
	evalcheck_proof: EvalcheckProof<BinaryField128b>,
	second_sumcheck_proof: SumcheckProof<BinaryField128b>,
	second_evalcheck_proof: EvalcheckProof<BinaryField128b>,
}

fn verify<PCS, CH>(
	log_size: usize,
	pcs: &PCS,
	proof: Proof<PCS::Commitment, PCS::Proof>,
	verifier_trace: VerifierTrace,
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
	debug_assert_eq!(pcs.n_vars(), log_size);
	let Proof {
		trace_comm,
		trace_eval_proof,
		zerocheck_proof,
		sumcheck_proof,
		evalcheck_proof,
		second_sumcheck_proof,
		second_evalcheck_proof,
	} = proof;

	// Round 1
	challenger.observe(trace_comm.clone());

	// Round 2

	// make mixing claim for batching zerochecks
	tracing::debug!("Making batched zerocheck claim");
	let zerocheck_claim =
		make_batched_zerocheck_claim(log_size, verifier_trace.clone(), &mut challenger);

	// verify zerocheck with the mixed claim
	tracing::debug!("Verifying first zerocheck");
	let zerocheck_challenge = challenger.sample_vec(log_size);
	let sumcheck_claim =
		verify_zerocheck(&zerocheck_claim, zerocheck_proof, zerocheck_challenge).unwrap();

	// Run sumcheck protocol
	tracing::debug!("Verifying first sumcheck");
	let sumcheck_domain =
		EvaluationDomain::new(sumcheck_claim.poly.max_individual_degree() + 1).unwrap();
	let (_, evalcheck_claim) =
		full_verify(&sumcheck_claim, sumcheck_proof, &sumcheck_domain, &mut challenger);

	// Verify evalcheck
	tracing::debug!("Verifying first evalcheck");
	let eval_point = evalcheck_claim.eval_point.clone();
	let mut batch_committed_eval_claims =
		BatchCommittedEvalClaims::new(&[verifier_trace.trace_batch.clone()]);
	let mut shifted_eval_claims = Vec::new();
	let mut packed_eval_claims = Vec::new();
	verify_evalcheck(
		evalcheck_claim,
		evalcheck_proof,
		&mut batch_committed_eval_claims,
		&mut shifted_eval_claims,
		&mut packed_eval_claims,
	)
	.unwrap();
	debug_assert_eq!(batch_committed_eval_claims.n_batches(), 1);
	debug_assert_eq!(shifted_eval_claims.len(), 1);
	debug_assert_eq!(packed_eval_claims.len(), 0);

	// Get all claimed evaluations
	let shifted_eval_claim = shifted_eval_claims.pop().unwrap();
	let mut all_claimed_evals = batch_committed_eval_claims
		.try_extract_same_query_pcs_claim(0)
		.unwrap()
		.unwrap()
		.evals;
	all_claimed_evals.push(shifted_eval_claim.eval);

	// Run sumcheck protocol again
	let same_query_pcs_claim = batch_committed_eval_claims
		.try_extract_same_query_pcs_claim(0)
		.unwrap()
		.unwrap();
	let second_sumcheck_claim = make_second_sumcheck_mixed_claim(
		shifted_eval_claim,
		same_query_pcs_claim,
		eval_point,
		verifier_trace.clone(),
		&mut challenger,
	);

	tracing::debug!("Verifying second sumcheck");
	let second_sumcheck_domain =
		EvaluationDomain::new(second_sumcheck_claim.poly.max_individual_degree() + 1).unwrap();
	let (_, second_evalcheck_claim) = full_verify(
		&second_sumcheck_claim,
		second_sumcheck_proof,
		&second_sumcheck_domain,
		&mut challenger,
	);

	// do second evalcheck verification
	tracing::debug!("Verifying second evalcheck");
	let mut batch_committed_eval_claims =
		BatchCommittedEvalClaims::new(&[verifier_trace.trace_batch.clone()]);

	verify_evalcheck(
		second_evalcheck_claim,
		second_evalcheck_proof,
		&mut batch_committed_eval_claims,
		&mut shifted_eval_claims,
		&mut packed_eval_claims,
	)
	.unwrap();
	debug_assert_eq!(batch_committed_eval_claims.n_batches(), 1);
	debug_assert_eq!(shifted_eval_claims.len(), 0);
	debug_assert_eq!(packed_eval_claims.len(), 0);

	let same_query_pcs_claim = batch_committed_eval_claims
		.try_extract_same_query_pcs_claim(0)
		.unwrap()
		.unwrap();

	tracing::debug!("Verifying xyzc PCS eval");
	pcs.verify_evaluation(
		&mut challenger,
		&trace_comm,
		&same_query_pcs_claim.eval_point,
		trace_eval_proof,
		&same_query_pcs_claim.evals,
	)
	.unwrap();
}

fn main() {
	tracing_subscriber::fmt::init();

	const SECURITY_BITS: usize = 100;

	// log_size is the log of the number of rows in the trace
	let log_size = 20;
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

	tracing::info!("Generating the trace");
	// prover view of trace
	let len = (1 << log_size) / PackedBinaryField128x1b::WIDTH;
	let mut x_vals = vec![PackedBinaryField128x1b::default(); len];
	let mut y_vals = vec![PackedBinaryField128x1b::default(); len];
	let mut z_vals = vec![PackedBinaryField128x1b::default(); len];
	let mut c_in_vals = vec![PackedBinaryField128x1b::default(); len];
	let mut c_out_vals = vec![PackedBinaryField128x1b::default(); len];
	x_vals
		.par_iter_mut()
		.zip(y_vals.par_iter_mut())
		.zip(z_vals.par_iter_mut())
		.zip(c_in_vals.par_iter_mut())
		.zip(c_out_vals.par_iter_mut())
		.for_each_init(thread_rng, |rng, ((((x, y), z), c_in), c_out)| {
			*x = PackedBinaryField128x1b::random(&mut *rng);
			*y = PackedBinaryField128x1b::random(&mut *rng);

			let x_u32 = must_cast_ref::<_, [u32; 4]>(x);
			let y_u32 = must_cast_ref::<_, [u32; 4]>(y);
			let z_u32 = must_cast_mut::<_, [u32; 4]>(z);
			let c_in_u32 = must_cast_mut::<_, [u32; 4]>(c_in);
			let c_out_u32 = must_cast_mut::<_, [u32; 4]>(c_out);

			for i in 0..4 {
				let carry;
				(z_u32[i], carry) = x_u32[i].overflowing_add(y_u32[i]);
				c_in_u32[i] = x_u32[i] ^ y_u32[i] ^ z_u32[i];
				c_out_u32[i] = (c_in_u32[i] >> 1) | (if carry { 1 << 31 } else { 0 });
			}
		});

	let x = MultilinearExtension::from_values(x_vals).unwrap();
	let y = MultilinearExtension::from_values(y_vals).unwrap();
	let z = MultilinearExtension::from_values(z_vals).unwrap();
	let c_in = MultilinearExtension::from_values(c_in_vals).unwrap();
	let c_out = MultilinearExtension::from_values(c_out_vals).unwrap();
	let trace = ProverTrace {
		x,
		y,
		z,
		c_in,
		c_out,
	};

	// verifier view of trace
	let trace_batch = CommittedBatch {
		id: 0,
		round_id: 1,
		n_vars: log_size,
		n_polys: 4,
		tower_level: 0,
	};

	let x_oracle = trace_batch.oracle(X_COL_IDX).unwrap();
	let y_oracle = trace_batch.oracle(Y_COL_IDX).unwrap();
	let z_oracle = trace_batch.oracle(Z_COL_IDX).unwrap();
	let c_out_oracle = trace_batch.oracle(C_OUT_COL_IDX).unwrap();

	let logical_right_shift_offset = 1;
	let logical_right_shift_block_size = LOG_BLOCK_SIZE;
	let shifted = Shifted::new(
		c_out_oracle.clone(),
		logical_right_shift_offset,
		logical_right_shift_block_size,
		ShiftVariant::LogicalRight,
	)
	.unwrap();
	let c_in_oracle = MultilinearPolyOracle::Shifted(shifted);
	let verifier_trace = VerifierTrace {
		trace_batch,
		x_oracle,
		y_oracle,
		z_oracle,
		c_in_oracle,
		c_out_oracle,
	};

	// Set up the challenger
	let challenger = <HashChallenger<_, GroestlHasher<_>>>::new();

	tracing::info!("Proving");
	let proof = prove(log_size, &pcs, trace, verifier_trace.clone(), challenger.clone());
	tracing::info!("Verifying");
	verify(log_size, &pcs, proof, verifier_trace, challenger.clone());
}
