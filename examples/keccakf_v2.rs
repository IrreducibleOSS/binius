// Copyright 2024 Ulvetanna Inc.

//! Example of a Binius SNARK that proves execution of Keccak-f[1600] permutations.
//!
//! The Keccak-f permutation is the core of the SHA-3 and Keccak-256 hash functions. This example
//! proves and verifies a commit-and-prove SNARK for many independent Keccak-f permutations. That
//! means there are not boundary constraints, this simple proves a relation between the data
//! committed by the input and output columns.
//!
//! The arithmetization uses 1-bit committed columns. Each column treats chunks of 64 contiguous
//! bits as a 64-bit state element of the 25 x 64-bit Keccak-f state. Every row of 64-bit chunks
//! attests to the validity of one Keccak-f round. Each permutation consists of 24 chained rounds.
//!
//! For Keccak-f specification and pseudocode, see
//! [Keccak specifications summary](https://keccak.team/keccak_specs_summary.html).

#![feature(step_trait)]

use anyhow::Result;
use binius_backend_provider::make_backend;
use binius_core::{
	challenger::{
		new_hasher_challenger, CanObserve, CanSample, CanSampleBits, IsomorphicChallenger,
	},
	oracle::{
		BatchId, ConstraintSet, ConstraintSetBuilder, MultilinearOracleSet, OracleId, ShiftVariant,
	},
	poly_commit::{tensor_pcs, PolyCommitScheme},
	polynomial::{CompositionPoly, Error as PolynomialError},
	protocols::{
		abstract_sumcheck::standard_switchover_heuristic,
		greedy_evalcheck::{self, GreedyEvalcheckProof, GreedyEvalcheckProveOutput},
		sumcheck_v2::{self, Proof as ZerocheckProof},
		test_utils::conflate_multilinear_evalchecks,
	},
	transparent::{multilinear_extension::MultilinearExtensionTransparent, step_down::StepDown},
	witness::MultilinearExtensionIndex,
};
use binius_field::{
	arch::packed_64::PackedBinaryField64x1b,
	as_packed_field::{PackScalar, PackedType},
	underlier::{UnderlierType, WithUnderlier},
	BinaryField, BinaryField128b, BinaryField128bPolyval, BinaryField16b, BinaryField1b,
	ExtensionField, Field, PackedBinaryField128x1b, PackedField, PackedFieldIndexable, TowerField,
};
use binius_hal::ComputationBackend;
use binius_hash::GroestlHasher;
use binius_macros::{composition_poly, IterOracles};
use binius_math::{EvaluationDomainFactory, IsomorphicEvaluationDomainFactory};
use binius_utils::{
	examples::get_log_trace_size, rayon::adjust_thread_pool, tracing::init_tracing,
};
use bytemuck::{must_cast_slice, must_cast_slice_mut, Pod};
use bytesize::ByteSize;
use itertools::chain;
use rand::{thread_rng, Rng};
use std::{array, fmt::Debug, iter};
use tiny_keccak::keccakf;
use tracing::instrument;

const LOG_ROWS_PER_ROUND: usize = 6;
const LOG_ROUNDS_PER_PERMUTATION: usize = 5;
const LOG_ROWS_PER_PERMUTATION: usize = LOG_ROWS_PER_ROUND + LOG_ROUNDS_PER_PERMUTATION;
const ROUNDS_PER_PERMUTATION: usize = 24;

const KECCAKF_RC: [u64; 32] = [
	0x0000000000000001,
	0x0000000000008082,
	0x800000000000808A,
	0x8000000080008000,
	0x000000000000808B,
	0x0000000080000001,
	0x8000000080008081,
	0x8000000000008009,
	0x000000000000008A,
	0x0000000000000088,
	0x0000000080008009,
	0x000000008000000A,
	0x000000008000808B,
	0x800000000000008B,
	0x8000000000008089,
	0x8000000000008003,
	0x8000000000008002,
	0x8000000000000080,
	0x000000000000800A,
	0x800000008000000A,
	0x8000000080008081,
	0x8000000000008080,
	0x0000000080000001,
	0x8000000080008008,
	// Pad to 32 entries
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
];

#[rustfmt::skip]
const RHO: [u32; 25] = [
	 0, 44, 43, 21, 14,
	28, 20,  3, 45, 61,
	 1,  6, 25,  8, 18,
	27, 36, 10, 15, 56,
	62, 55, 39, 41,  2,
];

#[rustfmt::skip]
const PI: [usize; 25] = [
	0, 6, 12, 18, 24,
	3, 9, 10, 16, 22,
	1, 7, 13, 19, 20,
	4, 5, 11, 17, 23,
	2, 8, 14, 15, 21,
];

#[derive(Clone, Debug)]
struct SumComposition {
	n_vars: usize,
}

impl<P: PackedField> CompositionPoly<P> for SumComposition {
	fn n_vars(&self) -> usize {
		self.n_vars
	}

	fn degree(&self) -> usize {
		1
	}

	fn evaluate(&self, query: &[P]) -> Result<P, PolynomialError> {
		if query.len() != self.n_vars {
			return Err(PolynomialError::IncorrectQuerySize {
				expected: self.n_vars,
			});
		}
		// Sum of scalar values at the corresponding positions of the packed values.
		Ok(query.iter().copied().sum())
	}

	fn binary_tower_level(&self) -> usize {
		0
	}
}

/// Fixed (ie. statement independent) trace columns.
#[derive(Debug, IterOracles)]
struct FixedOracle {
	round_consts: OracleId,
	selector: OracleId,
}

impl FixedOracle {
	pub fn new<F: TowerField>(
		oracles: &mut MultilinearOracleSet<F>,
		log_size: usize,
		backend: impl ComputationBackend + 'static,
	) -> Result<Self> {
		let round_const_values = must_cast_slice::<_, PackedBinaryField64x1b>(&KECCAKF_RC);
		let round_consts_single = oracles.add_named("round_consts_single").transparent(
			MultilinearExtensionTransparent::<_, F, _, _>::from_values(
				round_const_values,
				backend,
			)?,
		)?;
		let round_consts = oracles
			.add_named("round_consts")
			.repeating(round_consts_single, log_size - LOG_ROWS_PER_PERMUTATION)?;

		let selector_single = oracles
			.add_named("selector_single")
			.transparent(StepDown::new(
				LOG_ROWS_PER_PERMUTATION,
				ROUNDS_PER_PERMUTATION << LOG_ROWS_PER_ROUND,
			)?)?;
		let selector = oracles
			.add_named("selector")
			.repeating(selector_single, log_size - LOG_ROWS_PER_PERMUTATION)?;

		Ok(Self {
			round_consts,
			selector,
		})
	}
}

/// Instance and witness trace columns.
#[derive(IterOracles)]
struct TraceOracle {
	batch_id: BatchId,
	state_in: [OracleId; 25],
	state_out: [OracleId; 25],
	c: [OracleId; 5],
	d: [OracleId; 5],
	c_shift: [OracleId; 5],
	a_theta: [OracleId; 25],
	b: [OracleId; 25],
	next_state_in: [OracleId; 25],
}

impl TraceOracle {
	pub fn new<F: TowerField>(oracles: &mut MultilinearOracleSet<F>, log_size: usize) -> Self {
		let batch_id = oracles.add_committed_batch(log_size, BinaryField1b::TOWER_LEVEL);
		let state_in_oracle = oracles
			.add_named("state_in")
			.committed_multiple::<25>(batch_id);
		let state_out_oracle = oracles
			.add_named("state_out")
			.committed_multiple::<25>(batch_id);
		let c_oracle = oracles.add_named("c").committed_multiple::<5>(batch_id);
		let d_oracle = oracles.add_named("d").committed_multiple::<5>(batch_id);

		let c_shift_oracle = c_oracle.map(|c_x| {
			oracles
				.add_named("c_shift")
				.shifted(c_x, 1, 6, ShiftVariant::CircularLeft)
				.unwrap()
		});

		let a_theta_oracle = array::from_fn(|xy| {
			let x = xy % 5;
			oracles
				.add_named("a_theta")
				.linear_combination(
					log_size,
					[(state_in_oracle[xy], F::ONE), (d_oracle[x], F::ONE)],
				)
				.unwrap()
		});

		let b_oracle: [_; 25] = array::from_fn(|xy| {
			if xy == 0 {
				a_theta_oracle[0]
			} else {
				oracles
					.add_named(format!("b_{xy}"))
					.shifted(
						a_theta_oracle[PI[xy]],
						RHO[xy] as usize,
						6,
						ShiftVariant::CircularLeft,
					)
					.unwrap()
			}
		});

		let next_state_in = array::from_fn(|xy| {
			oracles
				.add_named(format!("next_state_in_{xy}"))
				.shifted(state_in_oracle[xy], 64, 11, ShiftVariant::LogicalRight)
				.unwrap()
		});

		TraceOracle {
			batch_id,
			state_in: state_in_oracle,
			state_out: state_out_oracle,
			c: c_oracle,
			c_shift: c_shift_oracle,
			d: d_oracle,
			a_theta: a_theta_oracle,
			b: b_oracle,
			next_state_in,
		}
	}
}

struct Proof<F: Field, PCSComm, PCSProof> {
	trace_comm: PCSComm,
	zerocheck_proof: ZerocheckProof<F>,
	evalcheck_proof: GreedyEvalcheckProof<F>,
	trace_open_proof: PCSProof,
}

#[instrument(skip_all)]
fn generate_trace<U, FW>(
	log_size: usize,
	fixed_oracle: &FixedOracle,
	trace_oracle: &TraceOracle,
) -> Result<MultilinearExtensionIndex<'static, U, FW>>
where
	U: UnderlierType + PackScalar<BinaryField1b> + PackScalar<FW> + Pod,
	FW: BinaryField,
{
	assert!(log_size >= <PackedType<U, BinaryField1b>>::LOG_WIDTH);
	let len = 1 << (log_size - <PackedType<U, BinaryField1b>>::LOG_WIDTH);
	let build_trace_column = || vec![U::default(); len].into_boxed_slice();

	let mut state_in = array::from_fn::<_, 25, _>(|_xy| build_trace_column());
	let mut state_out = array::from_fn::<_, 25, _>(|_xy| build_trace_column());
	let mut c = array::from_fn::<_, 5, _>(|_x| build_trace_column());
	let mut d = array::from_fn::<_, 5, _>(|_x| build_trace_column());
	let mut c_shift = array::from_fn::<_, 5, _>(|_x| build_trace_column());
	let mut a_theta = array::from_fn::<_, 25, _>(|_xy| build_trace_column());
	let mut b = array::from_fn::<_, 25, _>(|_xy| build_trace_column());
	let mut next_state_in = array::from_fn::<_, 25, _>(|_xy| build_trace_column());
	let mut round_consts = build_trace_column();
	let mut selector = build_trace_column();

	fn cast_u64_cols<U: Pod, const N: usize>(cols: &mut [Box<[U]>; N]) -> [&mut [u64]; N] {
		cols.each_mut()
			.map(|col| must_cast_slice_mut::<_, u64>(&mut *col))
	}

	let state_in_u64 = cast_u64_cols(&mut state_in);
	let state_out_u64 = cast_u64_cols(&mut state_out);
	let c_u64 = cast_u64_cols(&mut c);
	let d_u64 = cast_u64_cols(&mut d);
	let c_shift_u64 = cast_u64_cols(&mut c_shift);
	let a_theta_u64 = cast_u64_cols(&mut a_theta);
	let b_u64 = cast_u64_cols(&mut b);
	let next_state_in_u64 = cast_u64_cols(&mut next_state_in);
	let round_consts_u64 = must_cast_slice_mut(&mut round_consts);
	let selector_u64 = must_cast_slice_mut(&mut selector);

	let mut rng = thread_rng();

	// Each round state is 64 rows
	// Each permutation is 24 round states
	for perm_i in 0..1 << (log_size - LOG_ROWS_PER_PERMUTATION) {
		let i = perm_i << LOG_ROUNDS_PER_PERMUTATION;

		// Randomly generate the initial permutation input
		let input: [u64; 25] = rng.gen();
		let output = {
			let mut output = input;
			keccakf(&mut output);
			output
		};

		// Assign the permutation input
		for xy in 0..25 {
			state_in_u64[xy][i] = input[xy];
		}

		// Expand trace columns for each round
		for round_i in 0..(1 << LOG_ROUNDS_PER_PERMUTATION) {
			let i = i | round_i;

			for x in 0..5 {
				c_u64[x][i] = (0..5).fold(0, |acc, y| acc ^ state_in_u64[x + 5 * y][i]);
				c_shift_u64[x][i] = c_u64[x][i].rotate_left(1);
			}

			for x in 0..5 {
				d_u64[x][i] = c_u64[(x + 4) % 5][i] ^ c_shift_u64[(x + 1) % 5][i];
			}

			for x in 0..5 {
				for y in 0..5 {
					a_theta_u64[x + 5 * y][i] = state_in_u64[x + 5 * y][i] ^ d_u64[x][i];
				}
			}

			for xy in 0..25 {
				b_u64[xy][i] = a_theta_u64[PI[xy]][i].rotate_left(RHO[xy]);
			}

			for x in 0..5 {
				for y in 0..5 {
					let b0 = b_u64[x + 5 * y][i];
					let b1 = b_u64[(x + 1) % 5 + 5 * y][i];
					let b2 = b_u64[(x + 2) % 5 + 5 * y][i];
					state_out_u64[x + 5 * y][i] = b0 ^ (!b1 & b2);
				}
			}

			round_consts_u64[i] = KECCAKF_RC[round_i];
			state_out_u64[0][i] ^= round_consts_u64[i];
			if round_i < 31 {
				for xy in 0..25 {
					state_in_u64[xy][i + 1] = state_out_u64[xy][i];
					next_state_in_u64[xy][i] = state_out_u64[xy][i];
				}
			}

			selector_u64[i] = if round_i < 24 { u64::MAX } else { 0 };
		}

		// Assert correct output
		for xy in 0..25 {
			assert_eq!(state_out_u64[xy][i + 23], output[xy]);
		}
	}

	let index = MultilinearExtensionIndex::new().update_owned::<BinaryField1b, _>(iter::zip(
		chain!(
			[fixed_oracle.round_consts, fixed_oracle.selector],
			trace_oracle.state_in,
			trace_oracle.state_out,
			trace_oracle.c,
			trace_oracle.d,
			trace_oracle.c_shift,
			trace_oracle.a_theta,
			trace_oracle.b,
			trace_oracle.next_state_in,
		),
		chain!(
			[round_consts, selector],
			state_in,
			state_out,
			c,
			d,
			c_shift,
			a_theta,
			b,
			next_state_in,
		),
	))?;

	Ok(index)
}

#[allow(clippy::identity_op, clippy::erasing_op)]
fn make_constraints<'a, P: PackedField>(
	fixed_oracle: &'a FixedOracle,
	trace_oracle: &'a TraceOracle,
) -> ConstraintSet<P> {
	let mut builder = ConstraintSetBuilder::new();

	// C_x - \sum_{y=0}^4 A_{x,y} = 0
	for x in 0..5 {
		builder.add_zerocheck(
			[
				trace_oracle.c[x],
				trace_oracle.state_in[x + 5 * 0],
				trace_oracle.state_in[x + 5 * 1],
				trace_oracle.state_in[x + 5 * 2],
				trace_oracle.state_in[x + 5 * 3],
				trace_oracle.state_in[x + 5 * 4],
			],
			SumComposition { n_vars: 6 },
		);
	}

	// C_{x-1} + shift_{6,1}(C_{x+1}) - D_x = 0
	for x in 0..5 {
		builder.add_zerocheck(
			[
				trace_oracle.c[(x + 4) % 5],
				trace_oracle.c_shift[(x + 1) % 5],
				trace_oracle.d[x],
			],
			SumComposition { n_vars: 3 },
		);
	}

	composition_poly!(ChiIota[s, b0, b1, b2, rc] = s - (rc + b0 + (1 - b1) * b2));
	composition_poly!(Chi[s, b0, b1, b2] = s - (b0 + (1 - b1) * b2));
	for x in 0..5 {
		for y in 0..5 {
			if x == 0 && y == 0 {
				builder.add_zerocheck(
					[
						trace_oracle.state_out[x + 5 * y],
						trace_oracle.b[x + 5 * y],
						trace_oracle.b[(x + 1) % 5 + 5 * y],
						trace_oracle.b[(x + 2) % 5 + 5 * y],
						fixed_oracle.round_consts,
					],
					ChiIota,
				);
			} else {
				builder.add_zerocheck(
					[
						trace_oracle.state_out[x + 5 * y],
						trace_oracle.b[x + 5 * y],
						trace_oracle.b[(x + 1) % 5 + 5 * y],
						trace_oracle.b[(x + 2) % 5 + 5 * y],
					],
					Chi,
				)
			}
		}
	}

	composition_poly!(Consistency[state_out, next_state_in, select] = (state_out - next_state_in) * select);
	for xy in 0..25 {
		builder.add_zerocheck(
			[
				trace_oracle.state_out[xy],
				trace_oracle.next_state_in[xy],
				fixed_oracle.selector,
			],
			Consistency,
		)
	}

	builder.build()
}

#[allow(clippy::too_many_arguments)]
#[instrument(skip_all)]
fn prove<U, F, FW, DomainField, PCS, CH, Backend>(
	log_size: usize,
	oracles: &mut MultilinearOracleSet<F>,
	fixed_oracle: &FixedOracle,
	trace_oracle: &TraceOracle,
	pcs: &PCS,
	mut challenger: CH,
	mut witness: MultilinearExtensionIndex<U, FW>,
	domain_factory: impl EvaluationDomainFactory<DomainField>,
	backend: Backend,
) -> Result<Proof<F, PCS::Commitment, PCS::Proof>>
where
	U: UnderlierType + PackScalar<BinaryField1b> + PackScalar<FW> + PackScalar<DomainField>,
	PackedType<U, FW>: PackedFieldIndexable<Scalar = FW>,
	F: TowerField + From<FW>,
	FW: TowerField + From<F> + ExtensionField<DomainField>,
	DomainField: TowerField,
	PCS: PolyCommitScheme<PackedType<U, BinaryField1b>, F, Error: Debug, Proof: 'static>,
	CH: Clone + CanObserve<F> + CanObserve<PCS::Commitment> + CanSample<F> + CanSampleBits<usize>,
	Backend: ComputationBackend,
{
	let constraint_set = make_constraints(fixed_oracle, trace_oracle);

	// Round 1
	let trace_commit_polys = oracles
		.committed_oracle_ids(trace_oracle.batch_id)
		.map(|oracle_id| witness.get::<BinaryField1b>(oracle_id))
		.collect::<Result<Vec<_>, _>>()?;
	let (trace_comm, trace_committed) = pcs.commit(&trace_commit_polys)?;
	challenger.observe(trace_comm.clone());

	// Zerocheck
	let mut iso_challenger = IsomorphicChallenger::<_, _, FW>::new(&mut challenger);

	let zerocheck_challenges = iso_challenger.sample_vec(log_size);

	let switchover_fn = standard_switchover_heuristic(-2);

	let (zerocheck_claim, meta) =
		sumcheck_v2::constraint_set_zerocheck_claim(constraint_set.clone(), oracles)?;

	let prover = sumcheck_v2::prove::constraint_set_zerocheck_prover(
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

	// Evalcheck
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
	assert_eq!(batch_id, trace_oracle.batch_id);

	let trace_commit_polys = oracles
		.committed_oracle_ids(trace_oracle.batch_id)
		.map(|oracle_id| witness.get::<BinaryField1b>(oracle_id))
		.collect::<Result<Vec<_>, _>>()?;
	let trace_open_proof = pcs.prove_evaluation(
		&mut challenger,
		&trace_committed,
		&trace_commit_polys,
		&same_query_claim.eval_point,
		backend,
	)?;

	Ok(Proof {
		trace_comm,
		zerocheck_proof: zerocheck_proof.isomorphic(),
		evalcheck_proof,
		trace_open_proof,
	})
}

#[allow(clippy::too_many_arguments)]
#[instrument(skip_all)]
fn verify<P, F, PCS, CH, Backend>(
	log_size: usize,
	oracles: &mut MultilinearOracleSet<F>,
	fixed_oracle: &FixedOracle,
	trace_oracle: &TraceOracle,
	pcs: &PCS,
	mut challenger: CH,
	proof: Proof<F, PCS::Commitment, PCS::Proof>,
	backend: Backend,
) -> Result<()>
where
	P: PackedField<Scalar = BinaryField1b>,
	F: TowerField,
	PCS: PolyCommitScheme<P, F, Error: Debug, Proof: 'static>,
	CH: CanObserve<F> + CanObserve<PCS::Commitment> + CanSample<F> + CanSampleBits<usize>,
	Backend: ComputationBackend,
{
	let constraint_set = make_constraints::<F>(fixed_oracle, trace_oracle);

	let Proof {
		trace_comm,
		zerocheck_proof,
		evalcheck_proof,
		trace_open_proof,
	} = proof;

	// Round 1
	challenger.observe(trace_comm.clone());

	// Zerocheck
	let zerocheck_challenges = challenger.sample_vec(log_size);

	let (zerocheck_claim, meta) =
		sumcheck_v2::constraint_set_zerocheck_claim(constraint_set, oracles)?;
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
		sumcheck_v2::make_eval_claims(oracles, [meta], zerocheck_output)?;

	let evalcheck_claims = conflate_multilinear_evalchecks(evalcheck_multilinear_claims)?;

	// Evalcheck
	let same_query_claims =
		greedy_evalcheck::verify(oracles, evalcheck_claims, evalcheck_proof, &mut challenger)?;

	assert_eq!(same_query_claims.len(), 1);
	let (batch_id, same_query_claim) = same_query_claims
		.into_iter()
		.next()
		.expect("length is asserted to be 1");
	assert_eq!(batch_id, trace_oracle.batch_id);

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

	let log_size = get_log_trace_size().unwrap_or(14);
	let log_inv_rate = 1;

	type U = <PackedBinaryField128x1b as WithUnderlier>::Underlier;

	let backend = make_backend();
	let mut oracles = MultilinearOracleSet::new();
	let fixed_oracle = FixedOracle::new(&mut oracles, log_size, backend.clone()).unwrap();
	let trace_oracle = TraceOracle::new(&mut oracles, log_size);

	let trace_batch = oracles.committed_batch(trace_oracle.batch_id);

	// Set up the public parameters
	let pcs = tensor_pcs::find_proof_size_optimal_pcs::<
		U,
		BinaryField1b,
		BinaryField16b,
		BinaryField16b,
		BinaryField128b,
	>(SECURITY_BITS, log_size, trace_batch.n_polys, log_inv_rate, false)
	.unwrap();

	const KECCAK_256_RATE_BYTES: u64 = 1088 / 8;
	let num_of_perms = 1 << (log_size - 11) as u64;
	let data_hashed_256 = ByteSize::b(num_of_perms * KECCAK_256_RATE_BYTES);
	let tensorpcs_size = ByteSize::b(pcs.proof_size(trace_batch.n_polys) as u64);
	tracing::info!("Size of hashable Keccak-256 data: {}", data_hashed_256);
	tracing::info!("Size of PCS opening proof: {}", tensorpcs_size);

	let witness =
		generate_trace::<U, BinaryField128bPolyval>(log_size, &fixed_oracle, &trace_oracle)
			.unwrap();

	let challenger = new_hasher_challenger::<_, GroestlHasher<_>>();
	let domain_factory = IsomorphicEvaluationDomainFactory::<BinaryField128b>::default();
	let proof =
		prove::<_, BinaryField128b, BinaryField128bPolyval, BinaryField128bPolyval, _, _, _>(
			log_size,
			&mut oracles.clone(),
			&fixed_oracle,
			&trace_oracle,
			&pcs,
			challenger.clone(),
			witness,
			domain_factory,
			backend.clone(),
		)
		.unwrap();

	verify(
		log_size,
		&mut oracles.clone(),
		&fixed_oracle,
		&trace_oracle,
		&pcs,
		challenger.clone(),
		proof,
		backend,
	)
	.unwrap();
}
