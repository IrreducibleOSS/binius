// Copyright 2024 Ulvetanna Inc.

#![feature(step_trait)]

extern crate core;

use binius::{
	challenger::{CanObserve, CanSample, CanSampleBits, HashChallenger},
	field::{
		util::inner_product_unchecked, BinaryField, BinaryField1b, Field, PackedBinaryField128x1b,
		PackedBinaryField1x128b, PackedBinaryField8x16b, PackedField, TowerField,
	},
	hash::GroestlHasher,
	oracle::{
		BatchId, CommittedBatch, CommittedBatchSpec, CommittedId, CompositePolyOracle,
		MultilinearOracleSet, MultilinearPolyOracle, MultivariatePolyOracle, ProjectionVariant,
		ShiftVariant,
	},
	poly_commit::{tensor_pcs, PolyCommitScheme},
	polynomial::{
		composition::BivariateProduct,
		multilinear_query::MultilinearQuery,
		transparent::{
			disjoint_product::DisjointProduct, eq_ind::EqIndPartialEval,
			shift_ind::ShiftIndPartialEval, step_down::StepDown,
		},
		util::tensor_prod_eq_ind,
		CompositionPoly, Error as PolynomialError, EvaluationDomain, MultilinearExtension,
		MultilinearPoly, MultivariatePoly,
	},
	protocols::{
		evalcheck,
		evalcheck::{BatchCommittedEvalClaims, EvalcheckProof, EvalcheckWitness, ShiftedEvalClaim},
		sumcheck,
		sumcheck::{SumcheckClaim, SumcheckProof, SumcheckProveOutput, SumcheckWitness},
		test_utils::{full_prove_with_switchover, full_verify},
		zerocheck,
		zerocheck::{ZerocheckClaim, ZerocheckProof, ZerocheckProveOutput},
	},
};
use bytemuck::{must_cast_slice_mut, Pod};
use rand::{thread_rng, Rng};
use rayon::prelude::*;
use std::{array, env, fmt::Debug, iter, iter::Step, sync::Arc};
use tiny_keccak::keccakf;
use tracing::instrument;
use tracing_profile::{CsvLayer, PrintTreeLayer};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

type DynMultilinearPoly<'a, F> = dyn MultilinearPoly<F> + Send + Sync + 'a;

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

fn init_tracing() {
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
	}
}

#[derive(Debug)]
struct SumComposition {
	n_vars: usize,
}

impl<F: Field> CompositionPoly<F> for SumComposition {
	fn n_vars(&self) -> usize {
		self.n_vars
	}

	fn degree(&self) -> usize {
		1
	}

	fn evaluate(&self, query: &[F]) -> Result<F, PolynomialError> {
		self.evaluate_packed(query)
	}

	fn evaluate_packed(&self, query: &[F]) -> Result<F, PolynomialError> {
		if query.len() != self.n_vars {
			return Err(PolynomialError::IncorrectQuerySize {
				expected: self.n_vars,
			});
		}
		Ok(query.iter().sum())
	}

	fn binary_tower_level(&self) -> usize {
		0
	}
}

#[derive(Debug)]
struct ChiComposition;

impl<F: Field> CompositionPoly<F> for ChiComposition {
	fn n_vars(&self) -> usize {
		4
	}

	fn degree(&self) -> usize {
		2
	}

	fn evaluate(&self, query: &[F]) -> Result<F, PolynomialError> {
		self.evaluate_packed(query)
	}

	fn evaluate_packed(&self, query: &[F]) -> Result<F, PolynomialError> {
		let a = query[0];
		let b0 = query[1];
		let b1 = query[2];
		let b2 = query[3];
		Ok(a - (b0 + (F::ONE - b1) * b2))
	}

	fn binary_tower_level(&self) -> usize {
		0
	}
}

#[derive(Debug)]
struct ChiIotaComposition;

impl<F: Field> CompositionPoly<F> for ChiIotaComposition {
	fn n_vars(&self) -> usize {
		5
	}

	fn degree(&self) -> usize {
		2
	}

	fn evaluate(&self, query: &[F]) -> Result<F, PolynomialError> {
		self.evaluate_packed(query)
	}

	fn evaluate_packed(&self, query: &[F]) -> Result<F, PolynomialError> {
		let a = query[0];
		let b0 = query[1];
		let b1 = query[2];
		let b2 = query[3];
		let rc = query[4];
		Ok(a - (rc + b0 + (F::ONE - b1) * b2))
	}

	fn binary_tower_level(&self) -> usize {
		0
	}
}

#[derive(Debug)]
struct RoundConsistency;

impl<F: Field> CompositionPoly<F> for RoundConsistency {
	fn n_vars(&self) -> usize {
		3
	}

	fn degree(&self) -> usize {
		2
	}

	fn evaluate(&self, query: &[F]) -> Result<F, PolynomialError> {
		self.evaluate_packed(query)
	}

	fn evaluate_packed(&self, query: &[F]) -> Result<F, PolynomialError> {
		let state_out = query[0];
		let next_state_in = query[1];
		let select = query[2];
		Ok((state_out - next_state_in) * select)
	}

	fn binary_tower_level(&self) -> usize {
		0
	}
}

#[derive(Debug)]
struct RoundConstant<P: PackedField<Scalar = BinaryField1b>>(MultilinearExtension<'static, P>);

impl<P: PackedField<Scalar = BinaryField1b> + Pod> RoundConstant<P> {
	fn new() -> Self {
		let mut values = vec![P::default(); 1 << (11 - P::LOG_WIDTH)];
		must_cast_slice_mut::<_, u64>(&mut values).copy_from_slice(KECCAKF_RC.as_slice());
		Self(MultilinearExtension::from_values(values).unwrap())
	}
}

// TODO: Implement MultivariatePoly on MultilinearExtension?
impl<P, FE> MultivariatePoly<FE> for RoundConstant<P>
where
	P: PackedField<Scalar = BinaryField1b> + Debug,
	FE: BinaryField,
{
	fn n_vars(&self) -> usize {
		self.0.n_vars()
	}

	fn degree(&self) -> usize {
		self.0.n_vars()
	}

	fn evaluate(&self, query: &[FE]) -> Result<FE, PolynomialError> {
		self.0
			.evaluate(&MultilinearQuery::<FE>::with_full_query(query)?)
	}
}

#[derive(Debug)]
struct FixedOracle<F: Field> {
	round_consts: MultilinearPolyOracle<F>,
	selector: MultilinearPolyOracle<F>,
}

impl<F: TowerField> FixedOracle<F> {
	pub fn new(oracles: &mut MultilinearOracleSet<F>, log_size: usize) -> Self {
		let round_consts_single = oracles
			.add_transparent(Arc::new(RoundConstant::<PackedBinaryField128x1b>::new()), 0)
			.unwrap();
		let round_consts = oracles
			.add_repeating(round_consts_single, log_size - 11)
			.unwrap();

		let selector_single = oracles
			.add_transparent(Arc::new(StepDown::new(11, 24 * 64).unwrap()), 0)
			.unwrap();
		let selector = oracles
			.add_repeating(selector_single, log_size - 11)
			.unwrap();

		Self {
			round_consts: oracles.oracle(round_consts),
			selector: oracles.oracle(selector),
		}
	}
}

struct TraceOracle<F: Field> {
	state_in: [MultilinearPolyOracle<F>; 25],
	state_out: [MultilinearPolyOracle<F>; 25],
	c: [MultilinearPolyOracle<F>; 5],
	d: [MultilinearPolyOracle<F>; 5],
	c_shift: [MultilinearPolyOracle<F>; 5],
	a_theta: [MultilinearPolyOracle<F>; 25],
	b: [MultilinearPolyOracle<F>; 25],
	next_state_in: [MultilinearPolyOracle<F>; 25],
}

impl<F: TowerField> TraceOracle<F> {
	pub fn new(oracles: &mut MultilinearOracleSet<F>, batch_id: BatchId) -> Self {
		let trace_batch = oracles.committed_batch(batch_id);
		let log_size = trace_batch.n_vars;

		let state_in_oracle = array::from_fn(|xy| {
			oracles.committed_oracle_id(CommittedId {
				batch_id,
				index: xy,
			})
		});
		let state_out_oracle = array::from_fn(|xy| {
			oracles.committed_oracle_id(CommittedId {
				batch_id,
				index: 25 + xy,
			})
		});
		let c_oracle = array::from_fn(|x| {
			oracles.committed_oracle_id(CommittedId {
				batch_id,
				index: 50 + x,
			})
		});
		let d_oracle = array::from_fn(|x| {
			oracles.committed_oracle_id(CommittedId {
				batch_id,
				index: 55 + x,
			})
		});

		let c_shift_oracle = c_oracle.map(|c_x| {
			oracles
				.add_shifted(c_x, 1, 6, ShiftVariant::CircularLeft)
				.unwrap()
		});

		let a_theta_oracle = array::from_fn(|xy| {
			let x = xy % 5;
			oracles
				.add_linear_combination(
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
					.add_shifted(
						a_theta_oracle[PI[xy]],
						RHO[xy] as usize,
						6,
						ShiftVariant::CircularLeft,
					)
					.unwrap()
			}
		});

		let next_state_in = state_in_oracle.map(|state_in_xy| {
			oracles
				.add_shifted(state_in_xy, 64, 11, ShiftVariant::LogicalRight)
				.unwrap()
		});

		TraceOracle {
			state_in: state_in_oracle.map(|id| oracles.oracle(id)),
			state_out: state_out_oracle.map(|id| oracles.oracle(id)),
			c: c_oracle.map(|id| oracles.oracle(id)),
			c_shift: c_shift_oracle.map(|id| oracles.oracle(id)),
			d: d_oracle.map(|id| oracles.oracle(id)),
			a_theta: a_theta_oracle.map(|id| oracles.oracle(id)),
			b: b_oracle.map(|id| oracles.oracle(id)),
			next_state_in: next_state_in.map(|id| oracles.oracle(id)),
		}
	}
}

struct TraceWitness<P: PackedField> {
	state_in: [Vec<P>; 25],
	state_out: [Vec<P>; 25],
	c: [Vec<P>; 5],
	d: [Vec<P>; 5],
	c_shift: [Vec<P>; 5],
	a_theta: [Vec<P>; 25],
	b: [Vec<P>; 25],
	next_state_in: [Vec<P>; 25],
	round_consts: Vec<P>,
	selector: Vec<P>,
}

struct Proof<F: Field, PCSComm, PCSProof> {
	trace_comm: PCSComm,
	zerocheck_proof: ZerocheckProof,
	sumcheck_proof: SumcheckProof<F>,
	evalcheck_proof: EvalcheckProof<F>,
	second_sumcheck_proof: SumcheckProof<F>,
	second_evalcheck_proof: EvalcheckProof<F>,
	trace_open_proof: PCSProof,
}

#[instrument]
#[allow(clippy::needless_range_loop)]
fn generate_trace<P: PackedField + Pod>(log_size: usize) -> TraceWitness<P> {
	let build_trace_column = || vec![P::default(); 1 << (log_size - P::LOG_WIDTH)];
	let mut witness = TraceWitness {
		state_in: array::from_fn(|_xy| build_trace_column()),
		state_out: array::from_fn(|_xy| build_trace_column()),
		c: array::from_fn(|_x| build_trace_column()),
		d: array::from_fn(|_x| build_trace_column()),
		c_shift: array::from_fn(|_x| build_trace_column()),
		a_theta: array::from_fn(|_xy| build_trace_column()),
		b: array::from_fn(|_xy| build_trace_column()),
		next_state_in: array::from_fn(|_xy| build_trace_column()),
		round_consts: build_trace_column(),
		selector: build_trace_column(),
	};

	fn cast_u64_cols<P: PackedField + Pod, const N: usize>(
		cols: &mut [Vec<P>; N],
	) -> [&mut [u64]; N] {
		cols.each_mut()
			.map(|col| must_cast_slice_mut::<_, u64>(col.as_mut_slice()))
	}

	let state_in_u64 = cast_u64_cols(&mut witness.state_in);
	let state_out_u64 = cast_u64_cols(&mut witness.state_out);
	let c_u64 = cast_u64_cols(&mut witness.c);
	let d_u64 = cast_u64_cols(&mut witness.d);
	let c_shift_u64 = cast_u64_cols(&mut witness.c_shift);
	let a_theta_u64 = cast_u64_cols(&mut witness.a_theta);
	let b_u64 = cast_u64_cols(&mut witness.b);
	let next_state_in_u64 = cast_u64_cols(&mut witness.next_state_in);
	let round_consts_u64 = must_cast_slice_mut(witness.round_consts.as_mut_slice());
	let selector_u64 = must_cast_slice_mut(witness.selector.as_mut_slice());

	let mut rng = thread_rng();

	// Each round state is 64 rows
	// Each permutation is 24 round states
	for perm_i in 0..1 << (log_size - 11) {
		let i = perm_i << 5;

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
		for round_i in 0..32 {
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

	witness
}

#[allow(clippy::type_complexity)]
#[instrument(skip_all)]
fn make_second_round_sumcheck<'a, F, CH>(
	oracle_set: &mut MultilinearOracleSet<F>,
	trace_batch: &CommittedBatch,
	batch_committed_eval_claims: BatchCommittedEvalClaims<F>,
	shifted_eval_claims: Vec<ShiftedEvalClaim<F>>,
	evalcheck_witness: Option<
		&EvalcheckWitness<F, DynMultilinearPoly<'a, F>, Arc<DynMultilinearPoly<'a, F>>>,
	>,
	mut challenger: CH,
) -> (
	SumcheckClaim<F>,
	Option<SumcheckWitness<F, DynMultilinearPoly<'a, F>, Arc<DynMultilinearPoly<'a, F>>>>,
)
where
	F: TowerField,
	CH: CanSample<F>,
{
	let log_size = trace_batch.n_vars;

	let max_shift_block_size = shifted_eval_claims
		.iter()
		.map(|shift_claim| shift_claim.shifted.block_size())
		.max()
		.unwrap_or(0);
	assert_eq!(max_shift_block_size, 11);

	let mut sumcheck_claims = Vec::new();
	let mut oracles = Vec::new();
	let mut witnesses: Vec<Arc<dyn MultilinearPoly<F> + Send + Sync>> = Vec::new();

	let same_query_claim = batch_committed_eval_claims
		.try_extract_same_query_pcs_claim(trace_batch.id)
		.unwrap()
		.unwrap();

	let trace_mixing_challenges = challenger.sample_vec(trace_batch.n_polys);
	let mixed_trace_eval = inner_product_unchecked::<F, F>(
		same_query_claim.evals.iter().copied(),
		trace_mixing_challenges.iter().copied(),
	);

	let mixed_trace = oracle_set
		.add_linear_combination(
			log_size,
			trace_mixing_challenges
				.iter()
				.enumerate()
				.map(|(i, &coeff)| {
					(
						oracle_set.committed_oracle_id(CommittedId {
							batch_id: trace_batch.id,
							index: i,
						}),
						coeff,
					)
				})
				.collect::<Vec<_>>(),
		)
		.unwrap();
	let mixed_trace_projection = oracle_set
		.add_projected(
			mixed_trace,
			same_query_claim.eval_point[max_shift_block_size..].to_vec(),
			ProjectionVariant::LastVars,
		)
		.unwrap();
	oracles.push(oracle_set.oracle(mixed_trace_projection));

	let partial_eval_query = evalcheck_witness.map(|_| {
		MultilinearQuery::<F>::with_full_query(&same_query_claim.eval_point[max_shift_block_size..])
			.unwrap()
	});

	if let Some(evalcheck_witness) = evalcheck_witness {
		let partial_eval_query = partial_eval_query.as_ref().unwrap();

		let trace_witnesses = (0..trace_batch.n_polys)
			.map(|i| {
				let oracle = oracle_set.committed_oracle(CommittedId {
					batch_id: trace_batch.id,
					index: i,
				});
				evalcheck_witness.witness_for_oracle(&oracle).unwrap()
			})
			.collect::<Vec<_>>();
		let mixed_trace_values = (0..1 << log_size)
			.into_par_iter()
			.map(|i| {
				trace_witnesses
					.iter()
					.zip(trace_mixing_challenges.iter())
					.map(|(multilin, &coeff)| {
						multilin.evaluate_on_hypercube_and_scale(i, coeff).unwrap()
					})
					.sum::<F>()
			})
			.collect::<Vec<_>>();

		let mixed_trace_witness = MultilinearExtension::from_values(mixed_trace_values)
			.unwrap()
			.evaluate_partial_high(partial_eval_query)
			.unwrap();

		witnesses.push(Arc::new(mixed_trace_witness));
	}

	let eq_ind = EqIndPartialEval::new(
		max_shift_block_size,
		same_query_claim.eval_point[..max_shift_block_size].to_vec(),
	)
	.unwrap();

	if evalcheck_witness.is_some() {
		let eq_ind_witness = eq_ind.multilinear_extension().unwrap();
		witnesses.push(Arc::new(eq_ind_witness));
	}

	let eq_ind_oracle = oracle_set
		.add_transparent(Arc::new(eq_ind), F::TOWER_LEVEL)
		.unwrap();
	oracles.push(oracle_set.oracle(eq_ind_oracle));

	let trace_eval_oracle = MultivariatePolyOracle::Composite(
		CompositePolyOracle::new(
			max_shift_block_size,
			vec![
				oracle_set.oracle(mixed_trace_projection),
				oracle_set.oracle(eq_ind_oracle),
			],
			Arc::new(BivariateProduct),
		)
		.unwrap(),
	);
	sumcheck_claims.push(SumcheckClaim {
		poly: trace_eval_oracle,
		sum: mixed_trace_eval,
	});

	for claim in shifted_eval_claims {
		let shifted = claim.shifted;
		let shifted_oracle = oracle_set
			.add_projected(
				shifted.inner().id(),
				claim.eval_point[max_shift_block_size..].to_vec(),
				ProjectionVariant::LastVars,
			)
			.unwrap();
		oracles.push(oracle_set.oracle(shifted_oracle));

		if let Some(evalcheck_witness) = evalcheck_witness {
			let multilin = evalcheck_witness
				.witness_for_oracle(shifted.inner())
				.unwrap();
			let partial_eval_query = partial_eval_query.as_ref().unwrap();
			let projected_multilin = multilin.evaluate_partial_high(partial_eval_query).unwrap();
			witnesses.push(Arc::new(projected_multilin));
		}

		let shift_ind = ShiftIndPartialEval::new(
			shifted.block_size(),
			shifted.shift_offset(),
			shifted.shift_variant(),
			claim.eval_point[..shifted.block_size()].to_vec(),
		)
		.unwrap();

		let shift_ind_oracle = if shifted.block_size() < max_shift_block_size {
			oracle_set
				.add_transparent(
					Arc::new(DisjointProduct(
						shift_ind.clone(),
						EqIndPartialEval::new(
							max_shift_block_size - shifted.block_size(),
							claim.eval_point[shifted.block_size()..max_shift_block_size].to_vec(),
						)
						.unwrap(),
					)),
					F::TOWER_LEVEL,
				)
				.unwrap()
		} else {
			oracle_set
				.add_transparent(Arc::new(shift_ind.clone()), F::TOWER_LEVEL)
				.unwrap()
		};
		oracles.push(oracle_set.oracle(shift_ind_oracle));

		if evalcheck_witness.is_some() {
			let shift_ind_mle = shift_ind.multilinear_extension().unwrap();
			let shift_ind_witness = if shifted.block_size() < max_shift_block_size {
				let mut shift_ind_values = vec![F::ZERO; 1 << max_shift_block_size];
				shift_ind_values[..1 << shifted.block_size()]
					.copy_from_slice(shift_ind_mle.evals());
				tensor_prod_eq_ind(
					shifted.block_size(),
					&mut shift_ind_values,
					&claim.eval_point[shifted.block_size()..max_shift_block_size],
				)
				.unwrap();
				MultilinearExtension::from_values(shift_ind_values).unwrap()
			} else {
				shift_ind_mle
			};
			witnesses.push(Arc::new(shift_ind_witness));
		}

		let oracle = MultivariatePolyOracle::Composite(
			CompositePolyOracle::new(
				max_shift_block_size,
				vec![
					oracle_set.oracle(shifted_oracle),
					oracle_set.oracle(shift_ind_oracle),
				],
				Arc::new(BivariateProduct),
			)
			.unwrap(),
		);
		sumcheck_claims.push(SumcheckClaim {
			poly: oracle,
			sum: claim.eval,
		});
	}
	let mixed_sumcheck_claim = sumcheck::mix_claims(
		max_shift_block_size,
		oracles,
		sumcheck_claims.iter(),
		&mut challenger,
	)
	.unwrap();

	let mixed_sumcheck_claim_clone = mixed_sumcheck_claim.clone();
	let mixed_sumcheck_witness = evalcheck_witness
		.map(move |_| sumcheck::mix_witnesses(mixed_sumcheck_claim_clone, witnesses).unwrap());

	(mixed_sumcheck_claim, mixed_sumcheck_witness)
}

#[allow(clippy::too_many_arguments)]
#[instrument(skip_all)]
fn prove<P, F, PCS, CH>(
	log_size: usize,
	oracles: &mut MultilinearOracleSet<F>,
	trace_batch_id: BatchId,
	fixed_oracle: &FixedOracle<F>,
	trace_oracle: &TraceOracle<F>,
	constraints: &[MultivariatePolyOracle<F>],
	pcs: &PCS,
	mut challenger: CH,
	witness: TraceWitness<P>,
) -> Proof<F, PCS::Commitment, PCS::Proof>
where
	P: PackedField<Scalar = BinaryField1b> + Debug + Pod,
	F: TowerField + Step,
	PCS: PolyCommitScheme<P, F>,
	PCS::Error: Debug,
	PCS::Proof: 'static,
	CH: CanObserve<F> + CanObserve<PCS::Commitment> + CanSample<F> + CanSampleBits<usize>,
{
	// Round 1
	let trace_commit_polys = witness
		.state_in
		.iter()
		.chain(witness.state_out.iter())
		.chain(witness.c.iter())
		.chain(witness.d.iter())
		.map(|vals| MultilinearExtension::from_values_slice(vals.as_slice()).unwrap())
		.collect::<Vec<_>>();
	let trace_commit_polys = trace_commit_polys.iter().collect::<Vec<_>>();
	let (trace_comm, trace_committed) = pcs.commit(&trace_commit_polys).unwrap();
	challenger.observe(trace_comm.clone());

	// Zerocheck mixing
	let zerocheck_claims = constraints
		.iter()
		.cloned()
		.map(|poly| ZerocheckClaim { poly })
		.collect::<Vec<_>>();
	let zerocheck_claim = zerocheck::mix_claims(
		log_size,
		iter::once(&fixed_oracle.round_consts)
			.chain(iter::once(&fixed_oracle.selector))
			.chain(trace_oracle.state_in.iter())
			.chain(trace_oracle.state_out.iter())
			.chain(trace_oracle.c.iter())
			.chain(trace_oracle.d.iter())
			.chain(trace_oracle.c_shift.iter())
			.chain(trace_oracle.a_theta.iter())
			.chain(trace_oracle.b.iter())
			.chain(trace_oracle.next_state_in.iter())
			.cloned()
			.collect(),
		zerocheck_claims.iter(),
		&mut challenger,
	)
	.unwrap();

	let zerocheck_witness = zerocheck::mix_witness(
		zerocheck_claim.clone(),
		iter::once(&witness.round_consts)
			.chain(iter::once(&witness.selector))
			.chain(witness.state_in.iter())
			.chain(witness.state_out.iter())
			.chain(witness.c.iter())
			.chain(witness.d.iter())
			.chain(witness.c_shift.iter())
			.chain(witness.a_theta.iter())
			.chain(witness.b.iter())
			.chain(witness.next_state_in.iter())
			.map(|values| {
				Arc::new(MultilinearExtension::from_values_slice(values.as_slice()).unwrap())
					as Arc<dyn MultilinearPoly<F> + Send + Sync>
			})
			.collect(),
	)
	.unwrap();

	// Zerocheck
	let zerocheck_challenge = challenger.sample_vec(log_size);
	let ZerocheckProveOutput {
		sumcheck_claim,
		sumcheck_witness,
		zerocheck_proof,
	} = zerocheck::prove(oracles, zerocheck_witness, &zerocheck_claim, zerocheck_challenge).unwrap();

	// Sumcheck
	let sumcheck_domain =
		EvaluationDomain::new(sumcheck_claim.poly.max_individual_degree() + 1).unwrap();

	// TODO: Improve the logic to commit the optimal switchover.
	let switchover = sumcheck_claim.poly.n_vars() / 2;
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

	// Evalcheck
	let trace_batch = oracles.committed_batch(trace_batch_id);

	let mut shifted_eval_claims = Vec::new();
	let mut packed_eval_claims = Vec::new();
	let mut batch_committed_eval_claims = BatchCommittedEvalClaims::new(&[trace_batch.clone()]);
	let evalcheck_proof = evalcheck::prove(
		&evalcheck_witness,
		evalcheck_claim,
		&mut batch_committed_eval_claims,
		&mut shifted_eval_claims,
		&mut packed_eval_claims,
	)
	.unwrap();
	assert_eq!(batch_committed_eval_claims.n_batches(), 1);
	assert_eq!(shifted_eval_claims.len(), 54);
	assert_eq!(packed_eval_claims.len(), 0);

	let (second_sumcheck_claim, second_sumcheck_witness) = make_second_round_sumcheck(
		oracles,
		&trace_batch,
		batch_committed_eval_claims,
		shifted_eval_claims,
		Some(&evalcheck_witness),
		&mut challenger,
	);

	let sumcheck_domain =
		EvaluationDomain::new(second_sumcheck_claim.poly.max_individual_degree() + 1).unwrap();

	// TODO: Improve the logic to commit the optimal switchover.
	let switchover = second_sumcheck_claim.poly.n_vars() / 2;
	let (_, output) = full_prove_with_switchover(
		&second_sumcheck_claim,
		second_sumcheck_witness.unwrap(),
		&sumcheck_domain,
		&mut challenger,
		switchover,
	);

	let SumcheckProveOutput {
		evalcheck_claim: second_evalcheck_claim,
		evalcheck_witness: second_evalcheck_witness,
		sumcheck_proof: second_sumcheck_proof,
	} = output;

	let evalcheck_witness = evalcheck_witness.merge(second_evalcheck_witness);

	// Second Evalcheck
	let mut shifted_eval_claims = Vec::new();
	let mut packed_eval_claims = Vec::new();
	let mut batch_committed_eval_claims = BatchCommittedEvalClaims::new(&[trace_batch.clone()]);
	let second_evalcheck_proof = evalcheck::prove(
		&evalcheck_witness,
		second_evalcheck_claim,
		&mut batch_committed_eval_claims,
		&mut shifted_eval_claims,
		&mut packed_eval_claims,
	)
	.unwrap();
	assert_eq!(batch_committed_eval_claims.n_batches(), 1);
	assert_eq!(shifted_eval_claims.len(), 0);
	assert_eq!(packed_eval_claims.len(), 0);

	let same_query_claim = batch_committed_eval_claims
		.try_extract_same_query_pcs_claim(trace_batch.id)
		.unwrap()
		.unwrap();

	let trace_open_proof = pcs
		.prove_evaluation(
			&mut challenger,
			&trace_committed,
			&trace_commit_polys,
			&same_query_claim.eval_point,
		)
		.unwrap();

	Proof {
		trace_comm,
		zerocheck_proof,
		sumcheck_proof,
		evalcheck_proof,
		second_sumcheck_proof,
		second_evalcheck_proof,
		trace_open_proof,
	}
}

#[allow(clippy::too_many_arguments)]
#[instrument(skip_all)]
fn verify<P, F, PCS, CH>(
	log_size: usize,
	oracles: &mut MultilinearOracleSet<F>,
	trace_batch_id: BatchId,
	fixed_oracle: &FixedOracle<F>,
	trace_oracle: &TraceOracle<F>,
	constraints: &[MultivariatePolyOracle<F>],
	pcs: &PCS,
	mut challenger: CH,
	proof: Proof<F, PCS::Commitment, PCS::Proof>,
) where
	P: PackedField<Scalar = BinaryField1b> + Debug,
	F: TowerField + Step,
	PCS: PolyCommitScheme<P, F>,
	PCS::Error: Debug,
	PCS::Proof: 'static,
	CH: CanObserve<F> + CanObserve<PCS::Commitment> + CanSample<F> + CanSampleBits<usize>,
{
	let Proof {
		trace_comm,
		zerocheck_proof,
		sumcheck_proof,
		evalcheck_proof,
		second_sumcheck_proof,
		second_evalcheck_proof,
		trace_open_proof,
	} = proof;

	// Round 1
	challenger.observe(trace_comm.clone());

	// Zerocheck mixing
	let zerocheck_claims = constraints
		.iter()
		.cloned()
		.map(|poly| ZerocheckClaim { poly })
		.collect::<Vec<_>>();
	let zerocheck_claim = zerocheck::mix_claims(
		log_size,
		iter::once(&fixed_oracle.round_consts)
			.chain(iter::once(&fixed_oracle.selector))
			.chain(trace_oracle.state_in.iter())
			.chain(trace_oracle.state_out.iter())
			.chain(trace_oracle.c.iter())
			.chain(trace_oracle.d.iter())
			.chain(trace_oracle.c_shift.iter())
			.chain(trace_oracle.a_theta.iter())
			.chain(trace_oracle.b.iter())
			.chain(trace_oracle.next_state_in.iter())
			.cloned()
			.collect(),
		zerocheck_claims.iter(),
		&mut challenger,
	)
	.unwrap();

	// Zerocheck
	let zerocheck_challenge = challenger.sample_vec(log_size);
	let sumcheck_claim =
		zerocheck::verify(oracles, &zerocheck_claim, zerocheck_proof, zerocheck_challenge).unwrap();

	// Sumcheck
	let sumcheck_domain =
		EvaluationDomain::new(sumcheck_claim.poly.max_individual_degree() + 1).unwrap();

	let (_, evalcheck_claim) =
		full_verify(&sumcheck_claim, sumcheck_proof, &sumcheck_domain, &mut challenger);

	// Evalcheck
	let trace_batch = oracles.committed_batch(trace_batch_id);

	let mut shifted_eval_claims = Vec::new();
	let mut packed_eval_claims = Vec::new();
	let mut batch_committed_eval_claims = BatchCommittedEvalClaims::new(&[trace_batch.clone()]);
	evalcheck::verify(
		evalcheck_claim,
		evalcheck_proof,
		&mut batch_committed_eval_claims,
		&mut shifted_eval_claims,
		&mut packed_eval_claims,
	)
	.unwrap();
	assert_eq!(batch_committed_eval_claims.n_batches(), 1);
	assert_eq!(shifted_eval_claims.len(), 54);
	assert_eq!(packed_eval_claims.len(), 0);

	// Second Sumcheck
	let (second_sumcheck_claim, _) = make_second_round_sumcheck(
		oracles,
		&trace_batch,
		batch_committed_eval_claims,
		shifted_eval_claims,
		None,
		&mut challenger,
	);

	let sumcheck_domain =
		EvaluationDomain::new(second_sumcheck_claim.poly.max_individual_degree() + 1).unwrap();

	let (_, second_evalcheck_claim) = full_verify(
		&second_sumcheck_claim,
		second_sumcheck_proof,
		&sumcheck_domain,
		&mut challenger,
	);

	// Second Evalcheck
	let mut shifted_eval_claims = Vec::new();
	let mut packed_eval_claims = Vec::new();
	let mut batch_committed_eval_claims = BatchCommittedEvalClaims::new(&[trace_batch.clone()]);
	evalcheck::verify(
		second_evalcheck_claim,
		second_evalcheck_proof,
		&mut batch_committed_eval_claims,
		&mut shifted_eval_claims,
		&mut packed_eval_claims,
	)
	.unwrap();
	assert_eq!(batch_committed_eval_claims.n_batches(), 1);
	assert_eq!(shifted_eval_claims.len(), 0);
	assert_eq!(packed_eval_claims.len(), 0);

	let same_query_claim = batch_committed_eval_claims
		.try_extract_same_query_pcs_claim(trace_batch.id)
		.unwrap()
		.unwrap();

	pcs.verify_evaluation(
		&mut challenger,
		&trace_comm,
		&same_query_claim.eval_point,
		trace_open_proof,
		&same_query_claim.evals,
	)
	.unwrap();
}

fn make_constraints<F: Field>(
	log_size: usize,
	fixed_oracle: &FixedOracle<F>,
	trace_oracle: &TraceOracle<F>,
) -> Vec<MultivariatePolyOracle<F>> {
	let mut constraints = Vec::new();

	// C_x - \sum_{y=0}^4 A_{x,y} = 0
	for x in 0..1 {
		let constraint = CompositePolyOracle::new(
			log_size,
			iter::once(trace_oracle.c[x].clone())
				.chain((0..5).map(|y| trace_oracle.state_in[x + 5 * y].clone()))
				.collect(),
			Arc::new(SumComposition { n_vars: 6 }),
		)
		.unwrap();
		constraints.push(constraint.into());
	}

	// C_{x-1} + shift_{6,1}(C_{x+1}) - D_x = 0
	for x in 0..4 {
		let constraint = CompositePolyOracle::new(
			log_size,
			vec![
				trace_oracle.c[(x + 4) % 5].clone(),
				trace_oracle.c_shift[(x + 1) % 5].clone(),
				trace_oracle.d[x].clone(),
			],
			Arc::new(SumComposition { n_vars: 3 }),
		)
		.unwrap();
		constraints.push(constraint.into());
	}

	for x in 0..5 {
		for y in 0..5 {
			let constraint = if x == 0 && y == 0 {
				CompositePolyOracle::new(
					log_size,
					vec![
						trace_oracle.state_out[x + 5 * y].clone(),
						trace_oracle.b[x + 5 * y].clone(),
						trace_oracle.b[(x + 1) % 5 + 5 * y].clone(),
						trace_oracle.b[(x + 2) % 5 + 5 * y].clone(),
						fixed_oracle.round_consts.clone(),
					],
					Arc::new(ChiIotaComposition),
				)
				.unwrap()
			} else {
				CompositePolyOracle::new(
					log_size,
					vec![
						trace_oracle.state_out[x + 5 * y].clone(),
						trace_oracle.b[x + 5 * y].clone(),
						trace_oracle.b[(x + 1) % 5 + 5 * y].clone(),
						trace_oracle.b[(x + 2) % 5 + 5 * y].clone(),
					],
					Arc::new(ChiComposition),
				)
				.unwrap()
			};
			constraints.push(constraint.into());
		}
	}

	// Consistency checks with next round
	for xy in 0..25 {
		let constraint = CompositePolyOracle::new(
			log_size,
			vec![
				trace_oracle.state_out[xy].clone(),
				trace_oracle.next_state_in[xy].clone(),
				fixed_oracle.selector.clone(),
			],
			Arc::new(RoundConsistency),
		)
		.unwrap();
		constraints.push(constraint.into());
	}

	constraints
}

fn main() {
	const SECURITY_BITS: usize = 100;

	init_tracing();

	let log_size = 23;
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
	>(SECURITY_BITS, log_size, 60, log_inv_rate, false)
	.unwrap();

	let mut oracles = MultilinearOracleSet::new();
	let trace_batch_id = oracles.add_committed_batch(CommittedBatchSpec {
		round_id: 0,
		n_vars: log_size,
		n_polys: 60,
		tower_level: 0,
	});

	let fixed_oracle = FixedOracle::new(&mut oracles, log_size);
	let trace_oracle = TraceOracle::new(&mut oracles, trace_batch_id);
	let constraints = make_constraints(log_size, &fixed_oracle, &trace_oracle);

	let challenger = <HashChallenger<_, GroestlHasher<_>>>::new();

	let witness = generate_trace(log_size);
	let proof = prove(
		log_size,
		&mut oracles.clone(),
		trace_batch_id,
		&fixed_oracle,
		&trace_oracle,
		&constraints,
		&pcs,
		challenger.clone(),
		witness,
	);

	verify(
		log_size,
		&mut oracles.clone(),
		trace_batch_id,
		&fixed_oracle,
		&trace_oracle,
		&constraints,
		&pcs,
		challenger.clone(),
		proof,
	);
}
