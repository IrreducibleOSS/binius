// Copyright 2024 Ulvetanna Inc.

#![feature(step_trait)]

extern crate core;

use anyhow::Result;
use binius_core::{
	challenger::{CanObserve, CanSample, CanSampleBits, HashChallenger},
	oracle::{BatchId, CompositePolyOracle, MultilinearOracleSet, OracleId, ShiftVariant},
	poly_commit::{tensor_pcs, PolyCommitScheme},
	polynomial::{
		composition::{empty_mix_composition, index_composition},
		multilinear_query::MultilinearQuery,
		transparent::step_down::StepDown,
		CompositionPoly, Error as PolynomialError, EvaluationDomainFactory,
		IsomorphicEvaluationDomainFactory, MultilinearComposite, MultilinearExtension,
		MultivariatePoly,
	},
	protocols::{
		greedy_evalcheck,
		greedy_evalcheck::{GreedyEvalcheckProof, GreedyEvalcheckProveOutput},
		zerocheck::{self, ZerocheckClaim, ZerocheckProof, ZerocheckProveOutput},
	},
	witness::MultilinearExtensionIndex,
};
use binius_field::{
	as_packed_field::{PackScalar, PackedType},
	underlier::{UnderlierType, WithUnderlier},
	BinaryField, BinaryField128b, BinaryField128bPolyval, BinaryField16b, BinaryField1b,
	ExtensionField, Field, PackedBinaryField128x1b, PackedField, PackedFieldIndexable, TowerField,
};
use binius_hash::GroestlHasher;
use binius_macros::{composition_poly, IterOracles};
use binius_utils::{
	examples::get_log_trace_size, rayon::adjust_thread_pool, tracing::init_tracing,
};
use bytemuck::{must_cast_slice_mut, Pod};
use bytesize::ByteSize;
use itertools::chain;
use rand::{thread_rng, Rng};
use std::{array, fmt::Debug, iter};
use tiny_keccak::keccakf;
use tracing::instrument;

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

	fn evaluate_scalar(&self, query: &[P::Scalar]) -> Result<P::Scalar, PolynomialError> {
		if query.len() != self.n_vars {
			return Err(PolynomialError::IncorrectQuerySize {
				expected: self.n_vars,
			});
		}
		// Sum of scalar values at the corresponding positions of the packed values.
		Ok(query.iter().copied().sum())
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

composition_poly!(ChiComposition[a, b0, b1, b2] = a - (b0 + (1 - b1) * b2));
composition_poly!(ChiIotaComposition[a, b0, b1, b2, rc] = a - (rc + b0 + (1 - b1) * b2));
composition_poly!(RoundConsistency[state_out, next_state_in, select] = (state_out - next_state_in) * select);

#[derive(Debug)]
struct RoundConstant<P: PackedField<Scalar = BinaryField1b>>(MultilinearExtension<P>);

impl<P: PackedField<Scalar = BinaryField1b> + Pod> RoundConstant<P> {
	fn new() -> Result<Self> {
		let mut values = vec![P::default(); 1 << (11 - P::LOG_WIDTH)];
		must_cast_slice_mut::<_, u64>(&mut values).copy_from_slice(KECCAKF_RC.as_slice());
		let mle = MultilinearExtension::from_values(values)?;
		Ok(Self(mle))
	}
}

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

	fn binary_tower_level(&self) -> usize {
		0
	}
}

#[derive(Debug, IterOracles)]
struct FixedOracle {
	round_consts: OracleId,
	selector: OracleId,
}

impl FixedOracle {
	pub fn new<F: TowerField>(
		oracles: &mut MultilinearOracleSet<F>,
		log_size: usize,
	) -> Result<Self> {
		let round_consts_single =
			oracles.add_transparent(RoundConstant::<PackedBinaryField128x1b>::new()?)?;
		let round_consts = oracles.add_repeating(round_consts_single, log_size - 11)?;

		let selector_single = oracles.add_transparent(StepDown::new(11, 24 * 64)?)?;
		let selector = oracles.add_repeating(selector_single, log_size - 11)?;

		Ok(Self {
			round_consts,
			selector,
		})
	}
}

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
		let mut batch_scope = oracles.build_committed_batch(log_size, BinaryField1b::TOWER_LEVEL);
		let state_in_oracle = batch_scope.add_multiple::<25>();
		let state_out_oracle = batch_scope.add_multiple::<25>();
		let c_oracle = batch_scope.add_multiple::<5>();
		let d_oracle = batch_scope.add_multiple::<5>();
		let batch_id = batch_scope.build();

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

	fn iter_constrained_oracles(&self) -> impl Iterator<Item = OracleId> {
		chain!(
			self.state_in,
			self.state_out,
			self.c,
			self.c_shift,
			self.d,
			self.b,
			self.next_state_in,
		)
	}
}

fn constrained_oracles(
	fixed_oracle: &FixedOracle,
	trace_oracle: &TraceOracle,
) -> impl Iterator<Item = OracleId> {
	fixed_oracle
		.iter_oracles()
		.chain(trace_oracle.iter_constrained_oracles())
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

#[allow(clippy::too_many_arguments)]
#[instrument(skip_all)]
fn prove<U, F, FW, DomainField, PCS, CH>(
	log_size: usize,
	oracles: &mut MultilinearOracleSet<F>,
	fixed_oracle: &FixedOracle,
	trace_oracle: &TraceOracle,
	pcs: &PCS,
	mut challenger: CH,
	witness: MultilinearExtensionIndex<U, FW>,
	domain_factory: impl EvaluationDomainFactory<DomainField>,
) -> Result<Proof<F, PCS::Commitment, PCS::Proof>>
where
	U: UnderlierType + PackScalar<BinaryField1b> + PackScalar<FW>,
	PackedType<U, FW>: PackedFieldIndexable,
	F: TowerField + From<FW>,
	FW: TowerField + From<F> + ExtensionField<DomainField>,
	DomainField: TowerField,
	PCS: PolyCommitScheme<PackedType<U, BinaryField1b>, F, Error: Debug, Proof: 'static>,
	CH: CanObserve<F> + CanObserve<PCS::Commitment> + CanSample<F> + CanSampleBits<usize>,
{
	let mut trace_witness = witness.witness_index();

	// Round 1
	let trace_commit_polys = oracles
		.committed_oracle_ids(trace_oracle.batch_id)
		.map(|oracle_id| witness.get::<BinaryField1b>(oracle_id))
		.collect::<Result<Vec<_>, _>>()?;
	let (trace_comm, trace_committed) = pcs.commit(&trace_commit_polys)?;
	challenger.observe(trace_comm.clone());

	// Zerocheck mixing
	let mixing_challenge = challenger.sample();

	let mix_composition_verifier = make_constraints(fixed_oracle, trace_oracle, mixing_challenge)?;
	let mix_composition_prover =
		make_constraints(fixed_oracle, trace_oracle, FW::from(mixing_challenge))?;

	let zerocheck_column_oracles = constrained_oracles(fixed_oracle, trace_oracle)
		.map(|id| oracles.oracle(id))
		.collect();
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
		constrained_oracles(fixed_oracle, trace_oracle)
			.map(|oracle_id| witness.get_multilin_poly(oracle_id))
			.collect::<Result<_, _>>()?,
	)?;

	// Zerocheck
	let zerocheck_domain =
		domain_factory.create(zerocheck_claim.poly.max_individual_degree() + 1)?;
	let switchover_fn = |extension_degree| match extension_degree {
		128 => 5,
		_ => 1,
	};

	let ZerocheckProveOutput {
		evalcheck_claim,
		zerocheck_proof,
	} = zerocheck::prove(
		&zerocheck_claim,
		zerocheck_witness,
		&zerocheck_domain,
		&mut challenger,
		switchover_fn,
	)?;

	// Evalcheck
	let GreedyEvalcheckProveOutput {
		same_query_claims,
		proof: evalcheck_proof,
	} = greedy_evalcheck::prove(
		oracles,
		&mut trace_witness,
		[evalcheck_claim],
		switchover_fn,
		&mut challenger,
		domain_factory,
	)?;

	assert_eq!(same_query_claims.len(), 1);
	let (batch_id, same_query_claim) = same_query_claims
		.into_iter()
		.next()
		.expect("length is asserted to be 1");
	assert_eq!(batch_id, trace_oracle.batch_id);

	let trace_open_proof = pcs.prove_evaluation(
		&mut challenger,
		&trace_committed,
		&trace_commit_polys,
		&same_query_claim.eval_point,
	)?;

	Ok(Proof {
		trace_comm,
		zerocheck_proof,
		evalcheck_proof,
		trace_open_proof,
	})
}

#[allow(clippy::too_many_arguments)]
#[instrument(skip_all)]
fn verify<P, F, PCS, CH>(
	log_size: usize,
	oracles: &mut MultilinearOracleSet<F>,
	fixed_oracle: &FixedOracle,
	trace_oracle: &TraceOracle,
	pcs: &PCS,
	mut challenger: CH,
	proof: Proof<F, PCS::Commitment, PCS::Proof>,
) -> Result<()>
where
	P: PackedField<Scalar = BinaryField1b>,
	F: TowerField,
	PCS: PolyCommitScheme<P, F, Error: Debug, Proof: 'static>,
	CH: CanObserve<F> + CanObserve<PCS::Commitment> + CanSample<F> + CanSampleBits<usize>,
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
	let mix_composition = make_constraints(fixed_oracle, trace_oracle, mixing_challenge)?;

	// Zerocheck
	let zerocheck_column_oracles = constrained_oracles(fixed_oracle, trace_oracle)
		.map(|id| oracles.oracle(id))
		.collect();
	let zerocheck_claim = ZerocheckClaim {
		poly: CompositePolyOracle::new(log_size, zerocheck_column_oracles, mix_composition)?,
	};

	let evalcheck_claim = zerocheck::verify(&zerocheck_claim, zerocheck_proof, &mut challenger)?;

	// Evalcheck
	let same_query_claims =
		greedy_evalcheck::verify(oracles, [evalcheck_claim], evalcheck_proof, &mut challenger)?;

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
	)?;

	Ok(())
}

#[allow(clippy::identity_op, clippy::erasing_op)]
fn make_constraints<P: PackedField<Scalar: TowerField>>(
	fixed_oracle: &FixedOracle,
	trace_oracle: &TraceOracle,
	challenge: P::Scalar,
) -> Result<impl CompositionPoly<P> + Clone + 'static> {
	let zerocheck_column_ids = constrained_oracles(fixed_oracle, trace_oracle).collect::<Vec<_>>();
	let mix = empty_mix_composition(zerocheck_column_ids.len(), challenge);

	// C_x - \sum_{y=0}^4 A_{x,y} = 0
	let mix = mix.include((0..5).map(|x| {
		index_composition(
			&zerocheck_column_ids,
			[
				trace_oracle.c[x],
				trace_oracle.state_in[x + 5 * 0],
				trace_oracle.state_in[x + 5 * 1],
				trace_oracle.state_in[x + 5 * 2],
				trace_oracle.state_in[x + 5 * 3],
				trace_oracle.state_in[x + 5 * 4],
			],
			SumComposition { n_vars: 6 },
		)
		.unwrap()
	}))?;

	// C_{x-1} + shift_{6,1}(C_{x+1}) - D_x = 0
	let mix = mix.include((0..5).map(|x| {
		index_composition(
			&zerocheck_column_ids,
			[
				trace_oracle.c[(x + 4) % 5],
				trace_oracle.c_shift[(x + 1) % 5],
				trace_oracle.d[x],
			],
			SumComposition { n_vars: 3 },
		)
		.unwrap()
	}))?;

	// chi iota
	let chi_iota_constraint = {
		let x = 0;
		let y = 0;

		index_composition(
			&zerocheck_column_ids,
			[
				trace_oracle.state_out[x + 5 * y],
				trace_oracle.b[x + 5 * y],
				trace_oracle.b[(x + 1) % 5 + 5 * y],
				trace_oracle.b[(x + 2) % 5 + 5 * y],
				fixed_oracle.round_consts,
			],
			composition_poly!([a, b0, b1, b2, rc] = a - (rc + b0 + (1 - b1) * b2)), //ChiIotaComposition,
		)
		.unwrap()
	};

	let mix = mix.include([chi_iota_constraint])?;

	// chi
	let mix = mix.include((1..25).map(|i| {
		let x = i / 5;
		let y = i % 5;

		index_composition(
			&zerocheck_column_ids,
			[
				trace_oracle.state_out[x + 5 * y],
				trace_oracle.b[x + 5 * y],
				trace_oracle.b[(x + 1) % 5 + 5 * y],
				trace_oracle.b[(x + 2) % 5 + 5 * y],
			],
			ChiComposition,
		)
		.unwrap()
	}))?;

	// consistency checks with next round
	let mix = mix.include((0..25).map(|xy| {
		index_composition(
			&zerocheck_column_ids,
			[
				trace_oracle.state_out[xy],
				trace_oracle.next_state_in[xy],
				fixed_oracle.selector,
			],
			RoundConsistency,
		)
		.unwrap()
	}))?;

	Ok(mix)
}

fn main() {
	const SECURITY_BITS: usize = 100;

	adjust_thread_pool()
		.as_ref()
		.expect("failed to init thread pool");

	init_tracing();

	let log_size = get_log_trace_size().unwrap_or(14);
	let log_inv_rate = 1;

	type U = <PackedBinaryField128x1b as WithUnderlier>::Underlier;

	let mut oracles = MultilinearOracleSet::new();
	let fixed_oracle = FixedOracle::new(&mut oracles, log_size).unwrap();
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
	tracing::info!("Size of tensorpcs: {}", tensorpcs_size);

	let challenger = <HashChallenger<_, GroestlHasher<_>>>::new();

	let witness =
		generate_trace::<U, BinaryField128bPolyval>(log_size, &fixed_oracle, &trace_oracle)
			.unwrap();
	let domain_factory = IsomorphicEvaluationDomainFactory::<BinaryField128b>::default();
	// TODO: Ideally DomainField should be considerably smaller than F, something like BinaryField8b.
	//       However, BinaryField128bPolyval doesn't support any conversions other than to/from 1 and 128 bits.
	let proof = prove::<_, BinaryField128b, BinaryField128bPolyval, BinaryField128bPolyval, _, _>(
		log_size,
		&mut oracles.clone(),
		&fixed_oracle,
		&trace_oracle,
		&pcs,
		challenger.clone(),
		witness,
		domain_factory,
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
	)
	.unwrap();
}
