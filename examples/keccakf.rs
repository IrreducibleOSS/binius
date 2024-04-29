// Copyright 2024 Ulvetanna Inc.

#![feature(step_trait)]

extern crate core;

use anyhow::Result;
use binius_core::{
	challenger::{CanObserve, CanSample, CanSampleBits, HashChallenger},
	oracle::{
		BatchId, CommittedBatchSpec, CommittedId, CompositePolyOracle, MultilinearOracleSet,
		MultilinearPolyOracle, OracleId, ShiftVariant,
	},
	poly_commit::{tensor_pcs, PolyCommitScheme},
	polynomial::{
		composition::{empty_mix_composition, index_composition},
		multilinear_query::MultilinearQuery,
		transparent::step_down::StepDown,
		CompositionPoly, Error as PolynomialError, EvaluationDomain, MultilinearComposite,
		MultilinearExtension, MultilinearPoly, MultivariatePoly,
	},
	protocols::{
		evalcheck,
		evalcheck::{BatchCommittedEvalClaims, EvalcheckProof},
		sumcheck::{batch_verify, SumcheckBatchProof, SumcheckProof, SumcheckProveOutput},
		test_utils::{
			full_prove_with_switchover, full_verify, make_non_same_query_pcs_sumcheck_claims,
			make_non_same_query_pcs_sumchecks, prove_bivariate_sumchecks_with_switchover,
		},
		zerocheck,
		zerocheck::{ZerocheckClaim, ZerocheckProof, ZerocheckProveOutput},
	},
	witness::MultilinearWitnessIndex,
};
use binius_field::{
	BinaryField, BinaryField128bPolyval, BinaryField1b, ExtensionField, Field,
	PackedBinaryField128x1b, PackedBinaryField1x128b, PackedBinaryField8x16b, PackedField,
	TowerField,
};
use binius_hash::GroestlHasher;
use bytemuck::{must_cast_slice_mut, Pod};
use rand::{thread_rng, Rng};
use std::{array, env, fmt::Debug, iter, iter::Step, slice, sync::Arc};
use tiny_keccak::keccakf;
use tracing::instrument;
use tracing_profile::{CsvLayer, PrintTreeConfig, PrintTreeLayer};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

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
	} else {
		let _ = tracing_subscriber::registry()
			.with(PrintTreeLayer::new(PrintTreeConfig {
				attention_above_percent: 25.0,
				relevant_above_percent: 2.5,
				hide_below_percent: 1.0,
				display_unaccounted: false,
			}))
			.with(tracing_subscriber::fmt::layer())
			.try_init();
	}
}

#[derive(Clone, Debug)]
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

#[derive(Clone, Debug)]
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

#[derive(Clone, Debug)]
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

#[derive(Clone, Debug)]
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
	fn new() -> Result<Self> {
		let mut values = vec![P::default(); 1 << (11 - P::LOG_WIDTH)];
		must_cast_slice_mut::<_, u64>(&mut values).copy_from_slice(KECCAKF_RC.as_slice());
		let mle = MultilinearExtension::from_values(values)?;
		Ok(Self(mle))
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
	pub fn new(oracles: &mut MultilinearOracleSet<F>, log_size: usize) -> Result<Self> {
		let round_consts_single = oracles
			.add_transparent(Arc::new(RoundConstant::<PackedBinaryField128x1b>::new()?), 0)?;
		let round_consts = oracles.add_repeating(round_consts_single, log_size - 11)?;

		let selector_single = oracles.add_transparent(Arc::new(StepDown::new(11, 24 * 64)?), 0)?;
		let selector = oracles.add_repeating(selector_single, log_size - 11)?;

		Ok(Self {
			round_consts: oracles.oracle(round_consts),
			selector: oracles.oracle(selector),
		})
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

impl<P: PackedField> TraceWitness<P> {
	fn to_index<F, PE>(
		&self,
		fixed_oracle: &FixedOracle<F>,
		trace_oracle: &TraceOracle<F>,
	) -> MultilinearWitnessIndex<PE>
	where
		F: ExtensionField<P::Scalar>,
		PE: PackedField,
		PE::Scalar: ExtensionField<P::Scalar>,
	{
		let mut index = MultilinearWitnessIndex::new();

		for (oracle_arr, witness_arr) in [
			(&trace_oracle.state_in[..], &self.state_in[..]),
			(&trace_oracle.state_out[..], &self.state_out[..]),
			(&trace_oracle.c[..], &self.c[..]),
			(&trace_oracle.d[..], &self.d[..]),
			(&trace_oracle.c_shift[..], &self.c_shift[..]),
			(&trace_oracle.a_theta[..], &self.a_theta[..]),
			(&trace_oracle.b[..], &self.b[..]),
			(&trace_oracle.next_state_in[..], &self.next_state_in[..]),
			(slice::from_ref(&fixed_oracle.round_consts), slice::from_ref(&self.round_consts)),
			(slice::from_ref(&fixed_oracle.selector), slice::from_ref(&self.selector)),
		] {
			for (oracle, witness) in oracle_arr.iter().zip(witness_arr.iter()) {
				index.set(
					oracle.id(),
					MultilinearExtension::from_values_slice(witness.as_slice())
						.unwrap()
						.specialize_arc_dyn(),
				);
			}
		}

		index
	}
}

struct Proof<F: Field, PCSComm, PCSProof> {
	trace_comm: PCSComm,
	zerocheck_proof: ZerocheckProof,
	sumcheck_proof: SumcheckProof<F>,
	evalcheck_proof: EvalcheckProof<F>,
	second_round_batch_sumcheck_proof: SumcheckBatchProof<F>,
	second_round_evalcheck_proofs: Vec<EvalcheckProof<F>>,
	third_round_batch_sumcheck_proof: SumcheckBatchProof<F>,
	third_round_evalcheck_proofs: Vec<EvalcheckProof<F>>,
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

fn zerocheck_verifier_oracles<F: Field>(
	fixed_oracle: &FixedOracle<F>,
	trace_oracle: &TraceOracle<F>,
) -> Vec<MultilinearPolyOracle<F>> {
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
		.collect()
}

fn zerocheck_prover_witness<'a, P, PW, C>(
	log_size: usize,
	witness: &'a TraceWitness<P>,
	composition: C,
) -> Result<MultilinearComposite<PW, C, Arc<dyn MultilinearPoly<PW> + Send + Sync + 'a>>>
where
	P: PackedField<Scalar = BinaryField1b>,
	PW: ExtensionField<BinaryField1b>,
	C: CompositionPoly<PW>,
{
	let multilinears = iter::once(&witness.round_consts)
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
			MultilinearExtension::from_values_slice(values.as_slice())
				.unwrap()
				.specialize_arc_dyn()
		})
		.collect();

	Ok(MultilinearComposite::new(log_size, composition, multilinears)?)
}

#[allow(clippy::too_many_arguments)]
#[instrument(skip_all)]
fn prove<P, F, PW, PCS, CH>(
	log_size: usize,
	oracles: &mut MultilinearOracleSet<F>,
	trace_batch_id: BatchId,
	fixed_oracle: &FixedOracle<F>,
	trace_oracle: &TraceOracle<F>,
	pcs: &PCS,
	mut challenger: CH,
	witness: &TraceWitness<P>,
) -> Result<Proof<F, PCS::Commitment, PCS::Proof>>
where
	P: PackedField<Scalar = BinaryField1b> + Debug + Pod,
	F: TowerField + Step + From<PW>,
	PW: TowerField + From<F>,
	PCS: PolyCommitScheme<P, F>,
	PCS::Error: Debug,
	PCS::Proof: 'static,
	CH: CanObserve<F> + CanObserve<PCS::Commitment> + CanSample<F> + CanSampleBits<usize> + Clone,
{
	let mut trace_witness = witness.to_index::<_, PW>(fixed_oracle, trace_oracle);

	// Round 1
	let trace_commit_polys = witness
		.state_in
		.iter()
		.chain(witness.state_out.iter())
		.chain(witness.c.iter())
		.chain(witness.d.iter())
		.map(|vals| MultilinearExtension::from_values_slice(vals.as_slice()).unwrap())
		.collect::<Vec<_>>();
	let (trace_comm, trace_committed) = pcs.commit(&trace_commit_polys).unwrap();
	challenger.observe(trace_comm.clone());

	// Zerocheck mixing
	let zerocheck_column_oracles = zerocheck_verifier_oracles(fixed_oracle, trace_oracle);
	let zerocheck_column_ids = zerocheck_column_oracles
		.iter()
		.map(|oracle| oracle.id())
		.collect::<Vec<_>>();

	let mixing_challenge = challenger.sample();

	let mix_composition_verifier =
		make_constraints(fixed_oracle, trace_oracle, &zerocheck_column_ids, mixing_challenge)?;
	let mix_composition_prover = make_constraints(
		fixed_oracle,
		trace_oracle,
		&zerocheck_column_ids,
		PW::from(mixing_challenge),
	)?;

	let zerocheck_claim = ZerocheckClaim {
		poly: CompositePolyOracle::new(
			log_size,
			zerocheck_column_oracles,
			mix_composition_verifier,
		)?,
	};

	let zerocheck_witness = zerocheck_prover_witness(log_size, witness, mix_composition_prover)?;

	// Zerocheck
	let zerocheck_challenge = challenger.sample_vec(log_size - 1);

	let ZerocheckProveOutput {
		sumcheck_claim,
		sumcheck_witness,
		zerocheck_proof,
	} = zerocheck::prove(&zerocheck_claim, zerocheck_witness, zerocheck_challenge).unwrap();

	// Sumcheck
	let sumcheck_domain = EvaluationDomain::<PW>::new_isomorphic::<F>(
		sumcheck_claim.poly.max_individual_degree() + 1,
	)
	.unwrap();

	let switchover_fn = |extension_degree| match extension_degree {
		128 => 5,
		_ => 1,
	};

	let (_, output) = full_prove_with_switchover(
		&sumcheck_claim,
		sumcheck_witness,
		&sumcheck_domain,
		&mut challenger,
		switchover_fn,
	);

	let SumcheckProveOutput {
		evalcheck_claim,
		sumcheck_proof,
	} = output;

	// Evalcheck
	let trace_batch = oracles.committed_batch(trace_batch_id);

	let mut new_sumchecks = Vec::new();
	let mut batch_committed_eval_claims = BatchCommittedEvalClaims::new(&[trace_batch.clone()]);
	let evalcheck_proof = evalcheck::prove(
		oracles,
		&mut trace_witness,
		evalcheck_claim,
		&mut batch_committed_eval_claims,
		&mut new_sumchecks,
	)
	.unwrap();
	assert_eq!(batch_committed_eval_claims.n_batches(), 1);
	assert_eq!(new_sumchecks.len(), 54);

	// Second sumcheck
	let (second_round_batch_sumcheck_proof, second_round_evalcheck_claims) =
		prove_bivariate_sumchecks_with_switchover(new_sumchecks, &mut challenger, switchover_fn)?;

	// Second Evalchecks
	let mut new_sumchecks_2 = Vec::new();

	let second_round_evalcheck_proofs = second_round_evalcheck_claims
		.into_iter()
		.map(|claim| {
			evalcheck::prove(
				oracles,
				&mut trace_witness,
				claim,
				&mut batch_committed_eval_claims,
				&mut new_sumchecks_2,
			)
		})
		.collect::<Result<Vec<_>, _>>()?;
	assert_eq!(new_sumchecks_2.len(), 0);

	// Third sumcheck
	assert!(batch_committed_eval_claims
		.try_extract_same_query_pcs_claim(trace_batch.id)
		.transpose()
		.is_none());

	let non_sqpcs_claims = batch_committed_eval_claims.take_claims(trace_batch.id)?;
	let mut batch_committed_eval_claims_final =
		BatchCommittedEvalClaims::new(&[trace_batch.clone()]);

	let non_sqpcs_sumchecks = make_non_same_query_pcs_sumchecks(
		oracles,
		&mut trace_witness,
		&non_sqpcs_claims,
		&mut batch_committed_eval_claims_final,
	)?;

	let (third_round_batch_sumcheck_proof, third_round_evalcheck_claims) =
		prove_bivariate_sumchecks_with_switchover(
			non_sqpcs_sumchecks,
			&mut challenger,
			switchover_fn,
		)?;

	let mut new_sumchecks_3 = Vec::new();
	let third_round_evalcheck_proofs = third_round_evalcheck_claims
		.into_iter()
		.map(|claim| {
			evalcheck::prove(
				oracles,
				&mut trace_witness,
				claim,
				&mut batch_committed_eval_claims_final,
				&mut new_sumchecks_3,
			)
		})
		.collect::<Result<Vec<_>, _>>()?;
	assert_eq!(new_sumchecks_3.len(), 0);

	// Should be same query pcs claim
	let same_query_claim = batch_committed_eval_claims_final
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

	Ok(Proof {
		trace_comm,
		zerocheck_proof,
		sumcheck_proof,
		evalcheck_proof,
		second_round_batch_sumcheck_proof,
		second_round_evalcheck_proofs,
		third_round_batch_sumcheck_proof,
		third_round_evalcheck_proofs,
		trace_open_proof,
	})
}

#[allow(clippy::too_many_arguments)]
#[instrument(skip_all)]
fn verify<P, F, PCS, CH>(
	log_size: usize,
	oracles: &mut MultilinearOracleSet<F>,
	trace_batch_id: BatchId,
	fixed_oracle: &FixedOracle<F>,
	trace_oracle: &TraceOracle<F>,
	pcs: &PCS,
	mut challenger: CH,
	proof: Proof<F, PCS::Commitment, PCS::Proof>,
) -> Result<()>
where
	P: PackedField<Scalar = BinaryField1b> + Debug,
	F: TowerField,
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
		second_round_batch_sumcheck_proof,
		second_round_evalcheck_proofs,
		third_round_batch_sumcheck_proof,
		third_round_evalcheck_proofs,
		trace_open_proof,
	} = proof;

	// Round 1
	challenger.observe(trace_comm.clone());

	// Zerocheck mixing
	let zerocheck_column_oracles = zerocheck_verifier_oracles(fixed_oracle, trace_oracle);
	let zerocheck_column_ids = zerocheck_column_oracles
		.iter()
		.map(|oracle| oracle.id())
		.collect::<Vec<_>>();

	let mixing_challenge = challenger.sample();

	let mix_composition =
		make_constraints(fixed_oracle, trace_oracle, &zerocheck_column_ids, mixing_challenge)?;

	// Zerocheck
	let zerocheck_challenge = challenger.sample_vec(log_size - 1);

	let zerocheck_claim = ZerocheckClaim {
		poly: CompositePolyOracle::new(log_size, zerocheck_column_oracles, mix_composition)?,
	};

	let sumcheck_claim =
		zerocheck::verify(&zerocheck_claim, zerocheck_proof, zerocheck_challenge).unwrap();

	// Sumcheck
	let (_, evalcheck_claim) = full_verify(&sumcheck_claim, sumcheck_proof, &mut challenger);

	// Evalcheck
	let trace_batch = oracles.committed_batch(trace_batch_id);

	let mut new_sumcheck_claims = Vec::new();
	let mut batch_committed_eval_claims = BatchCommittedEvalClaims::new(&[trace_batch.clone()]);
	evalcheck::verify(
		oracles,
		evalcheck_claim,
		evalcheck_proof,
		&mut batch_committed_eval_claims,
		&mut new_sumcheck_claims,
	)
	.unwrap();
	assert_eq!(batch_committed_eval_claims.n_batches(), 1);
	assert_eq!(new_sumcheck_claims.len(), 54);

	// Second Sumcheck
	let second_round_evalcheck_claims =
		batch_verify(new_sumcheck_claims, second_round_batch_sumcheck_proof, &mut challenger)?;

	// Second Evalchecks
	let mut new_sumcheck_claims_2 = Vec::new();

	second_round_evalcheck_claims
		.into_iter()
		.zip(second_round_evalcheck_proofs)
		.try_for_each(|(claim, proof)| {
			evalcheck::verify(
				oracles,
				claim,
				proof,
				&mut batch_committed_eval_claims,
				&mut new_sumcheck_claims_2,
			)
		})?;

	assert_eq!(batch_committed_eval_claims.n_batches(), 1);
	assert_eq!(new_sumcheck_claims_2.len(), 0);

	// Third sumcheck
	assert!(batch_committed_eval_claims
		.try_extract_same_query_pcs_claim(trace_batch.id)
		.transpose()
		.is_none());

	let non_sqpcs_claims = batch_committed_eval_claims.take_claims(trace_batch.id)?;
	let mut batch_committed_eval_claims_final =
		BatchCommittedEvalClaims::new(&[trace_batch.clone()]);

	let third_round_sumcheck_claims = make_non_same_query_pcs_sumcheck_claims(
		oracles,
		&non_sqpcs_claims,
		&mut batch_committed_eval_claims_final,
	)?;

	let third_round_evalcheck_claims = batch_verify(
		third_round_sumcheck_claims,
		third_round_batch_sumcheck_proof,
		&mut challenger,
	)?;

	let mut new_sumcheck_claims_3 = Vec::new();

	third_round_evalcheck_claims
		.into_iter()
		.zip(third_round_evalcheck_proofs)
		.try_for_each(|(claim, proof)| {
			evalcheck::verify(
				oracles,
				claim,
				proof,
				&mut batch_committed_eval_claims_final,
				&mut new_sumcheck_claims_3,
			)
		})?;
	assert_eq!(new_sumcheck_claims_3.len(), 0);

	// Should be same query pcs claim
	let same_query_claim = batch_committed_eval_claims_final
		.try_extract_same_query_pcs_claim(trace_batch.id)?
		.expect("eval queries must be at a single point");

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
fn make_constraints<F: TowerField, FI: TowerField>(
	fixed_oracle: &FixedOracle<F>,
	trace_oracle: &TraceOracle<F>,
	zerocheck_column_ids: &[OracleId],
	challenge: FI,
) -> Result<impl CompositionPoly<FI> + Clone> {
	let mix = empty_mix_composition(zerocheck_column_ids.len(), challenge);

	// C_x - \sum_{y=0}^4 A_{x,y} = 0
	let mix = mix.include((0..5).map(|x| {
		index_composition(
			zerocheck_column_ids,
			[
				trace_oracle.c[x].id(),
				trace_oracle.state_in[x + 5 * 0].id(),
				trace_oracle.state_in[x + 5 * 1].id(),
				trace_oracle.state_in[x + 5 * 2].id(),
				trace_oracle.state_in[x + 5 * 3].id(),
				trace_oracle.state_in[x + 5 * 4].id(),
			],
			SumComposition { n_vars: 6 },
		)
		.unwrap()
	}))?;

	// C_{x-1} + shift_{6,1}(C_{x+1}) - D_x = 0
	let mix = mix.include((0..5).map(|x| {
		index_composition(
			zerocheck_column_ids,
			[
				trace_oracle.c[(x + 4) % 5].id(),
				trace_oracle.c_shift[(x + 1) % 5].id(),
				trace_oracle.d[x].id(),
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
			zerocheck_column_ids,
			[
				trace_oracle.state_out[x + 5 * y].id(),
				trace_oracle.b[x + 5 * y].id(),
				trace_oracle.b[(x + 1) % 5 + 5 * y].id(),
				trace_oracle.b[(x + 2) % 5 + 5 * y].id(),
				fixed_oracle.round_consts.id(),
			],
			ChiIotaComposition,
		)
		.unwrap()
	};

	let mix = mix.include([chi_iota_constraint])?;

	// chi
	let mix = mix.include((1..24).map(|i| {
		let x = (i + 1) / 5;
		let y = (i + 1) % 5;

		index_composition(
			zerocheck_column_ids,
			[
				trace_oracle.state_out[x + 5 * y].id(),
				trace_oracle.b[x + 5 * y].id(),
				trace_oracle.b[(x + 1) % 5 + 5 * y].id(),
				trace_oracle.b[(x + 2) % 5 + 5 * y].id(),
			],
			ChiComposition,
		)
		.unwrap()
	}))?;

	// consistency checks with next round
	let mix = mix.include((0..25).map(|xy| {
		index_composition(
			zerocheck_column_ids,
			[
				trace_oracle.state_out[xy].id(),
				trace_oracle.next_state_in[xy].id(),
				fixed_oracle.selector.id(),
			],
			RoundConsistency,
		)
		.unwrap()
	}))?;

	Ok(mix)
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
	let fixed_oracle = FixedOracle::new(&mut oracles, log_size).unwrap();

	let trace_batch_id = oracles.add_committed_batch(CommittedBatchSpec {
		round_id: 0,
		n_vars: log_size,
		n_polys: 60,
		tower_level: 0,
	});
	let trace_oracle = TraceOracle::new(&mut oracles, trace_batch_id);

	let challenger = <HashChallenger<_, GroestlHasher<_>>>::new();

	let witness = generate_trace(log_size);
	let proof = prove::<_, _, BinaryField128bPolyval, _, _>(
		log_size,
		&mut oracles.clone(),
		trace_batch_id,
		&fixed_oracle,
		&trace_oracle,
		&pcs,
		challenger.clone(),
		&witness,
	)
	.unwrap();

	verify(
		log_size,
		&mut oracles.clone(),
		trace_batch_id,
		&fixed_oracle,
		&trace_oracle,
		&pcs,
		challenger.clone(),
		proof,
	)
	.unwrap();
}
