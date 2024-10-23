// Copyright 2024 Ulvetanna Inc.

#![feature(step_trait)]

/// An example computing the lowest 32 bits of consecutive fibonacci numbers.
/// The example performs the computation, generates a proof and verifies it.
use anyhow::Result;
use binius_core::{
	challenger::{new_hasher_challenger, IsomorphicChallenger},
	oracle::{BatchId, ConstraintSetBuilder, MultilinearOracleSet, OracleId, ShiftVariant},
	poly_commit::{tensor_pcs, PolyCommitScheme},
	protocols::{
		abstract_sumcheck::standard_switchover_heuristic,
		greedy_evalcheck_v2::{self, GreedyEvalcheckProof, GreedyEvalcheckProveOutput},
		sumcheck_v2::{self, Proof as ZerocheckProof},
	},
	transparent::step_down::StepDown,
	witness::MultilinearExtensionIndex,
};
use binius_field::{
	as_packed_field::{PackScalar, PackedType},
	underlier::{UnderlierType, WithUnderlier},
	BinaryField128b, BinaryField16b, BinaryField1b, ExtensionField, Field, PackedBinaryField128x1b,
	PackedExtension, PackedField, PackedFieldIndexable, TowerField,
};
use binius_hal::{make_portable_backend, ComputationBackend};
use binius_hash::GroestlHasher;
use binius_macros::{composition_poly, IterOracles};
use binius_math::{EvaluationDomainFactory, IsomorphicEvaluationDomainFactory};
use binius_utils::{
	examples::get_log_trace_size, rayon::adjust_thread_pool, tracing::init_tracing,
};
use bytemuck::{must_cast_slice_mut, Pod};
use itertools::Itertools;
use p3_challenger::{CanObserve, CanSample, CanSampleBits};
use std::fmt::Debug;
use tracing::{debug, info, instrument};

/// mod field_types is a selector of different sets of types which provide
/// equivalent functionality but may differ significantly in performance.
#[cfg(feature = "aes-tower")]
mod field_types {
	pub type Field = binius_field::AESTowerField128b;
	pub type DomainField = binius_field::AESTowerField8b;
	pub type DomainFieldWithStep = binius_field::AESTowerField8b;
	pub type Fbase = binius_field::AESTowerField8b;
}

#[cfg(not(feature = "aes-tower"))]
mod field_types {
	pub type Field = binius_field::BinaryField128bPolyval;
	pub type DomainField = binius_field::BinaryField128bPolyval;
	pub type DomainFieldWithStep = binius_field::BinaryField128b;
	pub type Fbase = binius_field::BinaryField128bPolyval;
}

// The fields must be in exactly the same order as the fields in U32FibOracle.
struct U32FibTrace<U>
where
	U: UnderlierType + PackScalar<BinaryField1b>,
{
	/// Fibonacci number computed in the current step.
	fib_out: Vec<U>,
	/// Fibonacci number computed one step ago.
	fib_prev1: Vec<U>,
	/// Fibonacci number computed two step ago.
	fib_prev2: Vec<U>,

	/// U32 carry addition information.
	c_out: Vec<U>,
	c_in: Vec<U>,

	/// Shifting pads with zeros, which will mess with constraints
	/// so we need a transparent StepDown to ignore the first row.
	disabled: Vec<U>,
}

impl<U: UnderlierType + PackScalar<BinaryField1b> + Pod> U32FibTrace<U> {
	fn new(log_size: usize) -> Self {
		Self {
			fib_out: vec![
				U::default();
				1 << (log_size - <PackedType<U, BinaryField1b>>::LOG_WIDTH)
			],
			fib_prev1: vec![
				U::default();
				1 << (log_size - <PackedType<U, BinaryField1b>>::LOG_WIDTH)
			],
			fib_prev2: vec![
				U::default();
				1 << (log_size - <PackedType<U, BinaryField1b>>::LOG_WIDTH)
			],
			c_out: vec![U::default(); 1 << (log_size - <PackedType<U, BinaryField1b>>::LOG_WIDTH)],
			c_in: vec![U::default(); 1 << (log_size - <PackedType<U, BinaryField1b>>::LOG_WIDTH)],
			disabled: vec![
				U::default();
				1 << (log_size - <PackedType<U, BinaryField1b>>::LOG_WIDTH)
			],
		}
	}

	fn fill_trace(mut self) -> Self {
		let fib_out = must_cast_slice_mut::<_, u32>(&mut self.fib_out);
		let fib_prev1 = must_cast_slice_mut::<_, u32>(&mut self.fib_prev1);
		let fib_prev2 = must_cast_slice_mut::<_, u32>(&mut self.fib_prev2);
		let cout = must_cast_slice_mut::<_, u32>(&mut self.c_out);
		let cin = must_cast_slice_mut::<_, u32>(&mut self.c_in);
		let disabled = must_cast_slice_mut::<_, u32>(&mut self.disabled);

		// Initialize the first row in a special manner.
		// It asserts that the initial fib number is 1, but doesn't represent a valid
		// addition operation, therefore this row is disabled.
		fib_prev2[0] = 0;
		fib_prev1[0] = 0;
		fib_out[0] = 1;

		cin[0] = 0;
		cout[0] = 0;

		// The three initial fib values are 0,0,1 and therefore don't affect bits other than the 0-th bit.
		// If we disable the constraints for that bit only, then we get a rather simple proof.
		disabled[0] = 1;

		// Every other row takes two previous fib numbers and adds them together.
		for i in 1..fib_prev2.len() {
			fib_prev1[i] = fib_out[i - 1];
			fib_prev2[i] = fib_prev1[i - 1];
			let carry;
			(fib_out[i], carry) = (fib_prev2[i]).overflowing_add(fib_prev1[i]);
			cin[i] = (fib_prev2[i]) ^ (fib_prev1[i]) ^ (fib_out[i]);
			cout[i] = cin[i] >> 1;
			if carry {
				cout[i] |= 1 << 31;
			}
			disabled[i] = 0;
		}

		self
	}

	fn borrowed_data(&self) -> impl Iterator<Item = &[U]> {
		vec![
			&self.fib_out,
			&self.fib_prev1,
			&self.fib_prev2,
			&self.c_out,
			&self.c_in,
			&self.disabled,
		]
		.into_iter()
		.map(|x| {
			let y: &[U] = x;
			y
		})
	}
}

#[derive(IterOracles)]
struct U32FibOracle {
	fib_out: OracleId,
	fib_prev1: OracleId,
	fib_prev2: OracleId,
	c_out: OracleId,
	c_in: OracleId,
	disabled: OracleId,

	batch_id: BatchId,
}

impl U32FibOracle {
	pub fn new<F: TowerField>(oracles: &mut MultilinearOracleSet<F>, n_vars: usize) -> Self {
		let batch_id = oracles.add_committed_batch(n_vars, BinaryField1b::TOWER_LEVEL);
		let [fib_out, c_out] = oracles.add_committed_multiple(batch_id);

		let fib_prev1 = oracles
			.add_named("fib_prev_1")
			.shifted(fib_out, 32, n_vars, ShiftVariant::LogicalLeft)
			.unwrap();
		let fib_prev2 = oracles
			.add_named("fib_prev_2")
			.shifted(fib_out, 64, n_vars, ShiftVariant::LogicalLeft)
			.unwrap();
		let c_in = oracles
			.add_named("cin")
			.shifted(c_out, 1, 5, ShiftVariant::LogicalLeft)
			.unwrap();
		let disabled = oracles
			.add_named("diabled")
			.transparent(StepDown::new(n_vars, 1).unwrap())
			.unwrap();

		Self {
			fib_out,
			fib_prev1,
			fib_prev2,
			c_out,
			c_in,
			disabled,

			batch_id,
		}
	}

	pub fn mixed_constraints<P: PackedField<Scalar: TowerField>>(&self) -> ConstraintSetBuilder<P> {
		let mut builder = ConstraintSetBuilder::new();

		builder.add_zerocheck(
			[
				self.fib_prev2,
				self.fib_prev1,
				self.c_in,
				self.fib_out,
				self.disabled,
			],
			composition_poly!([x, y, cin, z, disabled] = (1 - disabled) * (x + y + cin - z)),
		);
		builder.add_zerocheck(
			[self.fib_prev2, self.fib_prev1, self.c_in, self.c_out],
			composition_poly!([x, y, cin, cout] = (x + cin) * (y + cin) + cin - cout),
		);

		builder
	}
}

/// Joins U32FibOracle and U32FibTrace in a MultilinearExtensionIndex.
fn to_index<'a, U, F>(
	oracle: &'a U32FibOracle,
	trace: &'a U32FibTrace<U>,
) -> MultilinearExtensionIndex<'a, U, F>
where
	U: PackScalar<BinaryField1b> + PackScalar<F> + UnderlierType + Pod,
	F: TowerField + ExtensionField<BinaryField1b>,
{
	let oracle_id_and_data = oracle.iter_oracles().zip_eq(trace.borrowed_data());
	MultilinearExtensionIndex::new()
		.update_borrowed(oracle_id_and_data)
		.unwrap()
}

#[instrument(skip_all, level = "debug")]
#[allow(clippy::too_many_arguments)]
fn prove<U, F, FBase, DomainField, FEPCS, PCS, CH, Backend>(
	log_size: usize,
	oracles: &mut MultilinearOracleSet<F>,
	pcs: &PCS,
	oracle: &U32FibOracle,
	mut challenger: CH,
	mut witness: MultilinearExtensionIndex<U, F>,
	domain_factory: impl EvaluationDomainFactory<DomainField>,
	backend: Backend,
) -> Result<Proof<F, PCS::Commitment, PCS::Proof>>
where
	U: UnderlierType
		+ PackScalar<BinaryField1b>
		+ PackScalar<F>
		+ PackScalar<DomainField>
		+ PackScalar<FBase>,
	FEPCS: TowerField + From<F>,
	FBase: TowerField + ExtensionField<DomainField>,
	F: TowerField
		+ From<FEPCS>
		+ ExtensionField<FBase>
		+ ExtensionField<DomainField>
		+ PackedExtension<DomainField, Scalar = F>,
	PackedType<U, F>: PackedFieldIndexable<Scalar = F>,
	DomainField: TowerField,
	PCS: PolyCommitScheme<PackedType<U, BinaryField1b>, FEPCS, Error: Debug, Proof: 'static>,
	CH: CanObserve<FEPCS>
		+ CanObserve<PCS::Commitment>
		+ CanSample<FEPCS>
		+ CanSampleBits<usize>
		+ Clone,
	Backend: ComputationBackend,
{
	assert_eq!(pcs.n_vars(), log_size);

	let trace_commit_polys = oracles
		.committed_oracle_ids(oracle.batch_id)
		.map(|oracle_id| witness.get::<BinaryField1b>(oracle_id))
		.collect::<Result<Vec<_>, _>>()?;
	let (trace_comm, trace_committed) = pcs.commit(&trace_commit_polys)?;
	challenger.observe(trace_comm.clone());

	// Zerocheck
	let mut iso_challenger = IsomorphicChallenger::<_, _, F>::new(&mut challenger);

	let zerocheck_challenges = iso_challenger.sample_vec(log_size);

	let switchover_fn = standard_switchover_heuristic(-2);

	let constraint_set = oracle.mixed_constraints().build_one(oracles)?;
	let constraint_set_base = oracle.mixed_constraints().build_one(oracles)?;

	let (zerocheck_claim, meta) =
		sumcheck_v2::constraint_set_zerocheck_claim(constraint_set.clone())?;

	let prover = sumcheck_v2::prove::constraint_set_zerocheck_prover::<_, FBase, _, _, _>(
		constraint_set_base,
		constraint_set,
		&witness,
		domain_factory.clone(),
		switchover_fn,
		zerocheck_challenges.as_slice(),
		&backend,
	)?;

	let (sumcheck_output, zerocheck_proof) =
		sumcheck_v2::prove::batch_prove(vec![prover], &mut iso_challenger)?;

	let zerocheck_output = sumcheck_v2::zerocheck::verify_sumcheck_outputs(
		&[zerocheck_claim],
		&zerocheck_challenges,
		sumcheck_output,
	)?;

	let evalcheck_multilinear_claims =
		sumcheck_v2::make_eval_claims(oracles, [meta], zerocheck_output)?;

	// Prove evaluation claims
	let GreedyEvalcheckProveOutput {
		same_query_claims,
		proof: evalcheck_proof,
	} = greedy_evalcheck_v2::prove::<U, F, _, _, _>(
		oracles,
		&mut witness,
		evalcheck_multilinear_claims,
		switchover_fn,
		&mut iso_challenger,
		domain_factory,
		&backend,
	)?;

	assert_eq!(same_query_claims.len(), 1);
	let (batch_id, same_query_claim) = same_query_claims
		.into_iter()
		.next()
		.expect("length is asserted to be 1");
	assert_eq!(batch_id, oracle.batch_id);

	let trace_commit_polys = oracles
		.committed_oracle_ids(oracle.batch_id)
		.map(|oracle_id| witness.get::<BinaryField1b>(oracle_id))
		.collect::<Result<Vec<_>, _>>()?;

	// Prove commitment openings
	let eval_point: Vec<FEPCS> = same_query_claim
		.eval_point
		.into_iter()
		.map(|x| x.into())
		.collect();
	let trace_open_proof = pcs.prove_evaluation(
		&mut challenger,
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
fn verify<P, F, PCS, CH, Backend>(
	log_size: usize,
	oracles: &mut MultilinearOracleSet<F>,
	oracle: &U32FibOracle,
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

	let constraint_set = oracle.mixed_constraints::<F>().build_one(oracles)?;

	let (zerocheck_claim, meta) = sumcheck_v2::constraint_set_zerocheck_claim(constraint_set)?;
	let zerocheck_claims = [zerocheck_claim];

	let sumcheck_claims = sumcheck_v2::zerocheck::reduce_to_sumchecks(&zerocheck_claims)?;

	let sumcheck_output =
		sumcheck_v2::batch_verify(&sumcheck_claims, zerocheck_proof, &mut challenger)?;

	let zerocheck_output = sumcheck_v2::zerocheck::verify_sumcheck_outputs(
		&zerocheck_claims,
		&zerocheck_challenges,
		sumcheck_output,
	)?;

	let evalcheck_multilinear_claims =
		sumcheck_v2::make_eval_claims(oracles, [meta], zerocheck_output)?;

	// Evalcheck
	let same_query_claims = greedy_evalcheck_v2::verify(
		oracles,
		evalcheck_multilinear_claims,
		evalcheck_proof,
		&mut challenger,
	)?;

	assert_eq!(same_query_claims.len(), 1);
	let (batch_id, same_query_claim) = same_query_claims
		.into_iter()
		.next()
		.expect("length is asserted to be 1");
	assert_eq!(batch_id, oracle.batch_id);

	pcs.verify_evaluation(
		&mut challenger,
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
	let backend = make_portable_backend();

	// Note that values below 14 are rejected by `find_proof_size_optimal_pcs()`.
	let log_size = get_log_trace_size().unwrap_or(14);
	let log_inv_rate = 1;

	type U = <PackedBinaryField128x1b as WithUnderlier>::Underlier;
	type F = field_types::Field;

	debug!(num_bits = 1 << log_size, num_u32s = 1 << (log_size - 5), "U32 Fibonacci");

	let pcs = tensor_pcs::find_proof_size_optimal_pcs::<
		U,
		BinaryField1b,
		BinaryField16b,
		BinaryField16b,
		BinaryField128b,
	>(SECURITY_BITS, log_size, 2, log_inv_rate, false)
	.unwrap();

	let mut prover_oracles = MultilinearOracleSet::new();
	let prover_oracle = U32FibOracle::new::<field_types::Field>(&mut prover_oracles, log_size);

	let challenger = new_hasher_challenger::<_, GroestlHasher<_>>();
	let witness = U32FibTrace::<U>::new(log_size).fill_trace();
	let witness = to_index::<U, F>(&prover_oracle, &witness);

	let domain_factory =
		IsomorphicEvaluationDomainFactory::<field_types::DomainFieldWithStep>::default();

	info!("Proving");
	let proof = prove::<
		_,
		field_types::Field,
		field_types::Fbase,
		field_types::DomainField,
		BinaryField128b,
		_,
		_,
		_,
	>(
		log_size,
		&mut prover_oracles,
		&pcs,
		&prover_oracle,
		challenger.clone(),
		witness,
		domain_factory,
		&backend,
	)
	.unwrap();

	let mut verifier_oracles = MultilinearOracleSet::new();
	let verifier_oracle = U32FibOracle::new::<BinaryField128b>(&mut verifier_oracles, log_size);

	info!("Verifying");
	verify(
		log_size,
		&mut verifier_oracles,
		&verifier_oracle,
		&pcs,
		challenger.clone(),
		proof.isomorphic(),
		backend,
	)
	.unwrap();
}
