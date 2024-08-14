// Copyright 2024 Ulvetanna Inc.

#![feature(step_trait)]

/// An example computing the lowest 32 bits of consecutive fibonacci numbers.
/// The example performs the computation, generates a proof and verifies it.
use anyhow::Result;
use binius_core::{
	challenger::new_hasher_challenger,
	oracle::{BatchId, CompositePolyOracle, MultilinearOracleSet, OracleId, ShiftVariant},
	poly_commit::{tensor_pcs, PolyCommitScheme},
	polynomial::{
		composition::{empty_mix_composition, index_composition},
		transparent::step_down::StepDown,
		CompositionPoly, EvaluationDomainFactory, IsomorphicEvaluationDomainFactory,
		MultilinearComposite, MultilinearExtension,
	},
	protocols::{
		abstract_sumcheck::standard_switchover_heuristic,
		greedy_evalcheck::{self, GreedyEvalcheckProof, GreedyEvalcheckProveOutput},
		zerocheck::{self, ZerocheckBatchProof, ZerocheckBatchProveOutput, ZerocheckClaim},
	},
	witness::MultilinearWitnessIndex,
};
use binius_field::{
	underlier::WithUnderlier, BinaryField128b, BinaryField16b, BinaryField1b, ExtensionField,
	Field, PackedBinaryField128x1b, PackedBinaryField1x128b, PackedExtension, PackedField,
	TowerField,
};
use binius_hash::GroestlHasher;
use binius_macros::{composition_poly, IterOracles, IterPolys};
use binius_utils::{
	examples::get_log_trace_size, rayon::adjust_thread_pool, tracing::init_tracing,
};
use bytemuck::{must_cast_slice_mut, Pod};
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
}

#[cfg(not(feature = "aes-tower"))]
mod field_types {
	pub type Field = binius_field::BinaryField128bPolyval;
	pub type DomainField = binius_field::BinaryField128bPolyval;
	pub type DomainFieldWithStep = binius_field::BinaryField128b;
}

// The fields must be in exactly the same order as the fields in U32FibOracle.
#[derive(IterPolys)]
struct U32FibTrace<P: PackedField<Scalar = BinaryField1b>> {
	/// Fibonacci number computed in the current step.
	fib_out: Vec<P>,
	/// Fibonacci number computed one step ago.
	fib_prev1: Vec<P>,
	/// Fibonacci number computed two step ago.
	fib_prev2: Vec<P>,

	/// U32 carry addition information.
	c_out: Vec<P>,
	c_in: Vec<P>,

	/// Shifting pads with zeros, which will mess with constraints
	/// so we need a transparent StepDown to ignore the first row.
	disabled: Vec<P>,
}

impl<P: PackedField<Scalar = BinaryField1b> + Pod> U32FibTrace<P> {
	fn new(log_size: usize) -> Self {
		Self {
			fib_out: vec![P::default(); 1 << (log_size - P::LOG_WIDTH)],
			fib_prev1: vec![P::default(); 1 << (log_size - P::LOG_WIDTH)],
			fib_prev2: vec![P::default(); 1 << (log_size - P::LOG_WIDTH)],
			c_out: vec![P::default(); 1 << (log_size - P::LOG_WIDTH)],
			c_in: vec![P::default(); 1 << (log_size - P::LOG_WIDTH)],
			disabled: vec![P::default(); 1 << (log_size - P::LOG_WIDTH)],
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

	fn commit_polys(&self) -> impl Iterator<Item = MultilinearExtension<P, &[P]>> {
		[&self.fib_out, &self.c_out]
			.into_iter()
			.map(|values| MultilinearExtension::from_values_slice(values.as_slice()).unwrap())
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
			.add_shifted(fib_out, 32, n_vars, ShiftVariant::LogicalLeft)
			.unwrap();
		let fib_prev2 = oracles
			.add_shifted(fib_out, 64, n_vars, ShiftVariant::LogicalLeft)
			.unwrap();
		let c_in = oracles
			.add_shifted(c_out, 1, 5, ShiftVariant::LogicalLeft)
			.unwrap();
		let disabled = oracles
			.add_transparent(StepDown::new(n_vars, 1).unwrap())
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

	pub fn mixed_constraints<F: TowerField>(
		&self,
		challenge: F,
	) -> Result<impl CompositionPoly<F> + Clone> {
		let all_columns = &self.iter_oracles().collect::<Vec<_>>();
		let mix = empty_mix_composition(all_columns.len(), challenge);
		let mix = mix.include([index_composition(
			all_columns,
			[
				self.fib_prev2,
				self.fib_prev1,
				self.c_in,
				self.fib_out,
				self.disabled,
			],
			composition_poly!([x, y, cin, z, disabled] = (1 - disabled) * (x + y + cin - z)),
		)?])?;
		let mix = mix.include([index_composition(
			all_columns,
			[self.fib_prev2, self.fib_prev1, self.c_in, self.c_out],
			composition_poly!([x, y, cin, cout] = (x + cin) * (y + cin) + cin - cout),
		)?])?;
		Ok(mix)
	}
}

/// Joins U32FibOracle and U32FibTrace in a MultilinearWitnessIndex.
fn to_index<'a, P, PE>(
	oracle: &'a U32FibOracle,
	trace: &'a U32FibTrace<P>,
) -> MultilinearWitnessIndex<'a, PE>
where
	P: PackedField<Scalar = BinaryField1b> + Pod,
	PE: PackedField,
	PE::Scalar: ExtensionField<P::Scalar>,
{
	let mut index = MultilinearWitnessIndex::new();
	for (oracle, witness) in oracle.iter_oracles().zip(trace.iter_polys()) {
		index.set(oracle, witness.specialize_arc_dyn());
	}
	index
}

#[instrument(skip_all, level = "debug")]
fn prove<P, F, PW, DomainField, PCS, CH>(
	log_size: usize,
	oracles: &mut MultilinearOracleSet<F>,
	pcs: &PCS,
	oracle: &U32FibOracle,
	witness: &U32FibTrace<P>,
	mut challenger: CH,
	domain_factory: impl EvaluationDomainFactory<DomainField>,
) -> Result<Proof<F, PCS::Commitment, PCS::Proof>>
where
	P: PackedField<Scalar = BinaryField1b> + Pod,
	F: TowerField + From<PW>,
	PW: TowerField
		+ From<F>
		+ ExtensionField<DomainField>
		+ PackedExtension<DomainField, Scalar = PW>,
	DomainField: TowerField,
	PCS: PolyCommitScheme<P, F, Error: Debug, Proof: 'static>,
	CH: CanObserve<F> + CanObserve<PCS::Commitment> + CanSample<F> + CanSampleBits<usize> + Clone,
{
	assert_eq!(pcs.n_vars(), log_size);

	let mut witness_index = to_index::<P, PW>(oracle, witness);

	let trace_commit_polys = witness.commit_polys().collect::<Vec<_>>();
	let (trace_comm, trace_committed) = pcs.commit(&trace_commit_polys)?;
	challenger.observe(trace_comm.clone());

	// Zerocheck mixing
	let mixing_challenge = challenger.sample();

	let mix_composition_verifier = oracle.mixed_constraints(mixing_challenge)?;
	let mix_composition_prover = oracle.mixed_constraints(PW::from(mixing_challenge))?;

	let zerocheck_column_oracles = oracle.iter_oracles().map(|id| oracles.oracle(id)).collect();
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
		witness
			.iter_polys()
			.map(|mle| mle.specialize_arc_dyn::<PW>())
			.collect(),
	)?;

	// zerocheck::prove is instrumented
	let switchover_fn = standard_switchover_heuristic(-2);

	let ZerocheckBatchProveOutput {
		evalcheck_claims,
		proof: zerocheck_proof,
	} = zerocheck::batch_prove(
		[(zerocheck_claim, zerocheck_witness)],
		domain_factory.clone(),
		switchover_fn,
		&mut challenger,
	)?;

	// Prove evaluation claims
	let GreedyEvalcheckProveOutput {
		same_query_claims,
		proof: evalcheck_proof,
	} = greedy_evalcheck::prove::<_, _, DomainField, _>(
		oracles,
		&mut witness_index,
		evalcheck_claims,
		switchover_fn,
		&mut challenger,
		domain_factory,
	)?;

	assert_eq!(same_query_claims.len(), 1);
	let (batch_id, same_query_claim) = same_query_claims
		.into_iter()
		.next()
		.expect("length is asserted to be 1");
	assert_eq!(batch_id, oracle.batch_id);

	// Prove commitment openings
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

struct Proof<F: Field, PCSComm, PCSProof> {
	trace_comm: PCSComm,
	zerocheck_proof: ZerocheckBatchProof<F>,
	evalcheck_proof: GreedyEvalcheckProof<F>,
	trace_open_proof: PCSProof,
}

#[allow(clippy::too_many_arguments)]
#[instrument(skip_all, level = "debug")]
fn verify<P, F, PCS, CH>(
	log_size: usize,
	oracles: &mut MultilinearOracleSet<F>,
	oracle: &U32FibOracle,
	pcs: &PCS,
	mut challenger: CH,
	proof: Proof<F, PCS::Commitment, PCS::Proof>,
) -> Result<()>
where
	P: PackedField<Scalar = BinaryField1b> + Pod,
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
	let mix_composition = oracle.mixed_constraints(mixing_challenge)?;

	// Zerocheck
	let zerocheck_column_oracles = oracle.iter_oracles().map(|id| oracles.oracle(id)).collect();
	let zerocheck_claim = ZerocheckClaim {
		poly: CompositePolyOracle::new(log_size, zerocheck_column_oracles, mix_composition)?,
	};

	let evalcheck_claims =
		zerocheck::batch_verify([zerocheck_claim], zerocheck_proof, &mut challenger)?;

	// Evalcheck
	let same_query_claims =
		greedy_evalcheck::verify(oracles, evalcheck_claims, evalcheck_proof, &mut challenger)?;

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
	)?;

	Ok(())
}

fn main() {
	const SECURITY_BITS: usize = 100;

	adjust_thread_pool()
		.as_ref()
		.expect("failed to init thread pool");

	init_tracing().expect("failed to initialize tracing");

	// Note that values below 14 are rejected by `find_proof_size_optimal_pcs()`.
	let log_size = get_log_trace_size().unwrap_or(14);
	let log_inv_rate = 1;

	debug!(num_bits = 1 << log_size, num_u32s = 1 << (log_size - 5), "U32 Fibonacci");

	let pcs = tensor_pcs::find_proof_size_optimal_pcs::<
		<PackedBinaryField1x128b as WithUnderlier>::Underlier,
		BinaryField1b,
		BinaryField16b,
		BinaryField16b,
		BinaryField128b,
	>(SECURITY_BITS, log_size, 2, log_inv_rate, false)
	.unwrap();

	let mut oracles = MultilinearOracleSet::new();
	let oracle = U32FibOracle::new(&mut oracles, log_size);
	let challenger = new_hasher_challenger::<_, GroestlHasher<_>>();
	let witness = U32FibTrace::<PackedBinaryField128x1b>::new(log_size).fill_trace();
	let domain_factory =
		IsomorphicEvaluationDomainFactory::<field_types::DomainFieldWithStep>::default();

	info!("Proving");
	let proof = prove::<_, _, field_types::Field, field_types::DomainField, _, _>(
		log_size,
		&mut oracles,
		&pcs,
		&oracle,
		&witness,
		challenger.clone(),
		domain_factory,
	)
	.unwrap();

	info!("Verifying");
	verify(log_size, &mut oracles.clone(), &oracle, &pcs, challenger.clone(), proof).unwrap();
}
