// Copyright 2024 Ulvetanna Inc.

#![feature(step_trait)]

use anyhow::Result;
use binius_core::{
	challenger::HashChallenger,
	oracle::{BatchId, CompositePolyOracle, MultilinearOracleSet, OracleId, ShiftVariant},
	poly_commit::{tensor_pcs, PolyCommitScheme},
	polynomial::{
		composition::{empty_mix_composition, index_composition},
		CompositionPoly, EvaluationDomainFactory, IsomorphicEvaluationDomainFactory,
		MultilinearComposite, MultilinearExtension,
	},
	protocols::{
		greedy_evalcheck,
		greedy_evalcheck::{GreedyEvalcheckProof, GreedyEvalcheckProveOutput},
		zerocheck,
		zerocheck::{ZerocheckClaim, ZerocheckProof, ZerocheckProveOutput},
	},
	witness::MultilinearWitnessIndex,
};
use binius_field::{
	underlier::WithUnderlier, BinaryField128b, BinaryField16b, BinaryField1b, ExtensionField,
	Field, PackedBinaryField128x1b, PackedField, TowerField,
};
use binius_hash::GroestlHasher;
use binius_macros::{composition_poly, IterOracles, IterPolys};
use binius_utils::{
	examples::get_log_trace_size, rayon::adjust_thread_pool, tracing::init_tracing,
};
use bytemuck::{must_cast_slice_mut, Pod};
use p3_challenger::{CanObserve, CanSample, CanSampleBits};
use rand::{thread_rng, Rng};
use rayon::prelude::*;
use std::fmt::Debug;
use tracing::{debug, info, instrument};

// mod field_types is a selector of different sets of types which provide
// equivalent functionality but may differ significantly in performance.
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

#[derive(IterPolys)]
struct U32AddTrace<P: PackedField<Scalar = BinaryField1b>> {
	x_in: Vec<P>,
	y_in: Vec<P>,
	z_out: Vec<P>,
	c_out: Vec<P>,
	c_in: Vec<P>,
}

impl<P: PackedField<Scalar = BinaryField1b> + Pod> U32AddTrace<P> {
	fn new(log_size: usize) -> Self {
		Self {
			x_in: vec![P::default(); 1 << (log_size - P::LOG_WIDTH)],
			y_in: vec![P::default(); 1 << (log_size - P::LOG_WIDTH)],
			z_out: vec![P::default(); 1 << (log_size - P::LOG_WIDTH)],
			c_out: vec![P::default(); 1 << (log_size - P::LOG_WIDTH)],
			c_in: vec![P::default(); 1 << (log_size - P::LOG_WIDTH)],
		}
	}

	fn fill_trace(mut self) -> Self {
		(
			must_cast_slice_mut::<_, u32>(&mut self.x_in),
			must_cast_slice_mut::<_, u32>(&mut self.y_in),
			must_cast_slice_mut::<_, u32>(&mut self.z_out),
			must_cast_slice_mut::<_, u32>(&mut self.c_out),
			must_cast_slice_mut::<_, u32>(&mut self.c_in),
		)
			.into_par_iter()
			.for_each_init(thread_rng, |rng, (x, y, z, cout, cin)| {
				*x = rng.gen();
				*y = rng.gen();
				let carry;
				(*z, carry) = (*x).overflowing_add(*y);
				*cin = (*x) ^ (*y) ^ (*z);
				*cout = *cin >> 1;
				if carry {
					*cout |= 1 << 31;
				}
			});
		self
	}

	fn to_index<PE>(&self, oracle: &U32AddOracle) -> MultilinearWitnessIndex<PE>
	where
		PE: PackedField,
		PE::Scalar: ExtensionField<P::Scalar>,
	{
		let mut index = MultilinearWitnessIndex::new();

		for (oracle, witness) in oracle.iter_oracles().zip(self.iter_polys()) {
			index.set(oracle, witness.specialize_arc_dyn());
		}

		index
	}

	fn commit_polys(&self) -> impl Iterator<Item = MultilinearExtension<P>> {
		[&self.x_in, &self.y_in, &self.z_out, &self.c_out]
			.into_iter()
			.map(|values| MultilinearExtension::from_values_slice(values.as_slice()).unwrap())
	}
}

#[derive(IterOracles)]
struct U32AddOracle {
	x_in: OracleId,
	y_in: OracleId,
	z_out: OracleId,
	c_out: OracleId,
	c_in: OracleId,

	batch_id: BatchId,
}

impl U32AddOracle {
	pub fn new<F: TowerField>(oracles: &mut MultilinearOracleSet<F>, n_vars: usize) -> Self {
		let mut batch_scope = oracles.build_committed_batch(n_vars, BinaryField1b::TOWER_LEVEL);
		let x_in = batch_scope.add_one();
		let y_in = batch_scope.add_one();
		let z_out = batch_scope.add_one();
		let c_out = batch_scope.add_one();
		let batch_id = batch_scope.build();

		let c_in = oracles
			.add_shifted(c_out, 1, 5, ShiftVariant::LogicalLeft)
			.unwrap();
		Self {
			x_in,
			y_in,
			z_out,
			c_out,
			c_in,
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
			[self.x_in, self.y_in, self.c_in, self.z_out],
			composition_poly!([x, y, cin, z] = x + y + cin - z),
		)?])?;
		let mix = mix.include([index_composition(
			all_columns,
			[self.x_in, self.y_in, self.c_in, self.c_out],
			composition_poly!([x, y, cin, cout] = (x + cin) * (y + cin) + cin - cout),
		)?])?;
		Ok(mix)
	}
}

#[instrument(skip_all)]
fn prove<P, F, PW, DomainField, PCS, CH>(
	log_size: usize,
	oracles: &mut MultilinearOracleSet<F>,
	pcs: &PCS,
	oracle: &U32AddOracle,
	witness: &U32AddTrace<P>,
	mut challenger: CH,
	domain_factory: impl EvaluationDomainFactory<DomainField>,
) -> Result<Proof<F, PCS::Commitment, PCS::Proof>>
where
	P: PackedField<Scalar = BinaryField1b> + Pod,
	F: TowerField + From<PW>,
	PW: TowerField + From<F> + ExtensionField<DomainField>,
	DomainField: TowerField,
	PCS: PolyCommitScheme<P, F, Error: Debug, Proof: 'static>,
	CH: CanObserve<F> + CanObserve<PCS::Commitment> + CanSample<F> + CanSampleBits<usize> + Clone,
{
	assert_eq!(pcs.n_vars(), log_size);

	let mut witness_index = witness.to_index::<PW>(oracle);

	// Round 1
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
	let zerocheck_domain =
		domain_factory.create(zerocheck_claim.poly.max_individual_degree() + 1)?;
	let switchover_fn = |extension_degree| match extension_degree {
		128 => 5,
		_ => 1,
	};

	let ZerocheckProveOutput {
		evalcheck_claim,
		zerocheck_proof,
	} = zerocheck::prove::<F, PW, DomainField, _, _>(
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
	} = greedy_evalcheck::prove::<_, _, DomainField, _>(
		oracles,
		&mut witness_index,
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
	zerocheck_proof: ZerocheckProof<F>,
	evalcheck_proof: GreedyEvalcheckProof<F>,
	trace_open_proof: PCSProof,
}

#[allow(clippy::too_many_arguments)]
#[instrument(skip_all)]
fn verify<P, F, PCS, CH>(
	log_size: usize,
	oracles: &mut MultilinearOracleSet<F>,
	oracle: &U32AddOracle,
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

	let evalcheck_claim = zerocheck::verify(&zerocheck_claim, zerocheck_proof, &mut challenger)?;

	// Evalcheck
	let same_query_claims =
		greedy_evalcheck::verify(oracles, [evalcheck_claim], evalcheck_proof, &mut challenger)?;

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

	init_tracing();

	// Values below 14 are rejected by `find_proof_size_optimal_pcs()`.
	let log_size = get_log_trace_size().unwrap_or(14);
	let log_inv_rate = 1;

	debug!(num_bits = 1 << log_size, num_u32s = 1 << (log_size - 5), "U32 Addition");

	let pcs = tensor_pcs::find_proof_size_optimal_pcs::<
		<PackedBinaryField128x1b as WithUnderlier>::Underlier,
		BinaryField1b,
		BinaryField16b,
		BinaryField16b,
		BinaryField128b,
	>(SECURITY_BITS, log_size, 4, log_inv_rate, false)
	.unwrap();

	let mut oracles = MultilinearOracleSet::new();
	let oracle = U32AddOracle::new(&mut oracles, log_size);
	let challenger = <HashChallenger<_, GroestlHasher<_>>>::new();
	let witness = U32AddTrace::<PackedBinaryField128x1b>::new(log_size).fill_trace();
	let domain_factory =
		IsomorphicEvaluationDomainFactory::<field_types::DomainFieldWithStep>::default();

	info!("Proving");
	let proof = prove::<_, BinaryField128b, field_types::Field, field_types::DomainField, _, _>(
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
