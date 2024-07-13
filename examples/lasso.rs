// Copyright 2024 Ulvetanna Inc.

use anyhow::{anyhow, Result};
use binius_core::{
	challenger::{CanObserve, CanSample, CanSampleBits, HashChallenger},
	oracle::{BatchId, CommittedBatchSpec, CommittedId, MultilinearOracleSet, OracleId},
	poly_commit::{tensor_pcs, PolyCommitScheme},
	polynomial::{
		EvaluationDomainFactory, IsomorphicEvaluationDomainFactory, MultilinearExtensionBorrowed,
	},
	protocols::{
		abstract_sumcheck::standard_switchover_heuristic,
		greedy_evalcheck::{self, GreedyEvalcheckProof},
		lasso::{self, LassoBatch, LassoClaim, LassoWitness},
		msetcheck, prodcheck,
		zerocheck::{self, ZerocheckBatchProof},
	},
	witness::MultilinearExtensionIndex,
};
use binius_field::{
	as_packed_field::{PackScalar, PackedType},
	underlier::{UnderlierType, WithUnderlier},
	BinaryField, BinaryField128b, BinaryField16b, BinaryField1b, BinaryField32b, BinaryField8b,
	ExtensionField, Field, PackedBinaryField128x1b, PackedField, PackedFieldIndexable, TowerField,
};
use binius_hash::GroestlHasher;
use binius_utils::{
	examples::get_log_trace_size, rayon::adjust_thread_pool, tracing::init_tracing,
};
use itertools::izip;
use rand::thread_rng;
use std::{fmt::Debug, marker::PhantomData};
use tracing::instrument;

type B1 = BinaryField1b;
type B8 = BinaryField8b;
type B16 = BinaryField16b;
type B32 = BinaryField32b;
type B128 = BinaryField128b;

type Underlier = <PackedBinaryField128x1b as WithUnderlier>::Underlier;

struct TraceOracle {
	lasso_batch: LassoBatch,
	mults_batch: BatchId,
	product_batch: BatchId,
	lookup_t_batch: BatchId,
	grand_prod_batch: BatchId,
	mult_a: OracleId,
	mult_b: OracleId,
	product: OracleId,
	lookup_t: OracleId,
	lookup_u: OracleId,
}

impl TraceOracle {
	pub fn new<F: TowerField + From<u128>>(
		oracles: &mut MultilinearOracleSet<F>,
		log_size: usize,
	) -> Result<Self> {
		let mut mults_batch = oracles.build_committed_batch(log_size, 3);
		let [mult_a, mult_b] = mults_batch.add_multiple();
		let mults_batch = mults_batch.build();

		let mut product_batch = oracles.build_committed_batch(log_size, 4);
		let product = product_batch.add_one();
		let product_batch = product_batch.build();

		let mut lookup_t_batch = oracles.build_committed_batch(log_size, 5);
		let lookup_t = lookup_t_batch.add_one();
		let lookup_t_batch = lookup_t_batch.build();

		let lasso_batch = LassoBatch::new_in::<B32, _>(oracles, log_size);

		let grand_prod_batch = oracles.add_committed_batch(CommittedBatchSpec {
			n_vars: log_size + 2,
			n_polys: 1,
			tower_level: 7,
		});

		let lookup_u = oracles.add_linear_combination(
			log_size,
			[
				(mult_a, <F as TowerField>::basis(3, 3)?),
				(mult_b, <F as TowerField>::basis(3, 2)?),
				(product, <F as TowerField>::basis(3, 0)?),
			],
		)?;

		Ok(TraceOracle {
			lasso_batch,
			mults_batch,
			product_batch,
			lookup_t_batch,
			grand_prod_batch,
			mult_a,
			mult_b,
			product,
			lookup_t,
			lookup_u,
		})
	}
}

struct TraceWitness<U: UnderlierType + PackScalar<FW>, FW: BinaryField> {
	index: MultilinearExtensionIndex<'static, U, FW>,
	u_to_t_mapping: Vec<usize>,
}

fn underliers_unpack_scalars<U: UnderlierType + PackScalar<F>, F: Field>(underliers: &[U]) -> &[F]
where
	PackedType<U, F>: PackedFieldIndexable,
{
	PackedType::<U, F>::unpack_scalars(PackedType::<U, F>::from_underliers_ref(underliers))
}

fn underliers_unpack_scalars_mut<U: UnderlierType + PackScalar<F>, F: Field>(
	underliers: &mut [U],
) -> &mut [F]
where
	PackedType<U, F>: PackedFieldIndexable,
{
	PackedType::<U, F>::unpack_scalars_mut(PackedType::<U, F>::from_underliers_ref_mut(underliers))
}

#[instrument(skip_all)]
fn generate_trace<U, FW>(log_size: usize, trace_oracle: &TraceOracle) -> Result<TraceWitness<U, FW>>
where
	U: UnderlierType + PackScalar<B8> + PackScalar<B16> + PackScalar<B32> + PackScalar<FW>,
	PackedType<U, B8>: PackedFieldIndexable,
	PackedType<U, B16>: PackedFieldIndexable,
	PackedType<U, B32>: PackedFieldIndexable,
	FW: BinaryField + ExtensionField<B8> + ExtensionField<B16> + ExtensionField<B32>,
{
	let b8_packing_log_width = PackedType::<U, B8>::LOG_WIDTH;
	assert!(log_size >= b8_packing_log_width);

	let make_underliers =
		|log_size: usize| vec![U::default(); 1 << (log_size - b8_packing_log_width)];

	let mut mult_a = make_underliers(log_size);
	let mut mult_b = make_underliers(log_size);
	let mut product = make_underliers(log_size + 1);
	let mut lookup_u = make_underliers(log_size + 2);
	let mut lookup_t = make_underliers(log_size + 2);
	let mut u_to_t_mapping = vec![0; 1 << log_size];

	let mut rng = thread_rng();

	PackedType::<U, B8>::from_underliers_ref_mut(mult_a.as_mut_slice())
		.iter_mut()
		.for_each(|a| *a = PackedType::<U, B8>::random(&mut rng));
	PackedType::<U, B8>::from_underliers_ref_mut(mult_b.as_mut_slice())
		.iter_mut()
		.for_each(|b| *b = PackedType::<U, B8>::random(&mut rng));

	let mult_a_scalars = underliers_unpack_scalars::<_, B8>(mult_a.as_slice());
	let mult_b_scalars = underliers_unpack_scalars::<_, B8>(mult_b.as_slice());
	let product_scalars = underliers_unpack_scalars_mut::<_, B16>(product.as_mut_slice());
	let lookup_u_scalars = underliers_unpack_scalars_mut::<_, B32>(lookup_u.as_mut_slice());
	let lookup_t_scalars = underliers_unpack_scalars_mut::<_, B32>(lookup_t.as_mut_slice());

	for (a, b, lookup_u, product, u_to_t) in izip!(
		mult_a_scalars,
		mult_b_scalars,
		lookup_u_scalars.iter_mut(),
		product_scalars.iter_mut(),
		u_to_t_mapping.iter_mut()
	) {
		let a_int = u8::from(*a) as usize;
		let b_int = u8::from(*b) as usize;
		let ab_product = a_int * b_int;
		let lookup_index = a_int << 8 | b_int;
		*lookup_u = BinaryField32b::new((lookup_index << 16 | ab_product) as u32);
		*product = BinaryField16b::new(ab_product as u16);
		*u_to_t = lookup_index;
	}

	// the real table size is 2^16, but we repeat it up to log_size to meet Lasso requirements
	for (i, lookup_t) in lookup_t_scalars.iter_mut().enumerate() {
		let a_int = (i >> 8) & 0xff;
		let b_int = i & 0xff;
		let ab_product = a_int * b_int;
		let lookup_index = a_int << 8 | b_int;
		*lookup_t = BinaryField32b::new((lookup_index << 16 | ab_product) as u32);
	}

	let index = MultilinearExtensionIndex::new()
		.update_owned::<B8, _>([(trace_oracle.mult_a, mult_a), (trace_oracle.mult_b, mult_b)])?
		.update_owned::<B16, _>([(trace_oracle.product, product)])?
		.update_owned::<B32, _>([
			(trace_oracle.lookup_u, lookup_u),
			(trace_oracle.lookup_t, lookup_t),
		])?;

	Ok(TraceWitness {
		index,
		u_to_t_mapping,
	})
}

struct Proof<U, PCS1, PCS8, PCS16, PCS32, PCS128>
where
	U: UnderlierType
		+ PackScalar<B1>
		+ PackScalar<B8>
		+ PackScalar<B16>
		+ PackScalar<B32>
		+ PackScalar<B128>,
	PCS1: PolyCommitScheme<PackedType<U, B1>, B128, Error: Debug, Proof: 'static>,
	PCS8: PolyCommitScheme<PackedType<U, B8>, B128, Error: Debug, Proof: 'static>,
	PCS16: PolyCommitScheme<PackedType<U, B16>, B128, Error: Debug, Proof: 'static>,
	PCS32: PolyCommitScheme<PackedType<U, B32>, B128, Error: Debug, Proof: 'static>,
	PCS128: PolyCommitScheme<PackedType<U, B128>, B128, Error: Debug, Proof: 'static>,
{
	lasso_comm: PCS1::Commitment,
	lasso_proof: PCS1::Proof,
	mults_comm: PCS8::Commitment,
	mults_proof: PCS8::Proof,
	product_comm: PCS16::Commitment,
	product_proof: PCS16::Proof,
	lookup_t_comm: PCS32::Commitment,
	lookup_t_proof: PCS32::Proof,
	grand_prod_comm: PCS128::Commitment,
	grand_prod_proof: PCS128::Proof,
	zerocheck_proof: ZerocheckBatchProof<B128>,
	greedy_evalcheck_proof: GreedyEvalcheckProof<B128>,
	_u_marker: PhantomData<U>,
}

// witness column extractors

fn lasso_committed_polys<'a, U, F, FW>(
	oracles: &MultilinearOracleSet<F>,
	witness_index: &'a MultilinearExtensionIndex<'a, U, FW>,
	lasso_batch: &LassoBatch,
) -> Result<[MultilinearExtensionBorrowed<'a, PackedType<U, B1>>; 3]>
where
	F: TowerField,
	U: UnderlierType + PackScalar<FW> + PackScalar<B1>,
	FW: BinaryField + ExtensionField<B1>,
{
	let counts =
		witness_index.get::<B1>(oracles.committed_oracle_id(lasso_batch.counts_committed_id()))?;
	let carry_out = witness_index
		.get::<B1>(oracles.committed_oracle_id(lasso_batch.carry_out_committed_id()))?;
	let final_counts = witness_index
		.get::<B1>(oracles.committed_oracle_id(lasso_batch.final_counts_committed_id()))?;

	Ok([counts, carry_out, final_counts])
}

fn mults_committed_polys<'a, U, FW>(
	witness_index: &'a MultilinearExtensionIndex<'a, U, FW>,
	trace_oracle: &TraceOracle,
) -> Result<[MultilinearExtensionBorrowed<'a, PackedType<U, B8>>; 2]>
where
	U: UnderlierType + PackScalar<FW> + PackScalar<B8>,
	FW: BinaryField + ExtensionField<B8>,
{
	Ok([
		witness_index.get::<B8>(trace_oracle.mult_a)?,
		witness_index.get::<B8>(trace_oracle.mult_b)?,
	])
}

fn product_committed_polys<'a, U, FW>(
	witness_index: &'a MultilinearExtensionIndex<'a, U, FW>,
	trace_oracle: &TraceOracle,
) -> Result<[MultilinearExtensionBorrowed<'a, PackedType<U, B16>>; 1]>
where
	U: UnderlierType + PackScalar<FW> + PackScalar<B16>,
	FW: BinaryField + ExtensionField<B16>,
{
	Ok([witness_index.get::<B16>(trace_oracle.product)?])
}

fn lookup_t_committed_polys<'a, U, FW>(
	witness_index: &'a MultilinearExtensionIndex<'a, U, FW>,
	trace_oracle: &TraceOracle,
) -> Result<[MultilinearExtensionBorrowed<'a, PackedType<U, B32>>; 1]>
where
	U: UnderlierType + PackScalar<FW> + PackScalar<B32>,
	FW: BinaryField + ExtensionField<B32>,
{
	Ok([witness_index.get::<B32>(trace_oracle.lookup_t)?])
}

fn grand_prod_committed_polys<'a, U, F, FW>(
	oracles: &MultilinearOracleSet<F>,
	witness_index: &'a MultilinearExtensionIndex<'a, U, FW>,
	f_prime_committed_id: CommittedId,
) -> Result<[MultilinearExtensionBorrowed<'a, PackedType<U, B128>>; 1]>
where
	F: TowerField,
	U: UnderlierType + PackScalar<FW> + PackScalar<B128>,
	FW: BinaryField + ExtensionField<B128>,
{
	Ok([witness_index.get::<B128>(oracles.committed_oracle_id(f_prime_committed_id))?])
}

#[instrument(skip_all)]
#[allow(clippy::too_many_arguments)]
fn prove<U, PCS1, PCS8, PCS16, PCS32, PCS128, CH>(
	oracles: &mut MultilinearOracleSet<B128>,
	trace_oracle: &TraceOracle,
	witness: TraceWitness<U, B128>,
	pcs1: &PCS1,
	pcs8: &PCS8,
	pcs16: &PCS16,
	pcs32: &PCS32,
	pcs128: &PCS128,
	mut challenger: CH,
	domain_factory: impl EvaluationDomainFactory<B128>,
) -> Result<Proof<U, PCS1, PCS8, PCS16, PCS32, PCS128>>
where
	U: UnderlierType
		+ PackScalar<B1>
		+ PackScalar<B8>
		+ PackScalar<B16>
		+ PackScalar<B32>
		+ PackScalar<B128>,
	PCS1: PolyCommitScheme<PackedType<U, B1>, B128, Error: Debug, Proof: 'static>,
	PCS8: PolyCommitScheme<PackedType<U, B8>, B128, Error: Debug, Proof: 'static>,
	PCS16: PolyCommitScheme<PackedType<U, B16>, B128, Error: Debug, Proof: 'static>,
	PCS32: PolyCommitScheme<PackedType<U, B32>, B128, Error: Debug, Proof: 'static>,
	PCS128: PolyCommitScheme<PackedType<U, B128>, B128, Error: Debug, Proof: 'static>,
	CH: CanObserve<B128>
		+ CanSample<B128>
		+ CanSampleBits<usize>
		+ Clone
		+ CanObserve<PCS1::Commitment>
		+ CanObserve<PCS8::Commitment>
		+ CanObserve<PCS16::Commitment>
		+ CanObserve<PCS32::Commitment>
		+ CanObserve<PCS128::Commitment>,
	PackedType<U, B32>: PackedFieldIndexable,
	PackedType<U, B128>: PackedFieldIndexable,
{
	// Round 1 - trace commitments & Lasso deterministic reduction
	let (mults_comm, mults_committed) =
		pcs8.commit(&mults_committed_polys(&witness.index, trace_oracle)?)?;

	let (product_comm, product_committed) =
		pcs16.commit(&product_committed_polys(&witness.index, trace_oracle)?)?;

	let (lookup_t_comm, lookup_t_committed) =
		pcs32.commit(&lookup_t_committed_polys(&witness.index, trace_oracle)?)?;

	let lookup_t_oracle = oracles.oracle(trace_oracle.lookup_t);
	let lookup_u_oracle = oracles.oracle(trace_oracle.lookup_u);

	let lookup_t = witness.index.get_multilin_poly(trace_oracle.lookup_t)?;

	let lookup_u = witness.index.get_multilin_poly(trace_oracle.lookup_u)?;

	let lasso_claim = LassoClaim::new(lookup_t_oracle, lookup_u_oracle)?;

	let lasso_witness = LassoWitness::new(lookup_t, lookup_u, &witness.u_to_t_mapping)?;

	let lasso_batch = &trace_oracle.lasso_batch;

	let lasso_prove_output = lasso::prove::<B32, U, _, _, _>(
		oracles,
		witness.index,
		&lasso_claim,
		lasso_witness,
		lasso_batch,
	)?;

	let witness_index = lasso_prove_output.witness_index;

	let (lasso_comm, lasso_committed) =
		pcs1.commit(&lasso_committed_polys(oracles, &witness_index, lasso_batch)?)?;

	challenger.observe(lasso_comm.clone());
	challenger.observe(mults_comm.clone());
	challenger.observe(product_comm.clone());
	challenger.observe(lookup_t_comm.clone());

	// Round 2 - Msetcheck & Prodcheck
	let gamma = challenger.sample();
	let alpha = challenger.sample();

	let msetcheck_prove_output = msetcheck::prove(
		oracles,
		witness_index,
		&lasso_prove_output.reduced_lasso_claims.msetcheck_claim,
		lasso_prove_output.msetcheck_witness,
		gamma,
		Some(alpha),
	)?;

	let f_prime_committed_id = CommittedId {
		batch_id: trace_oracle.grand_prod_batch,
		index: 0,
	};

	let witness_index = msetcheck_prove_output.witness_index;
	let prodcheck_prove_output = prodcheck::prove(
		oracles,
		witness_index,
		&msetcheck_prove_output.prodcheck_claim,
		msetcheck_prove_output.prodcheck_witness,
		f_prime_committed_id,
	)?;

	let witness_index = prodcheck_prove_output.witness_index;
	let (grand_prod_comm, grand_prod_committed) = pcs128.commit(&grand_prod_committed_polys(
		oracles,
		&witness_index,
		f_prime_committed_id,
	)?)?;

	challenger.observe(grand_prod_comm.clone());

	// Prove reduced zerocheck originating from prodcheck

	let switchover_fn = standard_switchover_heuristic(-2);

	let lasso_zerocheck_claim = lasso_prove_output.reduced_lasso_claims.zerocheck_claim;
	let lasso_zerocheck_witness = lasso_prove_output
		.zerocheck_witness
		.to_arc_dyn_composition();

	let prodcheck_zerocheck_claim = prodcheck_prove_output
		.reduced_product_check_claims
		.t_prime_claim;
	let prodcheck_zerocheck_witness = prodcheck_prove_output
		.t_prime_witness
		.to_arc_dyn_composition();

	let zerocheck_prove_output = zerocheck::batch_prove(
		[
			(prodcheck_zerocheck_claim, prodcheck_zerocheck_witness),
			(lasso_zerocheck_claim, lasso_zerocheck_witness),
		],
		domain_factory.clone(),
		switchover_fn,
		&mut challenger,
	)?;

	// Greedy Evalcheck

	let mut legacy_witness_index = witness_index.witness_index();
	let greedy_evalcheck_prove_output = greedy_evalcheck::prove::<_, _, B128, _>(
		oracles,
		&mut legacy_witness_index,
		zerocheck_prove_output.evalcheck_claims,
		switchover_fn,
		&mut challenger,
		domain_factory,
	)?;

	// PCS opening proofs

	let batch_id_to_eval_point = |batch_id| {
		greedy_evalcheck_prove_output
			.same_query_claims
			.iter()
			.find(|(id, _)| *id == batch_id)
			.map(|(_, same_query_claim)| same_query_claim.eval_point.as_slice())
			.expect("present by greedy_evalcheck invariants")
	};

	let lasso_proof = pcs1.prove_evaluation(
		&mut challenger,
		&lasso_committed,
		&lasso_committed_polys(oracles, &witness_index, lasso_batch)?,
		batch_id_to_eval_point(trace_oracle.lasso_batch.batch_id()),
	)?;

	let mults_proof = pcs8.prove_evaluation(
		&mut challenger,
		&mults_committed,
		&mults_committed_polys(&witness_index, trace_oracle)?,
		batch_id_to_eval_point(trace_oracle.mults_batch),
	)?;

	let product_proof = pcs16.prove_evaluation(
		&mut challenger,
		&product_committed,
		&product_committed_polys(&witness_index, trace_oracle)?,
		batch_id_to_eval_point(trace_oracle.product_batch),
	)?;

	let lookup_t_proof = pcs32.prove_evaluation(
		&mut challenger,
		&lookup_t_committed,
		&lookup_t_committed_polys(&witness_index, trace_oracle)?,
		batch_id_to_eval_point(trace_oracle.lookup_t_batch),
	)?;

	let grand_prod_proof = pcs128.prove_evaluation(
		&mut challenger,
		&grand_prod_committed,
		&grand_prod_committed_polys(oracles, &witness_index, f_prime_committed_id)?,
		batch_id_to_eval_point(trace_oracle.grand_prod_batch),
	)?;

	Ok(Proof {
		lasso_comm,
		lasso_proof,
		mults_comm,
		mults_proof,
		product_comm,
		product_proof,
		lookup_t_comm,
		lookup_t_proof,
		grand_prod_comm,
		grand_prod_proof,
		zerocheck_proof: zerocheck_prove_output.proof,
		greedy_evalcheck_proof: greedy_evalcheck_prove_output.proof,
		_u_marker: PhantomData,
	})
}

#[instrument(skip_all)]
#[allow(clippy::too_many_arguments)]
fn verify<U, PCS1, PCS8, PCS16, PCS32, PCS128, CH>(
	oracles: &mut MultilinearOracleSet<B128>,
	trace_oracle: &TraceOracle,
	pcs1: &PCS1,
	pcs8: &PCS8,
	pcs16: &PCS16,
	pcs32: &PCS32,
	pcs128: &PCS128,
	mut challenger: CH,
	proof: Proof<U, PCS1, PCS8, PCS16, PCS32, PCS128>,
) -> Result<()>
where
	U: UnderlierType
		+ PackScalar<B1>
		+ PackScalar<B8>
		+ PackScalar<B16>
		+ PackScalar<B32>
		+ PackScalar<B128>,
	PCS1: PolyCommitScheme<PackedType<U, B1>, B128, Error: Debug, Proof: 'static>,
	PCS8: PolyCommitScheme<PackedType<U, B8>, B128, Error: Debug, Proof: 'static>,
	PCS16: PolyCommitScheme<PackedType<U, B16>, B128, Error: Debug, Proof: 'static>,
	PCS32: PolyCommitScheme<PackedType<U, B32>, B128, Error: Debug, Proof: 'static>,
	PCS128: PolyCommitScheme<PackedType<U, B128>, B128, Error: Debug, Proof: 'static>,
	CH: CanObserve<B128>
		+ CanSample<B128>
		+ CanSampleBits<usize>
		+ Clone
		+ CanObserve<PCS1::Commitment>
		+ CanObserve<PCS8::Commitment>
		+ CanObserve<PCS16::Commitment>
		+ CanObserve<PCS32::Commitment>
		+ CanObserve<PCS128::Commitment>,
{
	// Round 1 - Lasso deterministic reduction
	let lookup_t_oracle = oracles.oracle(trace_oracle.lookup_t);
	let lookup_u_oracle = oracles.oracle(trace_oracle.lookup_u);

	let lasso_claim = LassoClaim::new(lookup_t_oracle, lookup_u_oracle)?;

	let reduced_lasso_claims =
		lasso::verify::<B32, _>(oracles, &lasso_claim, &trace_oracle.lasso_batch)?;

	challenger.observe(proof.lasso_comm.clone());
	challenger.observe(proof.mults_comm.clone());
	challenger.observe(proof.product_comm.clone());
	challenger.observe(proof.lookup_t_comm.clone());

	// Round 2 - Msetcheck & Prodcheck
	let gamma = challenger.sample();
	let alpha = challenger.sample();

	let prodcheck_claim =
		msetcheck::verify(oracles, &reduced_lasso_claims.msetcheck_claim, gamma, Some(alpha))?;

	let f_prime_committed_id = CommittedId {
		batch_id: trace_oracle.grand_prod_batch,
		index: 0,
	};

	let grand_prod_oracle = oracles.committed_oracle(f_prime_committed_id);

	let reduced_prodcheck_claims = prodcheck::verify(oracles, &prodcheck_claim, grand_prod_oracle)?;

	challenger.observe(proof.grand_prod_comm.clone());

	let evalcheck_claims = zerocheck::batch_verify(
		[
			reduced_prodcheck_claims.t_prime_claim,
			reduced_lasso_claims.zerocheck_claim,
		],
		proof.zerocheck_proof,
		&mut challenger,
	)
	.unwrap();

	// Greedy evalcheck
	let same_query_pcs_claims = greedy_evalcheck::verify(
		oracles,
		evalcheck_claims,
		proof.greedy_evalcheck_proof,
		&mut challenger,
	)?;

	// PCS opening proofs
	let batch_id_to_eval_claim = |batch_id: BatchId| {
		same_query_pcs_claims
			.iter()
			.find(|(id, _)| *id == batch_id)
			.map(|(_, same_query_claim)| same_query_claim)
			.expect("present by greedy_evalcheck invariants")
	};

	let lasso_eval_claim = batch_id_to_eval_claim(trace_oracle.lasso_batch.batch_id());

	pcs1.verify_evaluation(
		&mut challenger,
		&proof.lasso_comm,
		&lasso_eval_claim.eval_point,
		proof.lasso_proof,
		&lasso_eval_claim.evals,
	)?;

	let mults_eval_claim = batch_id_to_eval_claim(trace_oracle.mults_batch);

	pcs8.verify_evaluation(
		&mut challenger,
		&proof.mults_comm,
		&mults_eval_claim.eval_point,
		proof.mults_proof,
		&mults_eval_claim.evals,
	)?;

	let product_eval_claim = batch_id_to_eval_claim(trace_oracle.product_batch);

	pcs16.verify_evaluation(
		&mut challenger,
		&proof.product_comm,
		&product_eval_claim.eval_point,
		proof.product_proof,
		&product_eval_claim.evals,
	)?;

	let lookup_t_eval_claim = batch_id_to_eval_claim(trace_oracle.lookup_t_batch);

	pcs32.verify_evaluation(
		&mut challenger,
		&proof.lookup_t_comm,
		&lookup_t_eval_claim.eval_point,
		proof.lookup_t_proof,
		&lookup_t_eval_claim.evals,
	)?;

	let grand_prod_eval_claim = batch_id_to_eval_claim(trace_oracle.grand_prod_batch);

	pcs128.verify_evaluation(
		&mut challenger,
		&proof.grand_prod_comm,
		&grand_prod_eval_claim.eval_point,
		proof.grand_prod_proof,
		&grand_prod_eval_claim.evals,
	)?;

	Ok(())
}

fn main() -> Result<()> {
	// mostly copied verbatim from the Keccak example initialization
	const SECURITY_BITS: usize = 100;

	adjust_thread_pool()
		.as_ref()
		.expect("failed to init thread pool");
	init_tracing().expect("failed to initialize tracing");

	let log_size = get_log_trace_size().unwrap_or(16);
	let log_inv_rate = 1;

	assert!(
		log_size >= 16,
		"log_size should be big enough to fit the 8b x 8b multiplication table"
	);

	// Set up the public parameters

	macro_rules! optimal_block_pcs {
		($pcs_var: ident := $n_polys:literal x [$f:ty > $fa:ty > $fi:ty > $fe:ty] ^ ($log_size:expr)) => {
			let $pcs_var = tensor_pcs::find_proof_size_optimal_pcs::<Underlier, $f, $fa, $fi, $fe>(
				SECURITY_BITS,
				$log_size,
				$n_polys,
				log_inv_rate,
				false,
			)
			.ok_or_else(|| anyhow!("Cannot determite optimal TensorPCS"))?;
		};
	}

	optimal_block_pcs!(pcs1 := 3 x [B1 > B16 > B16 > B128] ^ (log_size + B32::TOWER_LEVEL));
	optimal_block_pcs!(pcs8 := 2 x [B8 > B16 > B16 > B128] ^ (log_size));
	optimal_block_pcs!(pcs16 := 1 x [B16 > B16 > B16 > B128] ^ (log_size));
	optimal_block_pcs!(pcs32 := 1 x [B32 > B16 > B32 > B128] ^ (log_size));

	// relying on Field -> PackedField impl until prodcheck is aware of packed fields
	optimal_block_pcs!(pcs128 := 1 x [B128 > B16 > B128 > B128] ^ (log_size + 2));

	let mut oracles = MultilinearOracleSet::<B128>::new();
	let trace_oracle = TraceOracle::new(&mut oracles, log_size)?;
	let challenger = <HashChallenger<_, GroestlHasher<_>>>::new();
	let trace_witness = generate_trace::<Underlier, _>(log_size, &trace_oracle)?;
	let domain_factory = IsomorphicEvaluationDomainFactory::<B128>::default();

	let proof = prove(
		&mut oracles.clone(),
		&trace_oracle,
		trace_witness,
		&pcs1,
		&pcs8,
		&pcs16,
		&pcs32,
		&pcs128,
		challenger.clone(),
		domain_factory,
	)?;

	verify(
		&mut oracles.clone(),
		&trace_oracle,
		&pcs1,
		&pcs8,
		&pcs16,
		&pcs32,
		&pcs128,
		challenger.clone(),
		proof,
	)?;

	Ok(())
}
