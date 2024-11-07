// Copyright 2024 Irreducible Inc.

use anyhow::{anyhow, Result};
use binius_core::{
	challenger::{new_hasher_challenger, CanObserve, CanSample, CanSampleBits},
	oracle::{BatchId, MultilinearOracleSet, OracleId},
	poly_commit::{tensor_pcs, PolyCommitScheme},
	protocols::{
		gkr_gpa::{self, GrandProductBatchProof, GrandProductBatchProveOutput},
		greedy_evalcheck::{self, GreedyEvalcheckProof},
		lasso::{self, LassoBatches, LassoClaim, LassoProof, LassoProveOutput, LassoWitness},
		sumcheck::standard_switchover_heuristic,
	},
	witness::MultilinearExtensionIndex,
};
use binius_field::{
	as_packed_field::{PackScalar, PackedType},
	underlier::{UnderlierType, WithUnderlier},
	BinaryField, BinaryField128b, BinaryField16b, BinaryField32b, BinaryField8b, ExtensionField,
	Field, PackedBinaryField128x1b, PackedField, PackedFieldIndexable, TowerField,
};
use binius_hal::{make_portable_backend, ComputationBackend};
use binius_hash::GroestlHasher;
use binius_math::{
	EvaluationDomainFactory, IsomorphicEvaluationDomainFactory, MultilinearExtensionBorrowed,
};
use binius_utils::{
	examples::get_log_trace_size, rayon::adjust_thread_pool, tracing::init_tracing,
};
use itertools::izip;
use rand::thread_rng;
use std::{fmt::Debug, marker::PhantomData};
use tracing::instrument;

type B8 = BinaryField8b;
type B16 = BinaryField16b;
type B32 = BinaryField32b;
type B128 = BinaryField128b;

type Underlier = <PackedBinaryField128x1b as WithUnderlier>::Underlier;

const T_LOG_SIZE: usize = 16;

struct TraceOracle {
	lasso_batches: LassoBatches,
	mults_batch: BatchId,
	product_batch: BatchId,
	lookup_t_batch: BatchId,
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
		let mults_batch = oracles.add_committed_batch(log_size, 3);
		let [mult_a, mult_b] = oracles.add_named("mults").committed_multiple(mults_batch);

		let product_batch = oracles.add_committed_batch(log_size, 4);
		let product = oracles.add_named("product").committed(product_batch);

		let lookup_t_batch = oracles.add_committed_batch(T_LOG_SIZE, 5);
		let lookup_t = oracles.add_named("lookup_t").committed(lookup_t_batch);

		let lasso_batches = LassoBatches::new_in::<B32, _>(oracles, &[log_size], T_LOG_SIZE);

		let lookup_u = oracles.add_named("lookup_u").linear_combination(
			log_size,
			[
				(mult_a, <F as TowerField>::basis(3, 3)?),
				(mult_b, <F as TowerField>::basis(3, 2)?),
				(product, <F as TowerField>::basis(3, 0)?),
			],
		)?;

		Ok(TraceOracle {
			lasso_batches,
			mults_batch,
			product_batch,
			lookup_t_batch,
			mult_a,
			mult_b,
			product,
			lookup_t,
			lookup_u,
		})
	}
}

struct TraceWitness<U: UnderlierType + PackScalar<F>, F: BinaryField> {
	index: MultilinearExtensionIndex<'static, U, F>,
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

fn make_underliers<U: UnderlierType + PackScalar<FS>, FS: Field>(log_size: usize) -> Vec<U> {
	let packing_log_width = PackedType::<U, FS>::LOG_WIDTH;
	vec![U::default(); 1 << (log_size - packing_log_width)]
}
#[instrument(skip_all, level = "debug")]
fn generate_trace<U, F>(log_size: usize, trace_oracle: &TraceOracle) -> Result<TraceWitness<U, F>>
where
	U: UnderlierType + PackScalar<B8> + PackScalar<B16> + PackScalar<B32> + PackScalar<F>,
	PackedType<U, B8>: PackedFieldIndexable,
	PackedType<U, B16>: PackedFieldIndexable,
	PackedType<U, B32>: PackedFieldIndexable,
	F: BinaryField + ExtensionField<B8> + ExtensionField<B16> + ExtensionField<B32>,
{
	let mut mult_a = make_underliers::<_, B8>(log_size);
	let mut mult_b = make_underliers::<_, B8>(log_size);
	let mut product = make_underliers::<_, B16>(log_size);
	let mut lookup_u = make_underliers::<_, B32>(log_size);
	let mut lookup_t = make_underliers::<_, B32>(T_LOG_SIZE);
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

	for (i, lookup_t) in lookup_t_scalars.iter_mut().enumerate() {
		let a_int = (i >> 8) & 0xff;
		let b_int = i & 0xff;
		let ab_product = a_int * b_int;
		let lookup_index = a_int << 8 | b_int;
		*lookup_t = BinaryField32b::new((lookup_index << 16 | ab_product) as u32);
	}

	let mut index = MultilinearExtensionIndex::new();
	index.set_owned::<B8, _>([(trace_oracle.mult_a, mult_a), (trace_oracle.mult_b, mult_b)])?;
	index.set_owned::<B16, _>([(trace_oracle.product, product)])?;
	index.set_owned::<B32, _>([
		(trace_oracle.lookup_u, lookup_u),
		(trace_oracle.lookup_t, lookup_t),
	])?;

	Ok(TraceWitness {
		index,
		u_to_t_mapping,
	})
}

struct Proof<U, PCS8, PCS16, PCS32>
where
	U: UnderlierType + PackScalar<B8> + PackScalar<B16> + PackScalar<B32> + PackScalar<B128>,
	PCS8: PolyCommitScheme<PackedType<U, B8>, B128, Error: Debug, Proof: 'static>,
	PCS16: PolyCommitScheme<PackedType<U, B16>, B128, Error: Debug, Proof: 'static>,
	PCS32: PolyCommitScheme<PackedType<U, B32>, B128, Error: Debug, Proof: 'static>,
{
	lasso_counts_comm: PCS32::Commitment,
	lasso_counts_proof: PCS32::Proof,
	lasso_final_counts_comm: PCS32::Commitment,
	lasso_final_counts_proof: PCS32::Proof,
	lasso_proof: LassoProof<B128>,
	mults_comm: PCS8::Commitment,
	mults_proof: PCS8::Proof,
	product_comm: PCS16::Commitment,
	product_proof: PCS16::Proof,
	lookup_t_comm: PCS32::Commitment,
	lookup_t_proof: PCS32::Proof,
	gpa_proof: GrandProductBatchProof<B128>,
	greedy_evalcheck_proof: GreedyEvalcheckProof<B128>,
	_u_marker: PhantomData<U>,
}

// witness column extractors

fn extract_batch_id_polys<'a, U, F, FS>(
	batch_id: BatchId,
	witness_index: &'a MultilinearExtensionIndex<'a, U, F>,
	oracles: &MultilinearOracleSet<B128>,
) -> Result<Vec<MultilinearExtensionBorrowed<'a, PackedType<U, FS>>>>
where
	U: UnderlierType + PackScalar<F> + PackScalar<FS>,
	FS: TowerField,
	F: TowerField + ExtensionField<FS>,
{
	let trace_commit_polys = oracles
		.committed_oracle_ids(batch_id)
		.map(|oracle_id| witness_index.get::<FS>(oracle_id))
		.collect::<Result<Vec<_>, _>>()?;
	Ok(trace_commit_polys)
}

#[instrument(skip_all, level = "debug")]
#[allow(clippy::too_many_arguments)]
fn prove<U, PCS8, PCS16, PCS32, CH, Backend>(
	oracles: &mut MultilinearOracleSet<B128>,
	trace_oracle: &TraceOracle,
	witness: TraceWitness<U, B128>,
	lasso_counts_pcs: &PCS32,
	pcs8: &PCS8,
	pcs16: &PCS16,
	pcs32: &PCS32,
	mut challenger: CH,
	domain_factory: impl EvaluationDomainFactory<B128>,
	backend: &Backend,
) -> Result<Proof<U, PCS8, PCS16, PCS32>>
where
	U: UnderlierType + PackScalar<B8> + PackScalar<B16> + PackScalar<B32> + PackScalar<B128>,
	PCS8: PolyCommitScheme<PackedType<U, B8>, B128, Error: Debug, Proof: 'static>,
	PCS16: PolyCommitScheme<PackedType<U, B16>, B128, Error: Debug, Proof: 'static>,
	PCS32: PolyCommitScheme<PackedType<U, B32>, B128, Error: Debug, Proof: 'static>,
	CH: CanObserve<B128>
		+ CanSample<B128>
		+ CanSampleBits<usize>
		+ Clone
		+ CanObserve<PCS8::Commitment>
		+ CanObserve<PCS16::Commitment>
		+ CanObserve<PCS32::Commitment>,
	PackedType<U, B32>: PackedFieldIndexable,
	PackedType<U, B128>: PackedFieldIndexable,
	Backend: ComputationBackend + 'static,
{
	// Round 1 - trace commitments & Lasso deterministic reduction
	let (mults_comm, mults_committed) = pcs8.commit(&extract_batch_id_polys::<_, _, B8>(
		trace_oracle.mults_batch,
		&witness.index,
		oracles,
	)?)?;

	let (product_comm, product_committed) = pcs16.commit(&extract_batch_id_polys::<_, _, B16>(
		trace_oracle.product_batch,
		&witness.index,
		oracles,
	)?)?;

	let (lookup_t_comm, lookup_t_committed) = pcs32.commit(
		&extract_batch_id_polys::<_, _, B32>(trace_oracle.lookup_t_batch, &witness.index, oracles)?,
	)?;

	// Round 2 - Lasso prove

	let lookup_t_oracle = oracles.oracle(trace_oracle.lookup_t);
	let lookup_u_oracle = oracles.oracle(trace_oracle.lookup_u);

	let lookup_t = witness.index.get_multilin_poly(trace_oracle.lookup_t)?;

	let lookup_u = witness.index.get_multilin_poly(trace_oracle.lookup_u)?;

	let lasso_claim = LassoClaim::new(lookup_t_oracle, [lookup_u_oracle].to_vec())?;

	let lasso_witness =
		LassoWitness::new(lookup_t, [lookup_u].to_vec(), [&witness.u_to_t_mapping].to_vec())?;

	let lasso_batches = &trace_oracle.lasso_batches;

	challenger.observe(mults_comm.clone());
	challenger.observe(product_comm.clone());
	challenger.observe(lookup_t_comm.clone());

	let gamma = challenger.sample();
	let alpha = challenger.sample();

	let lasso_prove_output = lasso::prove::<B32, U, _, _>(
		oracles,
		witness.index,
		&lasso_claim,
		lasso_witness,
		lasso_batches,
		gamma,
		alpha,
	)?;

	let LassoProveOutput {
		reduced_gpa_claims,
		reduced_gpa_witnesses,
		gpa_metas,
		witness_index,
		lasso_proof,
	} = lasso_prove_output;

	let mut witness_index = witness_index;

	let (lasso_counts_comm, lasso_counts_committed) =
		lasso_counts_pcs.commit(&extract_batch_id_polys::<_, _, B32>(
			trace_oracle.lasso_batches.counts_batch_ids[0],
			&witness_index,
			oracles,
		)?)?;

	let (lasso_final_counts_comm, lasso_final_counts_committed) =
		pcs32.commit(&extract_batch_id_polys::<_, _, B32>(
			trace_oracle.lasso_batches.final_counts_batch_id,
			&witness_index,
			oracles,
		)?)?;

	challenger.observe(lasso_counts_comm.clone());
	challenger.observe(lasso_final_counts_comm.clone());

	let GrandProductBatchProveOutput {
		final_layer_claims,
		proof: gpa_proof,
	} = gkr_gpa::batch_prove(
		reduced_gpa_witnesses,
		&reduced_gpa_claims,
		domain_factory.clone(),
		&mut challenger,
		&backend,
	)?;

	let evalcheck_multilinear_claims =
		gkr_gpa::make_eval_claims(oracles, gpa_metas, final_layer_claims)?;

	// Greedy Evalcheck
	let switchover_fn = standard_switchover_heuristic(-2);
	let greedy_evalcheck_prove_output = greedy_evalcheck::prove::<U, B128, B128, _, _>(
		oracles,
		&mut witness_index,
		evalcheck_multilinear_claims,
		switchover_fn,
		&mut challenger,
		domain_factory,
		&backend,
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

	let lasso_counts_proof = lasso_counts_pcs.prove_evaluation(
		&mut challenger,
		&lasso_counts_committed,
		&extract_batch_id_polys::<_, _, B32>(
			trace_oracle.lasso_batches.counts_batch_ids[0],
			&witness_index,
			oracles,
		)?,
		batch_id_to_eval_point(trace_oracle.lasso_batches.counts_batch_ids[0]),
		&backend,
	)?;

	let lasso_final_counts_proof = pcs32.prove_evaluation(
		&mut challenger,
		&lasso_final_counts_committed,
		&extract_batch_id_polys::<_, _, B32>(
			trace_oracle.lasso_batches.final_counts_batch_id,
			&witness_index,
			oracles,
		)?,
		batch_id_to_eval_point(trace_oracle.lasso_batches.final_counts_batch_id),
		&backend,
	)?;

	let mults_proof = pcs8.prove_evaluation(
		&mut challenger,
		&mults_committed,
		&extract_batch_id_polys::<_, _, B8>(trace_oracle.mults_batch, &witness_index, oracles)?,
		batch_id_to_eval_point(trace_oracle.mults_batch),
		&backend,
	)?;

	let product_proof = pcs16.prove_evaluation(
		&mut challenger,
		&product_committed,
		&extract_batch_id_polys::<_, _, B16>(trace_oracle.product_batch, &witness_index, oracles)?,
		batch_id_to_eval_point(trace_oracle.product_batch),
		&backend,
	)?;

	let lookup_t_proof = pcs32.prove_evaluation(
		&mut challenger,
		&lookup_t_committed,
		&extract_batch_id_polys::<_, _, B32>(trace_oracle.lookup_t_batch, &witness_index, oracles)?,
		batch_id_to_eval_point(trace_oracle.lookup_t_batch),
		&backend,
	)?;

	Ok(Proof {
		lasso_counts_comm,
		lasso_counts_proof,
		lasso_final_counts_comm,
		lasso_final_counts_proof,
		mults_comm,
		mults_proof,
		product_comm,
		product_proof,
		lookup_t_comm,
		lookup_t_proof,
		lasso_proof,
		gpa_proof,
		greedy_evalcheck_proof: greedy_evalcheck_prove_output.proof,
		_u_marker: PhantomData,
	})
}

#[instrument(skip_all, level = "debug")]
#[allow(clippy::too_many_arguments)]
fn verify<U, PCS8, PCS16, PCS32, CH, Backend>(
	oracles: &mut MultilinearOracleSet<B128>,
	trace_oracle: &TraceOracle,
	lasso_counts_pcs: &PCS32,
	pcs8: &PCS8,
	pcs16: &PCS16,
	pcs32: &PCS32,
	mut challenger: CH,
	proof: Proof<U, PCS8, PCS16, PCS32>,
	backend: &Backend,
) -> Result<()>
where
	U: UnderlierType + PackScalar<B8> + PackScalar<B16> + PackScalar<B32> + PackScalar<B128>,
	PCS8: PolyCommitScheme<PackedType<U, B8>, B128, Error: Debug, Proof: 'static>,
	PCS16: PolyCommitScheme<PackedType<U, B16>, B128, Error: Debug, Proof: 'static>,
	PCS32: PolyCommitScheme<PackedType<U, B32>, B128, Error: Debug, Proof: 'static>,
	CH: CanObserve<B128>
		+ CanSample<B128>
		+ CanSampleBits<usize>
		+ Clone
		+ CanObserve<PCS8::Commitment>
		+ CanObserve<PCS16::Commitment>
		+ CanObserve<PCS32::Commitment>,
	Backend: ComputationBackend + 'static,
{
	// Unpack the proof
	let Proof {
		lasso_counts_comm,
		lasso_counts_proof,
		lasso_final_counts_comm,
		lasso_final_counts_proof,
		mults_comm,
		mults_proof,
		product_comm,
		product_proof,
		lookup_t_comm,
		lookup_t_proof,
		lasso_proof,
		gpa_proof,
		greedy_evalcheck_proof,
		..
	} = proof;

	// Round 1 - Lasso verify
	let lookup_t_oracle = oracles.oracle(trace_oracle.lookup_t);
	let lookup_u_oracle = oracles.oracle(trace_oracle.lookup_u);

	let lasso_claim = LassoClaim::new(lookup_t_oracle, [lookup_u_oracle].to_vec())?;

	challenger.observe(mults_comm.clone());
	challenger.observe(product_comm.clone());
	challenger.observe(lookup_t_comm.clone());

	let gamma = challenger.sample();
	let alpha = challenger.sample();

	let (reduced_gpa_claims, gpa_metas) = lasso::verify::<B32, _>(
		oracles,
		&lasso_claim,
		&trace_oracle.lasso_batches,
		gamma,
		alpha,
		lasso_proof,
	)?;

	challenger.observe(lasso_counts_comm.clone());
	challenger.observe(lasso_final_counts_comm.clone());

	let final_layer_claims = gkr_gpa::batch_verify(reduced_gpa_claims, gpa_proof, &mut challenger)?;

	let evalcheck_multilinear_claims =
		gkr_gpa::make_eval_claims(oracles, gpa_metas, final_layer_claims)?;

	// Greedy evalcheck
	let same_query_pcs_claims = greedy_evalcheck::verify(
		oracles,
		evalcheck_multilinear_claims,
		greedy_evalcheck_proof,
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

	let lasso_counts_eval_claim =
		batch_id_to_eval_claim(trace_oracle.lasso_batches.counts_batch_ids[0]);

	lasso_counts_pcs.verify_evaluation(
		&mut challenger,
		&lasso_counts_comm,
		&lasso_counts_eval_claim.eval_point,
		lasso_counts_proof,
		&lasso_counts_eval_claim.evals,
		&backend,
	)?;

	let lasso_final_counts_eval_claim =
		batch_id_to_eval_claim(trace_oracle.lasso_batches.final_counts_batch_id);

	pcs32.verify_evaluation(
		&mut challenger,
		&lasso_final_counts_comm,
		&lasso_final_counts_eval_claim.eval_point,
		lasso_final_counts_proof,
		&lasso_final_counts_eval_claim.evals,
		&backend,
	)?;

	let mults_eval_claim = batch_id_to_eval_claim(trace_oracle.mults_batch);

	pcs8.verify_evaluation(
		&mut challenger,
		&mults_comm,
		&mults_eval_claim.eval_point,
		mults_proof,
		&mults_eval_claim.evals,
		&backend,
	)?;

	let product_eval_claim = batch_id_to_eval_claim(trace_oracle.product_batch);

	pcs16.verify_evaluation(
		&mut challenger,
		&product_comm,
		&product_eval_claim.eval_point,
		product_proof,
		&product_eval_claim.evals,
		&backend,
	)?;

	let lookup_t_eval_claim = batch_id_to_eval_claim(trace_oracle.lookup_t_batch);

	pcs32.verify_evaluation(
		&mut challenger,
		&lookup_t_comm,
		&lookup_t_eval_claim.eval_point,
		lookup_t_proof,
		&lookup_t_eval_claim.evals,
		&backend,
	)?;

	Ok(())
}

fn main() -> Result<()> {
	// mostly copied verbatim from the Keccak example initialization
	const SECURITY_BITS: usize = 100;

	adjust_thread_pool()
		.as_ref()
		.expect("failed to init thread pool");
	let _guard = init_tracing().expect("failed to initialize tracing");

	let log_size = get_log_trace_size().unwrap_or(20);
	let log_inv_rate = 1;

	let backend = make_portable_backend();

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

	optimal_block_pcs!(lasso_counts_pcs := 2 x [B32 > B16 > B32 > B128] ^ (log_size));
	optimal_block_pcs!(pcs8 := 2 x [B8 > B16 > B16 > B128] ^ (log_size));
	optimal_block_pcs!(pcs16 := 1 x [B16 > B16 > B16 > B128] ^ (log_size));
	optimal_block_pcs!(pcs32 := 1 x [B32 > B16 > B32 > B128] ^ (T_LOG_SIZE));

	let mut oracles = MultilinearOracleSet::<B128>::new();
	let trace_oracle = TraceOracle::new(&mut oracles, log_size)?;
	let challenger = new_hasher_challenger::<_, GroestlHasher<_>>();
	let trace_witness = generate_trace::<Underlier, _>(log_size, &trace_oracle)?;
	let domain_factory = IsomorphicEvaluationDomainFactory::<B128>::default();

	let proof = prove(
		&mut oracles.clone(),
		&trace_oracle,
		trace_witness,
		&lasso_counts_pcs,
		&pcs8,
		&pcs16,
		&pcs32,
		challenger.clone(),
		domain_factory,
		&backend,
	)?;

	verify(
		&mut oracles.clone(),
		&trace_oracle,
		&lasso_counts_pcs,
		&pcs8,
		&pcs16,
		&pcs32,
		challenger.clone(),
		proof,
		&backend,
	)?;

	Ok(())
}
