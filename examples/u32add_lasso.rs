// Copyright 2024 Ulvetanna Inc.

use anyhow::{anyhow, Result};
use binius_core::{
	challenger::{new_hasher_challenger, CanObserve, CanSample, CanSampleBits},
	oracle::{BatchId, MultilinearOracleSet, OracleId, ShiftVariant},
	poly_commit::{tensor_pcs, PolyCommitScheme},
	protocols::{
		gkr_gpa::{self, GrandProductBatchProof, GrandProductBatchProveOutput},
		greedy_evalcheck_v2::{self, GreedyEvalcheckProof},
		lasso::{self, LassoBatches, LassoClaim, LassoProof, LassoProveOutput, LassoWitness},
		sumcheck_v2::standard_switchover_heuristic,
	},
	witness::MultilinearExtensionIndex,
};
use binius_field::{
	as_packed_field::{PackScalar, PackedType},
	packed::set_packed_slice,
	underlier::{UnderlierType, WithUnderlier, U1},
	BinaryField, BinaryField128b, BinaryField16b, BinaryField1b, BinaryField32b, BinaryField8b,
	ExtensionField, Field, PackedBinaryField128x1b, PackedField, PackedFieldIndexable, TowerField,
};
use binius_hal::{make_portable_backend, ComputationBackend, MultilinearExtensionBorrowed};
use binius_hash::GroestlHasher;
use binius_math::{EvaluationDomainFactory, IsomorphicEvaluationDomainFactory};
use binius_utils::{
	examples::get_log_trace_size, rayon::adjust_thread_pool, tracing::init_tracing,
};
use itertools::izip;
use rand::thread_rng;
use std::{fmt::Debug, marker::PhantomData};
use tracing::{debug, instrument};

type B1 = BinaryField1b;
type B8 = BinaryField8b;
type B16 = BinaryField16b;
type B32 = BinaryField32b;
type B128 = BinaryField128b;

type Underlier = <PackedBinaryField128x1b as WithUnderlier>::Underlier;

const ADD_T_LOG_SIZE: usize = 17;

struct TraceOracle {
	lasso_batch: LassoBatches,
	b8_batch: BatchId,
	b1_batch: BatchId,
	lookup_t_batch: BatchId,
	a: OracleId,
	b: OracleId,
	cin: OracleId,
	cout: OracleId,
	sum: OracleId,
	lookup_t: OracleId,
	lookup_u: OracleId,
}

impl TraceOracle {
	pub fn new<F: TowerField + From<u128>>(
		oracles: &mut MultilinearOracleSet<F>,
		log_size: usize,
	) -> Result<Self> {
		let b8_batch = oracles.add_committed_batch(log_size + 2, B8::TOWER_LEVEL);
		let [a, b, sum] = oracles.add_committed_multiple(b8_batch);

		let b1_batch = oracles.add_committed_batch(log_size + 2, B1::TOWER_LEVEL);
		let cout = oracles.add_committed(b1_batch);

		let cin = oracles.add_shifted(cout, 1, 2, ShiftVariant::LogicalLeft)?;

		let lookup_t_batch = oracles.add_committed_batch(ADD_T_LOG_SIZE, B32::TOWER_LEVEL);
		let lookup_t = oracles.add_committed(lookup_t_batch);

		let lasso_batch = LassoBatches::new_in::<B32, _>(oracles, &[log_size + 2], ADD_T_LOG_SIZE);

		let lookup_u = oracles.add_linear_combination(
			log_size + 2,
			[
				(cin, <F as TowerField>::basis(0, 25)?),
				(cout, <F as TowerField>::basis(0, 24)?),
				(a, <F as TowerField>::basis(3, 2)?),
				(b, <F as TowerField>::basis(3, 1)?),
				(sum, <F as TowerField>::basis(3, 0)?),
			],
		)?;

		Ok(TraceOracle {
			lasso_batch,
			b8_batch,
			b1_batch,
			lookup_t_batch,
			a,
			b,
			cin,
			cout,
			sum,
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
	assert!(log_size >= packing_log_width);
	vec![U::default(); 1 << (log_size - packing_log_width)]
}

#[instrument(skip_all)]
fn generate_trace<U, F>(log_size: usize, trace_oracle: &TraceOracle) -> Result<TraceWitness<U, F>>
where
	U: UnderlierType + PackScalar<B1> + PackScalar<B8> + PackScalar<B32> + PackScalar<F>,
	PackedType<U, B8>: PackedFieldIndexable,
	PackedType<U, B32>: PackedFieldIndexable,
	F: BinaryField + ExtensionField<B8> + ExtensionField<B32>,
{
	let mut a = make_underliers::<_, B32>(log_size);
	let mut b = make_underliers::<_, B32>(log_size);
	let mut sum = make_underliers::<_, B32>(log_size);
	let mut cin = make_underliers::<_, B1>(log_size + 2);
	let mut cout = make_underliers::<_, B1>(log_size + 2);
	let mut lookup_u = make_underliers::<_, B32>(log_size + 2);
	let mut lookup_t = make_underliers::<_, B32>(ADD_T_LOG_SIZE);
	let mut u_to_t_mapping = vec![0; 1 << (log_size + 2)];

	let mut rng = thread_rng();

	PackedType::<U, B32>::from_underliers_ref_mut(a.as_mut_slice())
		.iter_mut()
		.for_each(|a| *a = PackedType::<U, B32>::random(&mut rng));
	PackedType::<U, B32>::from_underliers_ref_mut(b.as_mut_slice())
		.iter_mut()
		.for_each(|b| *b = PackedType::<U, B32>::random(&mut rng));

	let a_scalars = underliers_unpack_scalars::<_, B8>(a.as_slice());
	let b_scalars = underliers_unpack_scalars::<_, B8>(b.as_slice());
	let sum_scalars = underliers_unpack_scalars_mut::<_, B8>(sum.as_mut_slice());
	let packed_slice_cin = PackedType::<U, B1>::from_underliers_ref_mut(cin.as_mut_slice());
	let packed_slice_cout = PackedType::<U, B1>::from_underliers_ref_mut(cout.as_mut_slice());
	let lookup_u_scalars = underliers_unpack_scalars_mut::<_, B32>(lookup_u.as_mut_slice());
	let lookup_t_scalars = underliers_unpack_scalars_mut::<_, B32>(lookup_t.as_mut_slice());

	let mut temp_cout = 0;

	for (i, (a, b, sum, lookup_u, u_to_t)) in izip!(
		a_scalars,
		b_scalars,
		sum_scalars.iter_mut(),
		lookup_u_scalars.iter_mut(),
		u_to_t_mapping.iter_mut()
	)
	.enumerate()
	{
		let a_int = u8::from(*a) as usize;
		let b_int = u8::from(*b) as usize;
		let cin = if i % 4 == 0 { 0 } else { temp_cout };

		let ab_sum = a_int + b_int + cin;

		temp_cout = ab_sum >> 8;

		set_packed_slice(packed_slice_cin, i, BinaryField1b::new(U1::new(cin as u8)));
		set_packed_slice(packed_slice_cout, i, BinaryField1b::new(U1::new(temp_cout as u8)));

		*u_to_t = (a_int << 8 | b_int) << 1 | cin;

		let ab_sum = ab_sum & 0xff;

		*sum = BinaryField8b::new(ab_sum as u8);

		let lookup_u_u32 =
			(((((((cin << 1 | temp_cout) << 8) | a_int) << 8) | b_int) << 8) | ab_sum) as u32;

		*lookup_u = BinaryField32b::new(lookup_u_u32);
	}

	for (i, lookup_t) in lookup_t_scalars.iter_mut().enumerate() {
		let a_int = (i >> 9) & 0xff;
		let b_int = (i >> 1) & 0xff;
		let cin = i & 1;
		let ab_sum = a_int + b_int + cin;
		let cout = ab_sum >> 8;
		let ab_sum = ab_sum & 0xff;

		let lookup_t_u32 =
			(((((((cin << 1 | cout) << 8) | a_int) << 8) | b_int) << 8) | ab_sum) as u32;

		*lookup_t = BinaryField32b::new(lookup_t_u32);
	}

	let index = MultilinearExtensionIndex::new()
		.update_owned::<B1, _>([(trace_oracle.cin, cin), (trace_oracle.cout, cout)])?
		.update_owned::<B8, _>([
			(trace_oracle.a, a),
			(trace_oracle.b, b),
			(trace_oracle.sum, sum),
		])?
		.update_owned::<B32, _>([
			(trace_oracle.lookup_t, lookup_t),
			(trace_oracle.lookup_u, lookup_u),
		])?;

	Ok(TraceWitness {
		index,
		u_to_t_mapping,
	})
}

struct Proof<U, PCS1, PCS8, PCS32>
where
	U: UnderlierType + PackScalar<B1> + PackScalar<B8> + PackScalar<B32> + PackScalar<B128>,
	PCS1: PolyCommitScheme<PackedType<U, B1>, B128, Error: Debug, Proof: 'static>,
	PCS8: PolyCommitScheme<PackedType<U, B8>, B128, Error: Debug, Proof: 'static>,
	PCS32: PolyCommitScheme<PackedType<U, B32>, B128, Error: Debug, Proof: 'static>,
{
	lasso_counts_comm: PCS32::Commitment,
	lasso_counts_proof: PCS32::Proof,
	lasso_final_counts_comm: PCS32::Commitment,
	lasso_final_counts_proof: PCS32::Proof,
	b8_comm: PCS8::Commitment,
	b8_proof: PCS8::Proof,
	b1_comm: PCS1::Commitment,
	b1_proof: PCS1::Proof,
	lookup_t_comm: PCS32::Commitment,
	lookup_t_proof: PCS32::Proof,
	gpa_proof: GrandProductBatchProof<B128>,
	greedy_evalcheck_proof: GreedyEvalcheckProof<B128>,
	lasso_proof: LassoProof<B128>,
	_u_marker: PhantomData<U>,
}

// witness column extractor

fn extract_batch_id_polys<'a, U, F, FS>(
	batch_id: BatchId,
	witness_index: &'a MultilinearExtensionIndex<'a, U, F>,
	oracles: &MultilinearOracleSet<F>,
) -> Result<Vec<MultilinearExtensionBorrowed<'a, PackedType<U, FS>>>>
where
	U: UnderlierType + PackScalar<F> + PackScalar<FS>,
	F: TowerField + ExtensionField<FS>,
	FS: TowerField,
{
	let trace_commit_polys = oracles
		.committed_oracle_ids(batch_id)
		.map(|oracle_id| witness_index.get::<FS>(oracle_id))
		.collect::<Result<Vec<_>, _>>()?;
	Ok(trace_commit_polys)
}

#[instrument(skip_all)]
#[allow(clippy::too_many_arguments)]
fn prove<U, PCS1, PCS8, PCS32, CH, Backend>(
	oracles: &mut MultilinearOracleSet<B128>,
	trace_oracle: &TraceOracle,
	witness: TraceWitness<U, B128>,
	pcs1: &PCS1,
	pcs8: &PCS8,
	pcs32: &PCS32,
	pcs_counts_lasso: &PCS32,
	pcs_final_counts_lasso: &PCS32,
	mut challenger: CH,
	domain_factory: impl EvaluationDomainFactory<B128>,
	backend: &Backend,
) -> Result<Proof<U, PCS1, PCS8, PCS32>>
where
	U: UnderlierType + PackScalar<B1> + PackScalar<B8> + PackScalar<B32> + PackScalar<B128>,
	PCS1: PolyCommitScheme<PackedType<U, B1>, B128, Error: Debug, Proof: 'static>,
	PCS8: PolyCommitScheme<PackedType<U, B8>, B128, Error: Debug, Proof: 'static>,
	PCS32: PolyCommitScheme<PackedType<U, B32>, B128, Error: Debug, Proof: 'static>,
	CH: CanObserve<B128>
		+ CanSample<B128>
		+ CanSampleBits<usize>
		+ Clone
		+ CanObserve<PCS1::Commitment>
		+ CanObserve<PCS8::Commitment>
		+ CanObserve<PCS32::Commitment>,
	PackedType<U, B32>: PackedFieldIndexable,
	PackedType<U, B128>: PackedFieldIndexable,
	Backend: ComputationBackend + 'static,
{
	// Round 1 - trace commitments & Lasso deterministic reduction
	let (b8_comm, b8_committed) = pcs8.commit(&extract_batch_id_polys::<_, _, B8>(
		trace_oracle.b8_batch,
		&witness.index,
		oracles,
	)?)?;

	let (b1_comm, b1_committed) = pcs1.commit(&extract_batch_id_polys::<_, _, B1>(
		trace_oracle.b1_batch,
		&witness.index,
		oracles,
	)?)?;

	let (lookup_t_comm, lookup_t_committed) = pcs32.commit(
		&extract_batch_id_polys::<_, _, B32>(trace_oracle.lookup_t_batch, &witness.index, oracles)?,
	)?;

	challenger.observe(b8_comm.clone());
	challenger.observe(b1_comm.clone());
	challenger.observe(lookup_t_comm.clone());

	let lookup_t_oracle = oracles.oracle(trace_oracle.lookup_t);
	let lookup_u_oracle = oracles.oracle(trace_oracle.lookup_u);

	let lookup_t = witness.index.get_multilin_poly(trace_oracle.lookup_t)?;

	let lookup_u = witness.index.get_multilin_poly(trace_oracle.lookup_u)?;

	let lasso_claim = LassoClaim::new(lookup_t_oracle, vec![lookup_u_oracle])?;

	let lasso_witness = LassoWitness::new(lookup_t, vec![lookup_u], vec![&witness.u_to_t_mapping])?;

	let lasso_batch = &trace_oracle.lasso_batch;

	let gamma = challenger.sample();
	let alpha = challenger.sample();

	let lasso_prove_output = lasso::prove::<B32, U, _, _>(
		oracles,
		witness.index,
		&lasso_claim,
		lasso_witness,
		lasso_batch,
		gamma,
		alpha,
	)?;

	let LassoProveOutput {
		reduced_gpa_claims,
		reduced_gpa_witnesses,
		gpa_metas,
		lasso_proof,
		witness_index,
	} = lasso_prove_output;

	let (lasso_counts_comm, lasso_counts_committed) =
		pcs_counts_lasso.commit(&extract_batch_id_polys::<_, _, B32>(
			trace_oracle.lasso_batch.counts_batch_ids[0],
			&witness_index,
			oracles,
		)?)?;

	let (lasso_final_counts_comm, lasso_final_counts_committed) =
		pcs_final_counts_lasso.commit(&extract_batch_id_polys::<_, _, B32>(
			trace_oracle.lasso_batch.final_counts_batch_id,
			&witness_index,
			oracles,
		)?)?;

	challenger.observe(lasso_counts_comm.clone());
	challenger.observe(lasso_final_counts_comm.clone());

	let switchover_fn = standard_switchover_heuristic(-2);

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
		gkr_gpa::make_eval_claims(oracles, gpa_metas, &final_layer_claims)?;

	// Greedy Evalcheck
	let mut witness_index = witness_index;

	let greedy_evalcheck_prove_output = greedy_evalcheck_v2::prove::<U, B128, _, _, _>(
		oracles,
		&mut witness_index,
		evalcheck_multilinear_claims,
		switchover_fn,
		&mut challenger,
		domain_factory,
		backend,
	)?;

	// PCS opening proofs

	let batch_id_to_eval_point = |batch_id| {
		greedy_evalcheck_prove_output
			.same_query_claims
			.iter()
			.find(|(id, _)| *id == batch_id)
			.map(|(_, same_query_claim)| same_query_claim.eval_point.as_slice())
			.expect("present by greedy_evalcheck_v2 invariants")
	};

	let lasso_counts_proof = pcs_counts_lasso.prove_evaluation(
		&mut challenger,
		&lasso_counts_committed,
		&extract_batch_id_polys::<_, _, B32>(
			trace_oracle.lasso_batch.counts_batch_ids[0],
			&witness_index,
			oracles,
		)?,
		batch_id_to_eval_point(trace_oracle.lasso_batch.counts_batch_ids[0]),
		&backend,
	)?;

	let lasso_final_counts_proof = pcs_final_counts_lasso.prove_evaluation(
		&mut challenger,
		&lasso_final_counts_committed,
		&extract_batch_id_polys::<_, _, B32>(
			trace_oracle.lasso_batch.final_counts_batch_id,
			&witness_index,
			oracles,
		)?,
		batch_id_to_eval_point(trace_oracle.lasso_batch.final_counts_batch_id),
		&backend,
	)?;

	let b8_proof = pcs8.prove_evaluation(
		&mut challenger,
		&b8_committed,
		&extract_batch_id_polys::<_, _, B8>(trace_oracle.b8_batch, &witness_index, oracles)?,
		batch_id_to_eval_point(trace_oracle.b8_batch),
		&backend,
	)?;

	let b1_proof = pcs1.prove_evaluation(
		&mut challenger,
		&b1_committed,
		&extract_batch_id_polys::<_, _, B1>(trace_oracle.b1_batch, &witness_index, oracles)?,
		batch_id_to_eval_point(trace_oracle.b1_batch),
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
		b8_comm,
		b8_proof,
		b1_comm,
		b1_proof,
		lookup_t_comm,
		lookup_t_proof,
		gpa_proof,
		greedy_evalcheck_proof: greedy_evalcheck_prove_output.proof,
		lasso_proof,
		_u_marker: PhantomData,
	})
}

#[instrument(skip_all)]
#[allow(clippy::too_many_arguments)]
fn verify<U, PCS1, PCS8, PCS32, CH, Backend>(
	oracles: &mut MultilinearOracleSet<B128>,
	trace_oracle: &TraceOracle,
	pcs1: &PCS1,
	pcs8: &PCS8,
	pcs32: &PCS32,
	pcs_counts_lasso: &PCS32,
	pcs_final_counts_lasso: &PCS32,
	mut challenger: CH,
	proof: Proof<U, PCS1, PCS8, PCS32>,
	backend: &Backend,
) -> Result<()>
where
	U: UnderlierType + PackScalar<B1> + PackScalar<B8> + PackScalar<B32> + PackScalar<B128>,
	PCS1: PolyCommitScheme<PackedType<U, B1>, B128, Error: Debug, Proof: 'static>,
	PCS8: PolyCommitScheme<PackedType<U, B8>, B128, Error: Debug, Proof: 'static>,
	PCS32: PolyCommitScheme<PackedType<U, B32>, B128, Error: Debug, Proof: 'static>,
	CH: CanObserve<B128>
		+ CanSample<B128>
		+ CanSampleBits<usize>
		+ Clone
		+ CanObserve<PCS1::Commitment>
		+ CanObserve<PCS8::Commitment>
		+ CanObserve<PCS32::Commitment>,
	Backend: ComputationBackend + 'static,
{
	// Unpack the proof
	let Proof {
		lasso_counts_comm,
		lasso_counts_proof,
		lasso_final_counts_comm,
		lasso_final_counts_proof,
		b8_comm,
		b8_proof,
		b1_comm,
		b1_proof,
		lookup_t_comm,
		lookup_t_proof,
		gpa_proof,
		greedy_evalcheck_proof,
		lasso_proof,
		..
	} = proof;

	challenger.observe(b8_comm.clone());
	challenger.observe(b1_comm.clone());
	challenger.observe(lookup_t_comm.clone());

	// Round 1 - Lasso deterministic reduction
	let lookup_t_oracle = oracles.oracle(trace_oracle.lookup_t);
	let lookup_u_oracle = oracles.oracle(trace_oracle.lookup_u);

	let lasso_claim = LassoClaim::new(lookup_t_oracle, vec![lookup_u_oracle])?;

	let gamma = challenger.sample();
	let alpha = challenger.sample();

	let (reduced_gpa_claims, gpa_metas) = lasso::verify::<B32, _>(
		oracles,
		&lasso_claim,
		&trace_oracle.lasso_batch,
		gamma,
		alpha,
		lasso_proof,
	)?;

	challenger.observe(lasso_counts_comm.clone());
	challenger.observe(lasso_final_counts_comm.clone());

	let final_layer_claims = gkr_gpa::batch_verify(reduced_gpa_claims, gpa_proof, &mut challenger)?;

	let evalcheck_multilinear_claims =
		gkr_gpa::make_eval_claims(oracles, gpa_metas, &final_layer_claims)?;

	// Greedy evalcheck
	let same_query_pcs_claims = greedy_evalcheck_v2::verify(
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
			.expect("present by greedy_evalcheck_v2 invariants")
	};

	let lasso_counts_eval_claim =
		batch_id_to_eval_claim(trace_oracle.lasso_batch.counts_batch_ids[0]);

	pcs_counts_lasso.verify_evaluation(
		&mut challenger,
		&lasso_counts_comm,
		&lasso_counts_eval_claim.eval_point,
		lasso_counts_proof,
		&lasso_counts_eval_claim.evals,
		backend,
	)?;

	let lasso_final_counts_eval_claim =
		batch_id_to_eval_claim(trace_oracle.lasso_batch.final_counts_batch_id);

	pcs_final_counts_lasso.verify_evaluation(
		&mut challenger,
		&lasso_final_counts_comm,
		&lasso_final_counts_eval_claim.eval_point,
		lasso_final_counts_proof,
		&lasso_final_counts_eval_claim.evals,
		&backend,
	)?;

	let b8_eval_claim = batch_id_to_eval_claim(trace_oracle.b8_batch);

	pcs8.verify_evaluation(
		&mut challenger,
		&b8_comm,
		&b8_eval_claim.eval_point,
		b8_proof,
		&b8_eval_claim.evals,
		&backend,
	)?;

	let b1_eval_claim = batch_id_to_eval_claim(trace_oracle.b1_batch);

	pcs1.verify_evaluation(
		&mut challenger,
		&b1_comm,
		&b1_eval_claim.eval_point,
		b1_proof,
		&b1_eval_claim.evals,
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
	const SECURITY_BITS: usize = 100;

	adjust_thread_pool()
		.as_ref()
		.expect("failed to init thread pool");
	init_tracing().expect("failed to initialize tracing");

	let log_size = get_log_trace_size().unwrap_or(14);
	// to match num_bits between u32add and u32add_lasso
	let log_size = log_size - 5;
	let log_inv_rate = 1;
	let backend = make_portable_backend();

	debug!(num_bits = 1 << (log_size + 5), num_u32s = 1 << log_size, "U32 Addition");

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

	optimal_block_pcs!(pcs1 := 1 x [B1 > B16 > B16 > B128] ^ (log_size + 2));
	optimal_block_pcs!(pcs8 := 3 x [B8 > B16 > B16 > B128] ^ (log_size + 2));
	optimal_block_pcs!(pcs32 := 1 x [B32 > B16 > B32 > B128] ^ (ADD_T_LOG_SIZE));
	optimal_block_pcs!(pcs_counts_lasso := 1 x [B32 > B16 > B32 > B128] ^ (log_size + 2));
	optimal_block_pcs!(pcs_final_counts_lasso := 1 x [B32 > B16 > B32 > B128] ^ (ADD_T_LOG_SIZE));

	let mut oracles = MultilinearOracleSet::<B128>::new();
	let trace_oracle = TraceOracle::new(&mut oracles, log_size)?;
	let challenger = new_hasher_challenger::<_, GroestlHasher<_>>();
	let trace_witness = generate_trace::<Underlier, _>(log_size, &trace_oracle)?;
	let domain_factory = IsomorphicEvaluationDomainFactory::<B128>::default();

	let proof = prove(
		&mut oracles.clone(),
		&trace_oracle,
		trace_witness,
		&pcs1,
		&pcs8,
		&pcs32,
		&pcs_counts_lasso,
		&pcs_final_counts_lasso,
		challenger.clone(),
		domain_factory,
		&backend,
	)?;

	verify(
		&mut oracles.clone(),
		&trace_oracle,
		&pcs1,
		&pcs8,
		&pcs32,
		&pcs_counts_lasso,
		&pcs_final_counts_lasso,
		challenger.clone(),
		proof,
		&backend,
	)?;

	Ok(())
}
