// Copyright 2024 Ulvetanna Inc.

use anyhow::{anyhow, Result};
use binius_core::{
	challenger::{CanObserve, CanSample, CanSampleBits, HashChallenger},
	oracle::{BatchId, CommittedBatchSpec, CommittedId, MultilinearOracleSet, OracleId},
	poly_commit::{tensor_pcs, PolyCommitScheme},
	polynomial::{
		EvaluationDomainFactory, IsomorphicEvaluationDomainFactory, MultilinearExtension,
	},
	protocols::{
		greedy_evalcheck::{self, GreedyEvalcheckProof},
		lasso::{self, LassoBatch, LassoClaim, LassoWitness},
		msetcheck, prodcheck,
		zerocheck::{self, ZerocheckProof},
	},
	witness::MultilinearWitnessIndex,
};
use binius_field::{
	underlier::WithUnderlier, BinaryField128b, BinaryField16b, BinaryField1b, BinaryField32b,
	BinaryField8b, ExtensionField, PackedBinaryField128x1b, PackedBinaryField16x8b,
	PackedBinaryField1x128b, PackedBinaryField4x32b, PackedBinaryField8x16b, PackedField,
	PackedFieldIndexable, TowerField,
};
use binius_hash::GroestlHasher;
use binius_utils::{
	examples::get_log_trace_size, rayon::adjust_thread_pool, tracing::init_tracing,
};
use itertools::izip;
use rand::thread_rng;
use std::fmt::Debug;
use tracing::instrument;

type B1 = BinaryField1b;
type B8 = BinaryField8b;
type B16 = BinaryField16b;
type B32 = BinaryField32b;
type B128 = BinaryField128b;

type P1 = PackedBinaryField128x1b;
type P8 = PackedBinaryField16x8b;
type P16 = PackedBinaryField8x16b;
type P32 = PackedBinaryField4x32b;
type P128 = PackedBinaryField1x128b;

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

struct TraceWitness {
	mult_a: Vec<P8>,
	mult_b: Vec<P8>,
	product: Vec<P16>,
	lookup_u: Vec<P32>,
	lookup_t: Vec<P32>,
	u_to_t_mapping: Vec<usize>,
}

impl TraceWitness {
	fn to_index<PE>(&self, trace_oracle: &TraceOracle) -> MultilinearWitnessIndex<PE>
	where
		PE: PackedField,
		PE::Scalar: ExtensionField<B8> + ExtensionField<B16> + ExtensionField<B32>,
	{
		let mut index = MultilinearWitnessIndex::new();

		index.set_many([
			(trace_oracle.mult_a, ref_mle_column(&self.mult_a).specialize_arc_dyn()),
			(trace_oracle.mult_b, ref_mle_column(&self.mult_b).specialize_arc_dyn()),
			(trace_oracle.product, ref_mle_column(&self.product).specialize_arc_dyn()),
			(trace_oracle.lookup_u, ref_mle_column(&self.lookup_u).specialize_arc_dyn()),
			(trace_oracle.lookup_t, ref_mle_column(&self.lookup_t).specialize_arc_dyn()),
		]);

		index
	}
}

fn build_trace_column<P: PackedField>(log_size: usize) -> Vec<P> {
	vec![P::default(); 1 << (log_size - P::LOG_WIDTH)]
}

#[allow(clippy::ptr_arg)]
fn ref_mle_column<P: PackedField>(raw_vec: &Vec<P>) -> MultilinearExtension<P, &[P]> {
	MultilinearExtension::from_values_slice(raw_vec).expect("infallible")
}

#[instrument(skip_all)]
fn generate_trace(log_size: usize) -> TraceWitness {
	let mut witness = TraceWitness {
		mult_a: build_trace_column(log_size),
		mult_b: build_trace_column(log_size),
		product: build_trace_column(log_size),
		lookup_u: build_trace_column(log_size),
		lookup_t: build_trace_column(log_size),
		u_to_t_mapping: vec![0; 1 << log_size],
	};

	let mut rng = thread_rng();
	witness
		.mult_a
		.iter_mut()
		.for_each(|a| *a = P8::random(&mut rng));
	witness
		.mult_b
		.iter_mut()
		.for_each(|a| *a = P8::random(&mut rng));

	let mult_a = P8::unpack_scalars_mut(&mut witness.mult_a);
	let mult_b = P8::unpack_scalars_mut(&mut witness.mult_b);
	let product = P16::unpack_scalars_mut(&mut witness.product);
	let lookup_u = P32::unpack_scalars_mut(&mut witness.lookup_u);
	let lookup_t = P32::unpack_scalars_mut(&mut witness.lookup_t);

	for (a, b, lookup_u, product, u_to_t) in izip!(
		mult_a,
		mult_b,
		lookup_u.iter_mut(),
		product.iter_mut(),
		witness.u_to_t_mapping.iter_mut()
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
	for (i, lookup_t) in lookup_t.iter_mut().enumerate() {
		let a_int = (i >> 8) & 0xff;
		let b_int = i & 0xff;
		let ab_product = a_int * b_int;
		let lookup_index = a_int << 8 | b_int;
		*lookup_t = BinaryField32b::new((lookup_index << 16 | ab_product) as u32);
	}

	witness
}

struct Proof<PCS1, PCS8, PCS16, PCS32, PCS128>
where
	PCS1: PolyCommitScheme<P1, B128, Error: Debug, Proof: 'static>,
	PCS8: PolyCommitScheme<P8, B128, Error: Debug, Proof: 'static>,
	PCS16: PolyCommitScheme<P16, B128, Error: Debug, Proof: 'static>,
	PCS32: PolyCommitScheme<P32, B128, Error: Debug, Proof: 'static>,
	PCS128: PolyCommitScheme<P128, B128, Error: Debug, Proof: 'static>,
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
	lasso_zerocheck_proof: ZerocheckProof<B128>,
	prodcheck_zerocheck_proof: ZerocheckProof<B128>,
	greedy_evalcheck_proof: GreedyEvalcheckProof<B128>,
}

#[instrument(skip_all)]
#[allow(clippy::too_many_arguments)]
fn prove<PCS1, PCS8, PCS16, PCS32, PCS128, CH>(
	oracles: &mut MultilinearOracleSet<B128>,
	trace_oracle: &TraceOracle,
	witness: &TraceWitness,
	pcs1: &PCS1,
	pcs8: &PCS8,
	pcs16: &PCS16,
	pcs32: &PCS32,
	pcs128: &PCS128,
	mut challenger: CH,
	domain_factory: impl EvaluationDomainFactory<B128>,
) -> Result<Proof<PCS1, PCS8, PCS16, PCS32, PCS128>>
where
	PCS1: PolyCommitScheme<P1, B128, Error: Debug, Proof: 'static>,
	PCS8: PolyCommitScheme<P8, B128, Error: Debug, Proof: 'static>,
	PCS16: PolyCommitScheme<P16, B128, Error: Debug, Proof: 'static>,
	PCS32: PolyCommitScheme<P32, B128, Error: Debug, Proof: 'static>,
	PCS128: PolyCommitScheme<P128, B128, Error: Debug, Proof: 'static>,
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
	let mut witness_index = witness.to_index::<B128>(trace_oracle);

	// Round 1 - trace commitments & Lasso deterministic reduction
	let mults_polys = [&witness.mult_a, &witness.mult_b].map(ref_mle_column);
	let (mults_comm, mults_committed) = pcs8.commit(&mults_polys)?;

	let product_polys = [&witness.product].map(ref_mle_column);
	let (product_comm, product_committed) = pcs16.commit(&product_polys)?;

	let lookup_t_polys = [&witness.lookup_t].map(ref_mle_column);
	let (lookup_t_comm, lookup_t_committed) = pcs32.commit(&lookup_t_polys)?;

	let lookup_t_oracle = oracles.oracle(trace_oracle.lookup_t);
	let lookup_u_oracle = oracles.oracle(trace_oracle.lookup_u);

	let lookup_t = witness_index
		.get(trace_oracle.lookup_t)
		.expect("present by construction")
		.clone();
	let lookup_u = witness_index
		.get(trace_oracle.lookup_u)
		.expect("present by construction")
		.clone();

	let lasso_claim = LassoClaim::new(lookup_t_oracle, lookup_u_oracle)?;

	let lasso_witness = LassoWitness::new(lookup_t, lookup_u, &witness.u_to_t_mapping)?;

	let lasso_prove_output = lasso::prove::<P32, P1, _, _, _>(
		oracles,
		&mut witness_index,
		&lasso_claim,
		lasso_witness,
		&trace_oracle.lasso_batch,
	)?;

	let (lasso_comm, lasso_committed) = pcs1.commit(&lasso_prove_output.committed_polys)?;

	challenger.observe(lasso_comm.clone());
	challenger.observe(mults_comm.clone());
	challenger.observe(product_comm.clone());
	challenger.observe(lookup_t_comm.clone());

	// Round 2 - Msetcheck & Prodcheck
	let gamma = challenger.sample();
	let alpha = challenger.sample();

	let msetcheck_prove_output = msetcheck::prove(
		oracles,
		&mut witness_index,
		&lasso_prove_output.reduced_lasso_claims.msetcheck_claim,
		lasso_prove_output.msetcheck_witness,
		gamma,
		Some(alpha),
	)?;

	let f_prime_committed_id = CommittedId {
		batch_id: trace_oracle.grand_prod_batch,
		index: 0,
	};

	let prodcheck_prove_output = prodcheck::prove(
		oracles,
		&mut witness_index,
		&msetcheck_prove_output.prodcheck_claim,
		msetcheck_prove_output.prodcheck_witness,
		f_prime_committed_id,
	)?;

	let grand_prod_polys = [prodcheck_prove_output.f_prime_commit.to_single_packed()];
	let (grand_prod_comm, grand_prod_committed) = pcs128.commit(&grand_prod_polys)?;

	challenger.observe(grand_prod_comm.clone());

	// Prove reduced zerocheck originating from prodcheck

	let switchover_fn = |extension_degree| match extension_degree {
		128 => 5,
		_ => 1,
	};

	let prodcheck_zerocheck_domain = domain_factory.create(
		prodcheck_prove_output
			.reduced_product_check_claims
			.t_prime_claim
			.poly
			.max_individual_degree()
			+ 1,
	)?;

	let prodcheck_zerocheck_prove_output = zerocheck::prove(
		&prodcheck_prove_output
			.reduced_product_check_claims
			.t_prime_claim,
		prodcheck_prove_output.t_prime_witness,
		&prodcheck_zerocheck_domain,
		&mut challenger,
		switchover_fn,
	)?;

	// Prove Lasso counts zerocheck

	let lasso_zerocheck_domain = domain_factory.create(
		lasso_prove_output
			.reduced_lasso_claims
			.zerocheck_claim
			.poly
			.max_individual_degree()
			+ 1,
	)?;

	let lasso_zerocheck_prove_output = zerocheck::prove(
		&lasso_prove_output.reduced_lasso_claims.zerocheck_claim,
		lasso_prove_output.zerocheck_witness,
		&lasso_zerocheck_domain,
		&mut challenger,
		switchover_fn,
	)?;

	// Greedy Evalcheck

	let greedy_evalcheck_prove_output = greedy_evalcheck::prove::<_, _, B128, _>(
		oracles,
		&mut witness_index,
		[
			lasso_zerocheck_prove_output.evalcheck_claim,
			prodcheck_zerocheck_prove_output.evalcheck_claim,
		],
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
		&lasso_prove_output.committed_polys,
		batch_id_to_eval_point(trace_oracle.lasso_batch.batch_id()),
	)?;

	let mults_proof = pcs8.prove_evaluation(
		&mut challenger,
		&mults_committed,
		&mults_polys,
		batch_id_to_eval_point(trace_oracle.mults_batch),
	)?;

	let product_proof = pcs16.prove_evaluation(
		&mut challenger,
		&product_committed,
		&product_polys,
		batch_id_to_eval_point(trace_oracle.product_batch),
	)?;

	let lookup_t_proof = pcs32.prove_evaluation(
		&mut challenger,
		&lookup_t_committed,
		&lookup_t_polys,
		batch_id_to_eval_point(trace_oracle.lookup_t_batch),
	)?;

	let grand_prod_proof = pcs128.prove_evaluation(
		&mut challenger,
		&grand_prod_committed,
		&grand_prod_polys,
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
		lasso_zerocheck_proof: lasso_zerocheck_prove_output.zerocheck_proof,
		prodcheck_zerocheck_proof: prodcheck_zerocheck_prove_output.zerocheck_proof,
		greedy_evalcheck_proof: greedy_evalcheck_prove_output.proof,
	})
}

#[instrument(skip_all)]
#[allow(clippy::too_many_arguments)]
fn verify<PCS1, PCS8, PCS16, PCS32, PCS128, CH>(
	oracles: &mut MultilinearOracleSet<B128>,
	trace_oracle: &TraceOracle,
	pcs1: &PCS1,
	pcs8: &PCS8,
	pcs16: &PCS16,
	pcs32: &PCS32,
	pcs128: &PCS128,
	mut challenger: CH,
	proof: Proof<PCS1, PCS8, PCS16, PCS32, PCS128>,
) -> Result<()>
where
	PCS1: PolyCommitScheme<P1, B128, Error: Debug, Proof: 'static>,
	PCS8: PolyCommitScheme<P8, B128, Error: Debug, Proof: 'static>,
	PCS16: PolyCommitScheme<P16, B128, Error: Debug, Proof: 'static>,
	PCS32: PolyCommitScheme<P32, B128, Error: Debug, Proof: 'static>,
	PCS128: PolyCommitScheme<P128, B128, Error: Debug, Proof: 'static>,
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

	// Verify prodcheck originating zerocheck
	let prodcheck_evalcheck_claim = zerocheck::verify(
		&reduced_prodcheck_claims.t_prime_claim,
		proof.prodcheck_zerocheck_proof,
		&mut challenger,
	)?;

	// Verify Lasso originating zerocheck
	let lasso_evalcheck_claim = zerocheck::verify(
		&reduced_lasso_claims.zerocheck_claim,
		proof.lasso_zerocheck_proof,
		&mut challenger,
	)?;

	// Greedy evalcheck
	let same_query_pcs_claims = greedy_evalcheck::verify(
		oracles,
		[lasso_evalcheck_claim, prodcheck_evalcheck_claim],
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
	init_tracing();

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
	let witness = generate_trace(log_size);
	let domain_factory = IsomorphicEvaluationDomainFactory::<B128>::default();

	let proof = prove(
		&mut oracles.clone(),
		&trace_oracle,
		&witness,
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
