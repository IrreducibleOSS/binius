// Copyright 2025 Irreducible Inc.

use binius_field::{
	packed::{get_packed_slice, set_packed_slice},
	BinaryField, BinaryField128bPolyval, BinaryField1b, Field, PackedBinaryField128x1b,
	PackedBinaryPolyval1x128b, PackedExtension, PackedField,
};
use binius_hal::make_portable_backend;
use binius_hash::groestl::Groestl256;
use binius_math::{
	DefaultEvaluationDomainFactory, EvaluationOrder, MLEEmbeddingAdapter, MultilinearExtension,
	MultilinearPoly, MultilinearQuery, MultilinearQueryRef,
};
use rand::{thread_rng, Rng};

use super::{batch_prove, common::BaseExpReductionOutput};
use crate::{
	fiat_shamir::HasherChallenger,
	protocols::gkr_exp::{batch_verify, common::ExpClaim, witness::BaseExpWitness},
	transcript::ProverTranscript,
	witness::MultilinearWitness,
};

type PBits = PackedBinaryField128x1b;
type F = BinaryField128bPolyval;
type P = PackedBinaryPolyval1x128b;

fn generate_claim_witness<'a>(
	exponents_in_each_row: &[u128],
	exponent_bit_width: usize,
	base: Option<MultilinearWitness<'a, P>>,
	eval_point: &[F],
) -> (BaseExpWitness<'a, P>, ExpClaim<F>) {
	let column_len = exponents_in_each_row.len();

	let exponent_witnesses_as_vec: Vec<_> = (0..exponent_bit_width)
		.map(|i| {
			let mut column_witness =
				vec![PBits::default(); column_len / <PBits as PackedField>::WIDTH];

			for (row, this_row_exponent) in exponents_in_each_row.iter().enumerate() {
				let this_bit_of_exponent = (this_row_exponent >> i) & 1;

				let single_bit_value = if this_bit_of_exponent == 1 {
					BinaryField1b::ONE
				} else {
					BinaryField1b::ZERO
				};

				set_packed_slice(&mut column_witness, row, single_bit_value);
			}

			column_witness
		})
		.collect::<Vec<_>>();

	let exponent_witnesses = exponent_witnesses_as_vec
		.into_iter()
		.map(|column| {
			let mle = MultilinearExtension::from_values(column).unwrap();
			MLEEmbeddingAdapter::<PBits, P>::from(mle).upcast_arc_dyn()
		})
		.collect::<Vec<_>>();

	let witness = if let Some(base) = base {
		BaseExpWitness::<'_, P>::new_with_dynamic_base(exponent_witnesses, base).unwrap()
	} else {
		BaseExpWitness::<'_, P>::new_with_constant_base(
			exponent_witnesses,
			F::MULTIPLICATIVE_GENERATOR,
		)
		.unwrap()
	};

	let exponentiation_result_witness = witness.exponentiation_result_witness();

	let last_layer_query = MultilinearQuery::expand(eval_point);

	let constant_base = if witness.uses_dynamic_base() {
		None
	} else {
		Some(F::MULTIPLICATIVE_GENERATOR)
	};

	let claim = ExpClaim {
		eval_point: eval_point.to_vec(),
		eval: exponentiation_result_witness
			.evaluate(MultilinearQueryRef::new(&last_layer_query))
			.unwrap(),
		exponent_bit_width: witness.exponent.len(),
		n_vars: eval_point.len(),
		constant_base,
	};
	(witness, claim)
}

fn generate_mul_witnesses_claims<'a, const LOG_SIZE: usize, const COLUMN_LEN: usize>(
	exponent_bit_width: usize,
) -> (Vec<ExpClaim<F>>, Vec<BaseExpWitness<'a, P>>) {
	let mut rng = thread_rng();

	let a: Vec<_> = (0..COLUMN_LEN)
		.map(|_| rng.gen::<u128>() % (1 << exponent_bit_width))
		.collect();
	let b: Vec<_> = (0..COLUMN_LEN)
		.map(|_| rng.gen::<u128>() % (1 << exponent_bit_width))
		.collect();
	let c: Vec<_> = a.iter().zip(&b).map(|(ai, bi)| ai * bi).collect();

	let eval_point_1 = [F::default(); LOG_SIZE].map(|_| <F as Field>::random(&mut rng));

	let eval_point_2 = [F::default(); LOG_SIZE].map(|_| <F as Field>::random(&mut rng));

	let (a_witness, a_claim) = generate_claim_witness(&a, exponent_bit_width, None, &eval_point_1);
	let (b_witness, b_claim) = generate_claim_witness(
		&b,
		exponent_bit_width,
		Some(a_witness.exponentiation_result_witness()),
		&eval_point_2,
	);

	let (c_witness, c_claim) =
		generate_claim_witness(&c, exponent_bit_width * 2, None, &eval_point_2);

	assert_eq!(b_claim.eval, c_claim.eval);

	let witnesses = [c_witness, b_witness, a_witness];
	let claims = [c_claim, b_claim, a_claim];

	(claims.to_vec(), witnesses.to_vec())
}

#[allow(clippy::type_complexity)]
fn generate_mul_witnesses_claims_with_different_log_size<'a>(
) -> (Vec<BaseExpWitness<'a, P>>, Vec<ExpClaim<F>>) {
	const LOG_SIZE_1: usize = 14usize;
	const COLUMN_LEN_1: usize = 1usize << LOG_SIZE_1;
	const EXPONENT_BIT_WIDTH_1: usize = 3usize;

	const LOG_SIZE_2: usize = 13usize;
	const COLUMN_LEN_2: usize = 1usize << LOG_SIZE_2;

	const EXPONENT_BIT_WIDTH_2: usize = 2usize;

	let (claims_1, witnesses_1) =
		generate_mul_witnesses_claims::<LOG_SIZE_1, COLUMN_LEN_1>(EXPONENT_BIT_WIDTH_1);

	let (claims_2, witnesses_2) =
		generate_mul_witnesses_claims::<LOG_SIZE_2, COLUMN_LEN_2>(EXPONENT_BIT_WIDTH_2);

	let witnesses = [witnesses_1, witnesses_2].concat();

	let claims = [claims_1, claims_2].concat();

	(witnesses, claims)
}

#[test]
fn witness_gen_happens_correctly() {
	const LOG_SIZE: usize = 13usize;
	const COLUMN_LEN: usize = 1usize << LOG_SIZE;
	const EXPONENT_BIT_WIDTH: usize = 8usize;

	let mut rng = thread_rng();

	let a: Vec<_> = (0..COLUMN_LEN).map(|_| rng.gen::<u8>() as u128).collect();
	let b: Vec<_> = (0..COLUMN_LEN).map(|_| rng.gen::<u8>() as u128).collect();
	let c: Vec<_> = a.iter().zip(&b).map(|(ai, bi)| ai * bi).collect();

	let eval_point = [F::default(); LOG_SIZE].map(|_| <F as Field>::random(&mut rng));

	let (a_witness, _) = generate_claim_witness(&a, EXPONENT_BIT_WIDTH, None, &eval_point);
	let a_exponentiation_result = a_witness.exponentiation_result_witness();

	let a_exponentiation_result_evals =
		P::cast_bases(a_exponentiation_result.packed_evals().unwrap());

	for (row_idx, this_row_exponent) in a.into_iter().enumerate() {
		assert_eq!(
			F::MULTIPLICATIVE_GENERATOR.pow(this_row_exponent),
			get_packed_slice(a_exponentiation_result_evals, row_idx)
		);
	}

	let (b_witness, _) = generate_claim_witness(
		&b,
		EXPONENT_BIT_WIDTH,
		Some(a_exponentiation_result.clone()),
		&eval_point,
	);
	let b_exponentiation_result = b_witness.exponentiation_result_witness();

	let b_exponentiation_result_evals =
		P::cast_bases(b_exponentiation_result.packed_evals().unwrap());

	for (row_idx, this_row_exponent) in b.into_iter().enumerate() {
		assert_eq!(
			get_packed_slice(a_exponentiation_result_evals, row_idx).pow(this_row_exponent),
			get_packed_slice(b_exponentiation_result_evals, row_idx)
		);
	}

	let (c_witness, _) = generate_claim_witness(&c, EXPONENT_BIT_WIDTH * 2, None, &eval_point);

	let c_exponentiation_result = c_witness.exponentiation_result_witness();

	assert_eq!(
		b_exponentiation_result_evals,
		P::cast_bases(c_exponentiation_result.packed_evals().unwrap())
	);
}

#[test]
fn prove_reduces_to_correct_claims() {
	let mut transcript = ProverTranscript::<HasherChallenger<Groestl256>>::new();

	let (witnesses, claims): (Vec<_>, Vec<_>) =
		generate_mul_witnesses_claims_with_different_log_size();

	let evaluation_domain_factory =
		DefaultEvaluationDomainFactory::<BinaryField128bPolyval>::default();

	let backend = make_portable_backend();

	let BaseExpReductionOutput {
		layers_claims: layer_eval_claims,
	} = batch_prove::batch_prove(
		EvaluationOrder::LowToHigh,
		witnesses.clone(),
		&claims,
		evaluation_domain_factory,
		&mut transcript,
		&backend,
	)
	.unwrap();

	for (exponent_bit_number, layer_eval_claim) in layer_eval_claims.iter().enumerate() {
		let mut j = 0;
		for witness in &witnesses {
			let exponent_len = witness.exponent.len() - 1;

			if exponent_len < exponent_bit_number {
				continue;
			}

			let this_claim = &layer_eval_claim[j];
			let this_bit_query = MultilinearQuery::expand(&this_claim.eval_point);

			match &witness.base {
				super::witness::BaseWitness::Constant(_) => {
					let exponent = witness.exponent[exponent_len - exponent_bit_number].clone();

					let claimed_evaluation = this_claim.eval;

					let actual_evaluation = exponent
						.evaluate(MultilinearQueryRef::new(&this_bit_query))
						.unwrap();

					assert_eq!(claimed_evaluation, actual_evaluation);
				}
				super::witness::BaseWitness::Dynamic(multilinear_poly) => {
					let exponent = witness.exponent[exponent_bit_number].clone();

					let claimed_evaluation = this_claim.eval;

					let actual_evaluation = exponent
						.evaluate(MultilinearQueryRef::new(&this_bit_query))
						.unwrap();

					assert_eq!(claimed_evaluation, actual_evaluation);

					j += 1;

					let claimed_evaluation = layer_eval_claim[j].eval;

					let actual_evaluation = multilinear_poly
						.evaluate(MultilinearQueryRef::new(&this_bit_query))
						.unwrap();

					assert_eq!(claimed_evaluation, actual_evaluation);
				}
			}

			j += 1;
		}
	}
}

#[test]
fn good_proof_verifies() {
	for evaluation_order in [EvaluationOrder::HighToLow, EvaluationOrder::LowToHigh] {
		let mut transcript = ProverTranscript::<HasherChallenger<Groestl256>>::new();

		let (witnesses, claims): (Vec<_>, Vec<_>) =
			generate_mul_witnesses_claims_with_different_log_size();

		let evaluation_domain_factory =
			DefaultEvaluationDomainFactory::<BinaryField128bPolyval>::default();

		let backend = make_portable_backend();

		batch_prove::batch_prove(
			evaluation_order,
			witnesses,
			&claims,
			evaluation_domain_factory,
			&mut transcript,
			&backend,
		)
		.unwrap();

		let mut verifier_transcript = transcript.into_verifier();

		let _reduced_claims =
			batch_verify::batch_verify(evaluation_order, &claims, &mut verifier_transcript)
				.unwrap();

		verifier_transcript.finalize().unwrap()
	}
}
