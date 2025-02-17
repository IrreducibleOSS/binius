// Copyright 2024-2025 Irreducible Inc.

use std::array;

use binius_field::{
	packed::{get_packed_slice, set_packed_slice},
	underlier::SmallU,
	BinaryField, BinaryField128b, BinaryField1b, BinaryField64b, BinaryField8b, Field,
	PackedBinaryField128x1b, PackedBinaryField1x128b, PackedBinaryField2x64b, PackedField,
};
use binius_hal::make_portable_backend;
use binius_math::{
	DefaultEvaluationDomainFactory, MLEEmbeddingAdapter, MultilinearExtension, MultilinearPoly,
	MultilinearQuery, MultilinearQueryRef,
};
use groestl_crypto::Groestl256;
use itertools::izip;
use rand::{thread_rng, Rng};

use super::{common::GeneratorExponentReductionOutput, prove};
use crate::{
	fiat_shamir::{Challenger, HasherChallenger},
	protocols::{
		gkr_gpa::LayerClaim,
		gkr_int_mul::generator_exponent::{verify, witness::GeneratorExponentWitness},
	},
	transcript::ProverTranscript,
};

type PBits = PackedBinaryField128x1b;
type FGenerator = BinaryField64b;
type PGenerator = PackedBinaryField2x64b;
type F = BinaryField128b;
type PChallenge = PackedBinaryField1x128b;

fn generate_witness_for_testing<const COLUMN_LEN: usize>(
	exponents_in_each_row: [u128; COLUMN_LEN],
) -> GeneratorExponentWitness<'static, PBits, PGenerator, PChallenge, 64> {
	let exponent_witnesses_as_vec: [_; 64] = array::from_fn(|i| {
		let mut column_witness = vec![PBits::default(); COLUMN_LEN / <PBits as PackedField>::WIDTH];

		for (row, this_row_exponent) in exponents_in_each_row.iter().enumerate() {
			let this_bit_of_exponent = ((this_row_exponent >> i) & 1) as u8;

			let single_bit_value = BinaryField1b::from(SmallU::<1>::new(this_bit_of_exponent));

			set_packed_slice(&mut column_witness, row, single_bit_value);
		}

		column_witness
	});

	let exponent_witnesses = exponent_witnesses_as_vec.map(|column| {
		let mle = MultilinearExtension::from_values(column).unwrap();
		MLEEmbeddingAdapter::<PBits, PChallenge>::from(mle).upcast_arc_dyn()
	});

	let witness =
		GeneratorExponentWitness::<'_, PBits, PGenerator, PChallenge, 64>::new(exponent_witnesses)
			.unwrap();

	witness
}

fn generate_witness_and_prove<
	const LOG_SIZE: usize,
	const COLUMN_LEN: usize,
	const EXPONENT_BIT_WIDTH: usize,
	Challenger_,
>(
	transcript: &mut ProverTranscript<Challenger_>,
) -> (
	LayerClaim<F>,
	GeneratorExponentWitness<'static, PBits, PGenerator, PChallenge, 64>,
	GeneratorExponentReductionOutput<F, 64>,
)
where
	Challenger_: Challenger,
{
	let mut rng = thread_rng();

	let exponent: [u128; COLUMN_LEN] = array::from_fn(|_| rng.gen());
	let witness = generate_witness_for_testing::<COLUMN_LEN>(exponent);

	let eval_point = [F::default(); LOG_SIZE].map(|_| <F as Field>::random(&mut rng));

	let last_layer = witness.single_bit_output_layers_data[EXPONENT_BIT_WIDTH - 1].clone();

	let last_layer_mle = MultilinearExtension::from_values(last_layer).unwrap();

	let last_layer_witness = MLEEmbeddingAdapter::<PGenerator, PChallenge>::from(last_layer_mle);

	let last_layer_query = MultilinearQuery::expand(&eval_point);

	let claim = LayerClaim {
		eval_point: eval_point.to_vec(),
		eval: last_layer_witness
			.evaluate(MultilinearQueryRef::new(&last_layer_query))
			.unwrap(),
	};

	let evaluation_domain_factory = DefaultEvaluationDomainFactory::<BinaryField8b>::default();

	let backend = make_portable_backend();

	let reduced_claims =
		prove::prove(&witness, &claim, evaluation_domain_factory, transcript, &backend).unwrap();

	(claim, witness, reduced_claims)
}

#[test]
#[allow(clippy::large_stack_frames)]
fn witness_gen_happens_correctly() {
	const LOG_SIZE: usize = 13usize;
	const COLUMN_LEN: usize = 1usize << LOG_SIZE;

	let mut rng = thread_rng();
	let exponent: [u128; COLUMN_LEN] = array::from_fn(|_| rng.gen());

	let witness = generate_witness_for_testing(exponent);

	let results = &witness.single_bit_output_layers_data[63];
	for (row_idx, this_row_exponent) in exponent.into_iter().enumerate() {
		assert_eq!(
			<PGenerator as PackedField>::Scalar::MULTIPLICATIVE_GENERATOR
				.pow(this_row_exponent as u64),
			get_packed_slice(results, row_idx)
		);
	}
}

#[test]
fn prove_reduces_to_correct_claims() {
	const LOG_SIZE: usize = 13usize;
	const COLUMN_LEN: usize = 1usize << LOG_SIZE;
	const EXPONENT_BIT_WIDTH: usize = 64usize;

	let mut transcript = ProverTranscript::<HasherChallenger<Groestl256>>::new();
	let reduced_claims =
		generate_witness_and_prove::<LOG_SIZE, COLUMN_LEN, EXPONENT_BIT_WIDTH, _>(&mut transcript);

	let (_, witness, evals_on_bit_columns) = reduced_claims;

	// Check the evaluations of the exponent bit columns

	for (this_bit_witness, this_claim) in
		izip!(witness.exponent, evals_on_bit_columns.eval_claims_on_exponent_bit_columns)
	{
		let this_bit_query = MultilinearQuery::expand(&this_claim.eval_point);

		let claimed_evaluation = this_claim.eval;

		let actual_evaluation = this_bit_witness
			.evaluate(MultilinearQueryRef::new(&this_bit_query))
			.unwrap();

		assert_eq!(claimed_evaluation, actual_evaluation);
	}
}

#[test]
fn good_proof_verifies() {
	const LOG_SIZE: usize = 13usize;
	const COLUMN_LEN: usize = 1usize << LOG_SIZE;
	const EXPONENT_BIT_WIDTH: usize = 64usize;

	let mut transcript = ProverTranscript::<HasherChallenger<Groestl256>>::new();
	let reduced_claims =
		generate_witness_and_prove::<LOG_SIZE, COLUMN_LEN, EXPONENT_BIT_WIDTH, _>(&mut transcript);

	let (claim, _, _) = reduced_claims;

	let mut verifier_transcript = transcript.into_verifier();

	let _reduced_claims = verify::verify::<FGenerator, _, _, EXPONENT_BIT_WIDTH>(
		&claim,
		&mut verifier_transcript,
		LOG_SIZE,
	)
	.unwrap();

	verifier_transcript.finalize().unwrap()
}
