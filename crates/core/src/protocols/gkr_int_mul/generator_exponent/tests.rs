// Copyright 2024 Irreducible Inc.

use std::array;

use binius_field::{
	packed::{get_packed_slice, set_packed_slice},
	underlier::SmallU,
	BinaryField, BinaryField1b, Field, PackedBinaryField128x1b, PackedBinaryField1x128b,
	PackedBinaryField2x64b, PackedField,
};
use binius_math::{MLEEmbeddingAdapter, MultilinearExtension};
use rand::{thread_rng, Rng};

use crate::protocols::gkr_int_mul::generator_exponent::witness::GeneratorExponentWitness;

type P = PackedBinaryField128x1b;
type PE = PackedBinaryField2x64b;
type PChallenge = PackedBinaryField1x128b;

fn generate_witness_for_testing<const COLUMN_LEN: usize>(
	exponents_in_each_row: [u128; COLUMN_LEN],
) -> GeneratorExponentWitness<'static, P, PE, PChallenge, 64> {
	let exponent_witnesses_as_vec: [_; 64] = array::from_fn(|i| {
		let mut column_witness = vec![P::default(); COLUMN_LEN / <P as PackedField>::WIDTH];

		for (row, this_row_exponent) in exponents_in_each_row.iter().enumerate() {
			let this_bit_of_exponent = ((this_row_exponent >> i) & 1) as u8;

			let single_bit_value = BinaryField1b::from(SmallU::<1>::new(this_bit_of_exponent));

			set_packed_slice(&mut column_witness, row, single_bit_value);
		}

		column_witness
	});

	let exponent_witnesses = exponent_witnesses_as_vec.map(|column| {
		let mle = MultilinearExtension::from_values(column).unwrap();
		MLEEmbeddingAdapter::<P, PChallenge>::from(mle).upcast_arc_dyn()
	});

	let witness =
		GeneratorExponentWitness::<'_, P, PE, PChallenge, 64>::new(exponent_witnesses).unwrap();

	witness
}

#[test]
fn witness_gen_happens_correctly() {
	const LOG_SIZE: usize = 13usize;
	const COLUMN_LEN: usize = 1usize << LOG_SIZE;

	let mut rng = thread_rng();
	let exponent: [u128; COLUMN_LEN] = array::from_fn(|_| rng.gen());

	let witness = generate_witness_for_testing(exponent);

	let results = &witness.single_bit_output_layers_data[63];
	for (row_idx, this_row_exponent) in exponent.into_iter().enumerate() {
		assert_eq!(
			<PE as PackedField>::Scalar::MULTIPLICATIVE_GENERATOR.pow([this_row_exponent as u64]),
			get_packed_slice(results, row_idx)
		);
	}
}
