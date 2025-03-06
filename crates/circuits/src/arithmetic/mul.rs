// Copyright 2024-2025 Irreducible Inc.

use anyhow::Error;
use binius_core::{constraint_system::mul::Mul, oracle::OracleId};
use binius_field::{BinaryField128b, BinaryField1b, TowerField};
use binius_macros::arith_expr;
use binius_maybe_rayon::iter::{
	IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator,
};

use crate::builder::ConstraintSystemBuilder;

pub fn mul(
	builder: &mut ConstraintSystemBuilder,
	name: impl ToString,
	xin_bits: Vec<OracleId>,
	yin_bits: Vec<OracleId>,
) -> Result<Vec<OracleId>, anyhow::Error> {
	let name = name.to_string();

	let log_rows = builder.log_rows([xin_bits.clone(), yin_bits.clone()].into_iter().flatten())?;

	let xin_exp_result_id = builder.add_committed(
		format!("{} xin_exp_result", name),
		log_rows,
		BinaryField128b::TOWER_LEVEL,
	);

	let yin_exp_result_id = builder.add_committed(
		format!("{} yin_exp_result", name),
		log_rows,
		BinaryField128b::TOWER_LEVEL,
	);

	let cout_low_exp_result_id = builder.add_committed(
		format!("{} cout_low_exp_result", name),
		log_rows,
		BinaryField128b::TOWER_LEVEL,
	);

	let cout_high_exp_result_id = builder.add_committed(
		format!("{} cout_high_exp_result", name),
		log_rows,
		BinaryField128b::TOWER_LEVEL,
	);

	let result_bits = xin_bits.len() + yin_bits.len();

	let cout_bits = (0..result_bits)
		.map(|i| {
			builder.add_committed(
				format!("{} bit of {}", i, name),
				log_rows,
				BinaryField1b::TOWER_LEVEL,
			)
		})
		.collect::<Vec<_>>();

	let cout_low_bits = cout_bits[0..cout_bits.len() / 2].to_vec();
	let cout_high_bits = cout_bits[cout_bits.len() / 2..].to_vec();

	if let Some(witness) = builder.witness() {
		let xin_columns = xin_bits
			.iter()
			.map(|&id| {
				witness
					.get::<BinaryField1b>(id)
					.map(|x| x.as_slice::<u32>())
			})
			.collect::<Result<Vec<_>, Error>>()?;

		let yin_columns = yin_bits
			.iter()
			.map(|&id| {
				witness
					.get::<BinaryField1b>(id)
					.map(|x| x.as_slice::<u32>())
			})
			.collect::<Result<Vec<_>, Error>>()?;

		let result = columns_to_numbers(&xin_columns)
			.into_iter()
			.zip(columns_to_numbers(&yin_columns))
			.map(|(x, y)| x * y)
			.collect::<Vec<_>>();

		let mut cout_columns = cout_bits
			.iter()
			.map(|&id| witness.new_column::<BinaryField1b>(id))
			.collect::<Vec<_>>();

		let mut cout_columns_u32 = cout_columns
			.iter_mut()
			.map(|column| column.as_mut_slice::<u32>())
			.collect::<Vec<_>>();

		numbers_to_columns(&result, &mut cout_columns_u32);
	}

	builder.assert_zero(
		name.clone(),
		[xin_bits[0], yin_bits[0], cout_bits[0]],
		arith_expr!([xin, yin, cout] = xin * yin - cout).convert_field(),
	);

	// (g^a)^b == g^c_low * (g^(2^32))^c_high
	builder.assert_zero(
		name,
		[
			yin_exp_result_id,
			cout_low_exp_result_id,
			cout_high_exp_result_id,
		],
		arith_expr!([yin, low, high] = low * high - yin).convert_field(),
	);

	let mul = Mul {
		xin_bits,
		xin_exp_result_id,
		yin_bits,
		yin_exp_result_id,
		cout_low_bits,
		cout_high_bits,
		cout_low_exp_result_id,
		cout_high_exp_result_id,
	};

	builder.add_mul(mul);

	Ok(cout_bits)
}

fn columns_to_numbers(columns: &[&[u32]]) -> Vec<u128> {
	let mut numbers: Vec<u128> = vec![0; columns.first().map(|c| c.len()).unwrap_or(0) * 32];

	for (bit, column) in columns.iter().enumerate() {
		numbers.par_iter_mut().enumerate().for_each(|(i, number)| {
			let num_idx = i / 32;
			let bit_idx = i % 32;

			if (column[num_idx] >> bit_idx) & 1 == 1 {
				*number |= 1 << bit;
			}
		});
	}
	numbers
}

fn numbers_to_columns(numbers: &[u128], columns: &mut [&mut [u32]]) {
	columns
		.par_iter_mut()
		.enumerate()
		.for_each(|(bit, column)| {
			for (i, number) in numbers.iter().enumerate() {
				if (number >> bit) & 1 == 1 {
					let num_idx = i / 32;
					let bit_idx = i % 32;
					column[num_idx] |= 1 << bit_idx;
				}
			}
		});
}

#[cfg(test)]
mod tests {
	use binius_core::{
		constraint_system::{self},
		fiat_shamir::HasherChallenger,
		tower::CanonicalTowerFamily,
	};
	use binius_field::BinaryField1b;
	use binius_hal::make_portable_backend;
	use binius_hash::compress::Groestl256ByteCompression;
	use binius_math::DefaultEvaluationDomainFactory;
	use groestl_crypto::Groestl256;

	use super::mul;
	use crate::{
		builder::{types::U, ConstraintSystemBuilder},
		unconstrained::unconstrained,
	};

	#[test]
	fn test_mul() {
		let allocator = bumpalo::Bump::new();
		let mut builder = ConstraintSystemBuilder::new_with_witness(&allocator);

		let log_n_muls = 9;

		let in_a = (0..2)
			.map(|i| {
				unconstrained::<BinaryField1b>(&mut builder, format!("in_a_{}", i), log_n_muls)
					.unwrap()
			})
			.collect::<Vec<_>>();
		let in_b = (0..2)
			.map(|i| {
				unconstrained::<BinaryField1b>(&mut builder, format!("in_b_{}", i), log_n_muls)
					.unwrap()
			})
			.collect::<Vec<_>>();

		mul(&mut builder, "test", in_a, in_b).unwrap();

		let witness = builder
			.take_witness()
			.expect("builder created with witness");

		let constraint_system = builder.build().unwrap();

		let domain_factory = DefaultEvaluationDomainFactory::default();
		let backend = make_portable_backend();

		let proof = constraint_system::prove::<
			U,
			CanonicalTowerFamily,
			_,
			Groestl256,
			Groestl256ByteCompression,
			HasherChallenger<Groestl256>,
			_,
		>(&constraint_system, 1, 10, &[], witness, &domain_factory, &backend)
		.unwrap();

		constraint_system::verify::<
			U,
			CanonicalTowerFamily,
			Groestl256,
			Groestl256ByteCompression,
			HasherChallenger<Groestl256>,
		>(&constraint_system, 1, 10, &[], proof)
		.unwrap();
	}
}
