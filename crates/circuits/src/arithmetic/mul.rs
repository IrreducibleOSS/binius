// Copyright 2025 Irreducible Inc.

//! Multiplication based on exponentiation.
//!
//! The core idea of this method is to verify the equality $a \cdot b = c$
//! by checking if $(g^a)^b = g^{clow} \cdot (g^{2^{32}})^{chigh}$,
//! where exponentiation proofs can be efficiently verified using the GKR exponentiation protocol.
//!
//! You can read more information in [Integer Multiplication in Binius](https://www.irreducible.com/posts/integer-multiplication-in-binius).

use std::array;

use anyhow::Error;
use binius_core::{constraint_system::exp::ExpBase, oracle::OracleId};
use binius_field::{
	underlier::WithUnderlier, BinaryField, BinaryField16b, BinaryField1b, BinaryField64b,
	PackedField, TowerField,
};
use binius_macros::arith_expr;
use binius_maybe_rayon::iter::{
	IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator,
};
use binius_utils::bail;

use super::static_exp::static_exp_lookups;
use crate::builder::{types::F, ConstraintSystemBuilder};

pub fn mul<FExpBase>(
	builder: &mut ConstraintSystemBuilder,
	name: impl ToString,
	xin_bits: Vec<OracleId>,
	yin_bits: Vec<OracleId>,
) -> Result<Vec<OracleId>, anyhow::Error>
where
	FExpBase: TowerField,
	F: From<FExpBase>,
{
	let name = name.to_string();

	let log_rows = builder.log_rows([xin_bits.clone(), yin_bits.clone()].into_iter().flatten())?;

	// $g^x$
	let xin_exp_result_id =
		builder.add_committed(format!("{} xin_exp_result", name), log_rows, FExpBase::TOWER_LEVEL);

	// $(g^x)^y$
	let yin_exp_result_id =
		builder.add_committed(format!("{} yin_exp_result", name), log_rows, FExpBase::TOWER_LEVEL);

	// $g^{clow}$
	let cout_low_exp_result_id = builder.add_committed(
		format!("{} cout_low_exp_result", name),
		log_rows,
		FExpBase::TOWER_LEVEL,
	);

	// $(g^{2^{32}})^{chigh}$
	let cout_high_exp_result_id = builder.add_committed(
		format!("{} cout_high_exp_result", name),
		log_rows,
		FExpBase::TOWER_LEVEL,
	);

	let result_bits = xin_bits.len() + yin_bits.len();

	if result_bits > FExpBase::N_BITS {
		bail!(anyhow::anyhow!("FExpBase to small"));
	}

	let cout_bits = (0..result_bits)
		.map(|i| {
			builder.add_committed(
				format!("{} bit of {}", i, name),
				log_rows,
				BinaryField1b::TOWER_LEVEL,
			)
		})
		.collect::<Vec<_>>();

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

	// $(g^x)^y = g^{clow} * (g^{2^{32}})^{chigh}$
	builder.assert_zero(
		name,
		[
			yin_exp_result_id,
			cout_low_exp_result_id,
			cout_high_exp_result_id,
		],
		arith_expr!([yin, low, high] = low * high - yin).convert_field(),
	);

	let (cout_low_bits, cout_high_bits) = cout_bits.split_at(cout_bits.len() / 2);

	builder.add_exp(
		xin_bits,
		xin_exp_result_id,
		ExpBase::Constant(FExpBase::MULTIPLICATIVE_GENERATOR.into()),
		FExpBase::TOWER_LEVEL,
	);
	builder.add_exp(
		yin_bits,
		yin_exp_result_id,
		ExpBase::Dynamic(xin_exp_result_id),
		FExpBase::TOWER_LEVEL,
	);
	builder.add_exp(
		cout_low_bits.to_vec(),
		cout_low_exp_result_id,
		ExpBase::Constant(FExpBase::MULTIPLICATIVE_GENERATOR.into()),
		FExpBase::TOWER_LEVEL,
	);
	builder.add_exp(
		cout_high_bits.to_vec(),
		cout_high_exp_result_id,
		ExpBase::Constant(
			FExpBase::MULTIPLICATIVE_GENERATOR
				.pow(1 << cout_low_bits.len())
				.into(),
		),
		FExpBase::TOWER_LEVEL,
	);

	Ok(cout_bits)
}

pub fn u32_mul<const LOG_MAX_MULTIPLICITY: usize>(
	builder: &mut ConstraintSystemBuilder,
	name: impl ToString,
	xin: [OracleId; 2],
	yin_bits: [OracleId; 32],
) -> Result<Vec<OracleId>, anyhow::Error> {
	let log_rows = builder.log_rows(xin)?;

	let name = name.to_string();

	let [xin_low, xin_high] = xin;

	let (xin_low_exp_res, g) = static_exp_lookups::<LOG_MAX_MULTIPLICITY>(
		builder,
		"xin_low_exp_res",
		xin_low,
		BinaryField64b::MULTIPLICATIVE_GENERATOR,
		None,
	)?;

	let (xin_high_exp_res, g_16) = static_exp_lookups::<LOG_MAX_MULTIPLICITY>(
		builder,
		"xin_high_exp_res",
		xin_high,
		BinaryField64b::MULTIPLICATIVE_GENERATOR.pow(1 << 16),
		None,
	)?;

	let xin_exp_result_id =
		builder.add_committed("xin_exp_result_id", log_rows, BinaryField64b::TOWER_LEVEL);

	builder.assert_zero(
		"xin_exp_result_id assert_zero",
		[xin_low_exp_res, xin_high_exp_res, xin_exp_result_id],
		arith_expr!(
			[xin_low_exp_res, xin_high_exp_res, xin_exp_result_id] =
				xin_low_exp_res * xin_high_exp_res - xin_exp_result_id
		)
		.convert_field(),
	);

	if let Some(witness) = builder.witness() {
		let xin_low = witness.get::<BinaryField16b>(xin_low)?.as_slice::<u16>();

		let xin_high = witness.get::<BinaryField16b>(xin_high)?.as_slice::<u16>();

		let mut xin_exp_result_id = witness.new_column::<BinaryField64b>(xin_exp_result_id);
		let xin_exp_result_id = xin_exp_result_id.as_mut_slice::<u64>();
		xin_exp_result_id
			.par_iter_mut()
			.enumerate()
			.for_each(|(i, xin_exp_result)| {
				*xin_exp_result = BinaryField64b::MULTIPLICATIVE_GENERATOR
					.pow(xin_low[i] as u128 + (1 << 16) * xin_high[i] as u128)
					.to_underlier();
			});
	}

	let yin_exp_result_id = builder.add_committed(
		format!("{} yin_exp_result", name),
		log_rows,
		BinaryField64b::TOWER_LEVEL,
	);

	builder.add_exp(
		yin_bits.to_vec(),
		yin_exp_result_id,
		ExpBase::Dynamic(xin_exp_result_id),
		BinaryField64b::TOWER_LEVEL,
	);

	let cout: [OracleId; 4] =
		builder.add_committed_multiple("cout", log_rows, BinaryField16b::TOWER_LEVEL);

	if let Some(witness) = builder.witness() {
		let mut cout = cout.map(|id| witness.new_column::<BinaryField16b>(id));

		let mut cout = cout
			.iter_mut()
			.map(|cout| cout.as_mut_slice::<u16>())
			.collect::<Vec<_>>();

		let yin_columns = yin_bits
			.iter()
			.map(|&id| {
				witness
					.get::<BinaryField1b>(id)
					.map(|x| x.as_slice::<u32>())
			})
			.collect::<Result<Vec<_>, Error>>()?;

		let yin_numbers = columns_to_numbers(&yin_columns);

		let xin_low_number = witness.get::<BinaryField16b>(xin_low)?.as_slice::<u16>();

		let xin_high_number = witness.get::<BinaryField16b>(xin_high)?.as_slice::<u16>();

		cout.iter_mut().enumerate().for_each(|(j, cout)| {
			cout.par_iter_mut().enumerate().for_each(|(i, cout)| {
				let value = ((xin_low_number[i] as u32 + ((xin_high_number[i] as u32) << 16))
					as u64) * (yin_numbers[i] as u64);

				*cout = ((value >> (j * 16)) & 0xFFFF) as u16;
			});
		});
	}

	let cout_exp_result_id: [OracleId; 4] = array::from_fn(|i| {
		let g_table = match i {
			0 => Some(g),
			1 => Some(g_16),
			_ => None,
		};

		let (cout_exp_result_id, _) = static_exp_lookups::<LOG_MAX_MULTIPLICITY>(
			builder,
			format!("cout_exp_result_id {}", i),
			cout[i],
			BinaryField64b::MULTIPLICATIVE_GENERATOR.pow(1 << (16 * i)),
			g_table,
		)
		.unwrap();

		cout_exp_result_id
	});

	builder.assert_zero(
		name,
		[
			yin_exp_result_id,
			cout_exp_result_id[0],
			cout_exp_result_id[1],
			cout_exp_result_id[2],
			cout_exp_result_id[3],
		],
		arith_expr!(
			[yin, cout_0, cout_1, cout_2, cout_3] = cout_0 * cout_1 * cout_2 * cout_3 - yin
		)
		.convert_field(),
	);

	Ok(cout.to_vec())
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
	use binius_field::{BinaryField1b, BinaryField8b};
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

		mul::<BinaryField8b>(&mut builder, "test", in_a, in_b).unwrap();

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
