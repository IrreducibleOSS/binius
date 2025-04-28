// Copyright 2025 Irreducible Inc.

//! Multiplication based on exponentiation.
//!
//! The core idea of this method is to verify the equality $a \cdot b = c$
//! by checking if $(g^a)^b = g^{clow} \cdot (g^{2^{len(clow)}})^{chigh}$,
//! where exponentiation proofs can be efficiently verified using the GKR exponentiation protocol.
//!
//! You can read more information in [Integer Multiplication in Binius](https://www.irreducible.com/posts/integer-multiplication-in-binius).

use std::array;

use anyhow::Error;
use binius_core::oracle::OracleId;
use binius_field::{
	as_packed_field::PackedType,
	packed::{get_packed_slice, set_packed_slice},
	BinaryField, BinaryField16b, BinaryField1b, BinaryField64b, Field, TowerField,
};
use binius_macros::arith_expr;
use binius_maybe_rayon::iter::{
	IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator,
};
use binius_utils::bail;
use itertools::izip;

use super::static_exp::u16_static_exp_lookups;
use crate::builder::{
	types::{F, U},
	ConstraintSystemBuilder,
};

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
		builder.add_committed(format!("{name} xin_exp_result"), log_rows, FExpBase::TOWER_LEVEL);

	// $(g^x)^y$
	let yin_exp_result_id =
		builder.add_committed(format!("{name} yin_exp_result"), log_rows, FExpBase::TOWER_LEVEL);

	// $g^{clow}$
	let cout_low_exp_result_id = builder.add_committed(
		format!("{name} cout_low_exp_result"),
		log_rows,
		FExpBase::TOWER_LEVEL,
	);

	// $(g^{2^{len(clow)}})^{chigh}$
	let cout_high_exp_result_id = builder.add_committed(
		format!("{name} cout_high_exp_result"),
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
				format!("{i} bit of {name}"),
				log_rows,
				BinaryField1b::TOWER_LEVEL,
			)
		})
		.collect::<Vec<_>>();

	if let Some(witness) = builder.witness() {
		let xin_columns = xin_bits
			.iter()
			.map(|&id| witness.get::<BinaryField1b>(id).map(|x| x.packed()))
			.collect::<Result<Vec<_>, Error>>()?;

		let yin_columns = yin_bits
			.iter()
			.map(|&id| witness.get::<BinaryField1b>(id).map(|x| x.packed()))
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

		let mut cout_columns_u8 = cout_columns
			.iter_mut()
			.map(|column| column.packed())
			.collect::<Vec<_>>();

		numbers_to_columns(&result, &mut cout_columns_u8);
	}

	// Handling special case when $x == 0$ $y == 0$ $c == 2^{2 \cdot n} -1$
	builder.assert_zero(
		name.clone(),
		[xin_bits[0], yin_bits[0], cout_bits[0]],
		arith_expr!([xin, yin, cout] = xin * yin - cout).convert_field(),
	);

	// $(g^x)^y = g^{clow} * (g^{2^{len(clow)}})^{chigh}$
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

	builder.add_static_exp(
		xin_bits,
		xin_exp_result_id,
		FExpBase::MULTIPLICATIVE_GENERATOR.into(),
		FExpBase::TOWER_LEVEL,
	);
	builder.add_dynamic_exp(yin_bits, yin_exp_result_id, xin_exp_result_id);
	builder.add_static_exp(
		cout_low_bits.to_vec(),
		cout_low_exp_result_id,
		FExpBase::MULTIPLICATIVE_GENERATOR.into(),
		FExpBase::TOWER_LEVEL,
	);
	builder.add_static_exp(
		cout_high_bits.to_vec(),
		cout_high_exp_result_id,
		exp_pow2(FExpBase::MULTIPLICATIVE_GENERATOR, cout_low_bits.len()).into(),
		FExpBase::TOWER_LEVEL,
	);

	Ok(cout_bits)
}

/// u32 Multiplication based on plain lookups for static exponentiation
/// and gkr_exp for dynamic exponentiation
///
/// The core idea of this method is to verify the equality $x \cdot y = c$
/// by checking if
///
/// $(g^{xlow} \cdot (g^{2^{16}})^{xhigh})^y = \prod_{i=0}^{3} (g^{2^{(16 \cdot i)}})^{c_i} $,
/// where $c_i$ is a $i$ 16-bit
pub fn u32_mul<const LOG_MAX_MULTIPLICITY: usize>(
	builder: &mut ConstraintSystemBuilder,
	name: impl ToString,
	xin_bits: [OracleId; 32],
	yin_bits: [OracleId; 32],
) -> Result<[OracleId; 64], anyhow::Error> {
	let log_rows = builder.log_rows(xin_bits)?;

	let name = name.to_string();

	let [xin_low, xin_high] = array::from_fn(|i| {
		let bits: [(usize, F); 16] =
			array::from_fn(|j| (xin_bits[16 * i + j], <F as TowerField>::basis(0, j).unwrap()));

		builder
			.add_linear_combination("xin_low", log_rows, bits)
			.unwrap()
	});

	if let Some(witness) = builder.witness() {
		let xin_columns = xin_bits
			.iter()
			.map(|&id| witness.get::<BinaryField1b>(id).map(|x| x.packed()))
			.collect::<Result<Vec<_>, Error>>()?;

		let xin_numbers = columns_to_numbers(&xin_columns);

		let mut xin_low = witness.new_column::<BinaryField16b>(xin_low);
		let xin_low = xin_low.as_mut_slice::<u16>();

		let mut xin_high = witness.new_column::<BinaryField16b>(xin_high);
		let xin_high = xin_high.as_mut_slice::<u16>();

		izip!(xin_numbers, xin_low, xin_high).for_each(|(xin, low, high)| {
			*low = (xin & 0xFFFF) as u16;
			*high = (xin >> 16) as u16;
		});
	}

	//$g^{xlow}$
	let (xin_low_exp_res_id, g) = u16_static_exp_lookups::<LOG_MAX_MULTIPLICITY>(
		builder,
		"xin_low_exp_res",
		xin_low,
		BinaryField64b::MULTIPLICATIVE_GENERATOR,
		None,
	)?;

	//$(g^{2^{16}})^{xhigh}$
	let (xin_high_exp_res_id, g_16) = u16_static_exp_lookups::<LOG_MAX_MULTIPLICITY>(
		builder,
		"xin_high_exp_res",
		xin_high,
		exp_pow2(BinaryField64b::MULTIPLICATIVE_GENERATOR, 16),
		None,
	)?;

	//$g^{xin}$
	let xin_exp_res_id =
		builder.add_committed("xin_exp_result", log_rows, BinaryField64b::TOWER_LEVEL);

	builder.assert_zero(
		"xin_exp_res_id zerocheck",
		[xin_low_exp_res_id, xin_high_exp_res_id, xin_exp_res_id],
		arith_expr!(
			[xin_low_exp_res, xin_high_exp_res, xin_exp_result_id] =
				xin_low_exp_res * xin_high_exp_res - xin_exp_result_id
		)
		.convert_field(),
	);

	if let Some(witness) = builder.witness() {
		let xin_low_exp_res = witness
			.get::<BinaryField64b>(xin_low_exp_res_id)?
			.as_slice::<BinaryField64b>();

		let xin_high_exp_res = witness
			.get::<BinaryField64b>(xin_high_exp_res_id)?
			.as_slice::<BinaryField64b>();

		let mut xin_exp_res = witness.new_column::<BinaryField64b>(xin_exp_res_id);
		let xin_exp_res = xin_exp_res.as_mut_slice::<BinaryField64b>();
		xin_exp_res
			.par_iter_mut()
			.enumerate()
			.for_each(|(i, xin_exp_res)| {
				*xin_exp_res = xin_low_exp_res[i] * xin_high_exp_res[i];
			});
	}

	//$(g^{x})^{y}$
	let yin_exp_result_id = builder.add_committed(
		format!("{name} yin_exp_result"),
		log_rows,
		BinaryField64b::TOWER_LEVEL,
	);

	builder.add_dynamic_exp(yin_bits.to_vec(), yin_exp_result_id, xin_exp_res_id);

	let cout_bits: [OracleId; 64] =
		builder.add_committed_multiple("cout_bits", log_rows, BinaryField1b::TOWER_LEVEL);

	let cout: [OracleId; 4] = array::from_fn(|i| {
		let bits: [(usize, F); 16] =
			array::from_fn(|j| (cout_bits[16 * i + j], <F as TowerField>::basis(0, j).unwrap()));

		builder
			.add_linear_combination("cout 16b", log_rows, bits)
			.unwrap()
	});

	if let Some(witness) = builder.witness() {
		let xin_columns = xin_bits
			.iter()
			.map(|&id| witness.get::<BinaryField1b>(id).map(|x| x.packed()))
			.collect::<Result<Vec<_>, Error>>()?;

		let yin_columns = yin_bits
			.iter()
			.map(|&id| witness.get::<BinaryField1b>(id).map(|x| x.packed()))
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

		let mut cout_columns = cout_columns
			.iter_mut()
			.map(|column| column.packed())
			.collect::<Vec<_>>();

		numbers_to_columns(&result, &mut cout_columns);

		let mut cout = cout.map(|id| witness.new_column::<BinaryField16b>(id));

		let mut cout = cout
			.iter_mut()
			.map(|cout| cout.as_mut_slice::<u16>())
			.collect::<Vec<_>>();

		cout.iter_mut().enumerate().for_each(|(j, cout)| {
			cout.par_iter_mut().enumerate().for_each(|(i, cout)| {
				let value = result[i];

				*cout = ((value >> (j * 16)) & 0xFFFF) as u16;
			});
		});
	}

	//$(g^{2^{(16 \cdot i)}})^{c_i}$ where $c_i$ is a $i$ 16-bit
	let cout_exp_res_id = (0..4)
		.map(|i| {
			let g_table = match i {
				0 => Some(g),
				1 => Some(g_16),
				_ => None,
			};

			u16_static_exp_lookups::<LOG_MAX_MULTIPLICITY>(
				builder,
				format!("cout_exp_result_id {i}"),
				cout[i],
				exp_pow2(BinaryField64b::MULTIPLICATIVE_GENERATOR, 16 * i),
				g_table,
			)
			.map(|res| res.0)
		})
		.collect::<Result<Vec<_>, anyhow::Error>>()?;

	// Handling special case when $x == 0$ $y == 0$ $c == 2^{2 \cdot n} -1$
	builder.assert_zero(
		name.clone(),
		[xin_bits[0], yin_bits[0], cout_bits[0]],
		arith_expr!([x, y, c] = x * y - c).convert_field(),
	);

	// $(g^{xlow} \cdot (g^{2^{16}})^{xhigh})^y = \prod_{i=0}^{3} (g^{2^{(16 \cdot i)}})^{c_i} $
	builder.assert_zero(
		name,
		[
			yin_exp_result_id,
			cout_exp_res_id[0],
			cout_exp_res_id[1],
			cout_exp_res_id[2],
			cout_exp_res_id[3],
		],
		arith_expr!(
			[yin, cout_0, cout_1, cout_2, cout_3] = cout_0 * cout_1 * cout_2 * cout_3 - yin
		)
		.convert_field(),
	);

	Ok(cout_bits)
}

fn exp_pow2<F: BinaryField>(mut g: F, log_exp: usize) -> F {
	for _ in 0..log_exp {
		g *= g
	}
	g
}

fn columns_to_numbers(columns: &[&[PackedType<U, BinaryField1b>]]) -> Vec<u128> {
	let width = PackedType::<U, BinaryField1b>::WIDTH;
	let mut numbers: Vec<u128> = vec![0; columns.first().map(|c| c.len() * width).unwrap_or(0)];

	for (bit, column) in columns.iter().enumerate() {
		numbers.par_iter_mut().enumerate().for_each(|(i, number)| {
			if get_packed_slice(column, i) == BinaryField1b::ONE {
				*number |= 1 << bit;
			}
		});
	}
	numbers
}

fn numbers_to_columns(numbers: &[u128], columns: &mut [&mut [PackedType<U, BinaryField1b>]]) {
	columns
		.par_iter_mut()
		.enumerate()
		.for_each(|(bit, column)| {
			for (i, number) in numbers.iter().enumerate() {
				if (number >> bit) & 1 == 1 {
					set_packed_slice(column, i, BinaryField1b::ONE);
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
	use binius_hash::groestl::{Groestl256, Groestl256ByteCompression};

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
				unconstrained::<BinaryField1b>(&mut builder, format!("in_a_{i}"), log_n_muls)
					.unwrap()
			})
			.collect::<Vec<_>>();
		let in_b = (0..2)
			.map(|i| {
				unconstrained::<BinaryField1b>(&mut builder, format!("in_b_{i}"), log_n_muls)
					.unwrap()
			})
			.collect::<Vec<_>>();

		mul::<BinaryField8b>(&mut builder, "test", in_a, in_b).unwrap();

		let witness = builder
			.take_witness()
			.expect("builder created with witness");

		let constraint_system = builder.build().unwrap();

		let backend = make_portable_backend();

		let proof = constraint_system::prove::<
			U,
			CanonicalTowerFamily,
			Groestl256,
			Groestl256ByteCompression,
			HasherChallenger<Groestl256>,
			_,
		>(&constraint_system, 1, 10, &[], witness, &backend)
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
