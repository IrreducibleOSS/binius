// Copyright 2024-2025 Irreducible Inc.

use alloy_primitives::U512;
use anyhow::Result;
use binius_core::{oracle::OracleId, transparent::constant::Constant};
use binius_field::{
	as_packed_field::{PackScalar, PackedType},
	tower_levels::TowerLevel,
	underlier::{UnderlierType, WithUnderlier},
	BinaryField, BinaryField16b, BinaryField1b, BinaryField32b, BinaryField8b, ExtensionField,
	PackedFieldIndexable, TowerField,
};
use binius_macros::arith_expr;
use bytemuck::Pod;

use super::{byte_sliced_add_carryfree, byte_sliced_mul};
use crate::{
	builder::ConstraintSystemBuilder,
	lasso::{
		batch::LookupBatch,
		lookups::u8_arithmetic::{add_carryfree_lookup, add_lookup, dci_lookup, mul_lookup},
	},
};

type B1 = BinaryField1b;
type B8 = BinaryField8b;
type B16 = BinaryField16b;
type B32 = BinaryField32b;

#[allow(clippy::too_many_arguments)]
pub fn byte_sliced_modular_mul<
	U,
	F,
	LevelIn: TowerLevel<OracleId>,
	LevelOut: TowerLevel<OracleId, Base = LevelIn>,
>(
	builder: &mut ConstraintSystemBuilder<U, F>,
	name: impl ToString,
	mult_a: &LevelIn::Data,
	mult_b: &LevelIn::Data,
	modulus_input: &[u8],
	log_size: usize,
	zero_byte_oracle: OracleId,
	zero_carry_oracle: OracleId,
) -> Result<LevelIn::Data, anyhow::Error>
where
	U: Pod
		+ UnderlierType
		+ PackScalar<B1>
		+ PackScalar<B8>
		+ PackScalar<B16>
		+ PackScalar<B32>
		+ PackScalar<F>,
	PackedType<U, B8>: PackedFieldIndexable,
	PackedType<U, B16>: PackedFieldIndexable,
	PackedType<U, B32>: PackedFieldIndexable,
	F: TowerField + BinaryField + ExtensionField<B8> + ExtensionField<B16> + ExtensionField<B32>,
	<F as WithUnderlier>::Underlier: From<u8>,
{
	builder.push_namespace(name);

	let lookup_t_mul = mul_lookup(builder, "mul table")?;
	let lookup_t_add = add_lookup(builder, "add table")?;
	let lookup_t_add_carryfree = add_carryfree_lookup(builder, "add cf table")?;

	// The double conditional increment wont be used if we're at the base of the tower
	let lookup_t_dci = if LevelIn::WIDTH == 1 {
		usize::MAX
	} else {
		dci_lookup(builder, "dci table")?
	};

	let mut lookup_batch_mul = LookupBatch::new(lookup_t_mul);
	let mut lookup_batch_add = LookupBatch::new(lookup_t_add);
	let mut lookup_batch_add_carryfree = LookupBatch::new(lookup_t_add_carryfree);

	// This batch WILL NOT get executed if we are instantiating it for 8b mul
	let mut lookup_batch_dci = LookupBatch::new(lookup_t_dci);

	let mut quotient = LevelIn::default();
	let mut remainder = LevelIn::default();
	let mut modulus = LevelIn::default();

	for byte_idx in 0..LevelIn::WIDTH {
		quotient[byte_idx] = builder.add_committed("quotient", log_size, B8::TOWER_LEVEL);
		remainder[byte_idx] = builder.add_committed("remainder", log_size, B8::TOWER_LEVEL);
		modulus[byte_idx] = builder.add_transparent(
			"modulus",
			Constant::new(
				log_size,
				F::from_underlier(<u8 as Into<F::Underlier>>::into(modulus_input[byte_idx])),
			),
		)?;
	}

	let ab = byte_sliced_mul::<_, _, LevelIn, LevelOut>(
		builder,
		"ab",
		mult_a,
		mult_b,
		log_size,
		zero_carry_oracle,
		&mut lookup_batch_mul,
		&mut lookup_batch_add,
		&mut lookup_batch_dci,
	)?;

	if let Some(witness) = builder.witness() {
		let ab_bytes_as_u8: Vec<_> = (0..LevelOut::WIDTH)
			.map(|this_byte_idx| {
				let this_byte_oracle = ab[this_byte_idx];
				witness
					.get::<B8>(this_byte_oracle)
					.unwrap()
					.as_slice::<u8>()
			})
			.collect();

		let mut quotient: Vec<_> = (0..LevelIn::WIDTH)
			.map(|this_byte_idx| {
				let this_byte_oracle = quotient[this_byte_idx];
				witness.new_column::<B8>(this_byte_oracle)
			})
			.collect();

		let mut remainder: Vec<_> = (0..LevelIn::WIDTH)
			.map(|this_byte_idx| {
				let this_byte_oracle: usize = remainder[this_byte_idx];
				witness.new_column::<B8>(this_byte_oracle)
			})
			.collect();

		let mut modulus: Vec<_> = (0..LevelIn::WIDTH)
			.map(|this_byte_idx| {
				let this_byte_oracle = modulus[this_byte_idx];
				witness.new_column::<B8>(this_byte_oracle)
			})
			.collect();

		let mut modulus_u512 = U512::ZERO;

		for (byte_idx, modulus_byte_column) in modulus.iter_mut().enumerate() {
			let modulus_byte_column_u8 = modulus_byte_column.as_mut_slice::<u8>();
			modulus_u512 |= U512::from(modulus_input[byte_idx]) << (8 * byte_idx);
			modulus_byte_column_u8.fill(modulus_input[byte_idx]);
		}

		for row_idx in 0..1 << log_size {
			let mut ab_u512 = U512::ZERO;
			for (byte_idx, ab_byte_column) in ab_bytes_as_u8.iter().enumerate() {
				ab_u512 |= U512::from(ab_byte_column[row_idx]) << (8 * byte_idx);
			}

			let quotient_u512 = ab_u512 / modulus_u512;
			let remainder_u512 = ab_u512 % modulus_u512;

			for (byte_idx, quotient_byte_column) in quotient.iter_mut().enumerate() {
				let quotient_byte_column_u8 = quotient_byte_column.as_mut_slice::<u8>();
				quotient_byte_column_u8[row_idx] = quotient_u512.byte(byte_idx);
			}

			for (byte_idx, remainder_byte_column) in remainder.iter_mut().enumerate() {
				let remainder_byte_column_u8 = remainder_byte_column.as_mut_slice::<u8>();
				remainder_byte_column_u8[row_idx] = remainder_u512.byte(byte_idx);
			}
		}
	}

	let qm = byte_sliced_mul::<_, _, LevelIn, LevelOut>(
		builder,
		"qm",
		&quotient,
		&modulus,
		log_size,
		zero_carry_oracle,
		&mut lookup_batch_mul,
		&mut lookup_batch_add,
		&mut lookup_batch_dci,
	)?;

	let mut repeating_zero = LevelIn::default();
	for byte_idx in 0..LevelIn::WIDTH {
		repeating_zero[byte_idx] = zero_byte_oracle;
	}

	let qm_plus_r = byte_sliced_add_carryfree::<_, _, LevelOut>(
		builder,
		"hi*lo",
		&qm,
		&LevelOut::join(&remainder, &repeating_zero),
		zero_carry_oracle,
		log_size,
		&mut lookup_batch_add,
		&mut lookup_batch_add_carryfree,
	)?;

	lookup_batch_mul.execute::<_, _, BinaryField32b, BinaryField32b>(builder)?;
	lookup_batch_add.execute::<_, _, BinaryField32b, BinaryField32b>(builder)?;
	lookup_batch_add_carryfree.execute::<_, _, BinaryField32b, BinaryField32b>(builder)?;

	if LevelIn::WIDTH != 1 {
		lookup_batch_dci.execute::<_, _, BinaryField32b, BinaryField32b>(builder)?;
	}

	let consistency = arith_expr!([x, y] = x - y);

	for byte_idx in 0..LevelOut::WIDTH {
		builder.assert_zero(
			format!("byte_consistency_{byte_idx}"),
			[ab[byte_idx], qm_plus_r[byte_idx]],
			consistency.clone().convert_field(),
		);
	}

	builder.pop_namespace();
	Ok(remainder)
}
