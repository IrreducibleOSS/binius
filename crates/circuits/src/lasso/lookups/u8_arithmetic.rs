// Copyright 2024 Irreducible Inc.

use anyhow::Result;
use binius_core::oracle::OracleId;
use binius_field::{
	as_packed_field::{PackScalar, PackedType},
	underlier::UnderlierType,
	BinaryField, BinaryField16b, BinaryField32b, BinaryField8b, ExtensionField,
	PackedFieldIndexable, TowerField,
};
use bytemuck::Pod;

use crate::builder::ConstraintSystemBuilder;

type B8 = BinaryField8b;
type B16 = BinaryField16b;
type B32 = BinaryField32b;
const T_LOG_SIZE_MUL: usize = 16;
const T_LOG_SIZE_ADD: usize = 17;
const T_LOG_SIZE_DCI: usize = 10;

pub fn mul_lookup<U, F>(
	builder: &mut ConstraintSystemBuilder<U, F>,
	name: impl ToString + Clone,
) -> Result<OracleId, anyhow::Error>
where
	U: Pod + UnderlierType + PackScalar<B8> + PackScalar<B16> + PackScalar<B32> + PackScalar<F>,
	PackedType<U, B8>: PackedFieldIndexable,
	PackedType<U, B16>: PackedFieldIndexable,
	PackedType<U, B32>: PackedFieldIndexable,
	F: TowerField + BinaryField + ExtensionField<B8> + ExtensionField<B16> + ExtensionField<B32>,
{
	builder.push_namespace(name.clone());

	let lookup_t = builder.add_committed("lookup_t", T_LOG_SIZE_MUL, B32::TOWER_LEVEL);

	if let Some(witness) = builder.witness() {
		let mut lookup_t = witness.new_column::<B32>(lookup_t);

		let lookup_t_u32 = lookup_t.as_mut_slice::<u32>();

		for (i, lookup_t) in lookup_t_u32.iter_mut().enumerate() {
			let a_int = (i >> 8) & 0xff;
			let b_int = i & 0xff;
			let ab_product = a_int * b_int;
			let lookup_index = a_int << 8 | b_int;
			assert_eq!(lookup_index, i);
			*lookup_t = (lookup_index << 16 | ab_product) as u32;
		}
	}

	builder.pop_namespace();
	Ok(lookup_t)
}

pub fn add_lookup<U, F>(
	builder: &mut ConstraintSystemBuilder<U, F>,
	name: impl ToString + Clone,
) -> Result<OracleId, anyhow::Error>
where
	U: Pod + UnderlierType + PackScalar<B8> + PackScalar<B16> + PackScalar<B32> + PackScalar<F>,
	PackedType<U, B8>: PackedFieldIndexable,
	PackedType<U, B16>: PackedFieldIndexable,
	PackedType<U, B32>: PackedFieldIndexable,
	F: TowerField + BinaryField + ExtensionField<B8> + ExtensionField<B16> + ExtensionField<B32>,
{
	builder.push_namespace(name.clone());

	let lookup_t = builder.add_committed("lookup_t", T_LOG_SIZE_ADD, B32::TOWER_LEVEL);

	if let Some(witness) = builder.witness() {
		let mut lookup_t = witness.new_column::<B32>(lookup_t);

		let lookup_t_u32 = lookup_t.as_mut_slice::<u32>();

		for carry_in_usize in 0..(1 << 1) {
			for x_in_usize in 0..(1 << 8) {
				for y_in_usize in 0..(1 << 8) {
					let lookup_index = (carry_in_usize << 16) | (x_in_usize << 8) | y_in_usize;
					let xy_sum_with_carry_out = x_in_usize + y_in_usize + carry_in_usize;
					let xy_sum_usize = xy_sum_with_carry_out & 0xff;
					let carry_out_usize = xy_sum_with_carry_out >> 8;
					let lookup_value = (carry_in_usize << 25)
						| (carry_out_usize << 24)
						| (x_in_usize << 16)
						| (y_in_usize << 8)
						| xy_sum_usize;
					lookup_t_u32[lookup_index] = lookup_value as u32;
				}
			}
		}
	}

	builder.pop_namespace();
	Ok(lookup_t)
}

pub fn add_carryfree_lookup<U, F>(
	builder: &mut ConstraintSystemBuilder<U, F>,
	name: impl ToString + Clone,
) -> Result<OracleId, anyhow::Error>
where
	U: Pod + UnderlierType + PackScalar<B8> + PackScalar<B16> + PackScalar<B32> + PackScalar<F>,
	PackedType<U, B8>: PackedFieldIndexable,
	PackedType<U, B16>: PackedFieldIndexable,
	PackedType<U, B32>: PackedFieldIndexable,
	F: TowerField + BinaryField + ExtensionField<B8> + ExtensionField<B16> + ExtensionField<B32>,
{
	builder.push_namespace(name.clone());

	let lookup_t = builder.add_committed("lookup_t", T_LOG_SIZE_ADD, B32::TOWER_LEVEL);

	if let Some(witness) = builder.witness() {
		let mut lookup_t = witness.new_column::<B32>(lookup_t);

		let lookup_t_u32 = lookup_t.as_mut_slice::<u32>();

		for carry_in_usize in 0..(1 << 1) {
			for x_in_usize in 0..(1 << 8) {
				for y_in_usize in 0..(1 << 8) {
					let lookup_index = (carry_in_usize << 16) | (x_in_usize << 8) | y_in_usize;
					let xy_sum_usize = x_in_usize + y_in_usize + carry_in_usize;

					// Make it impossible to add numbers resulting in a carry
					let lookup_value = if xy_sum_usize <= 0xff {
						(carry_in_usize << 24)
							| (x_in_usize << 16) | (y_in_usize << 8)
							| xy_sum_usize
					} else {
						0
					};
					lookup_t_u32[lookup_index] = lookup_value as u32;
				}
			}
		}
	}

	builder.pop_namespace();
	Ok(lookup_t)
}

pub fn dci_lookup<U, F>(
	builder: &mut ConstraintSystemBuilder<U, F>,
	name: impl ToString + Clone,
) -> Result<OracleId, anyhow::Error>
where
	U: Pod + UnderlierType + PackScalar<B8> + PackScalar<B16> + PackScalar<B32> + PackScalar<F>,
	PackedType<U, B8>: PackedFieldIndexable,
	PackedType<U, B16>: PackedFieldIndexable,
	PackedType<U, B32>: PackedFieldIndexable,
	F: TowerField + BinaryField + ExtensionField<B8> + ExtensionField<B16> + ExtensionField<B32>,
{
	builder.push_namespace(name.clone());

	let lookup_t = builder.add_committed("lookup_t", T_LOG_SIZE_DCI, B32::TOWER_LEVEL);

	if let Some(witness) = builder.witness() {
		let mut lookup_t = witness.new_column::<B32>(lookup_t);

		let lookup_t_u32 = lookup_t.as_mut_slice::<u32>();

		for first_carry_in_usize in 0..(1 << 1) {
			for second_carry_in_usize in 0..(1 << 1) {
				for x_in_usize in 0..(1 << 8) {
					let lookup_index =
						(first_carry_in_usize << 9) | (second_carry_in_usize << 8) | x_in_usize;
					let sum_with_carry_out =
						x_in_usize + first_carry_in_usize + second_carry_in_usize;
					let sum_usize = sum_with_carry_out & 0xff;
					let carry_out_usize = sum_with_carry_out >> 8;
					let lookup_value = (first_carry_in_usize << 18)
						| (second_carry_in_usize << 17)
						| (carry_out_usize << 16)
						| (x_in_usize << 8)
						| sum_usize;
					lookup_t_u32[lookup_index] = lookup_value as u32;
				}
			}
		}
	}

	builder.pop_namespace();
	Ok(lookup_t)
}
