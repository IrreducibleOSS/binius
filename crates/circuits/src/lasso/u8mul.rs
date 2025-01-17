// Copyright 2024-2025 Irreducible Inc.

use anyhow::{ensure, Result};
use binius_core::oracle::OracleId;
use binius_field::{
	as_packed_field::{PackScalar, PackedType},
	underlier::UnderlierType,
	BinaryField, BinaryField16b, BinaryField32b, BinaryField8b, ExtensionField,
	PackedFieldIndexable, TowerField,
};
use bytemuck::Pod;
use itertools::izip;

use super::batch::LookupBatch;
use crate::builder::ConstraintSystemBuilder;

type B8 = BinaryField8b;
type B16 = BinaryField16b;
type B32 = BinaryField32b;

pub fn u8mul_bytesliced<U, F>(
	builder: &mut ConstraintSystemBuilder<U, F>,
	lookup_batch: &mut LookupBatch,
	name: impl ToString + Clone,
	mult_a: OracleId,
	mult_b: OracleId,
	n_multiplications: usize,
) -> Result<[OracleId; 2], anyhow::Error>
where
	U: Pod + UnderlierType + PackScalar<B8> + PackScalar<B16> + PackScalar<B32> + PackScalar<F>,
	PackedType<U, B8>: PackedFieldIndexable,
	PackedType<U, B16>: PackedFieldIndexable,
	PackedType<U, B32>: PackedFieldIndexable,
	F: TowerField + BinaryField + ExtensionField<B8> + ExtensionField<B16> + ExtensionField<B32>,
{
	builder.push_namespace(name.clone());
	let log_rows = builder.log_rows([mult_a, mult_b])?;
	let product = builder.add_committed_multiple("product", log_rows, B8::TOWER_LEVEL);

	let lookup_u = builder.add_linear_combination(
		"lookup_u",
		log_rows,
		[
			(mult_a, <F as TowerField>::basis(3, 3)?),
			(mult_b, <F as TowerField>::basis(3, 2)?),
			(product[1], <F as TowerField>::basis(3, 1)?),
			(product[0], <F as TowerField>::basis(3, 0)?),
		],
	)?;

	let mut u_to_t_mapping = Vec::new();

	if let Some(witness) = builder.witness() {
		let mut product_low_witness = witness.new_column::<B8>(product[0]);
		let mut product_high_witness = witness.new_column::<B8>(product[1]);
		let mut lookup_u_witness = witness.new_column::<B32>(lookup_u);
		let mut u_to_t_mapping_witness = vec![0; 1 << log_rows];

		let mult_a_ints = witness.get::<B8>(mult_a)?.as_slice::<u8>();
		let mult_b_ints = witness.get::<B8>(mult_b)?.as_slice::<u8>();

		let product_low_u8 = product_low_witness.as_mut_slice::<u8>();
		let product_high_u8 = product_high_witness.as_mut_slice::<u8>();
		let lookup_u_u32 = lookup_u_witness.as_mut_slice::<u32>();

		for (a, b, lookup_u, product_low, product_high, u_to_t) in izip!(
			mult_a_ints,
			mult_b_ints,
			lookup_u_u32.iter_mut(),
			product_low_u8.iter_mut(),
			product_high_u8.iter_mut(),
			u_to_t_mapping_witness.iter_mut()
		) {
			let a_int = *a as usize;
			let b_int = *b as usize;
			let ab_product = a_int * b_int;
			let lookup_index = a_int << 8 | b_int;
			*lookup_u = (lookup_index << 16 | ab_product) as u32;

			*product_high = (ab_product >> 8) as u8;
			*product_low = (ab_product & 0xff) as u8;

			*u_to_t = lookup_index;
		}

		u_to_t_mapping = u_to_t_mapping_witness;
	}

	lookup_batch.add([lookup_u], u_to_t_mapping, n_multiplications);

	builder.pop_namespace();
	Ok(product)
}

pub fn u8mul<U, F>(
	builder: &mut ConstraintSystemBuilder<U, F>,
	lookup_batch: &mut LookupBatch,
	name: impl ToString + Clone,
	mult_a: OracleId,
	mult_b: OracleId,
	n_multiplications: usize,
) -> Result<OracleId, anyhow::Error>
where
	U: Pod + UnderlierType + PackScalar<B8> + PackScalar<B16> + PackScalar<B32> + PackScalar<F>,
	PackedType<U, B8>: PackedFieldIndexable,
	PackedType<U, B16>: PackedFieldIndexable,
	PackedType<U, B32>: PackedFieldIndexable,
	F: TowerField + BinaryField + ExtensionField<B8> + ExtensionField<B16> + ExtensionField<B32>,
{
	builder.push_namespace(name.clone());

	let product_bytesliced =
		u8mul_bytesliced(builder, lookup_batch, name, mult_a, mult_b, n_multiplications)?;
	let log_rows = builder.log_rows(product_bytesliced)?;
	ensure!(n_multiplications <= 1 << log_rows);

	let product = builder.add_linear_combination(
		"bytes summed",
		log_rows,
		[
			(product_bytesliced[0], <F as TowerField>::basis(3, 0)?),
			(product_bytesliced[1], <F as TowerField>::basis(3, 1)?),
		],
	)?;

	if let Some(witness) = builder.witness() {
		let product_low_witness = witness.get::<B8>(product_bytesliced[0])?;
		let product_high_witness = witness.get::<B8>(product_bytesliced[1])?;

		let mut product_witness = witness.new_column::<B16>(product);

		let product_low_u8 = product_low_witness.as_slice::<u8>();
		let product_high_u8 = product_high_witness.as_slice::<u8>();

		let product_u16 = product_witness.as_mut_slice::<u16>();

		for (row_idx, row_product) in product_u16.iter_mut().enumerate() {
			*row_product = (product_high_u8[row_idx] as u16) << 8 | product_low_u8[row_idx] as u16;
		}
	}

	builder.pop_namespace();
	Ok(product)
}
