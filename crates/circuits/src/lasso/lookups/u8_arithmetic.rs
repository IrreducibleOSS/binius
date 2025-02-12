// Copyright 2024-2025 Irreducible Inc.

use anyhow::Result;
use binius_core::oracle::OracleId;
use binius_field::{BinaryField32b, TowerField};

use crate::builder::ConstraintSystemBuilder;

type B32 = BinaryField32b;
const T_LOG_SIZE_MUL: usize = 16;
const T_LOG_SIZE_ADD: usize = 17;
const T_LOG_SIZE_DCI: usize = 10;

pub fn mul_lookup(
	builder: &mut ConstraintSystemBuilder,
	name: impl ToString + Clone,
) -> Result<OracleId, anyhow::Error> {
	builder.push_namespace(name);

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

pub fn add_lookup(
	builder: &mut ConstraintSystemBuilder,
	name: impl ToString + Clone,
) -> Result<OracleId, anyhow::Error> {
	builder.push_namespace(name);

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

pub fn add_carryfree_lookup(
	builder: &mut ConstraintSystemBuilder,
	name: impl ToString + Clone,
) -> Result<OracleId, anyhow::Error> {
	builder.push_namespace(name);

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

pub fn dci_lookup(
	builder: &mut ConstraintSystemBuilder,
	name: impl ToString + Clone,
) -> Result<OracleId, anyhow::Error> {
	builder.push_namespace(name);

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

#[cfg(test)]
mod tests {
	use binius_core::constraint_system::validate::validate_witness;
	use binius_field::{BinaryField1b, BinaryField32b, BinaryField8b};

	use crate::{
		builder::ConstraintSystemBuilder,
		lasso::{self, batch::LookupBatch},
		unconstrained::unconstrained,
	};

	#[test]
	fn test_lasso_u8add_carryfree_rejects_carry() {
		// TODO: Make this test 100% certain to pass instead of 2^14 bits of security from randomness
		let allocator = bumpalo::Bump::new();
		let mut builder = ConstraintSystemBuilder::new_with_witness(&allocator);
		let log_size = 14;
		let x_in = unconstrained::<BinaryField8b>(&mut builder, "x", log_size).unwrap();
		let y_in = unconstrained::<BinaryField8b>(&mut builder, "y", log_size).unwrap();
		let c_in = unconstrained::<BinaryField1b>(&mut builder, "c", log_size).unwrap();

		let lookup_t = super::add_carryfree_lookup(&mut builder, "add cf table").unwrap();
		let mut lookup_batch = LookupBatch::new([lookup_t]);
		let _sum_and_cout = lasso::u8add_carryfree(
			&mut builder,
			&mut lookup_batch,
			"lasso_u8add",
			x_in,
			y_in,
			c_in,
			log_size,
		)
		.unwrap();

		lookup_batch
			.execute::<BinaryField32b>(&mut builder)
			.unwrap();

		let witness = builder.take_witness().unwrap();
		let constraint_system = builder.build().unwrap();
		let boundaries = vec![];
		validate_witness(&constraint_system, &boundaries, &witness)
			.expect_err("Rejected overflowing add");
	}

	#[test]
	fn test_lasso_u8mul() {
		let allocator = bumpalo::Bump::new();
		let mut builder = ConstraintSystemBuilder::new_with_witness(&allocator);
		let log_size = 10;

		let mult_a = unconstrained::<BinaryField8b>(&mut builder, "mult_a", log_size).unwrap();
		let mult_b = unconstrained::<BinaryField8b>(&mut builder, "mult_b", log_size).unwrap();

		let mul_lookup_table = super::mul_lookup(&mut builder, "mul table").unwrap();

		let mut lookup_batch = LookupBatch::new([mul_lookup_table]);

		let _product = lasso::u8mul(
			&mut builder,
			&mut lookup_batch,
			"lasso_u8mul",
			mult_a,
			mult_b,
			1 << log_size,
		)
		.unwrap();

		lookup_batch
			.execute::<BinaryField32b>(&mut builder)
			.unwrap();

		let witness = builder.take_witness().unwrap();
		let constraint_system = builder.build().unwrap();
		let boundaries = vec![];
		validate_witness(&constraint_system, &boundaries, &witness).unwrap();
	}

	#[test]
	fn test_lasso_batched_u8mul() {
		let allocator = bumpalo::Bump::new();
		let mut builder = ConstraintSystemBuilder::new_with_witness(&allocator);
		let log_size = 10;
		let mul_lookup_table = super::mul_lookup(&mut builder, "mul table").unwrap();

		let mut lookup_batch = LookupBatch::new([mul_lookup_table]);

		for _ in 0..10 {
			let mult_a = unconstrained::<BinaryField8b>(&mut builder, "mult_a", log_size).unwrap();
			let mult_b = unconstrained::<BinaryField8b>(&mut builder, "mult_b", log_size).unwrap();

			let _product = lasso::u8mul(
				&mut builder,
				&mut lookup_batch,
				"lasso_u8mul",
				mult_a,
				mult_b,
				1 << log_size,
			)
			.unwrap();
		}

		lookup_batch
			.execute::<BinaryField32b>(&mut builder)
			.unwrap();

		let witness = builder.take_witness().unwrap();
		let constraint_system = builder.build().unwrap();
		let boundaries = vec![];
		validate_witness(&constraint_system, &boundaries, &witness).unwrap();
	}

	#[test]
	fn test_lasso_batched_u8mul_rejects() {
		let allocator = bumpalo::Bump::new();
		let mut builder = ConstraintSystemBuilder::new_with_witness(&allocator);
		let log_size = 10;

		// We try to feed in the add table instead
		let mul_lookup_table = super::add_lookup(&mut builder, "mul table").unwrap();

		let mut lookup_batch = LookupBatch::new([mul_lookup_table]);

		// TODO?: Make this test fail 100% of the time, even though its almost impossible with rng
		for _ in 0..10 {
			let mult_a = unconstrained::<BinaryField8b>(&mut builder, "mult_a", log_size).unwrap();
			let mult_b = unconstrained::<BinaryField8b>(&mut builder, "mult_b", log_size).unwrap();

			let _product = lasso::u8mul(
				&mut builder,
				&mut lookup_batch,
				"lasso_u8mul",
				mult_a,
				mult_b,
				1 << log_size,
			)
			.unwrap();
		}

		lookup_batch
			.execute::<BinaryField32b>(&mut builder)
			.unwrap();

		let witness = builder.take_witness().unwrap();
		let constraint_system = builder.build().unwrap();
		let boundaries = vec![];
		validate_witness(&constraint_system, &boundaries, &witness)
			.expect_err("Channels should be unbalanced");
	}
}
