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
	use binius_field::{BinaryField1b, BinaryField8b, BinaryField32b};

	use crate::{
		builder::test_utils::test_circuit,
		lasso::{self, batch::LookupBatch},
		unconstrained::unconstrained,
	};

	#[test]
	fn test_lasso_u8add_carryfree_rejects_carry() {
		// TODO: Make this test 100% certain to pass instead of 2^14 bits of security from
		// randomness
		test_circuit(|builder| {
			let log_size = 14;
			let x_in = unconstrained::<BinaryField8b>(builder, "x", log_size)?;
			let y_in = unconstrained::<BinaryField8b>(builder, "y", log_size)?;
			let c_in = unconstrained::<BinaryField1b>(builder, "c", log_size)?;

			let lookup_t = super::add_carryfree_lookup(builder, "add cf table")?;
			let mut lookup_batch = LookupBatch::new([lookup_t]);
			let _sum_and_cout = lasso::u8add_carryfree(
				builder,
				&mut lookup_batch,
				"lasso_u8add",
				x_in,
				y_in,
				c_in,
				log_size,
			)?;
			lookup_batch.execute::<BinaryField32b>(builder)?;
			Ok(vec![])
		})
		.expect_err("Rejected overflowing add");
	}

	#[test]
	fn test_lasso_u8mul() {
		test_circuit(|builder| {
			let log_size = 10;

			let mult_a = unconstrained::<BinaryField8b>(builder, "mult_a", log_size)?;
			let mult_b = unconstrained::<BinaryField8b>(builder, "mult_b", log_size)?;

			let mul_lookup_table = super::mul_lookup(builder, "mul table")?;

			let mut lookup_batch = LookupBatch::new([mul_lookup_table]);

			let _product = lasso::u8mul(
				builder,
				&mut lookup_batch,
				"lasso_u8mul",
				mult_a,
				mult_b,
				1 << log_size,
			)?;

			lookup_batch.execute::<BinaryField32b>(builder)?;
			Ok(vec![])
		})
		.unwrap();
	}

	#[test]
	fn test_lasso_batched_u8mul() {
		test_circuit(|builder| {
			let log_size = 10;
			let mul_lookup_table = super::mul_lookup(builder, "mul table")?;

			let mut lookup_batch = LookupBatch::new([mul_lookup_table]);

			for _ in 0..10 {
				let mult_a = unconstrained::<BinaryField8b>(builder, "mult_a", log_size)?;
				let mult_b = unconstrained::<BinaryField8b>(builder, "mult_b", log_size)?;

				let _product = lasso::u8mul(
					builder,
					&mut lookup_batch,
					"lasso_u8mul",
					mult_a,
					mult_b,
					1 << log_size,
				)?;
			}

			lookup_batch.execute::<BinaryField32b>(builder)?;
			Ok(vec![])
		})
		.unwrap();
	}

	#[test]
	fn test_lasso_batched_u8mul_rejects() {
		test_circuit(|builder| {
			let log_size = 10;

			// We try to feed in the add table instead
			let mul_lookup_table = super::add_lookup(builder, "mul table")?;

			let mut lookup_batch = LookupBatch::new([mul_lookup_table]);

			// TODO?: Make this test fail 100% of the time, even though its almost impossible with
			// rng
			for _ in 0..10 {
				let mult_a = unconstrained::<BinaryField8b>(builder, "mult_a", log_size)?;
				let mult_b = unconstrained::<BinaryField8b>(builder, "mult_b", log_size)?;
				let _product = lasso::u8mul(
					builder,
					&mut lookup_batch,
					"lasso_u8mul",
					mult_a,
					mult_b,
					1 << log_size,
				)?;
			}

			lookup_batch.execute::<BinaryField32b>(builder)?;
			Ok(vec![])
		})
		.expect_err("Channels should be unbalanced");
	}
}
