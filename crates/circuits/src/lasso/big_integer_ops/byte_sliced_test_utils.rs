// Copyright 2024 Irreducible Inc.

use std::{array, fmt::Debug};

use alloy_primitives::U512;
use binius_core::{constraint_system::validate::validate_witness, oracle::OracleId};
use binius_field::{
	arch::OptimalUnderlier, tower_levels::TowerLevel, BinaryField128b, BinaryField1b,
	BinaryField32b, BinaryField8b, Field, TowerField,
};
use rand::{rngs::ThreadRng, thread_rng, Rng};

use super::{
	byte_sliced_add, byte_sliced_add_carryfree, byte_sliced_double_conditional_increment,
	byte_sliced_modular_mul, byte_sliced_mul,
};
use crate::{
	builder::ConstraintSystemBuilder,
	lasso::{
		batch::LookupBatch,
		lookups::u8_arithmetic::{add_carryfree_lookup, add_lookup, dci_lookup, mul_lookup},
	},
	transparent,
	unconstrained::unconstrained,
};

type B8 = BinaryField8b;
type B32 = BinaryField32b;

pub fn random_u512(rng: &mut ThreadRng) -> U512 {
	let limbs = array::from_fn(|_| rng.gen());
	U512::from_limbs(limbs)
}

pub fn test_bytesliced_add<const WIDTH: usize, TL>()
where
	TL: TowerLevel<OracleId, Data = [OracleId; WIDTH]>,
{
	type U = OptimalUnderlier;
	type F = BinaryField128b;
	let allocator = bumpalo::Bump::new();
	let mut builder = ConstraintSystemBuilder::<U, F>::new_with_witness(&allocator);
	let log_size = 14;

	let x_in = array::from_fn(|_| {
		unconstrained::<_, _, BinaryField8b>(&mut builder, "x", log_size).unwrap()
	});
	let y_in = array::from_fn(|_| {
		unconstrained::<_, _, BinaryField8b>(&mut builder, "y", log_size).unwrap()
	});
	let c_in = unconstrained::<_, _, BinaryField1b>(&mut builder, "cin first", log_size).unwrap();

	let lookup_t_add = add_lookup(&mut builder, "add table").unwrap();

	let mut lookup_batch_add = LookupBatch::new(lookup_t_add);
	let _sum_and_cout = byte_sliced_add::<_, _, TL>(
		&mut builder,
		"lasso_bytesliced_add",
		&x_in,
		&y_in,
		c_in,
		log_size,
		&mut lookup_batch_add,
	)
	.unwrap();

	lookup_batch_add
		.execute::<_, _, B32, B32>(&mut builder)
		.unwrap();

	let witness = builder.take_witness().unwrap();
	let constraint_system = builder.build().unwrap();
	let boundaries = vec![];
	validate_witness(&constraint_system, &boundaries, &witness).unwrap();
}

pub fn test_bytesliced_add_carryfree<const WIDTH: usize, TL>()
where
	TL: TowerLevel<OracleId, Data = [OracleId; WIDTH]>,
{
	type U = OptimalUnderlier;
	type F = BinaryField128b;
	let allocator = bumpalo::Bump::new();
	let mut builder = ConstraintSystemBuilder::<U, F>::new_with_witness(&allocator);
	let log_size = 14;
	let x_in = array::from_fn(|_| builder.add_committed("x", log_size, BinaryField8b::TOWER_LEVEL));
	let y_in = array::from_fn(|_| builder.add_committed("y", log_size, BinaryField8b::TOWER_LEVEL));
	let c_in = builder.add_committed("c", log_size, BinaryField1b::TOWER_LEVEL);

	if let Some(witness) = builder.witness() {
		let mut x_in: [_; WIDTH] =
			array::from_fn(|byte_idx| witness.new_column::<BinaryField8b>(x_in[byte_idx]));
		let mut y_in: [_; WIDTH] =
			array::from_fn(|byte_idx| witness.new_column::<BinaryField8b>(y_in[byte_idx]));
		let mut c_in = witness.new_column::<BinaryField1b>(c_in);

		let x_in_bytes_u8: [_; WIDTH] = x_in.each_mut().map(|col| col.as_mut_slice::<u8>());
		let y_in_bytes_u8: [_; WIDTH] = y_in.each_mut().map(|col| col.as_mut_slice::<u8>());
		let c_in_u8 = c_in.as_mut_slice::<u8>();

		for row_idx in 0..1 << log_size {
			let mut rng = thread_rng();
			let input_bitmask = (U512::from(1u8) << (8 * WIDTH)) - U512::from(1u8);
			let mut x = random_u512(&mut rng);
			x &= input_bitmask;
			let mut y = random_u512(&mut rng);
			y &= input_bitmask;

			let mut c: bool = rng.gen();

			while (x + y + U512::from(c)) > input_bitmask {
				x = random_u512(&mut rng);
				x &= input_bitmask;
				y = random_u512(&mut rng);
				y &= input_bitmask;
				c = rng.gen();
			}

			for byte_idx in 0..WIDTH {
				x_in_bytes_u8[byte_idx][row_idx] = x.byte(byte_idx);

				y_in_bytes_u8[byte_idx][row_idx] = y.byte(byte_idx);
			}

			c_in_u8[row_idx / 8] |= (c as u8) << (row_idx % 8);
		}
	}

	let lookup_t_add = add_lookup(&mut builder, "add table").unwrap();
	let lookup_t_add_carryfree = add_carryfree_lookup(&mut builder, "add table").unwrap();

	let mut lookup_batch_add = LookupBatch::new(lookup_t_add);
	let mut lookup_batch_add_carryfree = LookupBatch::new(lookup_t_add_carryfree);

	let _sum_and_cout = byte_sliced_add_carryfree::<_, _, TL>(
		&mut builder,
		"lasso_bytesliced_add_carryfree",
		&x_in,
		&y_in,
		c_in,
		log_size,
		&mut lookup_batch_add,
		&mut lookup_batch_add_carryfree,
	)
	.unwrap();

	lookup_batch_add
		.execute::<_, _, B32, B32>(&mut builder)
		.unwrap();
	lookup_batch_add_carryfree
		.execute::<_, _, B32, B32>(&mut builder)
		.unwrap();

	let witness = builder.take_witness().unwrap();
	let constraint_system = builder.build().unwrap();
	let boundaries = vec![];
	validate_witness(&constraint_system, &boundaries, &witness).unwrap();
}

pub fn test_bytesliced_double_conditional_increment<const WIDTH: usize, TL>()
where
	TL: TowerLevel<OracleId, Data = [OracleId; WIDTH]>,
{
	type U = OptimalUnderlier;
	type F = BinaryField128b;

	let allocator = bumpalo::Bump::new();
	let mut builder = ConstraintSystemBuilder::<U, F>::new_with_witness(&allocator);
	let log_size = 14;

	let x_in = array::from_fn(|_| {
		unconstrained::<_, _, BinaryField8b>(&mut builder, "x", log_size).unwrap()
	});

	let first_c_in =
		unconstrained::<_, _, BinaryField1b>(&mut builder, "cin first", log_size).unwrap();

	let second_c_in =
		unconstrained::<_, _, BinaryField1b>(&mut builder, "cin second", log_size).unwrap();

	let zero_oracle_carry =
		transparent::constant(&mut builder, "zero carry", log_size, BinaryField1b::ZERO).unwrap();
	let lookup_t_dci = dci_lookup(&mut builder, "add table").unwrap();

	let mut lookup_batch_dci = LookupBatch::new(lookup_t_dci);

	let _sum_and_cout = byte_sliced_double_conditional_increment::<_, _, TL>(
		&mut builder,
		"lasso_bytesliced_DCI",
		&x_in,
		first_c_in,
		second_c_in,
		log_size,
		zero_oracle_carry,
		&mut lookup_batch_dci,
	)
	.unwrap();

	lookup_batch_dci
		.execute::<_, _, B32, B32>(&mut builder)
		.unwrap();

	let witness = builder.take_witness().unwrap();
	let constraint_system = builder.build().unwrap();
	let boundaries = vec![];
	validate_witness(&constraint_system, &boundaries, &witness).unwrap();
}

pub fn test_bytesliced_mul<const WIDTH: usize, TL>()
where
	TL: TowerLevel<OracleId>,
	TL::Base: TowerLevel<OracleId, Data = [OracleId; WIDTH]>,
{
	type U = OptimalUnderlier;
	type F = BinaryField128b;

	let allocator = bumpalo::Bump::new();
	let mut builder = ConstraintSystemBuilder::<U, F>::new_with_witness(&allocator);
	let log_size = 14;

	let mult_a = array::from_fn(|_| {
		unconstrained::<_, _, BinaryField8b>(&mut builder, "a", log_size).unwrap()
	});
	let mult_b = array::from_fn(|_| {
		unconstrained::<_, _, BinaryField8b>(&mut builder, "b", log_size).unwrap()
	});

	let zero_oracle_carry =
		transparent::constant(&mut builder, "zero carry", log_size, BinaryField1b::ZERO).unwrap();

	let lookup_t_mul = mul_lookup(&mut builder, "mul lookup").unwrap();
	let lookup_t_add = add_lookup(&mut builder, "add lookup").unwrap();
	let lookup_t_dci = dci_lookup(&mut builder, "dci lookup").unwrap();

	let mut lookup_batch_mul = LookupBatch::new(lookup_t_mul);
	let mut lookup_batch_add = LookupBatch::new(lookup_t_add);
	let mut lookup_batch_dci = LookupBatch::new(lookup_t_dci);

	let _sum_and_cout = byte_sliced_mul::<_, _, TL::Base, TL>(
		&mut builder,
		"lasso_bytesliced_mul",
		&mult_a,
		&mult_b,
		log_size,
		zero_oracle_carry,
		&mut lookup_batch_mul,
		&mut lookup_batch_add,
		&mut lookup_batch_dci,
	)
	.unwrap();

	let witness = builder.take_witness().unwrap();
	let constraint_system = builder.build().unwrap();
	let boundaries = vec![];
	validate_witness(&constraint_system, &boundaries, &witness).unwrap();
}

pub fn test_bytesliced_modular_mul<const WIDTH: usize, TL>()
where
	TL: TowerLevel<OracleId>,
	TL::Base: TowerLevel<OracleId, Data = [OracleId; WIDTH]>,
	<TL as TowerLevel<usize>>::Data: Debug,
{
	type U = OptimalUnderlier;
	type F = BinaryField128b;

	let allocator = bumpalo::Bump::new();
	let mut builder = ConstraintSystemBuilder::<U, F>::new_with_witness(&allocator);
	let log_size = 14;

	let mut rng = thread_rng();

	let mult_a = builder.add_committed_multiple::<WIDTH>("a", log_size, B8::TOWER_LEVEL);
	let mult_b = builder.add_committed_multiple::<WIDTH>("b", log_size, B8::TOWER_LEVEL);

	let input_bitmask = (U512::from(1u8) << (8 * WIDTH)) - U512::from(1u8);

	let modulus = (random_u512(&mut rng) % input_bitmask) + U512::from(1u8);

	if let Some(witness) = builder.witness() {
		let mut mult_a: [_; WIDTH] =
			array::from_fn(|byte_idx| witness.new_column::<BinaryField8b>(mult_a[byte_idx]));

		let mult_a_u8 = mult_a.each_mut().map(|col| col.as_mut_slice::<u8>());

		let mut mult_b: [_; WIDTH] =
			array::from_fn(|byte_idx| witness.new_column::<BinaryField8b>(mult_b[byte_idx]));

		let mult_b_u8 = mult_b.each_mut().map(|col| col.as_mut_slice::<u8>());

		for row_idx in 0..1 << log_size {
			let mut a = random_u512(&mut rng);
			let mut b = random_u512(&mut rng);

			a %= modulus;
			b %= modulus;

			for byte_idx in 0..WIDTH {
				mult_a_u8[byte_idx][row_idx] = a.byte(byte_idx);
				mult_b_u8[byte_idx][row_idx] = b.byte(byte_idx);
			}
		}
	}

	let modulus_input: [_; WIDTH] = array::from_fn(|byte_idx| modulus.byte(byte_idx));

	let zero_oracle_byte =
		transparent::constant(&mut builder, "zero carry", log_size, BinaryField8b::ZERO).unwrap();

	let zero_oracle_carry =
		transparent::constant(&mut builder, "zero carry", log_size, BinaryField1b::ZERO).unwrap();

	let _modded_product = byte_sliced_modular_mul::<_, _, TL::Base, TL>(
		&mut builder,
		"lasso_bytesliced_mul",
		&mult_a,
		&mult_b,
		&modulus_input,
		log_size,
		zero_oracle_byte,
		zero_oracle_carry,
	)
	.unwrap();

	let witness = builder.take_witness().unwrap();
	let constraint_system = builder.build().unwrap();
	let boundaries = vec![];
	validate_witness(&constraint_system, &boundaries, &witness).unwrap();
}
