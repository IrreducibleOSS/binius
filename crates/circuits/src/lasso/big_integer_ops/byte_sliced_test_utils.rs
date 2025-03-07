// Copyright 2024-2025 Irreducible Inc.

use std::{array, fmt::Debug};

use alloy_primitives::U512;
use binius_core::oracle::OracleId;
use binius_field::{
	tower_levels::TowerLevel, BinaryField1b, BinaryField32b, BinaryField8b, Field, TowerField,
};
use rand::{rngs::StdRng, thread_rng, Rng, SeedableRng};

use super::{
	byte_sliced_add, byte_sliced_add_carryfree, byte_sliced_double_conditional_increment,
	byte_sliced_modular_mul, byte_sliced_mul,
};
use crate::{
	builder::test_utils::test_circuit,
	lasso::{
		batch::LookupBatch,
		lookups::u8_arithmetic::{add_carryfree_lookup, add_lookup, dci_lookup, mul_lookup},
	},
	transparent,
	unconstrained::unconstrained,
};

type B8 = BinaryField8b;
type B32 = BinaryField32b;

pub fn random_u512(rng: &mut impl Rng) -> U512 {
	let limbs = array::from_fn(|_| rng.gen());
	U512::from_limbs(limbs)
}

pub fn test_bytesliced_add<const WIDTH: usize, TL>()
where
	TL: TowerLevel,
{
	test_circuit(|builder| {
		let log_size = 14;
		let x_in = TL::from_fn(|_| unconstrained::<BinaryField8b>(builder, "x", log_size).unwrap());
		let y_in = TL::from_fn(|_| unconstrained::<BinaryField8b>(builder, "y", log_size).unwrap());
		let c_in = unconstrained::<BinaryField1b>(builder, "cin first", log_size)?;
		let lookup_t_add = add_lookup(builder, "add table")?;
		let mut lookup_batch_add = LookupBatch::new([lookup_t_add]);
		let _sum_and_cout = byte_sliced_add::<TL>(
			builder,
			"lasso_bytesliced_add",
			&x_in,
			&y_in,
			c_in,
			log_size,
			&mut lookup_batch_add,
		)?;
		lookup_batch_add.execute::<B32>(builder)?;
		Ok(vec![])
	})
	.unwrap();
}

pub fn test_bytesliced_add_carryfree<const WIDTH: usize, TL>()
where
	TL: TowerLevel,
{
	test_circuit(|builder| {
		let log_size = 14;
		let x_in =
			TL::from_fn(|_| builder.add_committed("x", log_size, BinaryField8b::TOWER_LEVEL));
		let y_in =
			TL::from_fn(|_| builder.add_committed("y", log_size, BinaryField8b::TOWER_LEVEL));
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

		let lookup_t_add = add_lookup(builder, "add table")?;
		let lookup_t_add_carryfree = add_carryfree_lookup(builder, "add table")?;

		let mut lookup_batch_add = LookupBatch::new([lookup_t_add]);
		let mut lookup_batch_add_carryfree = LookupBatch::new([lookup_t_add_carryfree]);

		let _sum_and_cout = byte_sliced_add_carryfree::<TL>(
			builder,
			"lasso_bytesliced_add_carryfree",
			&x_in,
			&y_in,
			c_in,
			log_size,
			&mut lookup_batch_add,
			&mut lookup_batch_add_carryfree,
		)?;

		lookup_batch_add.execute::<B32>(builder)?;
		lookup_batch_add_carryfree.execute::<B32>(builder)?;
		Ok(vec![])
	})
	.unwrap();
}

pub fn test_bytesliced_double_conditional_increment<const WIDTH: usize, TL>()
where
	TL: TowerLevel,
{
	test_circuit(|builder| {
		let log_size = 14;
		let x_in = TL::from_fn(|_| unconstrained::<BinaryField8b>(builder, "x", log_size).unwrap());
		let first_c_in = unconstrained::<BinaryField1b>(builder, "cin first", log_size)?;
		let second_c_in = unconstrained::<BinaryField1b>(builder, "cin second", log_size)?;
		let zero_oracle_carry =
			transparent::constant(builder, "zero carry", log_size, BinaryField1b::ZERO)?;
		let lookup_t_dci = dci_lookup(builder, "add table")?;
		let mut lookup_batch_dci = LookupBatch::new([lookup_t_dci]);
		let _sum_and_cout = byte_sliced_double_conditional_increment::<TL>(
			builder,
			"lasso_bytesliced_DCI",
			&x_in,
			first_c_in,
			second_c_in,
			log_size,
			zero_oracle_carry,
			&mut lookup_batch_dci,
		)?;
		lookup_batch_dci.execute::<B32>(builder)?;
		Ok(vec![])
	})
	.unwrap();
}

pub fn test_bytesliced_mul<const WIDTH: usize, TL>()
where
	TL: TowerLevel,
{
	test_circuit(|builder| {
		let log_size = 14;
		let mult_a =
			TL::Base::from_fn(|_| unconstrained::<BinaryField8b>(builder, "a", log_size).unwrap());
		let mult_b =
			TL::Base::from_fn(|_| unconstrained::<BinaryField8b>(builder, "b", log_size).unwrap());
		let zero_oracle_carry =
			transparent::constant(builder, "zero carry", log_size, BinaryField1b::ZERO)?;
		let lookup_t_mul = mul_lookup(builder, "mul lookup")?;
		let lookup_t_add = add_lookup(builder, "add lookup")?;
		let lookup_t_dci = dci_lookup(builder, "dci lookup")?;
		let mut lookup_batch_mul = LookupBatch::new([lookup_t_mul]);
		let mut lookup_batch_add = LookupBatch::new([lookup_t_add]);
		let mut lookup_batch_dci = LookupBatch::new([lookup_t_dci]);
		let _sum_and_cout = byte_sliced_mul::<TL::Base, TL>(
			builder,
			"lasso_bytesliced_mul",
			&mult_a,
			&mult_b,
			log_size,
			zero_oracle_carry,
			&mut lookup_batch_mul,
			&mut lookup_batch_add,
			&mut lookup_batch_dci,
		)?;
		Ok(vec![])
	})
	.unwrap();
}

pub fn test_bytesliced_modular_mul<const WIDTH: usize, TL>()
where
	TL: TowerLevel<Data<usize>: Debug>,
	TL::Base: TowerLevel<Data<usize> = [OracleId; WIDTH]>,
{
	test_circuit(|builder| {
		let log_size = 12;
		let mut rng = thread_rng();
		let mult_a = builder.add_committed_multiple::<WIDTH>("a", log_size, B8::TOWER_LEVEL);
		let mult_b = builder.add_committed_multiple::<WIDTH>("b", log_size, B8::TOWER_LEVEL);
		let input_bitmask = (U512::from(1u8) << (8 * WIDTH)) - U512::from(1u8);
		let modulus =
			(random_u512(&mut StdRng::from_seed([42; 32])) % input_bitmask) + U512::from(1u8);

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
			transparent::constant(builder, "zero carry", log_size, BinaryField8b::ZERO)?;
		let zero_oracle_carry =
			transparent::constant(builder, "zero carry", log_size, BinaryField1b::ZERO)?;
		let _modded_product = byte_sliced_modular_mul::<TL::Base, TL>(
			builder,
			"lasso_bytesliced_mul",
			&mult_a,
			&mult_b,
			&modulus_input,
			log_size,
			zero_oracle_byte,
			zero_oracle_carry,
		)?;
		Ok(vec![])
	})
	.unwrap();
}
