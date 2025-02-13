use binius_core::oracle::{OracleId, ShiftVariant};
use binius_field::{
	as_packed_field::PackScalar, underlier::UnderlierType, BinaryField1b, TowerField,
};
use binius_utils::checked_arithmetics::checked_log_2;
use bytemuck::Pod;

use crate::{arithmetic, arithmetic::Flags, builder::ConstraintSystemBuilder};

type F1 = BinaryField1b;
const LOG_U32_BITS: usize = checked_log_2(32);

// Gadget that performs two u32 variables XOR and then rotates the result
fn xor_rotate_right<U, F>(
	builder: &mut ConstraintSystemBuilder<U, F>,
	name: impl ToString,
	log_size: usize,
	a: OracleId,
	b: OracleId,
	rotate_right_offset: u32,
) -> Result<OracleId, anyhow::Error>
where
	U: PackScalar<F> + PackScalar<F1> + Pod,
	F: TowerField,
{
	assert!(rotate_right_offset <= 32);

	builder.push_namespace(name);

	let xor = builder
		.add_linear_combination("xor", log_size, [(a, F::ONE), (b, F::ONE)])
		.unwrap();

	let rotate = builder.add_shifted(
		"rotate",
		xor,
		32 - rotate_right_offset as usize,
		LOG_U32_BITS,
		ShiftVariant::CircularLeft,
	)?;

	if let Some(witness) = builder.witness() {
		let a_value = witness.get::<F1>(a)?.as_slice::<u32>();
		let b_value = witness.get::<F1>(b)?.as_slice::<u32>();

		let mut xor_witness = witness.new_column::<F1>(xor);
		let xor_value = xor_witness.as_mut_slice::<u32>();

		for (idx, v) in xor_value.iter_mut().enumerate() {
			*v = a_value[idx] ^ b_value[idx];
		}

		let mut rotate_witness = witness.new_column::<F1>(rotate);
		let rotate_value = rotate_witness.as_mut_slice::<u32>();
		for (idx, v) in rotate_value.iter_mut().enumerate() {
			*v = xor_value[idx].rotate_right(rotate_right_offset);
		}
	}

	builder.pop_namespace();

	Ok(rotate)
}

#[allow(clippy::too_many_arguments)]
pub fn blake3_g<U, F>(
	builder: &mut ConstraintSystemBuilder<U, F>,
	name: impl ToString,
	a_in: OracleId,
	b_in: OracleId,
	c_in: OracleId,
	d_in: OracleId,
	mx: OracleId,
	my: OracleId,
	log_size: usize,
) -> Result<[OracleId; 4], anyhow::Error>
where
	U: UnderlierType + Pod + PackScalar<F> + PackScalar<BinaryField1b>,
	F: TowerField,
{
	builder.push_namespace(name);

	let a1 = arithmetic::u32::add3(builder, "a_in + b_in + mx", a_in, b_in, mx, Flags::Unchecked)?;

	let d1 = xor_rotate_right(builder, "(d_in ^ a1).rotate_right(16)", log_size, d_in, a1, 16u32)?;

	let c1 = arithmetic::u32::add(builder, "c_in + d1", c_in, d1, Flags::Unchecked)?;

	let b1 = xor_rotate_right(builder, "(b_in ^ c1).rotate_right(12)", log_size, b_in, c1, 12u32)?;

	let a2 = arithmetic::u32::add3(builder, "a1 + b1 + my_in", a1, b1, my, Flags::Unchecked)?;

	let d2 = xor_rotate_right(builder, "(d1 ^ a2).rotate_right(8)", log_size, d1, a2, 8u32)?;

	let c2 = arithmetic::u32::add(builder, "c1 + d2", c1, d2, Flags::Unchecked)?;

	let b2 = xor_rotate_right(builder, "(b1 ^ c2).rotate_right(7)", log_size, b1, c2, 7u32)?;

	builder.pop_namespace();

	Ok([a2, b2, c2, d2])
}

#[cfg(test)]
mod tests {
	use binius_core::constraint_system::validate::validate_witness;
	use binius_field::{arch::OptimalUnderlier, BinaryField128b, BinaryField1b};
	use binius_maybe_rayon::prelude::*;

	use crate::{
		blake3::blake3_g,
		builder::ConstraintSystemBuilder,
		unconstrained::{fixed_u32, unconstrained},
	};

	type U = OptimalUnderlier;
	type F128 = BinaryField128b;
	type F1 = BinaryField1b;

	const LOG_SIZE: usize = 5;

	// The Blake3 mixing function, G, which mixes either a column or a diagonal.
	// https://github.com/BLAKE3-team/BLAKE3/blob/master/reference_impl/reference_impl.rs
	const fn g(
		a_in: u32,
		b_in: u32,
		c_in: u32,
		d_in: u32,
		mx: u32,
		my: u32,
	) -> (u32, u32, u32, u32) {
		let a1 = a_in.wrapping_add(b_in).wrapping_add(mx);
		let d1 = (d_in ^ a1).rotate_right(16);
		let c1 = c_in.wrapping_add(d1);
		let b1 = (b_in ^ c1).rotate_right(12);

		let a2 = a1.wrapping_add(b1).wrapping_add(my);
		let d2 = (d1 ^ a2).rotate_right(8);
		let c2 = c1.wrapping_add(d2);
		let b2 = (b1 ^ c2).rotate_right(7);

		(a2, b2, c2, d2)
	}

	#[test]
	fn test_vector() {
		// Let's use some fixed data input to check that our in-circuit computation
		// produces same output as out-of-circuit one
		let a = 0xaaaaaaaau32;
		let b = 0xbbbbbbbbu32;
		let c = 0xccccccccu32;
		let d = 0xddddddddu32;
		let mx = 0xffff00ffu32;
		let my = 0xff00ffffu32;

		let (expected_0, expected_1, expected_2, expected_3) = g(a, b, c, d, mx, my);

		let size = 1 << LOG_SIZE;

		let allocator = bumpalo::Bump::new();
		let mut builder = ConstraintSystemBuilder::<U, F128>::new_with_witness(&allocator);

		let a_in = fixed_u32::<U, F128, F1>(&mut builder, "a", LOG_SIZE, vec![a; size]).unwrap();
		let b_in = fixed_u32::<U, F128, F1>(&mut builder, "b", LOG_SIZE, vec![b; size]).unwrap();
		let c_in = fixed_u32::<U, F128, F1>(&mut builder, "c", LOG_SIZE, vec![c; size]).unwrap();
		let d_in = fixed_u32::<U, F128, F1>(&mut builder, "d", LOG_SIZE, vec![d; size]).unwrap();
		let mx_in = fixed_u32::<U, F128, F1>(&mut builder, "mx", LOG_SIZE, vec![mx; size]).unwrap();
		let my_in = fixed_u32::<U, F128, F1>(&mut builder, "my", LOG_SIZE, vec![my; size]).unwrap();

		let output =
			blake3_g(&mut builder, "g", a_in, b_in, c_in, d_in, mx_in, my_in, LOG_SIZE).unwrap();

		if let Some(witness) = builder.witness() {
			(
				witness.get::<F1>(output[0]).unwrap().as_slice::<u32>(),
				witness.get::<F1>(output[1]).unwrap().as_slice::<u32>(),
				witness.get::<F1>(output[2]).unwrap().as_slice::<u32>(),
				witness.get::<F1>(output[3]).unwrap().as_slice::<u32>(),
			)
				.into_par_iter()
				.for_each(|(actual_0, actual_1, actual_2, actual_3)| {
					assert_eq!(*actual_0, expected_0);
					assert_eq!(*actual_1, expected_1);
					assert_eq!(*actual_2, expected_2);
					assert_eq!(*actual_3, expected_3);
				});
		}

		let witness = builder.take_witness().unwrap();
		let constraints_system = builder.build().unwrap();

		validate_witness(&constraints_system, &[], &witness).unwrap();
	}

	#[test]
	fn test_random_input() {
		let allocator = bumpalo::Bump::new();
		let mut builder = ConstraintSystemBuilder::<U, F128>::new_with_witness(&allocator);

		let a_in = unconstrained::<U, F128, F1>(&mut builder, "a", LOG_SIZE).unwrap();
		let b_in = unconstrained::<U, F128, F1>(&mut builder, "b", LOG_SIZE).unwrap();
		let c_in = unconstrained::<U, F128, F1>(&mut builder, "c", LOG_SIZE).unwrap();
		let d_in = unconstrained::<U, F128, F1>(&mut builder, "d", LOG_SIZE).unwrap();
		let mx_in = unconstrained::<U, F128, F1>(&mut builder, "mx", LOG_SIZE).unwrap();
		let my_in = unconstrained::<U, F128, F1>(&mut builder, "my", LOG_SIZE).unwrap();

		blake3_g(&mut builder, "g", a_in, b_in, c_in, d_in, mx_in, my_in, LOG_SIZE).unwrap();

		let witness = builder.take_witness().unwrap();
		let constraints_system = builder.build().unwrap();

		validate_witness(&constraints_system, &[], &witness).unwrap();
	}
}
