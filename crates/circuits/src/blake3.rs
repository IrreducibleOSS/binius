// Copyright 2024-2025 Irreducible Inc.

use binius_core::oracle::{OracleId, ShiftVariant};
use binius_field::{BinaryField1b, Field};
use binius_utils::checked_arithmetics::checked_log_2;

use crate::{
	arithmetic,
	arithmetic::Flags,
	builder::{types::F, ConstraintSystemBuilder},
};

type F1 = BinaryField1b;
const LOG_U32_BITS: usize = checked_log_2(32);
const BLAKE3_STATE_LEN: usize = 16;

// Gadget that performs two u32 variables XOR and then rotates the result
fn xor_rotate_right(
	builder: &mut ConstraintSystemBuilder,
	name: impl ToString,
	log_size: usize,
	a: OracleId,
	b: OracleId,
	rotate_right_offset: u32,
) -> Result<OracleId, anyhow::Error> {
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

// Gadget for Blake3 g function
#[allow(clippy::too_many_arguments)]
pub fn g(
	builder: &mut ConstraintSystemBuilder,
	name: impl ToString,
	a_in: OracleId,
	b_in: OracleId,
	c_in: OracleId,
	d_in: OracleId,
	mx: OracleId,
	my: OracleId,
	log_size: usize,
) -> Result<(OracleId, OracleId, OracleId, OracleId), anyhow::Error> {
	builder.push_namespace(name);

	let ab = arithmetic::u32::add(builder, "a_in + b_in", a_in, b_in, Flags::Unchecked)?;
	let a1 = arithmetic::u32::add(builder, "a_in + b_in + mx", ab, mx, Flags::Unchecked)?;

	let d1 = xor_rotate_right(builder, "(d_in ^ a1).rotate_right(16)", log_size, d_in, a1, 16u32)?;

	let c1 = arithmetic::u32::add(builder, "c_in + d1", c_in, d1, Flags::Unchecked)?;

	let b1 = xor_rotate_right(builder, "(b_in ^ c1).rotate_right(12)", log_size, b_in, c1, 12u32)?;

	let a1b1 = arithmetic::u32::add(builder, "a1 + b1", a1, b1, Flags::Unchecked)?;
	let a2 = arithmetic::u32::add(builder, "a1 + b1 + my_in", a1b1, my, Flags::Unchecked)?;

	let d2 = xor_rotate_right(builder, "(d1 ^ a2).rotate_right(8)", log_size, d1, a2, 8u32)?;

	let c2 = arithmetic::u32::add(builder, "c1 + d2", c1, d2, Flags::Unchecked)?;

	let b2 = xor_rotate_right(builder, "(b1 ^ c2).rotate_right(7)", log_size, b1, c2, 7u32)?;

	builder.pop_namespace();

	Ok((a2, b2, c2, d2))
}

// Gadget for Blake3 round function
pub fn round(
	builder: &mut ConstraintSystemBuilder,
	name: impl ToString,
	state: &[OracleId],
	m: &[OracleId],
	log_size: usize,
) -> Result<[OracleId; BLAKE3_STATE_LEN], anyhow::Error> {
	assert_eq!(state.len(), m.len());
	assert_eq!(state.len(), BLAKE3_STATE_LEN);

	builder.push_namespace(name);

	// Mixing columns
	let (s0, s4, s8, s12) =
		g(builder, "mix-columns-0", state[0], state[4], state[8], state[12], m[0], m[1], log_size)?;

	let (s1, s5, s9, s13) =
		g(builder, "mix-columns-1", state[1], state[5], state[9], state[13], m[2], m[3], log_size)?;
	#[rustfmt::skip]
	let (s2, s6, s10, s14) =
		g(builder, "mix-columns-2", state[2], state[6], state[10], state[14], m[4], m[5], log_size)?;
	#[rustfmt::skip]
	let (s3, s7, s11, s15) =
		g(builder, "mix-columns-3", state[3], state[7], state[11], state[15], m[6], m[7], log_size)?;

	// Mixing diagonals
	let (s0, s5, s10, s15) = g(builder, "mix-diagonals-0", s0, s5, s10, s15, m[8], m[9], log_size)?;
	#[rustfmt::skip]
	let (s1, s6, s11, s12) = g(builder, "mix-diagonals-1", s1, s6, s11, s12, m[10], m[11], log_size)?;

	let (s2, s7, s8, s13) = g(builder, "mix-diagonals-2", s2, s7, s8, s13, m[12], m[13], log_size)?;

	let (s3, s4, s9, s14) = g(builder, "mix-diagonals-3", s3, s4, s9, s14, m[14], m[15], log_size)?;

	builder.pop_namespace();

	Ok([
		s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15,
	])
}

#[cfg(test)]
mod tests {
	use binius_core::{constraint_system::validate::validate_witness, oracle::OracleId};
	use binius_field::BinaryField1b;
	use binius_maybe_rayon::prelude::*;

	use crate::{
		blake3::{g, round, BLAKE3_STATE_LEN},
		builder::ConstraintSystemBuilder,
		unconstrained::{fixed_u32, unconstrained},
	};

	type F1 = BinaryField1b;

	const LOG_SIZE: usize = 5;

	// The Blake3 mixing function, G, which mixes either a column or a diagonal.
	// https://github.com/BLAKE3-team/BLAKE3/blob/master/reference_impl/reference_impl.rs#L42
	const fn g_out_of_circuit(
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
	fn test_vector_g() {
		// Let's use some fixed data input to check that our in-circuit computation
		// produces same output as out-of-circuit one
		let a = 0xaaaaaaaau32;
		let b = 0xbbbbbbbbu32;
		let c = 0xccccccccu32;
		let d = 0xddddddddu32;
		let mx = 0xffff00ffu32;
		let my = 0xff00ffffu32;

		let (expected_0, expected_1, expected_2, expected_3) = g_out_of_circuit(a, b, c, d, mx, my);

		let size = 1 << LOG_SIZE;

		let allocator = bumpalo::Bump::new();
		let mut builder = ConstraintSystemBuilder::new_with_witness(&allocator);

		let a_in = fixed_u32::<F1>(&mut builder, "a", LOG_SIZE, vec![a; size]).unwrap();
		let b_in = fixed_u32::<F1>(&mut builder, "b", LOG_SIZE, vec![b; size]).unwrap();
		let c_in = fixed_u32::<F1>(&mut builder, "c", LOG_SIZE, vec![c; size]).unwrap();
		let d_in = fixed_u32::<F1>(&mut builder, "d", LOG_SIZE, vec![d; size]).unwrap();
		let mx_in = fixed_u32::<F1>(&mut builder, "mx", LOG_SIZE, vec![mx; size]).unwrap();
		let my_in = fixed_u32::<F1>(&mut builder, "my", LOG_SIZE, vec![my; size]).unwrap();

		let output = g(&mut builder, "g", a_in, b_in, c_in, d_in, mx_in, my_in, LOG_SIZE).unwrap();

		if let Some(witness) = builder.witness() {
			(
				witness.get::<F1>(output.0).unwrap().as_slice::<u32>(),
				witness.get::<F1>(output.1).unwrap().as_slice::<u32>(),
				witness.get::<F1>(output.2).unwrap().as_slice::<u32>(),
				witness.get::<F1>(output.3).unwrap().as_slice::<u32>(),
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
	fn test_random_input_g() {
		let allocator = bumpalo::Bump::new();
		let mut builder = ConstraintSystemBuilder::new_with_witness(&allocator);

		let a_in = unconstrained::<F1>(&mut builder, "a", LOG_SIZE).unwrap();
		let b_in = unconstrained::<F1>(&mut builder, "b", LOG_SIZE).unwrap();
		let c_in = unconstrained::<F1>(&mut builder, "c", LOG_SIZE).unwrap();
		let d_in = unconstrained::<F1>(&mut builder, "d", LOG_SIZE).unwrap();
		let mx_in = unconstrained::<F1>(&mut builder, "mx", LOG_SIZE).unwrap();
		let my_in = unconstrained::<F1>(&mut builder, "my", LOG_SIZE).unwrap();

		g(&mut builder, "g", a_in, b_in, c_in, d_in, mx_in, my_in, LOG_SIZE).unwrap();

		let witness = builder.take_witness().unwrap();
		let constraints_system = builder.build().unwrap();

		validate_witness(&constraints_system, &[], &witness).unwrap();
	}

	// The Blake3 round function:
	// https://github.com/BLAKE3-team/BLAKE3/blob/master/reference_impl/reference_impl.rs#L53
	const fn round_out_of_circuit(
		state: &[u32; BLAKE3_STATE_LEN],
		m: &[u32; BLAKE3_STATE_LEN],
	) -> [u32; BLAKE3_STATE_LEN] {
		// Mix the columns.
		let (s0, s4, s8, s12) =
			g_out_of_circuit(state[0], state[4], state[8], state[12], m[0], m[1]);
		let (s1, s5, s9, s13) =
			g_out_of_circuit(state[1], state[5], state[9], state[13], m[2], m[3]);
		let (s2, s6, s10, s14) =
			g_out_of_circuit(state[2], state[6], state[10], state[14], m[4], m[5]);
		let (s3, s7, s11, s15) =
			g_out_of_circuit(state[3], state[7], state[11], state[15], m[6], m[7]);

		// Mix the diagonals.
		let (s0, s5, s10, s15) = g_out_of_circuit(s0, s5, s10, s15, m[8], m[9]);
		let (s1, s6, s11, s12) = g_out_of_circuit(s1, s6, s11, s12, m[10], m[11]);
		let (s2, s7, s8, s13) = g_out_of_circuit(s2, s7, s8, s13, m[12], m[13]);
		let (s3, s4, s9, s14) = g_out_of_circuit(s3, s4, s9, s14, m[14], m[15]);

		[
			s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15,
		]
	}

	#[test]
	fn test_vector_round() {
		let state = [
			0xfffffff0, 0xfffffff1, 0xfffffff2, 0xfffffff3, 0xfffffff4, 0xfffffff5, 0xfffffff6,
			0xfffffff7, 0xfffffff8, 0xfffffff9, 0xfffffffa, 0xfffffffb, 0xfffffffc, 0xfffffffd,
			0xfffffffe, 0xffffffff,
		];

		let m = [
			0x09ffffff, 0x08ffffff, 0x07ffffff, 0x06ffffff, 0x05ffffff, 0x04ffffff, 0x03ffffff,
			0x02ffffff, 0x01ffffff, 0x00ffffff, 0x0fffffff, 0x0effffff, 0x0dffffff, 0x0cffffff,
			0x0bffffff, 0x0affffff,
		];

		assert_eq!(state.len(), BLAKE3_STATE_LEN);
		assert_eq!(state.len(), m.len());

		let expected = round_out_of_circuit(&state, &m);

		let size = 1 << LOG_SIZE;

		let allocator = bumpalo::Bump::new();
		let mut builder = ConstraintSystemBuilder::new_with_witness(&allocator);

		// populate State and Message columns with some fixed values
		let state = (0..state.len())
			.map(|idx| {
				fixed_u32::<F1>(&mut builder, format!("s{}", idx), LOG_SIZE, vec![state[idx]; size])
					.unwrap()
			})
			.collect::<Vec<OracleId>>();
		let m = (0..m.len())
			.map(|idx| {
				fixed_u32::<F1>(&mut builder, format!("m{}", idx), LOG_SIZE, vec![m[idx]; size])
					.unwrap()
			})
			.collect::<Vec<OracleId>>();

		// execute 'round' gadget
		let actual = round(&mut builder, "round", &state, &m, LOG_SIZE).unwrap();

		// compare output values with expected ones
		if let Some(witness) = builder.witness() {
			for (i, expected_i) in expected.into_iter().enumerate() {
				let values = witness.get::<F1>(actual[i]).unwrap().as_slice::<u32>();
				assert!(values.iter().all(|v| *v == expected_i));
			}
		}

		let witness = builder.take_witness().unwrap();
		let constraints_system = builder.build().unwrap();

		validate_witness(&constraints_system, &[], &witness).unwrap();
	}

	#[test]
	fn test_random_input_round() {
		let allocator = bumpalo::Bump::new();
		let mut builder = ConstraintSystemBuilder::new_with_witness(&allocator);

		let state = (0..BLAKE3_STATE_LEN)
			.map(|idx| unconstrained::<F1>(&mut builder, format!("s{}", idx), LOG_SIZE).unwrap())
			.collect::<Vec<OracleId>>();

		let m = (0..BLAKE3_STATE_LEN)
			.map(|idx| unconstrained::<F1>(&mut builder, format!("m{}", idx), LOG_SIZE).unwrap())
			.collect::<Vec<OracleId>>();

		round(&mut builder, "round", &state, &m, LOG_SIZE).unwrap();

		let witness = builder.take_witness().unwrap();
		let constraints_system = builder.build().unwrap();

		validate_witness(&constraints_system, &[], &witness).unwrap();
	}
}
