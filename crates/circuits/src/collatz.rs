// Copyright 2024 Irreducible Inc.

use binius_core::{constraint_system::channel::ChannelId, oracle::OracleId};
use binius_field::{
	as_packed_field::PackScalar, BinaryField1b, BinaryField32b, ExtensionField, TowerField,
};
use binius_macros::arith_expr;
use bytemuck::Pod;

use crate::{arithmetic, builder::ConstraintSystemBuilder, transparent};

pub fn collatz<U, F>(
	builder: &mut ConstraintSystemBuilder<U, F>,
	x0: u32,
) -> Result<ChannelId, anyhow::Error>
where
	U: PackScalar<F> + PackScalar<BinaryField1b> + PackScalar<BinaryField32b> + Pod,
	F: TowerField + ExtensionField<BinaryField32b>,
{
	let channel = builder.add_channel();
	let (even, odd): (Vec<_>, Vec<_>) = collatz_orbit(x0).into_iter().partition(|x| x % 2 == 0);
	collatz_even(builder, channel, &even)?;
	collatz_odd(builder, channel, &odd)?;
	Ok(channel)
}

/// ```
/// assert_eq!(
///     binius_circuits::collatz::collatz_orbit(5),
///     vec![5, 16, 8, 4, 2]
/// )
/// ```
pub fn collatz_orbit(x0: u32) -> Vec<u32> {
	let mut res = vec![x0];
	let mut x = x0;
	while x != 1 {
		if x % 2 == 0 {
			x /= 2;
		} else {
			x = 3 * x + 1;
		}
		res.push(x);
	}
	// We ignore the final 1
	res.pop();
	res
}

fn collatz_even<U, F>(
	builder: &mut ConstraintSystemBuilder<U, F>,
	channel: ChannelId,
	advice: &[u32],
) -> Result<(), anyhow::Error>
where
	U: PackScalar<F> + PackScalar<BinaryField1b> + PackScalar<BinaryField32b> + Pod,
	F: TowerField + ExtensionField<BinaryField32b>,
{
	let log_1b_rows = 5 + binius_utils::checked_arithmetics::log2_ceil_usize(advice.len());
	let even = builder.add_committed("even", log_1b_rows, BinaryField1b::TOWER_LEVEL);
	if let Some(witness) = builder.witness() {
		witness
			.new_column::<BinaryField1b>(even)
			.as_mut_slice::<u32>()[..advice.len()]
			.copy_from_slice(advice);
	}

	// Passing Checked flag here makes sure the number is actually even
	let half = arithmetic::u32::half(builder, "half", even, arithmetic::Flags::Checked)?;

	let even_packed = arithmetic::u32::packed(builder, "even_packed", even)?;
	builder.receive(channel, advice.len(), [even_packed]);

	let half_packed = arithmetic::u32::packed(builder, "half_packed", half)?;
	builder.send(channel, advice.len(), [half_packed]);

	Ok(())
}

fn collatz_odd<U, F>(
	builder: &mut ConstraintSystemBuilder<U, F>,
	channel: ChannelId,
	advice: &[u32],
) -> Result<(), anyhow::Error>
where
	U: PackScalar<F> + PackScalar<BinaryField1b> + PackScalar<BinaryField32b> + Pod,
	F: TowerField + ExtensionField<BinaryField32b>,
{
	let log_32b_rows = binius_utils::checked_arithmetics::log2_ceil_usize(advice.len());
	let log_1b_rows = 5 + log_32b_rows;

	let odd = builder.add_committed("odd", log_1b_rows, BinaryField1b::TOWER_LEVEL);
	if let Some(witness) = builder.witness() {
		witness
			.new_column::<BinaryField1b>(odd)
			.as_mut_slice::<u32>()[..advice.len()]
			.copy_from_slice(advice);
	}

	// Ensure the number is odd
	ensure_odd(builder, odd, advice.len())?;

	let one = arithmetic::u32::constant(builder, "one", log_32b_rows, 1)?;
	let triple = arithmetic::u32::mul_const(builder, "triple", odd, 3, arithmetic::Flags::Checked)?;
	let triple_plus_one =
		arithmetic::u32::add(builder, "triple_plus_one", triple, one, arithmetic::Flags::Checked)?;

	let odd_packed = arithmetic::u32::packed(builder, "odd_packed", odd)?;
	builder.receive(channel, advice.len(), [odd_packed]);

	let triple_plus_one_packed =
		arithmetic::u32::packed(builder, "triple_plus_one_packed", triple_plus_one)?;
	builder.send(channel, advice.len(), [triple_plus_one_packed]);

	Ok(())
}

pub fn ensure_odd<U, F>(
	builder: &mut ConstraintSystemBuilder<U, F>,
	input: OracleId,
	count: usize,
) -> Result<(), anyhow::Error>
where
	U: PackScalar<F> + PackScalar<BinaryField1b> + Pod,
	F: TowerField,
{
	let log_32b_rows = builder.log_rows([input])? - 5;
	let lsb = arithmetic::u32::select_bit(builder, "lsb", input, 0)?;
	let selector = transparent::step_down(builder, "count", log_32b_rows, count)?;
	builder.assert_zero(
		[lsb, selector],
		arith_expr!([lsb, selector] = selector * (lsb + 1)).convert_field(),
	);
	Ok(())
}

#[cfg(test)]
mod tests {
	use binius_core::constraint_system::{
		channel::{Boundary, FlushDirection},
		validate::validate_witness,
	};
	use binius_field::{arch::OptimalUnderlier, BinaryField128b, BinaryField32b};

	use crate::{builder::ConstraintSystemBuilder, collatz::collatz};

	#[test]
	fn test_collatz() {
		let allocator = bumpalo::Bump::new();
		let mut builder =
			ConstraintSystemBuilder::<OptimalUnderlier, BinaryField128b>::new_with_witness(
				&allocator,
			);

		let x0 = 5;
		let channel_id = collatz(&mut builder, x0).unwrap();

		let boundaries = vec![
			Boundary {
				values: vec![BinaryField32b::new(x0).into()],
				channel_id,
				direction: FlushDirection::Push,
				multiplicity: 1,
			},
			Boundary {
				values: vec![BinaryField32b::new(1).into()],
				channel_id,
				direction: FlushDirection::Pull,
				multiplicity: 1,
			},
		];
		let witness = builder.take_witness().unwrap();
		let constraint_system = builder.build().unwrap();
		validate_witness(&constraint_system, &boundaries, &witness).unwrap();
	}
}
