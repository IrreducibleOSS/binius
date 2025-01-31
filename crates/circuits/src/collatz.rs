// Copyright 2024-2025 Irreducible Inc.

use binius_core::{
	constraint_system::channel::{Boundary, ChannelId, FlushDirection},
	oracle::OracleId,
};
use binius_field::{
	as_packed_field::PackScalar, BinaryField1b, BinaryField32b, ExtensionField, TowerField,
};
use binius_macros::arith_expr;
use bytemuck::Pod;

use crate::{arithmetic, builder::ConstraintSystemBuilder, transparent};

pub type Advice = (usize, usize);

pub struct Collatz {
	x0: u32,
	evens: Vec<u32>,
	odds: Vec<u32>,
}

impl Collatz {
	pub const fn new(x0: u32) -> Self {
		Self {
			x0,
			evens: vec![],
			odds: vec![],
		}
	}

	pub fn init_prover(&mut self) -> Advice {
		let (evens, odds) = collatz_orbit(self.x0).into_iter().partition(|x| x % 2 == 0);
		self.evens = evens;
		self.odds = odds;

		(self.evens.len(), self.odds.len())
	}

	pub fn build<U, F>(
		self,
		builder: &mut ConstraintSystemBuilder<U, F>,
		advice: Advice,
	) -> Result<Vec<Boundary<F>>, anyhow::Error>
	where
		U: PackScalar<F> + PackScalar<BinaryField1b> + PackScalar<BinaryField32b> + Pod,
		F: TowerField + ExtensionField<BinaryField32b>,
	{
		let (evens_count, odds_count) = advice;

		let channel = builder.add_channel();

		self.even(builder, channel, evens_count)?;
		self.odd(builder, channel, odds_count)?;

		let boundaries = self.get_boundaries(channel);

		Ok(boundaries)
	}

	fn even<U, F>(
		&self,
		builder: &mut ConstraintSystemBuilder<U, F>,
		channel: ChannelId,
		count: usize,
	) -> Result<(), anyhow::Error>
	where
		U: PackScalar<F> + PackScalar<BinaryField1b> + PackScalar<BinaryField32b> + Pod,
		F: TowerField + ExtensionField<BinaryField32b>,
	{
		let log_1b_rows = 5 + binius_utils::checked_arithmetics::log2_ceil_usize(count);
		let even = builder.add_committed("even", log_1b_rows, BinaryField1b::TOWER_LEVEL);
		if let Some(witness) = builder.witness() {
			debug_assert_eq!(count, self.evens.len());
			witness
				.new_column::<BinaryField1b>(even)
				.as_mut_slice::<u32>()[..count]
				.copy_from_slice(&self.evens);
		}

		// Passing Checked flag here makes sure the number is actually even
		let half = arithmetic::u32::half(builder, "half", even, arithmetic::Flags::Checked)?;

		let even_packed = arithmetic::u32::packed(builder, "even_packed", even)?;
		builder.receive(channel, count, [even_packed])?;

		let half_packed = arithmetic::u32::packed(builder, "half_packed", half)?;
		builder.send(channel, count, [half_packed])?;

		Ok(())
	}

	fn odd<U, F>(
		&self,
		builder: &mut ConstraintSystemBuilder<U, F>,
		channel: ChannelId,
		count: usize,
	) -> Result<(), anyhow::Error>
	where
		U: PackScalar<F> + PackScalar<BinaryField1b> + PackScalar<BinaryField32b> + Pod,
		F: TowerField + ExtensionField<BinaryField32b>,
	{
		let log_32b_rows = binius_utils::checked_arithmetics::log2_ceil_usize(count);
		let log_1b_rows = 5 + log_32b_rows;

		let odd = builder.add_committed("odd", log_1b_rows, BinaryField1b::TOWER_LEVEL);
		if let Some(witness) = builder.witness() {
			debug_assert_eq!(count, self.odds.len());
			witness
				.new_column::<BinaryField1b>(odd)
				.as_mut_slice::<u32>()[..count]
				.copy_from_slice(&self.odds);
		}

		// Ensure the number is odd
		ensure_odd(builder, odd, count)?;

		let one = arithmetic::u32::constant(builder, "one", log_32b_rows, 1)?;
		let triple =
			arithmetic::u32::mul_const(builder, "triple", odd, 3, arithmetic::Flags::Checked)?;
		let triple_plus_one = arithmetic::u32::add(
			builder,
			"triple_plus_one",
			triple,
			one,
			arithmetic::Flags::Checked,
		)?;

		let odd_packed = arithmetic::u32::packed(builder, "odd_packed", odd)?;
		builder.receive(channel, count, [odd_packed])?;

		let triple_plus_one_packed =
			arithmetic::u32::packed(builder, "triple_plus_one_packed", triple_plus_one)?;
		builder.send(channel, count, [triple_plus_one_packed])?;

		Ok(())
	}

	fn get_boundaries<F>(&self, channel_id: usize) -> Vec<Boundary<F>>
	where
		F: TowerField + From<BinaryField32b>,
	{
		vec![
			Boundary {
				channel_id,
				direction: FlushDirection::Push,
				values: vec![BinaryField32b::new(self.x0).into()],
				multiplicity: 1,
			},
			Boundary {
				channel_id,
				direction: FlushDirection::Pull,
				values: vec![BinaryField32b::new(1).into()],
				multiplicity: 1,
			},
		]
	}
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
		"is_odd",
		[lsb, selector],
		arith_expr!([lsb, selector] = selector * (lsb + 1)).convert_field(),
	);
	Ok(())
}

#[cfg(test)]
mod tests {
	use binius_core::constraint_system::validate::validate_witness;
	use binius_field::{arch::OptimalUnderlier, BinaryField128b};

	use crate::{builder::ConstraintSystemBuilder, collatz::Collatz};

	#[test]
	fn test_collatz() {
		let allocator = bumpalo::Bump::new();
		let mut builder =
			ConstraintSystemBuilder::<OptimalUnderlier, BinaryField128b>::new_with_witness(
				&allocator,
			);

		let x0 = 9999999;

		let mut collatz = Collatz::new(x0);
		let advice = collatz.init_prover();

		let boundaries = collatz.build(&mut builder, advice).unwrap();

		let witness = builder.take_witness().unwrap();
		let constraint_system = builder.build().unwrap();
		validate_witness(&constraint_system, &boundaries, &witness).unwrap();
	}
}
