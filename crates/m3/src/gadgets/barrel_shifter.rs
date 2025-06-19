// Copyright 2025 Irreducible Inc.

use std::cell::RefMut;

use binius_core::oracle::ShiftVariant;
use binius_field::{Field, PackedExtension, PackedFieldIndexable, packed::set_packed_slice};

use crate::builder::{B1, B32, B128, Col, Expr, TableBuilder, TableWitnessSegment, upcast_col};

/// Maximum number of bits of the shift amount, i.e. 0 < shift_amount < 1 <<
/// SHIFT_MAX_BITS - 1 = 31 where dst_val = src_val >> shift_amount or dst_val =
/// src_val << shift_amount
const MAX_SHIFT_BITS: usize = 5;

/// A gadget for performing a barrel shift circuit (<https://en.wikipedia.org/wiki/Barrel_shifter>).
///
/// The `BarrelShifter` gadget allows for left shifts, right shifts, and
/// rotations on 32-bit inputs, with a configurable shift amount and direction.
pub struct BarrelShifter {
	/// The input column representing the 32-bit value to be shifted.
	input: Col<B1, 32>,

	/// The shift amount column representing the 5 bits of positions to shift,
	/// ignoring the remaining 11.
	shift_amount: Col<B1, 16>,

	/// Virtual columns containing the binary decomposition of the shifted amount.
	shift_amount_bits: [Col<B1>; MAX_SHIFT_BITS],

	// TODO: Try to replace the Vec with an array.
	/// Partial shift virtual columns containing the partial_shift[i - 1]
	/// shifted by 2^i.
	shifted: Vec<Col<B1, 32>>, // Virtual

	/// Partial shift virtual columns containing either shifted[i] or partial_shit[i-1],
	/// depending on the value of `shift_amount_bits`.
	partial_shift: [Col<B1, 32>; MAX_SHIFT_BITS],

	/// The output column representing the result of the shift operation. This column is
	/// virtual or committed, depending on the flags
	pub output: Col<B1, 32>,

	/// The variant of the shift operation: logical left, logical right or
	/// circular left.
	pub variant: ShiftVariant,
}

impl BarrelShifter {
	/// Creates a new instance of the `BarrelShifter` gadget.
	///
	/// # Arguments
	///
	/// * `table` - A mutable reference to the `TableBuilder` used to define the gadget.
	/// * `input` - The input column of type `Col<B1, 32>`.
	/// * `shift_amount` - The shift amount column of type `Col<B1, 16>`. The 11 most significant
	///   bits are ignored.
	/// * `variant` - Indicates whether the circuits performs a logical left, logical right, or
	///   circular left shift.
	///
	/// # Returns
	///
	/// A new instance of the `BarrelShifter` gadget.
	pub fn new(
		table: &mut TableBuilder,
		input: Col<B1, 32>,
		shift_amount: Col<B1, 16>,
		variant: ShiftVariant,
	) -> Self {
		let partial_shift =
			core::array::from_fn(|i| table.add_committed(format!("partial_shift_{i}")));
		let shift_amount_bits: [_; MAX_SHIFT_BITS] = core::array::from_fn(|i| {
			table.add_selected(format!("shift_amount_bits_{i}"), shift_amount, i)
		});
		let mut shifted = Vec::with_capacity(MAX_SHIFT_BITS);
		let mut current_shift = input;
		for i in 0..MAX_SHIFT_BITS {
			shifted.push(table.add_shifted("shifted", current_shift, 5, 1 << i, variant));
			let partial_shift_packed: Col<B32> =
				table.add_packed(format!("partial_shift_packed_{i}"), partial_shift[i]);
			let shifted_packed: Expr<B32, 1> = table
				.add_packed(format!("shifted_packed_{i}"), shifted[i])
				.into();
			let current_shift_packed: Col<B32> =
				table.add_packed(format!("current_shift_packed_{i}"), current_shift);
			table.assert_zero(
				format!("correct_partial_shift_{i}"),
				partial_shift_packed
					- (shifted_packed * upcast_col(shift_amount_bits[i])
						+ current_shift_packed * (upcast_col(shift_amount_bits[i]) + B32::ONE)),
			);
			current_shift = partial_shift[i];
		}

		Self {
			input,
			shift_amount,
			shift_amount_bits,
			shifted,
			partial_shift,
			output: current_shift,
			variant,
		}
	}

	/// Populates the table with witness values for the barrel shifter.
	///
	/// # Arguments
	///
	/// * `index` - A mutable reference to the `TableWitness` used to populate the table.
	///
	/// # Returns
	///
	/// A `Result` indicating success or failure.
	pub fn populate<P>(&self, index: &mut TableWitnessSegment<P>) -> Result<(), anyhow::Error>
	where
		P: PackedFieldIndexable<Scalar = B128> + PackedExtension<B1>,
	{
		let input: RefMut<'_, [u32]> = index.get_mut_as(self.input).unwrap();
		let shift_amount: RefMut<'_, [u16]> = index.get_mut_as(self.shift_amount).unwrap();
		let mut partial_shift: [_; MAX_SHIFT_BITS] =
			array_util::try_from_fn(|i| index.get_mut_as(self.partial_shift[i]))?;
		let mut shifted: [_; MAX_SHIFT_BITS] =
			array_util::try_from_fn(|i| index.get_mut_as(self.shifted[i]))?;
		let mut shift_amount_bits: [_; MAX_SHIFT_BITS] =
			array_util::try_from_fn(|i| index.get_mut(self.shift_amount_bits[i]))?;

		for i in 0..index.size() {
			let mut current_shift = input[i];
			for j in 0..MAX_SHIFT_BITS {
				let bit = ((shift_amount[i] >> j) & 1) == 1;
				set_packed_slice(&mut shift_amount_bits[j], i, B1::from(bit));
				shifted[j][i] = match self.variant {
					ShiftVariant::LogicalLeft => current_shift << (1 << j),
					ShiftVariant::LogicalRight => current_shift >> (1 << j),
					ShiftVariant::CircularLeft => {
						(current_shift << (1 << j)) | (current_shift >> (32 - (1 << j)))
					}
				};
				if bit {
					current_shift = shifted[j][i];
				}
				partial_shift[j][i] = current_shift;
			}
		}
		Ok(())
	}
}

#[cfg(test)]
mod tests {
	use std::iter::repeat_with;

	use binius_compute::cpu::alloc::CpuComputeAllocator;
	use binius_field::{arch::OptimalUnderlier128b, as_packed_field::PackedType};
	use rand::{Rng, SeedableRng, rngs::StdRng};

	use super::*;
	use crate::builder::{ConstraintSystem, WitnessIndex};

	fn test_barrel_shifter(variant: ShiftVariant) {
		let mut cs = ConstraintSystem::new();
		let mut table = cs.add_table("BarrelShifterTable");
		let table_id = table.id();
		let mut allocator = CpuComputeAllocator::new(1 << 12);
		let allocator = allocator.into_bump_allocator();

		let input = table.add_committed::<B1, 32>("input");
		let shift_amount = table.add_committed::<B1, 16>("shift_amount");

		let shifter = BarrelShifter::new(&mut table, input, shift_amount, variant);

		let mut witness =
			WitnessIndex::<PackedType<OptimalUnderlier128b, B128>>::new(&cs, &allocator);
		let table_witness = witness.init_table(table_id, 1 << 8).unwrap();
		let mut segment = table_witness.full_segment();

		let mut rng = StdRng::seed_from_u64(0x1234);
		let test_inputs = repeat_with(|| rng.random())
			.take(1 << 8)
			.collect::<Vec<u32>>();

		for (i, (input, shift_amount)) in (*segment.get_mut_as(input).unwrap())
			.iter_mut()
			.zip(segment.get_mut_as(shift_amount).unwrap().iter_mut())
			.enumerate()
		{
			*input = test_inputs[i];
			*shift_amount = i as u16; // Only the first 5 bits are used
		}

		shifter.populate(&mut segment).unwrap();

		for (i, &output) in segment
			.get_as::<u32, B1, 32>(shifter.output)
			.unwrap()
			.iter()
			.enumerate()
		{
			let expected_output = match variant {
				ShiftVariant::LogicalLeft => test_inputs[i] << (i % 32),
				ShiftVariant::LogicalRight => test_inputs[i] >> (i % 32),
				ShiftVariant::CircularLeft => test_inputs[i].rotate_left(i as u32 % 32),
			};
			assert_eq!(output, expected_output);
		}

		let ccs = cs.compile().unwrap();
		let table_sizes = witness.table_sizes();
		let witness = witness.into_multilinear_extension_index();

		binius_core::constraint_system::validate::validate_witness(
			&ccs,
			&[],
			&table_sizes,
			&witness,
		)
		.unwrap();
	}

	#[test]
	fn test_barrel_shifter_logical_left() {
		test_barrel_shifter(ShiftVariant::LogicalLeft);
	}

	#[test]
	fn test_barrel_shifter_logical_right() {
		test_barrel_shifter(ShiftVariant::LogicalRight);
	}

	#[test]
	fn test_barrel_shifter_circular_left() {
		test_barrel_shifter(ShiftVariant::CircularLeft);
	}
}
