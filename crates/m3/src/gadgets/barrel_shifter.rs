// Copyright 2025 Irreducible Inc.

use std::cell::RefMut;

use binius_core::oracle::ShiftVariant;
use binius_field::{packed::set_packed_slice, Field, PackedExtension, PackedFieldIndexable};

use crate::builder::{upcast_col, Col, Expr, TableBuilder, TableWitnessSegment, B1, B128, B32};

/// Maximum number of bits of the shift amount, i.e. 0 < shift_amount < 1 <<
/// SHIFT_MAX_BITS - 1 = 31 where dst_val = src_val >> shift_amount or dst_val =
/// src_val << shift_amount
const MAX_SHIFT_BITS: usize = 5;

/// Flags to configure the behavior of the barrel shifter.
pub struct BarrelShifterFlags {
	/// The variant of the shift operation: logical left, logical right or
	/// circular left.
	pub(crate) variant: ShiftVariant,
	/// Whether the output column should be committed or computed.
	pub(crate) commit_output: bool,
}

/// A gadget for performing a barrel shift circuit (<https://en.wikipedia.org/wiki/Barrel_shifter>).
///
/// The `BarrelShifter` gadget allows for left shifts, right shifts, and
/// rotations on 32-bit inputs, with a configurable shift amount and direction.
pub struct BarrelShifter {
	/// The input column representing the 32-bit value to be shifted.
	input: Col<B1, 32>,

	/// The shift amount column representing the 5 of positions to shift,
	/// ignoring the remaining 11.
	shift_amount: Col<B1, 16>,

	/// Binary decomposition of the shifted amount.
	shift_amount_bits: [Col<B1>; MAX_SHIFT_BITS], // Virtual

	// TODO: Try to replace the Vec with an array.
	/// partial shift columns containing the partia_shift[i - 1]
	/// shifted by 2^i.
	shifted: Vec<Col<B1, 32>>, // Virtual

	/// Partial shift columns containing either shifted[i] or partial_shit[i-1],
	/// depending on the value of `shift_amount_bits`.
	partial_shift: [Col<B1, 32>; MAX_SHIFT_BITS], // Virtual

	/// The output column representing the result of the shift operation.
	pub output: Col<B1, 32>, // Virtual or commited, depending on the flags

	/// Flags to configure the behavior of the barrel shifter (e.g., rotation,
	/// right shift).
	flags: BarrelShifterFlags,
}

impl BarrelShifter {
	/// Creates a new instance of the `BarrelShifter` gadget.
	///
	/// # Arguments
	///
	/// * `table` - A mutable reference to the `TableBuilder` used to define the
	///   gadget.
	/// * `flags` - A `BarrelShifterFlags` struct that configures the behavior
	///   of the gadget.
	///
	/// # Returns
	///
	/// A new instance of the `BarrelShifter` gadget.
	pub fn new(
		table: &mut TableBuilder,
		input: Col<B1, 32>,
		shift_amount: Col<B1, 16>,
		flags: BarrelShifterFlags,
	) -> Self {
		let partial_shift =
			core::array::from_fn(|i| table.add_committed(format!("partial_shift_{i}")));
		let shift_amount_bits: [_; MAX_SHIFT_BITS] = core::array::from_fn(|i| {
			table.add_selected(format!("shift_amount_bits_{i}"), shift_amount, i)
		});
		let mut shifted = Vec::with_capacity(MAX_SHIFT_BITS);
		let mut current_shift = input;
		for i in 0..MAX_SHIFT_BITS {
			shifted.push(table.add_shifted("shifted", current_shift, 5, 1 << i, flags.variant));
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

		// Define the output column (32 bits).
		let output = if flags.commit_output {
			// If the output is committed, add a committed column and enforce constraints.
			let output = table.add_committed::<B1, 32>("output");
			table.assert_zero("output_constraint", current_shift - output);
			output
		} else {
			current_shift
		};

		Self {
			input,
			shift_amount,
			shift_amount_bits,
			shifted,
			partial_shift,
			output,
			flags,
		}
	}

	/// Populates the table with witness values for the barrel shifter.
	///
	/// # Arguments
	///
	/// * `witness` - A mutable reference to the `TableWitness` used to populate
	///   the table.
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
		// TODO: Propagate the errors
		let mut partial_shift: [_; MAX_SHIFT_BITS] =
			core::array::try_from_fn(|i| index.get_mut_as(self.partial_shift[i]))?;
		let mut shifted: [_; MAX_SHIFT_BITS] =
			core::array::try_from_fn(|i| index.get_mut_as(self.shifted[i]))?;
		let mut shift_amount_bits: [_; MAX_SHIFT_BITS] =
			core::array::try_from_fn(|i| index.get_mut(self.shift_amount_bits[i]))?;

		for i in 0..index.size() {
			let mut current_shift = input[i];
			for j in 0..MAX_SHIFT_BITS {
				let bit = ((shift_amount[i] >> j) & 1) == 1;
				set_packed_slice(&mut shift_amount_bits[j], i, B1::from(bit));
				shifted[j][i] = match self.flags.variant {
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
	use binius_field::{arch::OptimalUnderlier128b, as_packed_field::PackedType};
	use bumpalo::Bump;

	use super::*;
	use crate::builder::{ConstraintSystem, Statement, WitnessIndex};

	fn test_barrel_shifter(input_val: u32, variant: ShiftVariant) {
		let mut cs = ConstraintSystem::new();
		let mut table = cs.add_table("BarrelShifterTable");
		let table_id = table.id();
		let allocator = Bump::new();

		let input = table.add_committed::<B1, 32>("input");
		let shift_amount = table.add_committed::<B1, 16>("shift_amount");

		let shifter = BarrelShifter::new(
			&mut table,
			input,
			shift_amount,
			BarrelShifterFlags {
				variant,
				commit_output: false,
			},
		);

		let statement = Statement {
			boundaries: vec![],
			table_sizes: vec![1 << 8],
		};
		let mut witness =
			WitnessIndex::<PackedType<OptimalUnderlier128b, B128>>::new(&cs, &allocator);
		let table_witness = witness.init_table(table_id, 1 << 8).unwrap();
		let mut segment = table_witness.full_segment();

		for (i, (input, shift_amount)) in (*segment.get_mut_as(input).unwrap())
			.iter_mut()
			.zip(segment.get_mut_as(shift_amount).unwrap().iter_mut())
			.enumerate()
		{
			*input = input_val;
			*shift_amount = i as u16; // Only the first 5 bits are used
		}

		shifter.populate(&mut segment).unwrap();

		for (i, &output) in segment
			.get_as::<u32, B1, 32>(shifter.output)
			.unwrap()
			.iter()
			.enumerate()
		{
			let i = i % 32;
			println!("i: {}, output: {:#x}", i, output);
			let expected_output = match variant {
				ShiftVariant::LogicalLeft => input_val << (i % 32),
				ShiftVariant::LogicalRight => input_val >> (i % 32),
				ShiftVariant::CircularLeft => {
					(input_val << (i % 32)) | (input_val >> ((32 - i) % 32))
				}
			};
			assert_eq!(output, expected_output);
		}

		let ccs = cs.compile(&statement).unwrap();
		let witness = witness.into_multilinear_extension_index();

		binius_core::constraint_system::validate::validate_witness(&ccs, &[], &witness).unwrap();
	}

	#[test]
	fn test_barrel_shifter_logical_left() {
		test_barrel_shifter(0x1234, ShiftVariant::LogicalLeft);
	}

	#[test]
	fn test_barrel_shifter_logical_right() {
		test_barrel_shifter(0x1234, ShiftVariant::LogicalRight);
	}

	#[test]
	fn test_barrel_shifter_circular_left() {
		test_barrel_shifter(0x1234, ShiftVariant::CircularLeft);
	}
}
