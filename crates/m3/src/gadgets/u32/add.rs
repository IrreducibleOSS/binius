// Copyright 2025 Irreducible Inc.

use binius_core::oracle::ShiftVariant;
use binius_field::{packed::set_packed_slice, Field, PackedExtension, PackedFieldIndexable};

use crate::builder::{column::Col, types::B1, witness::TableWitnessSegment, TableBuilder, B128};

/// A gadget for performing 32-bit integer addition on vertically-packed bit columns.
///
/// This gadget has input columns `xin` and `yin` for the two 32-bit integers to be added, and an
/// output column `zout`, and it constrains that `xin + yin = zout` as integers.
#[derive(Debug)]
pub struct U32Add {
	// Inputs
	pub xin: Col<B1, 32>,
	pub yin: Col<B1, 32>,

	// Private
	cin: Col<B1, 32>,
	cout: Col<B1, 32>,
	cout_shl: Col<B1, 32>,

	// Outputs
	/// The output column, either committed if `flags.commit_zout` is set, otherwise a linear
	/// combination derived column.
	pub zout: Col<B1, 32>,
	/// This is `Some` if `flags.expose_final_carry` is set, otherwise it is `None`.
	pub final_carry: Option<Col<B1>>,
	/// Flags modifying the gadget's behavior.
	pub flags: U32AddFlags,
}

/// Flags modifying the behavior of the [`U32Add`] gadget.
#[derive(Debug, Default, Clone)]
pub struct U32AddFlags {
	// Optionally a column for a dynamic carry in bit. This *must* be zero in all bits except the
	// 0th.
	pub carry_in_bit: Option<Col<B1, 32>>,
	pub commit_zout: bool,
	pub expose_final_carry: bool,
}

impl U32Add {
	pub fn new(
		table: &mut TableBuilder,
		xin: Col<B1, 32>,
		yin: Col<B1, 32>,
		flags: U32AddFlags,
	) -> Self {
		let cout = table.add_committed::<B1, 32>("cout");
		let cout_shl = table.add_shifted("cout_shl", cout, 5, 1, ShiftVariant::LogicalLeft);

		let cin = if let Some(carry_in_bit) = flags.carry_in_bit {
			table.add_computed("cin", cout_shl + carry_in_bit)
		} else {
			cout_shl
		};

		let final_carry = flags
			.expose_final_carry
			.then(|| table.add_selected("final_carry", cout, 31));

		table.assert_zero("carry_out", (xin + cin) * (yin + cin) + cin - cout);

		let zout = if flags.commit_zout {
			let zout = table.add_committed::<B1, 32>("zout");
			table.assert_zero("zout", xin + yin + cin - zout);
			zout
		} else {
			table.add_computed("zout", xin + yin + cin)
		};

		Self {
			xin,
			yin,
			cin,
			cout,
			cout_shl,
			final_carry,
			zout,
			flags,
		}
	}

	pub fn populate<P>(&self, index: &mut TableWitnessSegment<P>) -> Result<(), anyhow::Error>
	where
		P: PackedFieldIndexable<Scalar = B128> + PackedExtension<B1>,
	{
		let xin: std::cell::RefMut<'_, [u32]> = index.get_mut_as(self.xin)?;
		let yin = index.get_mut_as(self.yin)?;
		let mut cout = index.get_mut_as(self.cout)?;
		let mut zout = index.get_mut_as(self.zout)?;
		let mut final_carry = if let Some(final_carry) = self.final_carry {
			let final_carry = index.get_mut(final_carry)?;
			Some(final_carry)
		} else {
			None
		};

		if let Some(carry_in_bit_col) = self.flags.carry_in_bit {
			// This is u32 assumed to be either 0 or 1.
			let carry_in_bit = index.get_mut_as(carry_in_bit_col)?;

			let mut cin = index.get_mut_as(self.cin)?;
			let mut cout_shl = index.get_mut_as(self.cout_shl)?;
			for i in 0..index.size() {
				let (x_plus_y, carry0) = xin[i].overflowing_add(yin[i]);
				let carry1;
				(zout[i], carry1) = x_plus_y.overflowing_add(carry_in_bit[i]);
				let carry = carry0 | carry1;

				cin[i] = xin[i] ^ yin[i] ^ zout[i];
				cout[i] = (carry as u32) << 31 | cin[i] >> 1;
				cout_shl[i] = cout[i] << 1;

				if let Some(ref mut final_carry) = final_carry {
					set_packed_slice(&mut *final_carry, i, if carry { B1::ONE } else { B1::ZERO });
				}
			}
		} else {
			// When the carry in bit is fixed to zero, we can simplify the logic.
			let mut cin = index.get_mut_as(self.cin)?;
			for i in 0..index.size() {
				let carry;
				(zout[i], carry) = xin[i].overflowing_add(yin[i]);
				cin[i] = xin[i] ^ yin[i] ^ zout[i];
				cout[i] = (carry as u32) << 31 | cin[i] >> 1;
				if let Some(ref mut final_carry) = final_carry {
					set_packed_slice(&mut *final_carry, i, if carry { B1::ONE } else { B1::ZERO });
				}
			}
		};
		Ok(())
	}
}

#[cfg(test)]
mod tests {
	use binius_field::{
		arch::OptimalUnderlier128b, as_packed_field::PackedType, packed::get_packed_slice,
	};
	use bumpalo::Bump;
	use rand::{prelude::StdRng, Rng as _, SeedableRng};

	use super::*;
	use crate::builder::{ConstraintSystem, Statement, WitnessIndex};

	#[test]
	fn test_basic() {
		const TABLE_SZ: usize = 1 << 14;

		let mut cs = ConstraintSystem::new();
		let mut table = cs.add_table("u32_add test");

		let xin = table.add_committed::<B1, 32>("xin");
		let yin = table.add_committed::<B1, 32>("yin");

		let adder = U32Add::new(&mut table, xin, yin, U32AddFlags::default());
		let table_id = table.id();
		let statement = Statement {
			boundaries: vec![],
			table_sizes: vec![TABLE_SZ],
		};
		let allocator = Bump::new();
		let mut witness =
			WitnessIndex::<PackedType<OptimalUnderlier128b, B128>>::new(&cs, &allocator);

		let table_witness = witness.init_table(table_id, TABLE_SZ).unwrap();
		let mut segment = table_witness.full_segment();

		let mut rng = StdRng::seed_from_u64(0);

		// Generate random u32 operands and expected results.
		//
		// (x, y, z)
		let test_vector: Vec<(u32, u32, u32)> = (0..segment.size())
			.map(|_| {
				let x = rng.gen::<u32>();
				let y = rng.gen::<u32>();
				let z = x.wrapping_add(y);
				(x, y, z)
			})
			.collect();

		{
			let mut xin_bits = segment.get_mut_as::<u32, _, 32>(adder.xin).unwrap();
			let mut yin_bits = segment.get_mut_as::<u32, _, 32>(adder.yin).unwrap();
			for (i, (x, y, _)) in test_vector.iter().enumerate() {
				xin_bits[i] = *x;
				yin_bits[i] = *y;
			}
		}

		// Populate the gadget
		adder.populate(&mut segment).unwrap();

		{
			// Verify results
			let zout_bits = segment.get_as::<u32, _, 32>(adder.zout).unwrap();
			for (i, (_, _, z)) in test_vector.iter().enumerate() {
				assert_eq!(zout_bits[i], *z);
			}
		}

		// Validate constraint system
		let ccs = cs.compile(&statement).unwrap();
		let witness = witness.into_multilinear_extension_index();

		binius_core::constraint_system::validate::validate_witness(&ccs, &[], &witness).unwrap();
	}

	#[test]
	fn test_add_with_carry() {
		const TABLE_SZ: usize = 1 << 2;

		let mut cs = ConstraintSystem::new();
		let mut table = cs.add_table("u32_add_with_carry test");

		let xin = table.add_committed::<B1, 32>("xin");
		let yin = table.add_committed::<B1, 32>("yin");
		let carry_in = table.add_committed::<B1, 32>("carry_in");

		let flags = U32AddFlags {
			carry_in_bit: Some(carry_in),
			expose_final_carry: true,
			commit_zout: false,
		};
		let adder = U32Add::new(&mut table, xin, yin, flags);
		let table_id = table.id();
		let statement = Statement {
			boundaries: vec![],
			table_sizes: vec![TABLE_SZ],
		};
		let allocator = Bump::new();
		let mut witness =
			WitnessIndex::<PackedType<OptimalUnderlier128b, B128>>::new(&cs, &allocator);

		let table_witness = witness.init_table(table_id, TABLE_SZ).unwrap();
		let mut segment = table_witness.full_segment();

		// The test vector that contains interesting cases.
		//
		// (x, y, carry_in, zout, final_carry)
		let test_vector = [
			(0xFFFFFFFF, 0x00000001, 0x00000000, 0x00000000, true),
			(0xFFFFFFFF, 0x00000000, 0x00000000, 0xFFFFFFFF, false),
			(0x7FFFFFFF, 0x00000001, 0x00000000, 0x80000000, false),
			(0xFFFF0000, 0x0000FFFF, 0x00000001, 0x00000000, true),
		];
		assert_eq!(test_vector.len(), segment.size());

		{
			// Populate the columns with the inputs from the test vector.
			let mut xin_bits = segment.get_mut_as::<u32, _, 32>(adder.xin).unwrap();
			let mut yin_bits = segment.get_mut_as::<u32, _, 32>(adder.yin).unwrap();
			let mut carry_in_bits = segment.get_mut_as::<u32, _, 32>(carry_in).unwrap();
			for (i, (x, y, carry, _, _)) in test_vector.iter().enumerate() {
				xin_bits[i] = *x;
				yin_bits[i] = *y;
				carry_in_bits[i] = *carry;
			}
		}

		// Populate the gadget
		adder.populate(&mut segment).unwrap();

		{
			// Verify results
			let zout_bits = segment.get_as::<u32, _, 32>(adder.zout).unwrap();
			let final_carry = segment.get(adder.final_carry.unwrap()).unwrap();

			for (i, (_, _, _, zout, expected_carry)) in test_vector.iter().enumerate() {
				assert_eq!(zout_bits[i], *zout);

				// Check final carry bit
				assert_eq!(get_packed_slice(&final_carry, i), B1::from(*expected_carry));
			}
		}

		// Validate constraint system
		let ccs = cs.compile(&statement).unwrap();
		let witness = witness.into_multilinear_extension_index();

		binius_core::constraint_system::validate::validate_witness(&ccs, &[], &witness).unwrap();
	}
}
