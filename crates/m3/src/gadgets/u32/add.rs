// Copyright 2025 Irreducible Inc.

use std::{array, marker::PhantomData};

use binius_core::oracle::ShiftVariant;
use binius_field::{
	packed::set_packed_slice, Field, PackedExtension, PackedField, PackedFieldIndexable,
	PackedSubfield, TowerField,
};
use itertools::izip;

use crate::builder::{
	column::Col, types::B1, witness::TableWitnessSegment, TableBuilder, B128, B32, B64,
};
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
use std::{array, marker::PhantomData, ops::Deref};
/// Gadget for unsigned addition using non-packed one-bit columns generic over `u32` and `u64`

#[derive(Debug)]
pub struct WideAdd<UX: UnsignedAddPrimitives, const BIT_LENGTH: usize> {
	pub x_in: [Col<B1>; BIT_LENGTH],
	pub y_in: [Col<B1>; BIT_LENGTH],

	cin: [Col<B1>; BIT_LENGTH],

	pub z_out: [Col<B1>; BIT_LENGTH],
	pub final_carry_out: Col<B1>,
	/// This gadget always exposes the final carry bit.
	pub flags: U32AddFlags,
	// TODO: Maybe make a serperate flag for handling unsigned adds of arbitrary bit length.
	_marker: PhantomData<UX>,
}
impl<UX: UnsignedAddPrimitives, const BIT_LENGTH: usize> WideAdd<UX, BIT_LENGTH> {
	pub fn new(
		table: &mut TableBuilder,
		x_in: [Col<B1>; BIT_LENGTH],
		y_in: [Col<B1>; BIT_LENGTH],
		flags: U32AddFlags,
	) -> Self {
		assert_eq!(BIT_LENGTH, UX::BIT_LENGTH);
		let cin = array::from_fn(|i| {
			if i != 0 {
				table.add_committed(format!("cin[{i}]"))
			} else {
				flags
					.carry_in_bit
					.map(|carry_col| table.add_selected("cin[0]", carry_col, 0))
					.unwrap_or_else(|| table.add_committed("cin[0]"))
			}
		});
		let zout = if flags.commit_zout {
			let zout = table.add_committed_multiple("z_out");
			for bit in 0..BIT_LENGTH {
				table.assert_zero(
					format!("sum_{bit}"),
					x_in[bit] + y_in[bit] + cin[bit] - zout[bit],
				);
			}
			zout
		} else {
			array::from_fn(|i| table.add_computed(format!("zout_{i}"), x_in[i] + y_in[i] + cin[i]))
		};
		let final_carry_out = table.add_committed("final_carry_out");
		let cout: [_; BIT_LENGTH] = array::from_fn(|i| {
			if i != BIT_LENGTH - 1 {
				cin[i + 1]
			} else {
				final_carry_out
			}
		});
		for bit in 0..BIT_LENGTH {
			table.assert_zero(
				format!("carry_{bit}"),
				(x_in[bit] + cin[bit]) * (y_in[bit] + cin[bit]) + cin[bit] - cout[bit],
			);
		}

		Self {
			x_in,
			y_in,
			cin,
			z_out: zout,
			final_carry_out,
			flags,
			_marker: PhantomData,
		}
	}

	pub fn populate<P>(&self, index: &mut TableWitnessSegment<P>) -> Result<(), anyhow::Error>
	where
		P: PackedField<Scalar = B128> + PackedExtension<B1>,
	{
		let x_in = array_util::try_map(self.x_in, |bit| index.get(bit))?;
		let y_in = array_util::try_map(self.y_in, |bit| index.get(bit))?;
		let mut cin = array_util::try_map(self.cin, |bit| index.get_mut(bit))?;
		let mut zout = array_util::try_map(self.z_out, |bit| index.get_mut(bit))?;
		let mut final_carry_out = index.get_mut(self.final_carry_out)?;

		let mut carry_in = if self.flags.carry_in_bit.is_none() {
			let initial_carry = PackedSubfield::<P, B1>::default();
			vec![initial_carry; cin[0].len()]
		} else {
			index
				.get(self.flags.carry_in_bit.expect("carry_in_bit is not None"))?
				.deref()
				.to_vec()
		};
		for bit in 0..BIT_LENGTH {
			for (x_in_packed, y_in_packed, c_in_packed, zout_packed, carry_in) in izip!(
				x_in[bit].iter(),
				y_in[bit].iter(),
				cin[bit].iter_mut(),
				zout[bit].iter_mut(),
				carry_in.iter_mut()
			) {
				let sum = *x_in_packed + *y_in_packed + *carry_in;
				let new_carry = (*x_in_packed + *carry_in) * (*y_in_packed + *carry_in) + *carry_in;
				*c_in_packed = *carry_in;
				*zout_packed = sum;
				*carry_in = new_carry;
			}
		}

		for (final_carry, carry) in izip!(final_carry_out.iter_mut(), carry_in.iter()) {
			*final_carry = *carry;
		}

		Ok(())
	}
}
#[derive(Debug)]

/// A very simple trait used in Addition gadgets for unsigned integers of different bit lengths.
pub trait UnsignedAddPrimitives {
	type F: TowerField;
	const BIT_LENGTH: usize;
}

impl UnsignedAddPrimitives for u32 {
	type F = B32;

	const BIT_LENGTH: usize = 32;
}

impl UnsignedAddPrimitives for u64 {
	type F = B64;

	const BIT_LENGTH: usize = 64;
}

/// Gadget for incrementing the values in a column by 1, generic over `u32` and `u64`
#[derive(Debug)]
pub struct Incr<UPrimitive: UnsignedAddPrimitives, const BIT_LENGTH: usize> {
	cin: [Col<B1>; BIT_LENGTH],

	/// Input column as bits.
	pub input: [Col<B1>; BIT_LENGTH],
	/// Output column as bits.
	pub zout: [Col<B1>; BIT_LENGTH],
	/// Output carry bit if there is any.
	pub final_carry_out: Col<B1>,

	_marker: PhantomData<UPrimitive>,
}

impl<UPrimitive: UnsignedAddPrimitives, const BIT_LENGTH: usize> Incr<UPrimitive, BIT_LENGTH> {
	pub fn new(table: &mut TableBuilder, input: [Col<B1>; BIT_LENGTH]) -> Self {
		assert_eq!(BIT_LENGTH, UPrimitive::BIT_LENGTH);
		let cin = array::from_fn(|i| {
			if i != 0 {
				table.add_committed(format!("cout[{i}]"))
			} else {
				table.add_constant("cout[0]", [B1::ONE])
			}
		});
		let zout = table.add_committed_multiple("zout");
		let final_carry_out = table.add_committed("final_carry_out");
		let cout: [_; BIT_LENGTH] = array::from_fn(|i| {
			if i != BIT_LENGTH - 1 {
				cin[i + 1]
			} else {
				final_carry_out
			}
		});

		for (i, &xin) in input.iter().enumerate() {
			table.assert_zero(format!("sum[{i}]"), xin + cin[i] - zout[i]);
			table.assert_zero(format!("carry[{i}]"), xin * cin[i] - cout[i]);
		}

		Self {
			cin,
			input,
			zout,
			final_carry_out,
			_marker: PhantomData,
		}
	}

	pub fn populate<P>(&self, index: &mut TableWitnessSegment<P>) -> Result<(), anyhow::Error>
	where
		P: PackedField<Scalar = B128> + PackedExtension<B1>,
	{
		// Get the input bits (read-only)
		let in_bits = array_util::try_map(self.input, |bit| index.get(bit))?;
		// Get the carry-in bits
		let mut cin = array_util::try_map(self.cin, |bit| index.get_mut(bit))?;
		// Get the output bits
		let mut zout = array_util::try_map(self.zout, |bit| index.get_mut(bit))?;
		let mut final_carry_out = index.get_mut(self.final_carry_out)?;

		let p_inv = PackedSubfield::<P, B1>::broadcast(B1::ONE);
		let mut carry_in = vec![p_inv; cin[0].len()];
		for bit in 0..BIT_LENGTH {
			for (in_bit, cin, zout, carry_in) in izip!(
				in_bits[bit].iter(),
				cin[bit].iter_mut(),
				zout[bit].iter_mut(),
				carry_in.iter_mut()
			) {
				let sum = *in_bit + *carry_in;
				let new_carry = (*in_bit) * (*carry_in);
				*cin = *carry_in;
				*zout = sum;
				*carry_in = new_carry;
			}
		}
		for (final_carry, carry) in izip!(final_carry_out.iter_mut(), carry_in.iter()) {
			*final_carry = *carry;
		}

		Ok(())
	}
}

pub type U32Incr = Incr<u32, 32>;
pub type U64Incr = Incr<u64, 64>;

#[cfg(test)]
mod tests {
	use std::iter::repeat_with;

	use binius_field::{
		arch::OptimalUnderlier128b, as_packed_field::PackedType, packed::get_packed_slice,
	};
	use bumpalo::Bump;
	use rand::{prelude::StdRng, Rng as _, SeedableRng};

	use super::*;
	use crate::builder::{
		test_utils::validate_system_witness, ConstraintSystem, Statement, WitnessIndex,
	};

	#[test]
	fn prop_test_no_carry() {
		const N_ITER: usize = 1 << 14;

		let mut rng = StdRng::seed_from_u64(0);
		let test_vector: Vec<(u32, u32, u32, u32, bool)> = (0..N_ITER)
			.map(|_| {
				let x: u32 = rng.gen();
				let y: u32 = rng.gen();
				let (z, carry) = x.overflowing_add(y);
				// (x, y, carry_in, zout, final_carry)
				(x, y, 0x00000000, z, carry)
			})
			.collect();

		TestPlan {
			dyn_carry_in: false,
			expose_final_carry: true,
			commit_zout: false,
			test_vector,
		}
		.execute();
	}

	#[test]
	fn prop_test_with_carry() {
		const N_ITER: usize = 1 << 14;

		let mut rng = StdRng::seed_from_u64(0);
		let test_vector: Vec<(u32, u32, u32, u32, bool)> = (0..N_ITER)
			.map(|_| {
				let x: u32 = rng.gen();
				let y: u32 = rng.gen();
				let carry_in = rng.gen::<bool>() as u32;
				let (x_plus_y, carry1) = x.overflowing_add(y);
				let (z, carry2) = x_plus_y.overflowing_add(carry_in);
				let final_carry = carry1 | carry2;
				(x, y, carry_in, z, final_carry)
			})
			.collect();

		TestPlan {
			dyn_carry_in: true,
			expose_final_carry: true,
			commit_zout: true,
			test_vector,
		}
		.execute();
	}

	#[test]
	fn test_add_with_carry() {
		// (x, y, carry_in, zout, final_carry)
		let test_vector = [
			(0xFFFFFFFF, 0x00000001, 0x00000000, 0x00000000, true), // max + 1 = 0 (overflow)
			(0xFFFFFFFF, 0x00000000, 0x00000000, 0xFFFFFFFF, false), // max + 0 = max (no overflow)
			(0x7FFFFFFF, 0x00000001, 0x00000000, 0x80000000, false), // Sign bit transition
			(0xFFFF0000, 0x0000FFFF, 0x00000001, 0x00000000, true), // overflow with carry_in
		];
		TestPlan {
			dyn_carry_in: true,
			expose_final_carry: true,
			commit_zout: false,
			test_vector: test_vector.to_vec(),
		}
		.execute();
	}

	struct TestPlan {
		dyn_carry_in: bool,
		expose_final_carry: bool,
		commit_zout: bool,
		/// (x, y, carry_in, zout, final_carry)
		test_vector: Vec<(u32, u32, u32, u32, bool)>,
	}

	impl TestPlan {
		fn execute(self) {
			let mut cs = ConstraintSystem::new();
			let mut table = cs.add_table("u32_add");

			let xin = table.add_committed::<B1, 32>("xin");
			let yin = table.add_committed::<B1, 32>("yin");

			let carry_in = self
				.dyn_carry_in
				.then_some(table.add_committed::<B1, 32>("carry_in"));

			let flags = U32AddFlags {
				carry_in_bit: carry_in,
				expose_final_carry: self.expose_final_carry,
				commit_zout: self.commit_zout,
			};
			let adder = U32Add::new(&mut table, xin, yin, flags);
			assert!(adder.final_carry.is_some() == self.expose_final_carry);

			let table_id = table.id();
			let statement = Statement {
				boundaries: vec![],
				table_sizes: vec![self.test_vector.len()],
			};
			let allocator = Bump::new();
			let mut witness =
				WitnessIndex::<PackedType<OptimalUnderlier128b, B128>>::new(&cs, &allocator);

			let table_witness = witness
				.init_table(table_id, self.test_vector.len())
				.unwrap();
			let mut segment = table_witness.full_segment();

			{
				let mut xin_bits = segment.get_mut_as::<u32, _, 32>(adder.xin).unwrap();
				let mut yin_bits = segment.get_mut_as::<u32, _, 32>(adder.yin).unwrap();
				let mut carry_in_bits =
					carry_in.map(|carry_in| segment.get_mut_as::<u32, _, 32>(carry_in).unwrap());
				for (i, (x, y, carry_in, _, _)) in self.test_vector.iter().enumerate() {
					xin_bits[i] = *x;
					yin_bits[i] = *y;
					if let Some(ref mut carry_in_bits) = carry_in_bits {
						carry_in_bits[i] = *carry_in;
					}
				}
			}

			// Populate the gadget
			adder.populate(&mut segment).unwrap();

			{
				// Verify results
				let zout_bits = segment.get_as::<u32, _, 32>(adder.zout).unwrap();
				let final_carry = adder
					.final_carry
					.map(|final_carry| segment.get(final_carry).unwrap());
				for (i, (_, _, _, zout, expected_carry)) in self.test_vector.iter().enumerate() {
					assert_eq!(zout_bits[i], *zout);

					if let Some(ref final_carry) = final_carry {
						assert_eq!(get_packed_slice(final_carry, i), B1::from(*expected_carry));
					}
				}
			}

			// Validate constraint system
			let ccs = cs.compile(&statement).unwrap();
			let witness = witness.into_multilinear_extension_index();

			binius_core::constraint_system::validate::validate_witness(&ccs, &[], &witness)
				.unwrap();
		}
	}

	#[test]
	fn test_incr() {
		const TABLE_SIZE: usize = 1 << 9;

		let mut cs = ConstraintSystem::new();
		let mut table = cs.add_table("u32_incr");

		let table_id = table.id();
		let mut rng = StdRng::seed_from_u64(0);
		let test_values = repeat_with(|| B32::new(rng.gen::<u32>()))
			.take(TABLE_SIZE)
			.collect::<Vec<_>>();

		let xin = table.add_committed_multiple("xin");

		let incr = U32Incr::new(&mut table, xin);

		let allocator = Bump::new();
		let mut witness =
			WitnessIndex::<PackedType<OptimalUnderlier128b, B128>>::new(&cs, &allocator);

		let table_witness = witness.init_table(table_id, TABLE_SIZE).unwrap();
		let mut segment = table_witness.full_segment();

		{
			let mut xin_witness =
				array_util::try_map(xin, |bit_col| segment.get_mut(bit_col)).unwrap();
			for (i, value) in test_values.iter().enumerate() {
				for (bit, packed) in xin_witness.iter_mut().enumerate() {
					set_packed_slice(packed, i, B1::from(((value.val() >> bit) & 1) == 1))
				}
			}
		}

		incr.populate(&mut segment).unwrap();

		{
			let zouts = array_util::try_map(incr.zout, |bit_col| segment.get(bit_col)).unwrap();
			for (i, value) in test_values.iter().enumerate() {
				let expected = value.val().wrapping_add(1);
				let mut got = 0u32;
				for bit in (0..32).rev() {
					got <<= 1;
					if get_packed_slice(&zouts[bit], i) == B1::ONE {
						got |= 1;
					}
				}
				assert_eq!(expected, got);
			}
		}
		// Validate constraint system
		validate_system_witness::<OptimalUnderlier128b>(&cs, witness, vec![]);
	}
}
