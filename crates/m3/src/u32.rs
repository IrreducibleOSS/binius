// Copyright 2025 Irreducible Inc.

use binius_core::oracle::ShiftVariant;
use binius_field::{
	as_packed_field::{PackScalar, PackedType},
	packed::set_packed_slice,
	underlier::UnderlierType,
	Field,
};
use bytemuck::{cast_slice_mut, must_cast_slice, must_cast_slice_mut, Pod};

use super::{
	builder::{Col, TableBuilder},
	types::{B1, B128, B16, B32, B64, B8},
	witness::WitnessIndex,
};
use crate::witness::TableWitnessIndexSegment;

// Concepts:
//
// - Gadgets used within a single table. They derive data from input witness values, plus advice.
// - Gadgets assume that their input columns have been populated already.
//
// - Table population just takes advice as a struct.
//
// - Provide functions to populate table rows and row segments. Row segments for better parallelism.

#[derive(Debug)]
pub struct U32Add {
	// Inputs
	pub xin: Col<B1, 5>,
	pub yin: Col<B1, 5>,

	// Private
	cin: Col<B1, 5>,
	cout: Col<B1, 5>,
	cout_shl: Col<B1, 5>,

	// Outputs
	pub final_carry: Option<Col<B1>>,
	pub zout: Col<B1, 5>,
	pub flags: U32AddFlags,
}

#[derive(Debug, Default, Clone)]
pub struct U32AddFlags {
	// Optionally a column for a dynamic carry in bit. This *must* be zero in all bits except the
	// 0th.
	pub carry_in_bit: Option<Col<B1, 5>>,
	pub commit_zout: bool,
	pub expose_final_carry: bool,
}

/// Note that these methods are not trait impls, they are just regular methods. This gives
/// flexibility to adapt method signatures to the specific needs of the component.
impl U32Add {
	/// Constructs a new `U32Add` component.
	pub fn new(
		table: &mut TableBuilder,
		xin: Col<B1, 5>,
		yin: Col<B1, 5>,
		flags: U32AddFlags,
	) -> Self {
		let cout = table.add_committed::<B1, 5>("cout");
		let cout_shl = table.add_shifted("cout_shl", cout, 5, 1, ShiftVariant::LogicalLeft);

		let cin = if let Some(carry_in_bit) = flags.carry_in_bit {
			table.add_linear_combination("cin", cout_shl + carry_in_bit)
		} else {
			cout_shl
		};

		let final_carry = flags
			.expose_final_carry
			.then(|| table.add_selected("final_carry", cout, 31));

		table.assert_zero("carry_out", (xin + cin) * (yin + cin) + cin - cout);

		// TODO: Depending on the
		let zout = if flags.commit_zout {
			let zout = table.add_committed::<B1, 5>("zout");
			table.assert_zero("zout", xin + yin + cin - zout);
			zout
		} else {
			table.add_linear_combination("zout", xin + yin + cin)
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

	pub fn populate<U>(&self, index: &mut TableWitnessIndexSegment<U>) -> Result<(), anyhow::Error>
	where
		U: Pod + PackScalar<B1>,
		// The index only returns packed fields, so knowing the underlier itself is pod is
		// insufficient.
		PackedType<U, B1>: Pod,
	{
		let xin = must_cast_slice::<_, u32>(&*index.get(self.xin)?);
		let yin = must_cast_slice::<_, u32>(&*index.get(self.yin)?);
		let cout = must_cast_slice_mut::<_, u32>(&mut *index.get_mut(self.cout)?);
		let final_carry = self
			.final_carry
			.map(|final_carry| &mut *index.get_mut(final_carry))?;

		if let Some(carry_in_bit_col) = self.flags.carry_in_bit {
			// This is u32 assumed to be either 0 or 1.
			let carry_in_bit = must_cast_slice_mut::<_, u32>(&**index.get_mut(carry_in_bit_col));

			let cin = must_cast_slice_mut::<_, u32>(&**index.get_mut(self.cin));
			let cout_shl = must_cast_slice_mut::<_, u32>(&**index.get_mut(self.cout_shl));
			for i in 0..index.size() {
				let (x_plus_y, carry0) = xin[i].overflowing_add(yin[i]);
				let (zout, carry1) = x_plus_y.overflowing_add(carry_in_bit[i]);
				let carry = carry0 | carry1;

				cin[i] = xin[i] ^ yin[i] ^ zout;
				cout[i] = (carry as u32) << 31 | cin[i] >> 1;
				cout_shl[i] = cout[i] << 1;

				if let Some(final_carry) = final_carry {
					set_packed_slice(final_carry, i, if carry { B1::ONE } else { B1::ZERO });
				}
			}
		} else {
			// When the carry in bit is fixed to zero, we can simplify the logic.
			let cin = must_cast_slice_mut::<_, u32>(&**index.get_mut(self.cin));
			for i in 0..index.n_rows() {
				let (zout, carry) = xin[i].overflowing_add(yin[i]);
				cin[i] = xin[i] ^ yin[i] ^ zout;
				cout[i] = (carry as u32) << 31 | cin[i] >> 1;
				if let Some(final_carry) = final_carry {
					set_packed_slice(final_carry, i, if carry { B1::ONE } else { B1::ZERO });
				}
			}
		};
	}
}
