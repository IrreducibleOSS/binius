// Copyright 2025 Irreducible Inc.

use binius_core::oracle::ShiftVariant;
use binius_field::{packed::set_packed_slice, Field};
use bytemuck::{cast_slice_mut, must_cast_slice, must_cast_slice_mut};

use super::{
	constraint_system::{Col, Expr, Table},
	types::{B1, B128, B16, B32, B64, B8},
	witness::WitnessIndex,
};

// Concepts:
//
// - Gadgets used within a single table. They derive data from input witness values, plus advice.
// - Gadgets assume that their input columns have been populated already.
//
// - Table population just takes advice as a struct.
//
// - Provide functions to populate table rows and row segments. Row segments for better parallelism.

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
	pub zout: Expr<B1, 5>,

	pub flags: U32AddFlags,
}

#[derive(Debug, Default, Clone)]
pub struct U32AddFlags {
	// Optionally a column for a dynamic carry in bit. This *must* be zero in all bits except the
	// 0th.
	pub carry_in_bit: Option<Col<B1, 5>>,
	pub expose_final_carry: bool,
}

/// Note that these methods are not trait impls, they are just regular methods. This gives
/// flexibility to adapt method signatures to the specific needs of the component.
impl U32Add {
	/// Constructs a new `U32Add` component.
	pub fn new(
		table: &mut Table<B128>,
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

		// REVIEW: This is awkward because we need to populate zout witness
		let zout = xin + yin + cin;

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

	/// index here is assumed to be a witness index segment
	pub fn populate<U: UnderlierType>(&self, index: &mut TableWitnessIndex<U>) {
		let xin = must_cast_slice::<_, u32>(&**index.get(self.xin));
		let yin = must_cast_slice::<_, u32>(&**index.get(self.yin));
		let cout = must_cast_slice_mut::<_, u32>(&**index.get_mut(self.cout));
		let final_carry = self
			.final_carry
			.map(|final_carry| &**index.get_mut(final_carry));
		// let zout = index.get_mut(self.zout);

		if let Some(carry_in_bit) = self.flags.carry_in_bit {
			let cin = must_cast_slice_mut::<_, u32>(&**index.get_mut(self.cin));
			let cout_shl = must_cast_slice_mut::<_, u32>(&**index.get_mut(self.cout_shl));
			for i in 0..index.n_rows() {
				// let carry;
				let (zout, carry) = xin[i].overflowing_add(yin[i]);
				cin[i] = xin[i] ^ yin[i] ^ zout;
				cout[i] = (carry as u32) << 31 | cin[i] >> 1;
				if let Some(final_carry) = final_carry {
					set_packed_slice(final_carry, i, if carry { B1::ONE } else { B1::ZERO });
				}
			}
		} else {
			let cin = must_cast_slice_mut::<_, u32>(&**index.get_mut(self.cin));
			for i in 0..index.n_rows() {
				let (zout, carry) = xin[i].overflowing_add(yin[i]);
				cin[i] = xin[i] ^ yin[i] ^ zout;
				cout[i] = (carry as u32) << 31 | cin[i] >> 1;
				if let Some(final_carry) = final_carry {
					set_packed_slice(final_carry, i, if carry { B1::ONE } else { B1::ZERO });
				}
			}
		}
	}
}
