// Copyright 2025 Irreducible Inc.

use binius_core::constraint_system::channel::ChannelId;
use binius_field::{PackedExtension, PackedField, PackedFieldIndexable, PackedSubfield};
use binius_math::ArithExpr;

use crate::builder::{
	B1, B8, B32, B128, Col, Expr, TableBuilder, TableFiller, TableId, TableWitnessSegment,
	column::upcast_col,
};

/// A gadget that computes the logical AND of two boolean columns using JOLT style
/// lookups.

pub struct And {
	/// The table ID
	pub id: TableId,
	/// The first argument.
	pub a: Col<B8>,
	/// The second argument.
	pub b: Col<B8>,
	/// The output column that holds the result of the AND operation.
	pub out: Col<B8>,
	/// Merged column for lookup operations
	pub merged: Col<B32>,
}

/// Returns an arithmetic expression that represents the AND operation.
pub fn and_circuit() -> ArithExpr<B128> {
	// The circuit is a lookup table for the and operation, which takes 2 8-bit inputs and
	// returns a 8-bit output.
	let mut circuit = ArithExpr::zero();
	for i in 0..8 {
		circuit += ArithExpr::Var(i) * ArithExpr::Var(i + 8) * ArithExpr::Const(B8::new(1 << i))
	}
	circuit
}

impl And {
	/// Creates a new AND table.
	pub fn new(table: &mut TableBuilder, lookup_chan: ChannelId) -> Self {
		let a = table.add_committed("a");
		let b = table.add_committed("b");
		let out = table.add_committed("out");

		// Merge the inputs and output into a single column for lookup
		// Format: a (8 bit) | b (8 bit) | out (8 bit)
		let merged = table.add_computed(
			"merged",
			upcast_col(a) + upcast_col(b) * B8::new(2) + upcast_col(out) * B8::new(4),
		);

		// Pull from lookup channel to verify against lookup table
		table.pull(lookup_chan, [merged]);

		Self {
			id: table.id(),
			a,
			b,
			out,
			merged,
		}
	}

	/// Populate the witness based on the input values
	pub fn populate<P>(&self, witness: &mut TableWitnessSegment<P>) -> anyhow::Result<()>
	where
		P: PackedFieldIndexable<Scalar = B128> + PackedExtension<B1> + PackedExtension<B8>,
	{
		let mut a = witness.get_mut(self.a)?;
		let mut b = witness.get_mut(self.b)?;
		let mut out = witness.get_mut(self.out)?;
		let mut merged = witness.get_mut(self.merged)?;

		for i in 0..witness.size() {
			// Compute the logical AND
			out[i] = a[i] * b[i];

			// Compute the merged value
			merged[i] = a[i].into() + b[i].into() * B8::new(2) + out[i].into() * B8::new(4);
		}

		Ok(())
	}
}

impl<P> TableFiller<P> for And
where
	P: PackedFieldIndexable<Scalar = B128> + PackedExtension<B1> + PackedExtension<B8>,
{
	type Event = (bool, bool);

	fn id(&self) -> TableId {
		self.id
	}

	fn fill<'a>(
		&'a self,
		rows: impl Iterator<Item = &'a Self::Event> + Clone,
		witness: &'a mut TableWitnessSegment<P>,
	) -> anyhow::Result<()> {
		let mut a = witness.get_mut(self.a)?;
		let mut b = witness.get_mut(self.b)?;
		let mut out = witness.get_mut(self.out)?;
		let mut merged = witness.get_mut(self.merged)?;

		for (i, &(a_val, b_val)) in rows.enumerate() {
			a[i] = B1::from(a_val);
			b[i] = B1::from(b_val);

			// Compute logical AND
			let out_val = a_val && b_val;
			out[i] = B1::from(out_val);

			// Compute merged value
			let merged_val = (a_val as u8) + ((b_val as u8) << 1) + ((out_val as u8) << 2);
			merged[i] = B8::new(merged_val);
		}

		Ok(())
	}
}

/// A lookup table for AND operations.
/// This table stores all possible combinations of inputs and outputs for AND.
pub struct AndLookupTable {
	pub id: TableId,
	pub entries: Col<B8>,
}

impl AndLookupTable {
	pub fn new(table: &mut TableBuilder, lookup_chan: ChannelId) -> Self {
		// Fixed size table with 4 entries (all possible input combinations)
		table.require_fixed_size(2);

		let entries = table.add_committed("entries");

		// Push to lookup channel to provide lookup values
		table.push(lookup_chan, [entries]);

		Self {
			id: table.id(),
			entries,
		}
	}
}

impl<P> TableFiller<P> for AndLookupTable
where
	P: PackedFieldIndexable<Scalar = B128> + PackedExtension<B8>,
{
	type Event = ();

	fn id(&self) -> TableId {
		self.id
	}

	fn fill<'a>(
		&'a self,
		_rows: impl Iterator<Item = &'a Self::Event> + Clone,
		witness: &'a mut TableWitnessSegment<P>,
	) -> anyhow::Result<()> {
		let mut entries = witness.get_mut(self.entries)?;

		// Fill in all possible combinations for a & b:
		// 0 & 0 = 0 -> 0
		// 0 & 1 = 0 -> 2
		// 1 & 0 = 0 -> 1
		// 1 & 1 = 1 -> 7 (1 + 2 + 4)
		entries[0] = B8::new(0); // 0 & 0 = 0
		entries[1] = B8::new(2); // 0 & 1 = 0
		entries[2] = B8::new(1); // 1 & 0 = 0
		entries[3] = B8::new(7); // 1 & 1 = 1

		Ok(())
	}
}
