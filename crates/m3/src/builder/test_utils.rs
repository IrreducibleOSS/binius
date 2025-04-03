// Copyright 2025 Irreducible Inc.

//! Utilities for testing M3 constraint systems and gadgets.

use anyhow::Result;
use binius_field::{PackedField, TowerField};

use crate::builder::{TableFiller, TableId, TableWitnessSegment};

/// An easy-to-use implementation of [`TableFiller`] that is constructed with a closure.
///
/// Using this [`TableFiller`] implementation carries some overhead, so it is best to use it only
/// for testing.
#[allow(clippy::type_complexity)]
pub struct ClosureFiller<'a, P, Event>
where
	P: PackedField,
	P::Scalar: TowerField,
{
	table_id: TableId,
	fill: Box<dyn for<'b> Fn(&'b [&'b Event], &'b mut TableWitnessSegment<P>) -> Result<()> + 'a>,
}

impl<'a, P: PackedField<Scalar: TowerField>, Event> ClosureFiller<'a, P, Event> {
	pub fn new(
		table_id: TableId,
		fill: impl for<'b> Fn(&'b [&'b Event], &'b mut TableWitnessSegment<P>) -> Result<()> + 'a,
	) -> Self {
		Self {
			table_id,
			fill: Box::new(fill),
		}
	}
}

impl<P: PackedField<Scalar: TowerField>, Event: Clone> TableFiller<P>
	for ClosureFiller<'_, P, Event>
{
	type Event = Event;

	fn id(&self) -> TableId {
		self.table_id
	}

	fn fill<'b>(
		&'b self,
		rows: impl Iterator<Item = &'b Self::Event> + Clone,
		witness: &'b mut TableWitnessSegment<P>,
	) -> Result<()> {
		(*self.fill)(&rows.collect::<Vec<_>>(), witness)
	}
}
