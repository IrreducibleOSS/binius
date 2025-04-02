// Copyright 2025 Irreducible Inc.

use anyhow::Result;
use binius_field::{underlier::UnderlierType, TowerField};

use crate::builder::{TableFiller, TableId, TableWitnessSegment};

///! Utilities for testing M3 constraint systems and gadgets.

pub struct ClosureFiller<'a, U: UnderlierType, F: TowerField, Event> {
	table_id: TableId,
	fill:
		Box<dyn for<'b> Fn(&'b [&'b Event], &'b mut TableWitnessSegment<U, F>) -> Result<()> + 'a>,
}

impl<'a, U: UnderlierType, F: TowerField, Event> ClosureFiller<'a, U, F, Event> {
	pub fn new(
		table_id: TableId,
		fill: impl for<'b> Fn(&'b [&'b Event], &'b mut TableWitnessSegment<U, F>) -> Result<()> + 'a,
	) -> Self {
		Self {
			table_id,
			fill: Box::new(fill),
		}
	}
}

impl<'a, U: UnderlierType, F: TowerField, Event: Clone> TableFiller<U, F>
	for ClosureFiller<'a, U, F, Event>
{
	type Event = Event;

	fn id(&self) -> TableId {
		self.table_id
	}

	fn fill<'b>(
		&'b self,
		rows: impl Iterator<Item = &'b Self::Event> + Clone,
		witness: &'b mut TableWitnessSegment<U, F>,
	) -> anyhow::Result<()> {
		(*self.fill)(&rows.collect::<Vec<_>>(), witness)
	}
}
