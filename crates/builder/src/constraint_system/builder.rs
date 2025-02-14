use std::{cell::RefCell, rc::Rc};

use super::{
	table_builder::TableBuilder, ChannelId, ConstraintSystem, OracleId, OracleInfo, Table, TableId,
};

#[derive(Default)]
pub struct ConstraintSystemBuilderMeta {
	pub oracle_infos: Vec<OracleInfo>,
	pub tables: Vec<Table>,
	pub tables_to_oracles: Vec<Vec<OracleId>>,
	pub channel_count: usize,
}

#[derive(Default)]
pub struct ConstraintSystemBuilder {
	meta: Rc<RefCell<ConstraintSystemBuilderMeta>>,
}

impl ConstraintSystemBuilder {
	pub fn new() -> Self {
		Self::default()
	}

	#[allow(clippy::type_complexity)]
	pub fn build(self) -> Result<ConstraintSystem, anyhow::Error> {
		// argue here that we're outputting a sound constraint system by assumption of sound table builders

		let mut meta = Rc::into_inner(self.meta)
			.expect("Failed to build ConstraintSystem: references still exist to meta")
			.into_inner();

		let ConstraintSystemBuilderMeta {
			oracle_infos,
			tables,
			tables_to_oracles,
			channel_count,
		} = meta;

		let oracles_to_tables = {
			let mut pairs: Vec<(OracleId, TableId)> = tables_to_oracles
				.iter()
				.enumerate()
				.flat_map(|(table_id, table_oracle_ids)| {
					table_oracle_ids
						.iter()
						.map(move |&oracle_id| (oracle_id, table_id))
				})
				.collect();
			pairs.sort();
			pairs.into_iter().map(|(_, table_id)| table_id).collect()
		};

		//
		Ok(ConstraintSystem {
			oracle_infos,
			tables,
			oracles_to_tables,
			tables_to_oracles: None,
			channel_count,
		})
	}

	pub fn new_table_builder(&mut self, name: impl ToString) -> TableBuilder {
		TableBuilder::new(name, self.meta.clone())
	}

	pub fn add_channel(&mut self) -> ChannelId {
		let mut meta = self.meta.borrow_mut();
		let channel_id = meta.channel_count;
		meta.channel_count += 1;
		channel_id
	}
}
