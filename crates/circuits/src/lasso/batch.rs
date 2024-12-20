// Copyright 2024 Irreducible Inc.

use anyhow::Ok;
use binius_core::oracle::OracleId;
use binius_field::{
	as_packed_field::{PackScalar, PackedType},
	ExtensionField, PackedFieldIndexable, TowerField,
};

use super::lasso::lasso;
use crate::builder::ConstraintSystemBuilder;
pub struct LookupBatch {
	lookup_us: Vec<OracleId>,
	u_to_t_mappings: Vec<Vec<usize>>,
	lookup_col_lens: Vec<usize>,
	lookup_t: OracleId,
}

impl LookupBatch {
	pub fn new(t_table_id: OracleId) -> Self {
		Self {
			lookup_t: t_table_id,
			lookup_us: vec![],
			u_to_t_mappings: vec![],
			lookup_col_lens: vec![],
		}
	}

	pub fn add(&mut self, lookup_u: OracleId, u_to_t_mapping: Vec<usize>, lookup_u_col_len: usize) {
		self.lookup_us.push(lookup_u);
		self.u_to_t_mappings.push(u_to_t_mapping);
		self.lookup_col_lens.push(lookup_u_col_len);
	}

	pub fn execute<U, F, FS, FC>(
		&mut self,
		builder: &mut ConstraintSystemBuilder<U, F>,
	) -> Result<(), anyhow::Error>
	where
		U: PackScalar<FC> + PackScalar<F>,
		PackedType<U, FC>: PackedFieldIndexable,
		FC: TowerField,
		FS: TowerField,
		F: ExtensionField<FC> + TowerField,
	{
		let channel = builder.add_channel();

		lasso::<_, _, FS, FC>(
			builder,
			"batched lasso",
			&self.lookup_col_lens,
			&self.u_to_t_mappings,
			&self.lookup_us,
			self.lookup_t,
			channel,
		)?;

		Ok(())
	}
}
