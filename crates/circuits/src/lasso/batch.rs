// Copyright 2024-2025 Irreducible Inc.

use anyhow::Ok;
use binius_core::oracle::OracleId;
use binius_field::{
	ExtensionField, PackedFieldIndexable, TowerField,
	as_packed_field::{PackScalar, PackedType},
};
use itertools::Itertools;

use super::lasso::lasso;
use crate::builder::{
	ConstraintSystemBuilder,
	types::{F, U},
};
pub struct LookupBatch {
	lookup_us: Vec<Vec<OracleId>>,
	u_to_t_mappings: Vec<Vec<usize>>,
	lookup_col_lens: Vec<usize>,
	lookup_t: Vec<OracleId>,
	executed: bool,
}

impl Drop for LookupBatch {
	fn drop(&mut self) {
		if !self.executed {
			tracing::warn!("LookupBatch dropped before calling execute!");
		}
	}
}

impl LookupBatch {
	pub fn new(t_oracles: impl IntoIterator<Item = OracleId>) -> Self {
		Self {
			lookup_t: t_oracles.into_iter().collect_vec(),
			lookup_us: vec![],
			u_to_t_mappings: vec![],
			lookup_col_lens: vec![],
			executed: false,
		}
	}

	pub fn add(
		&mut self,
		lookup_u: impl IntoIterator<Item = OracleId>,
		u_to_t_mapping: Vec<usize>,
		lookup_u_col_len: usize,
	) {
		self.lookup_us.push(lookup_u.into_iter().collect_vec());
		self.u_to_t_mappings.push(u_to_t_mapping);
		self.lookup_col_lens.push(lookup_u_col_len);
	}

	pub fn execute<FC>(mut self, builder: &mut ConstraintSystemBuilder) -> Result<(), anyhow::Error>
	where
		FC: TowerField,
		U: PackScalar<FC>,
		F: ExtensionField<FC>,
		PackedType<U, FC>: PackedFieldIndexable,
	{
		let channel = builder.add_channel();

		lasso::<FC>(
			builder,
			"batched lasso",
			&self.lookup_col_lens,
			&self.u_to_t_mappings,
			&self.lookup_us,
			&self.lookup_t,
			channel,
		)?;

		self.executed = true;

		Ok(())
	}
}
