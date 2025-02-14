use std::{cell::RefCell, marker::PhantomData, rc::Rc};

use anyhow::anyhow;
use binius_core::{
	constraint_system::{
		channel::{ChannelId, Flush, FlushDirection},
		ConstraintSystem,
	},
	oracle::{Constraint, Error as OracleError, MultilinearOracleSet, OracleId},
	polynomial::MultivariatePoly,
	witness::{MultilinearExtensionIndex, MultilinearWitness},
};
use binius_field::{
	arch::OptimalUnderlier, BinaryField128b as B128, BinaryField1b as B1, TowerField,
};
use binius_math::ArithExpr;
use binius_utils::bail;

use super::{
	builder::ConstraintSystemBuilderMeta, constraint_set::ConstraintSetBuilder, Derived, Filler,
	LinearCombination, Oracle, OracleInfo, OracleVariant, Packed, Projected, ProjectionVariant,
	Repeated, ShiftVariant, Table,
};
use crate::{derived_fillers, error::Error};

/// what are we doing here?
///
///
///
///
///
///
///
///

pub struct TableBuilder {
	name: String,
	meta: Rc<RefCell<ConstraintSystemBuilderMeta>>,
	oracle_infos: Vec<OracleInfo>,
	non_zero_oracle_ids: Vec<OracleId>,
	flushes: Vec<Flush>,
	constraint_set_builder: ConstraintSetBuilder, // un-parameterize?
	namespace_path: Vec<String>,
}

impl Drop for TableBuilder {
	fn drop(&mut self) {
		let constraint_set_builder = std::mem::take(&mut self.constraint_set_builder);
		let constraint_set = constraint_set_builder.build().unwrap();

		let mut meta = self.meta.borrow_mut();
		let table_id = meta.tables.len();
		meta.tables.push(Table {
			id: table_id,
			name: std::mem::take(&mut self.name),
			non_zero_oracle_ids: std::mem::take(&mut self.non_zero_oracle_ids),
			constraint_set,
			flushes: std::mem::take(&mut self.flushes),
		});
		let oracle_ids = std::mem::take(&mut self.oracle_infos)
			.into_iter()
			.map(|x| x.oracle.id)
			.collect::<Vec<_>>();
		meta.tables_to_oracles.push(oracle_ids);
	}
}

impl TableBuilder {
	fn get_oracle(&mut self, id: OracleId) -> Oracle {
		// check is valid
		let meta = self.meta.borrow();
		let x = meta.oracle_infos[id].oracle.clone();
		// .clone();
		x
		// self.oracles[id].clone()
	}

	//

	pub fn new(name: impl ToString, meta: Rc<RefCell<ConstraintSystemBuilderMeta>>) -> Self {
		Self {
			name: name.to_string(),
			meta,
			oracle_infos: Vec::new(),
			non_zero_oracle_ids: Vec::new(),
			flushes: Vec::new(),
			constraint_set_builder: ConstraintSetBuilder::new(),
			namespace_path: Vec::new(),
		}
	}

	fn check_oracle_ids_in_table(&self, oracle_ids: &[OracleId]) -> Result<(), anyhow::Error> {
		// so we've simply checked they are in our list, because we know our list is valid.
		// Self::is_subset_sorted(&oracle_ids, &self.oracle_ids_fillers);
		//
		// // check oracles are valid
		// let oracles = self.oracles.borrow();
		// // if !oracles.is_valid_oracle_id(id)
		// // now check all belong to table

		// // let map = self.oracles_to_tables.borrow();
		// // if oracle_ids
		// // 	.iter()
		// // 	.any(|&oracle_id| self.id != map[oracle_id])
		// // {
		// // 	bail!(anyhow!("flush oops"))
		// // }
		//
		Ok(())
	}

	fn is_subset_sorted<T: Ord + Clone>(a: &[T], b: &[T]) -> bool {
		let mut sorted_b = b.to_vec();
		sorted_b.sort();
		a.iter().all(|item| sorted_b.binary_search(item).is_ok())
	}

	// NAMING

	fn scoped_name(&self, name: impl ToString) -> String {
		let name = name.to_string();
		if self.namespace_path.is_empty() {
			name
		} else {
			format!("{}::{name}", self.namespace_path.join("::"))
		}
	}

	// /// Anything pushed to the namespace will become part of oracle name, which is useful for debugging.
	// ///
	// /// Use `pop_namespace(&mut self)` to remove the latest name.
	// ///
	// /// Example
	// /// ```
	// /// use binius_circuits::builder::GCSBuilder;
	// /// use binius_field::{TowerField, BinaryField128b, BinaryField1b, arch::OptimalUnderlier};
	// ///
	// /// let log_size = 14;
	// ///
	// /// let mut builder = GCSBuilder::<OptimalUnderlier, BinaryField128b>::new();
	// /// builder.push_namespace("a");
	// /// let x = builder.add_committed("x", log_size, BinaryField1b::TOWER_LEVEL);
	// /// builder.push_namespace("b");
	// /// let y = builder.add_committed("y", log_size, BinaryField1b::TOWER_LEVEL);
	// /// builder.pop_namespace();
	// /// builder.pop_namespace();
	// /// let z = builder.add_committed("z", log_size, BinaryField1b::TOWER_LEVEL);
	// ///
	// /// let system = builder.build().unwrap();
	// /// assert_eq!(system.oracles.oracle(x).name().unwrap(), "a::x");
	// /// assert_eq!(system.oracles.oracle(y).name().unwrap(), "a::b::y");
	// /// assert_eq!(system.oracles.oracle(z).name().unwrap(), "z");
	// /// ```
	pub fn push_namespace(&mut self, name: impl ToString) {
		self.namespace_path.push(name.to_string());
	}

	pub fn pop_namespace(&mut self) {
		self.namespace_path.pop();
	}

	// FLUSH METHODS

	pub fn add_channel(&mut self) -> ChannelId {
		let mut meta = self.meta.borrow_mut();
		let channel_id = meta.channel_count;
		meta.channel_count += 1;
		channel_id
	}

	pub fn flush(
		&mut self,
		direction: FlushDirection,
		channel_id: ChannelId,
		oracle_ids: impl IntoIterator<Item = OracleId>,
	) -> Result<(), anyhow::Error> {
		self.flush_with_multiplicity(direction, channel_id, oracle_ids, 1)?;

		Ok(())
	}

	pub fn flush_with_multiplicity(
		&mut self,
		direction: FlushDirection,
		channel_id: ChannelId,
		oracle_ids: impl IntoIterator<Item = OracleId>,
		multiplicity: u64,
	) -> Result<(), anyhow::Error> {
		let oracle_ids: Vec<_> = oracle_ids.into_iter().collect();
		self.check_oracle_ids_in_table(&oracle_ids)?;

		self.flushes.push(Flush {
			channel_id,
			direction,
			oracles: oracle_ids.into_iter().collect(),
			multiplicity,
			selector: 0,
		});

		Ok(())
	}

	pub fn push(
		&mut self,
		channel_id: ChannelId,
		oracle_ids: impl IntoIterator<Item = OracleId>,
	) -> Result<(), anyhow::Error> {
		self.flush(FlushDirection::Push, channel_id, oracle_ids)
	}

	pub fn pull(
		&mut self,
		channel_id: ChannelId,
		oracle_ids: impl IntoIterator<Item = OracleId>,
	) -> Result<(), anyhow::Error> {
		self.flush(FlushDirection::Pull, channel_id, oracle_ids)
	}

	// ASSERT METHODS

	pub fn assert_zero(
		&mut self,
		name: impl ToString,
		oracle_ids: impl IntoIterator<Item = OracleId>,
		composition: ArithExpr<B128>,
	) -> Result<(), anyhow::Error> {
		let oracle_ids: Vec<_> = oracle_ids.into_iter().collect();
		self.check_oracle_ids_in_table(&oracle_ids)?;

		self.constraint_set_builder
			.add_zerocheck(self.scoped_name(name), oracle_ids, composition);

		Ok(())
	}

	pub fn assert_not_zero(&mut self, oracle_id: OracleId) -> Result<(), anyhow::Error> {
		self.check_oracle_ids_in_table(&[oracle_id])?;

		self.non_zero_oracle_ids.push(oracle_id);

		Ok(())
	}

	// ADD ORACLE METHODS

	// // OracleOrOracles here should be either OracleId or [OracleId; N]
	// // assume closure returns an oracle only having added to oracle set
	// // up to us to update oracles_to_tables
	// // the oracle will not yet appear in the table, except in the oracles_to_tables map
	// fn add_to_oracles(
	// 	&mut self,
	// 	get_oracle: impl FnOnce(
	// 		&mut MultilinearOracleSet<B128>,
	// 	) -> Result<(OracleId, Option<FillerSig>), OracleError>,
	// ) -> Result<(OracleId, Option<FillerSig>), OracleError> {
	// 	let oracle_set = &mut self.meta.borrow_mut().oracle_set;
	// 	let id = get_oracle(oracle_set)?;

	// 	self.oracle_ids_fillers.push(id);
	// 	// before we i think just pushed into the oracles map our table id which of course is wrong.
	// 	// depending on this type here we might should have pushed many of them

	// 	Ok(id)
	// }

	// fn add_multiple_to_oracles<const N: usize>(
	// 	&mut self,
	// 	get_oracle: impl FnOnce(&mut MultilinearOracleSet<B128>) -> Result<[OracleId; N], OracleError>,
	// ) -> Result<[OracleId; N], OracleError> {
	// 	let oracle_set = &mut self.meta.borrow_mut().oracle_set; //self.oracles.borrow_mut();
	// 	let ids = get_oracle(oracle_set)?;

	// 	self.oracle_ids_fillers.extend(ids.iter());
	// 	// before we i think just pushed into the oracles map our table id which of course is wrong.
	// 	// depending on this type here we might should have pushed many of them

	// 	Ok(ids)
	// }

	// pub fn add_transparent(
	// 	&mut self,
	// 	name: impl ToString,
	// 	poly: impl MultivariatePoly<B128> + 'static,
	// ) -> Result<OracleId, OracleError> {
	// 	self.add_to_oracles(|oracles| oracles.add_transparent(name, poly))
	// }

	pub fn add_original(&mut self, name: impl ToString, tower_level: usize) -> OracleId {
		let mut meta = self.meta.borrow_mut();
		let id = meta.oracle_infos.len();

		let oracle =
			Oracle::new(id, self.scoped_name(name), None, tower_level, OracleVariant::Original);
		meta.oracle_infos.push(OracleInfo {
			oracle,
			filler: None,
		});

		id
	}

	pub fn add_derived(
		&mut self,
		name: impl ToString,
		tower_level: usize,
		dependencies: impl IntoIterator<Item = OracleId>,
		expr: ArithExpr<B128>,
		filler: Filler,
	) -> OracleId {
		let mut meta = self.meta.borrow_mut();
		let id = meta.oracle_infos.len();

		let oracle = Oracle::new(
			id,
			self.scoped_name(name),
			None,
			tower_level,
			OracleVariant::Derived(Derived {
				dependencies: dependencies.into_iter().collect(),
			}),
		);
		meta.oracle_infos.push(OracleInfo {
			oracle,
			filler: None,
		});

		id
	}

	// pub fn add_committed_multiple<const N: usize>(
	// 	&mut self,
	// 	name: impl ToString,
	// 	tower_level: usize,
	// ) -> [OracleId; N] {
	// 	// self.add_multiple_to_oracles(|oracles| {
	// 	// 	Ok(oracles.add_committed_multiple(name, None, tower_level))
	// 	// })
	// 	let ids = self
	// 		.meta
	// 		.borrow_mut()
	// 		.oracle_set
	// 		// self
	// 		// 	.oracles
	// 		// 	.borrow_mut()
	// 		.add_committed_multiple(name, None, tower_level);
	// 	let x = ids.iter().map(|&x| (x, None)).collect::<Vec<_>>();
	// 	self.oracle_ids_fillers.extend(x);
	// 	ids
	// }

	// pub fn add_repeating(
	// 	&mut self,
	// 	name: impl ToString,
	// 	inner_id: OracleId,
	// ) -> Result<OracleId, OracleError> {
	// 	self.check_is_valid_oracle_id(inner_id)?;

	// 	let inner = self.get_oracle(inner_id);

	// 	let mut meta = self.meta.borrow_mut();
	// 	let id = meta.oracle_infos.len();

	// 	let oracle = Oracle::new(
	// 		id,
	// 		self.scoped_name(name),
	// 		None,
	// 		inner.tower_level,
	// 		OracleVariant::Repeating(Repeated::new(id, None)),
	// 	);
	// 	meta.oracle_infos.push(OracleInfo {
	// 		oracle,
	// 		filler: None,
	// 	});

	// 	Ok(id)
	// }

	// pub fn add_linear_combination(
	// 	&mut self,
	// 	name: impl ToString,
	// 	inner: impl IntoIterator<Item = (OracleId, B128)>,
	// 	filler: FillerSig,
	// ) -> Result<OracleId, OracleError> {
	// 	self.add_to_oracles(|oracles| oracles.add_linear_combination(name, inner))
	// }

	// pub fn add_linear_combination_with_offset(
	// 	&mut self,
	// 	name: impl ToString,
	// 	offset: B128,
	// 	inner: impl IntoIterator<Item = (OracleId, B128)>,
	// ) -> Result<OracleId, OracleError> {
	// 	self.add_to_oracles(|oracles| {
	// 		oracles.add_linear_combination_with_offset(name, offset, inner)
	// 	})
	// }

	pub fn add_packed(
		&mut self,
		name: impl ToString,
		inner_id: OracleId,
		log_degree: usize,
	) -> Result<OracleId, OracleError> {
		// self.add_to_oracles(|oracles| oracles.add_packed(name, inner_id, log_degree))
		Ok(0)
	}

	pub fn add_projected_with_filler(
		&mut self,
		name: impl ToString,
		inner_id: OracleId,
		values: Vec<B1>,
		variant: ProjectionVariant,
		filler: Filler,
	) -> Result<usize, OracleError> {
		// self.add_to_oracles(|oracles| oracles.add_projected(name, inner_id, values, variant))
		Ok(0)
	}

	pub fn add_projected(
		&mut self,
		name: impl ToString,
		inner_id: OracleId,
		values: Vec<B1>,
		variant: ProjectionVariant,
	) -> Result<usize, OracleError> {
		// get tower_level from inner_id
		let tower_level = 0;
		// self.add_to_oracles(|oracles| oracles.add_projected(name, inner_id, values, variant))
		self.add_projected_with_filler(
			name,
			inner_id,
			values.clone(),
			variant,
			derived_fillers::projected(tower_level, values, variant),
		)?;
		Ok(0)
	}

	pub fn add_shifted_with_filler(
		&mut self,
		name: impl ToString,
		inner_id: OracleId,
		offset: usize,
		block_bits: Option<usize>,
		variant: ShiftVariant,
		filler: Filler,
	) -> Result<OracleId, OracleError> {
		// self.add_to_oracles(|oracles| {
		// 	oracles.add_shifted(name, inner_id, offset, block_bits, variant)
		// })
		// let id = self
		// 	.meta
		// 	.borrow_mut()
		// 	.oracle_set
		// 	// self
		// 	// 	.oracles
		// 	// 	.borrow_mut()
		// 	.add_shifted(name, inner_id, offset, block_bits, variant)?;
		// self.oracle_ids_fillers.push(id);
		Ok(0)
	}

	pub fn add_shifted(
		&mut self,
		name: impl ToString,
		inner_id: OracleId,
		offset: usize,
		block_bits: Option<usize>,
		variant: ShiftVariant,
	) -> Result<OracleId, OracleError> {
		// get tower level from inner_id
		let tower_level = 0;
		self.add_shifted_with_filler(
			name,
			inner_id,
			offset,
			block_bits,
			ShiftVariant::LogicalLeft,
			derived_fillers::shifted(tower_level, offset, block_bits, variant),
		)?;

		// self.add_to_oracles(|oracles| {
		// 	oracles.add_shifted(name, inner_id, offset, block_bits, variant)
		// })
		// let id = self
		// 	.meta
		// 	.borrow_mut()
		// 	.oracle_set
		// 	// self
		// 	// 	.oracles
		// 	// 	.borrow_mut()
		// 	.add_shifted(name, inner_id, offset, block_bits, variant)?;
		// self.oracle_ids_fillers.push(id);
		Ok(0)
	}

	// pub fn add_zero_padded(
	// 	&mut self,
	// 	name: impl ToString,
	// 	inner_id: OracleId,
	// ) -> Result<OracleId, OracleError> {
	// 	self.add_to_oracles(|oracles| oracles.add_zero_padded(name, inner_id))
	// }
}
