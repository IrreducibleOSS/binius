// Copyright 2024-2025 Irreducible Inc.

use std::{cell::RefCell, collections::HashMap, rc::Rc};

use anyhow::{anyhow, ensure};
use binius_core::{
	constraint_system::{
		channel::{ChannelId, Flush, FlushDirection, OracleOrConst},
		exp::Exp,
		ConstraintSystem,
	},
	oracle::{
		ConstraintSetBuilder, Error as OracleError, MultilinearOracleSet, OracleId,
		ProjectionVariant, ShiftVariant,
	},
	polynomial::MultivariatePoly,
	transparent::step_down::StepDown,
	witness::MultilinearExtensionIndex,
};
use binius_field::{
	as_packed_field::{PackScalar, PackedType},
	BinaryField1b,
};
use binius_math::ArithExpr;
use binius_utils::bail;

use crate::builder::{
	types::{F, U},
	witness,
};

#[derive(Default)]
pub struct ConstraintSystemBuilder<'arena> {
	oracles: Rc<RefCell<MultilinearOracleSet<F>>>,
	constraints: ConstraintSetBuilder<F>,
	non_zero_oracle_ids: Vec<OracleId>,
	flushes: Vec<Flush<F>>,
	exponents: Vec<Exp<F>>,
	step_down_dedup: HashMap<(usize, usize), OracleId>,
	witness: Option<witness::Builder<'arena>>,
	next_channel_id: ChannelId,
	namespace_path: Vec<String>,
}

impl<'arena> ConstraintSystemBuilder<'arena> {
	pub fn new() -> Self {
		Self::default()
	}

	pub fn new_with_witness(allocator: &'arena bumpalo::Bump) -> Self {
		let oracles = Rc::new(RefCell::new(MultilinearOracleSet::new()));
		Self {
			witness: Some(witness::Builder::new(allocator, oracles.clone())),
			oracles,
			..Default::default()
		}
	}

	#[allow(clippy::type_complexity)]
	pub fn build(self) -> Result<ConstraintSystem<F>, anyhow::Error> {
		let table_constraints = self.constraints.build(&self.oracles.borrow())?;
		Ok(ConstraintSystem {
			max_channel_id: self
				.flushes
				.iter()
				.map(|flush| flush.channel_id)
				.max()
				.unwrap_or(0),
			table_constraints,
			non_zero_oracle_ids: self.non_zero_oracle_ids,
			oracles: Rc::into_inner(self.oracles)
				.ok_or_else(|| {
					anyhow!("Failed to build ConstraintSystem: references still exist to oracles")
				})?
				.into_inner(),
			flushes: self.flushes,
			exponents: self.exponents,
		})
	}

	pub const fn witness(&mut self) -> Option<&mut witness::Builder<'arena>> {
		self.witness.as_mut()
	}

	pub fn take_witness(
		&mut self,
	) -> Result<MultilinearExtensionIndex<'arena, PackedType<U, F>>, anyhow::Error> {
		Option::take(&mut self.witness)
			.ok_or_else(|| {
				anyhow!("Witness is missing. Are you in verifier mode, or have you already extraced the witness?")
			})?
			.build()
	}

	pub fn flush(
		&mut self,
		direction: FlushDirection,
		channel_id: ChannelId,
		count: usize,
		oracle_ids: impl IntoIterator<Item = OracleOrConst<F>> + Clone,
	) -> anyhow::Result<()>
	where
		U: PackScalar<BinaryField1b>,
	{
		self.flush_with_multiplicity(direction, channel_id, count, oracle_ids, 1)
	}

	pub fn flush_with_multiplicity(
		&mut self,
		direction: FlushDirection,
		channel_id: ChannelId,
		count: usize,
		oracle_ids: impl IntoIterator<Item = OracleOrConst<F>> + Clone,
		multiplicity: u64,
	) -> anyhow::Result<()>
	where
		U: PackScalar<BinaryField1b>,
	{
		//We assume there is at least one non constant in the collection of oracle ids.
		let non_const_oracles = oracle_ids
			.clone()
			.into_iter()
			.filter_map(|id| match id {
				OracleOrConst::Oracle(oracle_id) => Some(oracle_id),
				_ => None,
			})
			.collect::<Vec<_>>();

		let n_vars = self.log_rows(non_const_oracles)?;

		let selector = if let Some(&selector) = self.step_down_dedup.get(&(n_vars, count)) {
			selector
		} else {
			let step_down = StepDown::new(n_vars, count)?;
			let selector = self.add_transparent(
				format!("internal step_down {count}-{n_vars}"),
				step_down.clone(),
			)?;

			if let Some(witness) = self.witness() {
				step_down.populate(witness.new_column::<BinaryField1b>(selector).packed());
			}

			self.step_down_dedup.insert((n_vars, count), selector);
			selector
		};

		self.flush_custom(direction, channel_id, selector, oracle_ids, multiplicity)
	}

	pub fn flush_custom(
		&mut self,
		direction: FlushDirection,
		channel_id: ChannelId,
		selector: OracleId,
		oracle_ids: impl IntoIterator<Item = OracleOrConst<F>> + Clone,
		multiplicity: u64,
	) -> anyhow::Result<()> {
		//We assume there is atleast one non constant in the collection of oracle ids.
		let non_const_oracles = oracle_ids
			.clone()
			.into_iter()
			.filter_map(|id| match id {
				OracleOrConst::Oracle(oracle_id) => Some(oracle_id),
				_ => None,
			})
			.collect::<Vec<_>>();

		let log_rows = self.log_rows(non_const_oracles.iter().copied())?;
		ensure!(
			log_rows == self.log_rows([selector])?,
			"Selector {} n_vars does not match flush {:?}",
			selector,
			non_const_oracles
		);

		let oracles = oracle_ids.into_iter().collect();
		self.flushes.push(Flush {
			channel_id,
			direction,
			selector,
			oracles,
			multiplicity,
		});

		Ok(())
	}

	pub fn send(
		&mut self,
		channel_id: ChannelId,
		count: usize,
		oracle_ids: impl IntoIterator<Item = OracleOrConst<F>> + Clone,
	) -> anyhow::Result<()>
	where
		U: PackScalar<BinaryField1b>,
	{
		self.flush(FlushDirection::Push, channel_id, count, oracle_ids)
	}

	pub fn receive(
		&mut self,
		channel_id: ChannelId,
		count: usize,
		oracle_ids: impl IntoIterator<Item = OracleOrConst<F>> + Clone,
	) -> anyhow::Result<()>
	where
		U: PackScalar<BinaryField1b>,
	{
		self.flush(FlushDirection::Pull, channel_id, count, oracle_ids)
	}

	pub fn assert_zero(
		&mut self,
		name: impl ToString,
		oracle_ids: impl IntoIterator<Item = OracleId>,
		composition: ArithExpr<F>,
	) {
		self.constraints
			.add_zerocheck(name, oracle_ids, composition);
	}

	pub fn assert_not_zero(&mut self, oracle_id: OracleId) {
		self.non_zero_oracle_ids.push(oracle_id);
	}

	pub const fn add_channel(&mut self) -> ChannelId {
		let channel_id = self.next_channel_id;
		self.next_channel_id += 1;
		channel_id
	}

	pub fn add_committed(
		&mut self,
		name: impl ToString,
		n_vars: usize,
		tower_level: usize,
	) -> OracleId {
		self.oracles
			.borrow_mut()
			.add_named(self.scoped_name(name))
			.committed(n_vars, tower_level)
	}

	/// Adds an exponentiation operation to the constraint system.
	///
	/// # Parameters
	/// - `bits_ids`: A vector of `OracleId` representing the exponent in little-endian bit order.
	/// - `exp_result_id`: The `OracleId` that holds the result of the exponentiation..
	/// - `base`: The static base value.
	/// - `base_tower_level`: Specifies the field level in the tower where `base` is defined
	pub fn add_static_exp(
		&mut self,
		bits_ids: Vec<OracleId>,
		exp_result_id: OracleId,
		base: F,
		base_tower_level: usize,
	) {
		self.exponents.push(Exp {
			bits_ids,
			exp_result_id,
			base: OracleOrConst::Const {
				base,
				tower_level: base_tower_level,
			},
		});
	}

	/// Adds an exponentiation operation to the constraint system.
	///
	/// # Parameters
	/// - `bits_ids`: A vector of `OracleId` representing the exponent in little-endian bit order.
	/// - `exp_result_id`: The `OracleId` that holds the result of the exponentiation..
	/// - `base`: The dynamic base value.
	pub fn add_dynamic_exp(
		&mut self,
		bits_ids: Vec<OracleId>,
		exp_result_id: OracleId,
		base: OracleId,
	) {
		self.exponents.push(Exp {
			bits_ids,
			exp_result_id,
			base: OracleOrConst::Oracle(base),
		});
	}

	pub fn add_committed_multiple<const N: usize>(
		&mut self,
		name: impl ToString,
		n_vars: usize,
		tower_level: usize,
	) -> [OracleId; N] {
		self.oracles
			.borrow_mut()
			.add_named(self.scoped_name(name))
			.committed_multiple(n_vars, tower_level)
	}

	pub fn add_linear_combination(
		&mut self,
		name: impl ToString,
		n_vars: usize,
		inner: impl IntoIterator<Item = (OracleId, F)>,
	) -> Result<OracleId, OracleError> {
		self.oracles
			.borrow_mut()
			.add_named(self.scoped_name(name))
			.linear_combination(n_vars, inner)
	}

	pub fn add_linear_combination_with_offset(
		&mut self,
		name: impl ToString,
		n_vars: usize,
		offset: F,
		inner: impl IntoIterator<Item = (OracleId, F)>,
	) -> Result<OracleId, OracleError> {
		self.oracles
			.borrow_mut()
			.add_named(self.scoped_name(name))
			.linear_combination_with_offset(n_vars, offset, inner)
	}

	pub fn add_composite_mle(
		&mut self,
		name: impl ToString,
		n_vars: usize,
		inner: impl IntoIterator<Item = OracleId>,
		comp: ArithExpr<F>,
	) -> Result<OracleId, OracleError> {
		self.oracles
			.borrow_mut()
			.add_named(self.scoped_name(name))
			.composite_mle(n_vars, inner, comp)
	}

	pub fn add_packed(
		&mut self,
		name: impl ToString,
		id: OracleId,
		log_degree: usize,
	) -> Result<OracleId, OracleError> {
		self.oracles
			.borrow_mut()
			.add_named(self.scoped_name(name))
			.packed(id, log_degree)
	}

	pub fn add_projected(
		&mut self,
		name: impl ToString,
		id: OracleId,
		values: Vec<F>,
		variant: ProjectionVariant,
	) -> Result<usize, OracleError> {
		self.oracles
			.borrow_mut()
			.add_named(self.scoped_name(name))
			.projected(id, values, variant)
	}

	pub fn add_repeating(
		&mut self,
		name: impl ToString,
		id: OracleId,
		log_count: usize,
	) -> Result<OracleId, OracleError> {
		self.oracles
			.borrow_mut()
			.add_named(self.scoped_name(name))
			.repeating(id, log_count)
	}

	pub fn add_shifted(
		&mut self,
		name: impl ToString,
		id: OracleId,
		offset: usize,
		block_bits: usize,
		variant: ShiftVariant,
	) -> Result<OracleId, OracleError> {
		self.oracles
			.borrow_mut()
			.add_named(self.scoped_name(name))
			.shifted(id, offset, block_bits, variant)
	}

	pub fn add_transparent(
		&mut self,
		name: impl ToString,
		poly: impl MultivariatePoly<F> + 'static,
	) -> Result<OracleId, OracleError> {
		self.oracles
			.borrow_mut()
			.add_named(self.scoped_name(name))
			.transparent(poly)
	}

	pub fn add_zero_padded(
		&mut self,
		name: impl ToString,
		id: OracleId,
		n_vars: usize,
	) -> Result<OracleId, OracleError> {
		self.oracles
			.borrow_mut()
			.add_named(self.scoped_name(name))
			.zero_padded(id, n_vars)
	}

	fn scoped_name(&self, name: impl ToString) -> String {
		let name = name.to_string();
		if self.namespace_path.is_empty() {
			name
		} else {
			format!("{}::{name}", self.namespace_path.join("::"))
		}
	}

	/// Anything pushed to the namespace will become part of oracle name, which is useful for debugging.
	///
	/// Use `pop_namespace(&mut self)` to remove the latest name.
	///
	/// Example
	/// ```
	/// use binius_circuits::builder::ConstraintSystemBuilder;
	/// use binius_field::{TowerField, BinaryField128b, BinaryField1b, arch::OptimalUnderlier};
	///
	/// let log_size = 14;
	///
	/// let mut builder = ConstraintSystemBuilder::new();
	/// builder.push_namespace("a");
	/// let x = builder.add_committed("x", log_size, BinaryField1b::TOWER_LEVEL);
	/// builder.push_namespace("b");
	/// let y = builder.add_committed("y", log_size, BinaryField1b::TOWER_LEVEL);
	/// builder.pop_namespace();
	/// builder.pop_namespace();
	/// let z = builder.add_committed("z", log_size, BinaryField1b::TOWER_LEVEL);
	///
	/// let system = builder.build().unwrap();
	/// assert_eq!(system.oracles.oracle(x).name().unwrap(), "a::x");
	/// assert_eq!(system.oracles.oracle(y).name().unwrap(), "a::b::y");
	/// assert_eq!(system.oracles.oracle(z).name().unwrap(), "z");
	/// ```
	pub fn push_namespace(&mut self, name: impl ToString) {
		self.namespace_path.push(name.to_string());
	}

	pub fn pop_namespace(&mut self) {
		self.namespace_path.pop();
	}

	/// Returns the number of rows shared by a set of columns.
	///
	/// Fails if no columns are provided, or not all columns have the same number of rows.
	///
	/// This is useful for writing circuits with internal columns that depend on the height of input columns.
	pub fn log_rows(
		&self,
		oracle_ids: impl IntoIterator<Item = OracleId>,
	) -> anyhow::Result<usize> {
		let mut oracle_ids = oracle_ids.into_iter();
		let oracles = self.oracles.borrow();
		let Some(first_id) = oracle_ids.next() else {
			bail!(anyhow!("log_rows: You need to specify at least one column"));
		};
		let log_rows = oracles.n_vars(first_id);
		if oracle_ids.any(|id| oracles.n_vars(id) != log_rows) {
			bail!(anyhow!("log_rows: All columns must have the same number of rows"))
		}
		Ok(log_rows)
	}
}

#[cfg(test)]
mod tests {
	use binius_field::{BinaryField128b, PackedField, TowerField};
	use rand::{seq::SliceRandom, thread_rng};

	use super::*;
	use crate::{builder::test_utils::test_circuit, unconstrained::unconstrained};

	#[test]
	fn test_flush_with_const() {
		test_circuit(|builder| {
			let channel_id = builder.add_channel();
			let oracle = unconstrained::<BinaryField1b>(builder, "oracle", 1)?;
			builder
				.flush(
					FlushDirection::Push,
					channel_id,
					1,
					vec![
						OracleOrConst::Oracle(oracle),
						OracleOrConst::Const {
							base: F::one(),
							tower_level: BinaryField1b::TOWER_LEVEL,
						},
					],
				)
				.unwrap();

			builder
				.flush(
					FlushDirection::Pull,
					channel_id,
					1,
					vec![
						OracleOrConst::Oracle(oracle),
						OracleOrConst::Const {
							base: F::one(),
							tower_level: BinaryField1b::TOWER_LEVEL,
						},
					],
				)
				.unwrap();

			Ok(vec![])
		})
		.unwrap()
	}

	//Testing with larger oracles, and random constants, in a random order. To see if given appropriate flushes with constants the channel balances.
	#[test]
	fn test_flush_with_const_large() {
		test_circuit(|builder| {
			let channel_id = builder.add_channel();
			let mut rng = thread_rng();
			let oracles = (0..5)
				.map(|i| unconstrained::<BinaryField128b>(builder, format!("oracle {i}"), 5))
				.collect::<Result<Vec<_>, _>>()?;
			let random_consts = (0..5).map(|_| OracleOrConst::Const {
				base: BinaryField128b::random(&mut rng),
				tower_level: BinaryField128b::TOWER_LEVEL,
			});
			//Places the oracles and consts in a random order
			//This is not a cryptographic random order, but it is good enough for testing
			let mut random_order = oracles
				.iter()
				.copied()
				.map(OracleOrConst::Oracle)
				.chain(random_consts)
				.collect::<Vec<_>>();
			random_order.shuffle(&mut rng);

			let random_order_iterator = random_order.iter().copied();
			for i in 0..1 << 5 {
				builder
					.flush(FlushDirection::Push, channel_id, i, random_order_iterator.clone())
					.unwrap();

				builder
					.flush(FlushDirection::Pull, channel_id, i, random_order_iterator.clone())
					.unwrap();
			}

			Ok(vec![])
		})
		.unwrap()
	}
}
