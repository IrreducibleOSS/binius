// Copyright 2025 Irreducible Inc.

use std::{ops::Index, sync::Arc};

use binius_core::{
	constraint_system::channel::{ChannelId, FlushDirection},
	oracle::ShiftVariant,
	transparent::MultilinearExtensionTransparent,
};
use binius_field::{
	ExtensionField, TowerField,
	arch::OptimalUnderlier,
	as_packed_field::{PackScalar, PackedType},
	packed::pack_slice,
};
use binius_math::ArithCircuit;
use binius_utils::{
	checked_arithmetics::{checked_log_2, log2_ceil_usize, log2_strict_usize},
	sparse_index::SparseIndex,
};

use super::{
	B1, ColumnIndex, ColumnPartitionIndex, FlushOpts,
	channel::Flush,
	column::{Col, ColumnDef, ColumnId, ColumnInfo, ColumnShape},
	expr::{Expr, ZeroConstraint},
	stat::TableStat,
	structured::StructuredDynSize,
	types::B128,
	upcast_col,
};

pub type TableId = usize;

#[derive(Debug)]
pub struct TableBuilder<'a, F: TowerField = B128> {
	namespace: Option<String>,
	table: &'a mut Table<F>,
}

impl<'a, F: TowerField> TableBuilder<'a, F> {
	/// Returns a new `TableBuilder` for the given table.
	pub fn new(table: &'a mut Table<F>) -> Self {
		Self {
			namespace: None,
			table,
		}
	}

	/// Declares that the table's size must be a power of two.
	///
	/// The table's size is decided by the prover, but it must be a power of two.
	///
	/// ## Pre-conditions
	///
	/// This cannot be called if [`Self::require_power_of_two_size`] or
	/// [`Self::require_fixed_size`] has already been called.
	pub fn require_power_of_two_size(&mut self) {
		assert!(matches!(self.table.table_size_spec, TableSizeSpec::Arbitrary));
		self.table.table_size_spec = TableSizeSpec::PowerOfTwo;
	}

	/// Declares that the table's size must be a fixed power of two.
	///
	/// ## Pre-conditions
	///
	/// This cannot be called if [`Self::require_power_of_two_size`] or
	/// [`Self::require_fixed_size`] has already been called.
	pub fn require_fixed_size(&mut self, log_size: usize) {
		assert!(matches!(self.table.table_size_spec, TableSizeSpec::Arbitrary));
		self.table.table_size_spec = TableSizeSpec::Fixed { log_size };
	}

	/// Returns a new `TableBuilder` with the specified namespace.
	///
	/// A namespace is a prefix that will be prepended to all column names and zero constraints
	/// created by this builder. The new namespace is nested within the current builder's namespace
	/// (if any exists). When nesting namespaces, they are joined with "::" separators, creating a
	/// hierarchical naming structure.
	///
	/// # Note
	///
	/// This method doesn't modify the original builder. It returns a new builder that shares
	/// the underlying table but has its own namespace configuration that builds upon the
	/// original builder's namespace.
	///
	/// # Examples
	///
	/// ```
	/// # use binius_m3::builder::{B128, Col, Table, TableBuilder};
	/// let mut table = Table::<B128>::new(0, "table");
	/// let mut tb = TableBuilder::new(&mut table);
	///
	/// // Create a builder with namespace "arithmetic"
	/// let mut arithmetic_tb = tb.with_namespace("arithmetic");
	/// let add_col: Col<B128> = arithmetic_tb.add_committed("add"); // Column name: "arithmetic::add"
	///
	/// // Create a nested namespace "arithmetic::mul"
	/// let mut mul_tb = arithmetic_tb.with_namespace("mul");
	/// let result_col: Col<B128> = mul_tb.add_committed("result"); // Column name: "arithmetic::mul::result"
	/// ```
	pub fn with_namespace(&mut self, namespace: impl ToString) -> TableBuilder<'_, F> {
		TableBuilder {
			namespace: Some(self.namespaced_name(namespace)),
			table: self.table,
		}
	}

	/// Returns the [`TableId`] of the underlying table.
	pub fn id(&self) -> TableId {
		self.table.id()
	}

	pub fn add_committed<FSub, const VALUES_PER_ROW: usize>(
		&mut self,
		name: impl ToString,
	) -> Col<FSub, VALUES_PER_ROW>
	where
		FSub: TowerField,
		F: ExtensionField<FSub>,
	{
		self.table.new_column(
			self.namespaced_name(name),
			ColumnDef::Committed {
				tower_level: FSub::TOWER_LEVEL,
			},
		)
	}

	pub fn add_committed_multiple<FSub, const VALUES_PER_ROW: usize, const N: usize>(
		&mut self,
		name: impl ToString,
	) -> [Col<FSub, VALUES_PER_ROW>; N]
	where
		FSub: TowerField,
		F: ExtensionField<FSub>,
	{
		std::array::from_fn(|i| self.add_committed(format!("{}[{}]", name.to_string(), i)))
	}

	pub fn add_shifted<FSub, const VALUES_PER_ROW: usize>(
		&mut self,
		name: impl ToString,
		col: Col<FSub, VALUES_PER_ROW>,
		log_block_size: usize,
		offset: usize,
		variant: ShiftVariant,
	) -> Col<FSub, VALUES_PER_ROW>
	where
		FSub: TowerField,
		F: ExtensionField<FSub>,
	{
		assert!(log_block_size <= log2_strict_usize(VALUES_PER_ROW));
		assert!(offset <= 1 << log_block_size);
		self.table.new_column(
			self.namespaced_name(name),
			ColumnDef::Shifted {
				col: col.id(),
				offset,
				log_block_size,
				variant,
			},
		)
	}

	pub fn add_packed<FSubSub, const VALUES_PER_ROW_SUB: usize, FSub, const VALUES_PER_ROW: usize>(
		&mut self,
		name: impl ToString,
		col: Col<FSubSub, VALUES_PER_ROW_SUB>,
	) -> Col<FSub, VALUES_PER_ROW>
	where
		FSub: TowerField + ExtensionField<FSubSub>,
		FSubSub: TowerField,
		F: ExtensionField<FSub>,
	{
		assert!(FSubSub::TOWER_LEVEL < FSub::TOWER_LEVEL);
		assert!(VALUES_PER_ROW_SUB > VALUES_PER_ROW);
		assert_eq!(
			FSub::TOWER_LEVEL + log2_strict_usize(VALUES_PER_ROW),
			FSubSub::TOWER_LEVEL + log2_strict_usize(VALUES_PER_ROW_SUB)
		);
		self.table.new_column(
			self.namespaced_name(name),
			ColumnDef::Packed {
				col: col.id(),
				log_degree: FSub::TOWER_LEVEL - FSubSub::TOWER_LEVEL,
			},
		)
	}

	/// Adds a derived column that is computed as an expression over other columns in the table.
	///
	/// The derived column has the same vertical stacking factor as the input columns and its
	/// values are computed independently. The cost of the column's evaluations are proportional
	/// to the polynomial degree of the expression. When the expression is linear, the column's
	/// cost is minimal. When the expression is non-linear, the column's evaluations are resolved
	/// by a sumcheck reduction.
	pub fn add_computed<FSub, const V: usize>(
		&mut self,
		name: impl ToString,
		expr: Expr<FSub, V>,
	) -> Col<FSub, V>
	where
		FSub: TowerField,
		F: ExtensionField<FSub>,
	{
		let expr_circuit = ArithCircuit::from(expr.expr());
		// Indicies within the partition.
		let indices_within_partition = expr_circuit
			.vars_usage()
			.iter()
			.enumerate()
			.filter(|(_, used)| **used)
			.map(|(i, _)| i)
			.collect::<Vec<_>>();
		let partition = &self.table.partitions[partition_id::<V>()];
		let cols = indices_within_partition
			.iter()
			.map(|&partition_index| partition.columns[partition_index])
			.collect::<Vec<_>>();

		let mut var_remapping = vec![0; expr_circuit.n_vars()];
		for (new_index, &old_index) in indices_within_partition.iter().enumerate() {
			var_remapping[old_index] = new_index;
		}
		let remapped_expr = expr_circuit
			.convert_field()
			.remap_vars(&var_remapping)
			.expect("var_remapping should be large enought");

		self.table.new_column(
			self.namespaced_name(name),
			ColumnDef::Computed {
				cols,
				expr: remapped_expr,
			},
		)
	}

	/// Add a derived column that selects a single value from a vertically stacked column.
	///
	/// The virtual column is derived from another column in the table passed as `col`, which we'll
	/// call the "inner" column. The inner column has `V` values vertically stacked per table cell.
	/// The `index` is in the range `0..V`, and it selects the `index`-th value from the inner
	/// column.
	pub fn add_selected<FSub, const V: usize>(
		&mut self,
		name: impl ToString,
		col: Col<FSub, V>,
		index: usize,
	) -> Col<FSub, 1>
	where
		FSub: TowerField,
		F: ExtensionField<FSub>,
	{
		assert!(index < V);
		self.table.new_column(
			self.namespaced_name(name),
			ColumnDef::Selected {
				col: col.id(),
				index,
				index_bits: log2_strict_usize(V),
			},
		)
	}

	/// Add a derived column that selects a subrange of values from a vertically stacked column.
	///
	/// The virtual column is derived from another column in the table passed as `col`, which we'll
	/// call the "inner" column. The inner column has `V` values vertically stacked per table cell.
	/// The `index` is in the range `0..(V - NEW_V)`, and it selects the values
	/// `(i * NEW_V)..((i + 1) * NEW_V)` from the inner column.
	pub fn add_selected_block<FSub, const V: usize, const NEW_V: usize>(
		&mut self,
		name: impl ToString,
		col: Col<FSub, V>,
		index: usize,
	) -> Col<FSub, NEW_V>
	where
		FSub: TowerField,
		F: ExtensionField<FSub>,
	{
		assert!(V.is_power_of_two());
		assert!(NEW_V.is_power_of_two());
		assert!(NEW_V < V);

		let log_values_per_row = log2_strict_usize(V);
		// This is also the value of the start_index.
		let log_new_values_per_row = log2_strict_usize(NEW_V);
		// Get the log size of the query.
		let log_query_size = log_values_per_row - log_new_values_per_row;

		self.table.new_column(
			self.namespaced_name(name),
			ColumnDef::Projected {
				col: col.id(),
				start_index: log_new_values_per_row,
				query_size: log_query_size,
				query_bits: index,
			},
		)
	}

	/// Given the representation at a tower level FSub (with `VALUES_PER_ROW` variables),
	/// returns the representation at a higher tower level F (with `NEW_VALUES_PER_ROW` variables)
	/// by left padding each FSub element with zeroes.
	pub fn add_zero_pad_upcast<FSub, const VALUES_PER_ROW: usize, const NEW_VALUES_PER_ROW: usize>(
		&mut self,
		name: impl ToString,
		col: Col<FSub, VALUES_PER_ROW>,
	) -> Col<FSub, NEW_VALUES_PER_ROW>
	where
		FSub: TowerField,
		F: ExtensionField<FSub>,
	{
		assert!(VALUES_PER_ROW.is_power_of_two());
		assert!(NEW_VALUES_PER_ROW.is_power_of_two());
		assert!(NEW_VALUES_PER_ROW > VALUES_PER_ROW);
		let log_new_values_per_row = log2_strict_usize(NEW_VALUES_PER_ROW);
		let log_values_per_row = log2_strict_usize(VALUES_PER_ROW);
		let n_pad_vars = log_new_values_per_row - log_values_per_row;
		let nonzero_index = (1 << n_pad_vars) - 1;
		self.table.new_column(
			self.namespaced_name(name),
			ColumnDef::ZeroPadded {
				col: col.id(),
				n_pad_vars,
				start_index: log_values_per_row,
				nonzero_index,
			},
		)
	}

	/// Given the representation at a tower level FSub (with `VALUES_PER_ROW` variables),
	/// returns the representation at a higher tower level F (with `NEW_VALUES_PER_ROW` variables).
	/// This is done by keeping the `nonzero-index`-th FSub element, and setting all the others to
	/// 0. Note that `0 <= nonzero_index < NEW_VALUES_PER_ROW / VALUES_PER_ROW`.
	pub fn add_zero_pad<FSub, const VALUES_PER_ROW: usize, const NEW_VALUES_PER_ROW: usize>(
		&mut self,
		name: impl ToString,
		col: Col<FSub, VALUES_PER_ROW>,
		nonzero_index: usize,
	) -> Col<FSub, NEW_VALUES_PER_ROW>
	where
		FSub: TowerField,
		F: ExtensionField<FSub>,
	{
		assert!(VALUES_PER_ROW.is_power_of_two());
		assert!(NEW_VALUES_PER_ROW.is_power_of_two());
		assert!(NEW_VALUES_PER_ROW > VALUES_PER_ROW);
		let log_new_values_per_row = log2_strict_usize(NEW_VALUES_PER_ROW);
		let log_values_per_row = log2_strict_usize(VALUES_PER_ROW);
		let n_pad_vars = log_new_values_per_row - log_values_per_row;
		assert!(nonzero_index < 1 << n_pad_vars);

		self.table.new_column(
			self.namespaced_name(name),
			ColumnDef::ZeroPadded {
				col: col.id(),
				n_pad_vars,
				start_index: log_values_per_row,
				nonzero_index,
			},
		)
	}

	/// Adds a column to the table with a constant cell value.
	///
	/// The cell is repeated for each row in the table, but the values stacked vertically within
	/// the cell are not necessarily all equal.
	pub fn add_constant<FSub, const V: usize>(
		&mut self,
		name: impl ToString,
		constants: [FSub; V],
	) -> Col<FSub, V>
	where
		FSub: TowerField,
		F: ExtensionField<FSub>,
		OptimalUnderlier: PackScalar<FSub> + PackScalar<F>,
	{
		let namespaced_name = self.namespaced_name(name);
		let n_vars = log2_strict_usize(V);
		let packed_values: Vec<PackedType<OptimalUnderlier, FSub>> = pack_slice(&constants);
		let mle = MultilinearExtensionTransparent::<
			PackedType<OptimalUnderlier, FSub>,
			PackedType<OptimalUnderlier, F>,
			_,
		>::from_values_and_mu(packed_values, n_vars)
		.unwrap();
		self.table.new_column(
			namespaced_name,
			ColumnDef::Constant {
				poly: Arc::new(mle),
				data: constants.map(|f_sub| f_sub.into()).to_vec(),
			},
		)
	}

	/// Adds field exponentiation column with a fixed base
	///
	/// ## Parameters
	/// - `name`: Name for the column
	/// - `pow_bits`: The bits of exponent columns from LSB to MSB
	/// - `base`: The base to exponentiate. The field used in exponentiation will be `FSub`
	///
	/// ## Preconditions
	/// * `pow_bits.len()` must be less than or equal to the width of the field `FSub`
	///
	/// ## NOTE
	/// * The witness generation for the return column will be done inside gkr_gpa *
	pub fn add_static_exp<FExpBase>(
		&mut self,
		name: impl ToString,
		pow_bits: &[Col<B1>],
		base: FExpBase,
	) -> Col<FExpBase>
	where
		FExpBase: TowerField,
		F: ExtensionField<FExpBase>,
	{
		assert!(pow_bits.len() <= (1 << FExpBase::TOWER_LEVEL));

		// TODO: Add check for F, FSub, VALUES_PER_ROW
		let namespaced_name = self.namespaced_name(name);
		let bit_cols = pow_bits
			.iter()
			.enumerate()
			.map(|(index, bit)| {
				assert_eq!(
					self.table.id(),
					bit.id().table_id,
					"passed foreign table column at index={index}"
				);
				bit.id()
			})
			.collect();
		self.table.new_column(
			namespaced_name,
			ColumnDef::StaticExp {
				bit_cols,
				base: base.into(),
				base_tower_level: FExpBase::TOWER_LEVEL,
			},
		)
	}

	/// Adds field exponentiation column with a base from another column
	///
	/// ## Parameters
	/// - `name`: Name for the column
	/// - `pow_bits`: The bits of exponent columns from LSB to MSB
	/// - `base`: The column of base to exponentiate. The field used in exponentiation will be
	///   `FSub`
	///
	/// ## Preconditions
	/// * `pow_bits.len()` must be less than or equal to the width of field `FSub`
	///
	/// ## NOTE
	/// * The witness generation for the return column will be done inside gkr_gpa *
	pub fn add_dynamic_exp<FExpBase>(
		&mut self,
		name: impl ToString,
		pow_bits: &[Col<B1>],
		base: Col<FExpBase>,
	) -> Col<FExpBase>
	where
		FExpBase: TowerField,
		F: ExtensionField<FExpBase>,
	{
		assert!(pow_bits.len() <= (1 << FExpBase::TOWER_LEVEL));

		// TODO: Add check for F, FSub, VALUES_PER_ROW
		let namespaced_name = self.namespaced_name(name);
		let bit_cols = pow_bits
			.iter()
			.enumerate()
			.map(|(index, bit)| {
				assert_eq!(
					self.table.id(),
					bit.id().table_id,
					"passed foreign table column at index={index}"
				);
				bit.id()
			})
			.collect();
		self.table.new_column(
			namespaced_name,
			ColumnDef::DynamicExp {
				bit_cols,
				base: base.id(),
				base_tower_level: FExpBase::TOWER_LEVEL,
			},
		)
	}

	/// Add a structured column to a table.
	///
	/// A structured column is one that has sufficient structure that its multilinear extension
	/// can be evaluated succinctly. See [`StructuredDynSize`] for more information.
	pub fn add_structured<FSub>(
		&mut self,
		name: impl ToString,
		variant: StructuredDynSize,
	) -> Col<FSub>
	where
		FSub: TowerField,
		F: ExtensionField<FSub>,
	{
		assert!(
			self.table.requires_any_po2_size(),
			"Structured dynamic size columns may only be added to tables that are power of two sized"
		);
		let namespaced_name = self.namespaced_name(name);
		self.table
			.new_column(namespaced_name, ColumnDef::StructuredDynSize(variant))
	}

	/// Add a structured fixed-size column to a table.
	pub fn add_fixed<FSub>(&mut self, name: impl ToString, expr: ArithCircuit<F>) -> Col<FSub>
	where
		FSub: TowerField,
		F: ExtensionField<FSub>,
	{
		assert!(
			matches!(self.table.table_size_spec, TableSizeSpec::Fixed { log_size } if log_size == expr.n_vars()),
			"Structured fixed-size columns may only be added to tables with a fixed log_size that matches the n_vars of the expression"
		);

		let namespaced_name = self.namespaced_name(name);
		self.table
			.new_column(namespaced_name, ColumnDef::StructuredFixedSize { expr })
	}

	/// Constrains that an expression computed over the table columns is zero.
	///
	/// The zero constraint applies to all values stacked vertically within the column cells. That
	/// means that the expression is evaluated independently `V` times per row, and each evaluation
	/// in the stack must be zero.
	pub fn assert_zero<FSub, const V: usize>(&mut self, name: impl ToString, expr: Expr<FSub, V>)
	where
		FSub: TowerField,
		F: ExtensionField<FSub>,
	{
		let namespaced_name = self.namespaced_name(name);
		self.table
			.partition_mut(V)
			.assert_zero(namespaced_name, expr)
	}

	/// Constrains that all values contained in this column are non-zero.
	pub fn assert_nonzero<FSub, const V: usize>(&mut self, expr: Col<FSub, V>)
	where
		FSub: TowerField,
		F: ExtensionField<FSub>,
	{
		assert_eq!(expr.table_id, self.id());
		assert!(expr.table_index.0 < self.table.columns.len());

		self.table.columns[expr.table_index.0].is_nonzero = true;
	}

	pub fn pull<FSub>(&mut self, channel: ChannelId, cols: impl IntoIterator<Item = Col<FSub>>)
	where
		FSub: TowerField,
		F: ExtensionField<FSub>,
	{
		self.pull_with_opts(channel, cols, FlushOpts::default());
	}

	pub fn push<FSub>(&mut self, channel: ChannelId, cols: impl IntoIterator<Item = Col<FSub>>)
	where
		FSub: TowerField,
		F: ExtensionField<FSub>,
	{
		self.push_with_opts(channel, cols, FlushOpts::default());
	}

	pub fn pull_with_opts<FSub>(
		&mut self,
		channel: ChannelId,
		cols: impl IntoIterator<Item = Col<FSub>>,
		opts: FlushOpts,
	) where
		FSub: TowerField,
		F: ExtensionField<FSub>,
	{
		self.table.partition_mut(1).flush(
			channel,
			FlushDirection::Pull,
			cols.into_iter().map(upcast_col),
			opts,
		);
	}

	pub fn push_with_opts<FSub>(
		&mut self,
		channel: ChannelId,
		cols: impl IntoIterator<Item = Col<FSub>>,
		opts: FlushOpts,
	) where
		FSub: TowerField,
		F: ExtensionField<FSub>,
	{
		self.table.partition_mut(1).flush(
			channel,
			FlushDirection::Push,
			cols.into_iter().map(upcast_col),
			opts,
		);
	}

	fn namespaced_name(&self, name: impl ToString) -> String {
		let name = name.to_string();
		match &self.namespace {
			Some(namespace) => format!("{namespace}::{name}"),
			None => name.to_string(),
		}
	}
}

/// A table in an M3 constraint system.
///
/// ## Invariants
///
/// * All expressions in `zero_constraints` have a number of variables less than or equal to the
///   number of table columns (the length of `column_info`).
/// * All flushes in `flushes` contain column indices less than the number of table columns (the
///   length of `column_info`).
#[derive(Debug)]
pub struct Table<F: TowerField = B128> {
	pub id: TableId,
	pub name: String,
	pub columns: Vec<ColumnInfo<F>>,
	/// the size specification of a table
	table_size_spec: TableSizeSpec,
	pub(super) partitions: SparseIndex<TablePartition<F>>,
}

/// A table partition describes a part of a table where everything has the same pack factor (as well
/// as height) Tower level does not need to be the same.
///
/// Zerocheck constraints can only be defined within table partitions.
#[derive(Debug)]
pub(super) struct TablePartition<F: TowerField = B128> {
	pub table_id: TableId,
	pub values_per_row: usize,
	pub flushes: Vec<Flush>,
	pub columns: Vec<ColumnId>,
	pub zero_constraints: Vec<ZeroConstraint<F>>,
}

impl<F: TowerField> TablePartition<F> {
	fn new(table_id: TableId, values_per_row: usize) -> Self {
		Self {
			table_id,
			values_per_row,
			flushes: Vec::new(),
			columns: Vec::new(),
			zero_constraints: Vec::new(),
		}
	}

	fn assert_zero<FSub, const VALUES_PER_ROW: usize>(
		&mut self,
		name: impl ToString,
		expr: Expr<FSub, VALUES_PER_ROW>,
	) where
		FSub: TowerField,
		F: ExtensionField<FSub>,
	{
		self.zero_constraints.push(ZeroConstraint {
			tower_level: FSub::TOWER_LEVEL,
			name: name.to_string(),
			expr: ArithCircuit::from(expr.expr()).convert_field(),
		});
	}

	fn flush(
		&mut self,
		channel_id: ChannelId,
		direction: FlushDirection,
		cols: impl IntoIterator<Item = Col<F>>,
		opts: FlushOpts,
	) {
		let columns = cols
			.into_iter()
			.map(|col| {
				assert_eq!(col.table_id, self.table_id);
				col.id()
			})
			.collect();
		let selectors = opts
			.selectors
			.iter()
			.map(|selector| {
				assert_eq!(selector.table_id, self.table_id);
				selector.id()
			})
			.collect::<Vec<_>>();
		self.flushes.push(Flush {
			columns,
			channel_id,
			direction,
			multiplicity: opts.multiplicity,
			selectors,
		});
	}
}

impl<F: TowerField> Table<F> {
	pub fn new(id: TableId, name: impl ToString) -> Self {
		Self {
			id,
			name: name.to_string(),
			columns: Vec::new(),
			table_size_spec: TableSizeSpec::Arbitrary,
			partitions: SparseIndex::new(),
		}
	}

	pub fn id(&self) -> TableId {
		self.id
	}

	fn new_column<FSub, const V: usize>(
		&mut self,
		name: impl ToString,
		col: ColumnDef<F>,
	) -> Col<FSub, V>
	where
		FSub: TowerField,
		F: ExtensionField<FSub>,
	{
		let table_id = self.id;
		let table_index = ColumnIndex(self.columns.len());
		let partition = self.partition_mut(V);
		let id = ColumnId {
			table_id,
			table_index,
		};
		let info = ColumnInfo {
			id,
			col,
			name: name.to_string(),
			shape: ColumnShape {
				tower_height: FSub::TOWER_LEVEL,
				log_values_per_row: log2_strict_usize(V),
			},
			is_nonzero: false,
		};

		let partition_index = ColumnPartitionIndex(partition.columns.len());
		partition.columns.push(id);
		self.columns.push(info);
		Col::new(id, partition_index)
	}

	fn partition_mut(&mut self, values_per_row: usize) -> &mut TablePartition<F> {
		self.partitions
			.entry(log2_strict_usize(values_per_row))
			.or_insert_with(|| TablePartition::new(self.id, values_per_row))
	}

	/// Returns true if this table requires to have any power-of-two size.
	pub fn requires_any_po2_size(&self) -> bool {
		matches!(self.table_size_spec, TableSizeSpec::PowerOfTwo)
	}

	/// Returns the size constraint of this table.
	pub(crate) fn size_spec(&self) -> TableSizeSpec {
		self.table_size_spec
	}

	pub fn stat(&self) -> TableStat {
		TableStat::new(self)
	}
}

impl<F: TowerField> Index<ColumnIndex> for Table<F> {
	type Output = ColumnInfo<F>;

	fn index(&self, index: ColumnIndex) -> &Self::Output {
		&self.columns[index.0]
	}
}

impl<F: TowerField> Index<ColumnId> for Table<F> {
	type Output = ColumnInfo<F>;

	fn index(&self, index: ColumnId) -> &Self::Output {
		assert_eq!(index.table_id, self.id());
		&self.columns[index.table_index.0]
	}
}

const fn partition_id<const V: usize>() -> usize {
	checked_log_2(V)
}

/// A category of the size specification of a table.
///
/// M3 tables can have size restrictions, where certain columns, specifically structured columns,
/// are only allowed for certain size specifications.
#[derive(Debug, Copy, Clone)]
pub(crate) enum TableSizeSpec {
	/// The table size may be arbitrary.
	Arbitrary,
	/// The table size may be any power of two.
	PowerOfTwo,
	/// The table size must be a fixed power of two.
	Fixed { log_size: usize },
}

/// Returns the binary logarithm of the table capacity required to accommodate the given number
/// of rows.
///
/// The table capacity must be a power of two (in order to be compatible with the multilinear
/// proof system, which associates each table index with a vertex of a boolean hypercube).
/// This is be the next power of two greater than the table size.
pub fn log_capacity(table_size: usize) -> usize {
	log2_ceil_usize(table_size)
}

#[cfg(test)]
mod tests {
	use super::{Table, TableBuilder};
	use crate::builder::B128;

	#[test]
	fn namespace_nesting() {
		let mut table = Table::<B128>::new(0, "table");
		let mut tb = TableBuilder::new(&mut table);
		let mut tb_ns_1 = tb.with_namespace("ns1");
		let tb_ns_2 = tb_ns_1.with_namespace("ns2");
		assert_eq!(tb_ns_2.namespaced_name("column"), "ns1::ns2::column");
	}
}
