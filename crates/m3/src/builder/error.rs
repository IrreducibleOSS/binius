// Copyright 2025 Irreducible Inc.

use std::cell::{BorrowError, BorrowMutError};

use binius_core::{oracle::Error as OracleError, polynomial::Error as PolynomialError};
use binius_math::Error as MathError;

use super::{column::ColumnId, structured::Error as StructuredError, table::TableId};

#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error("statement table sizes does not match the number of tables; expected {expected}, got {actual}")]
	StatementMissingTableSize { expected: usize, actual: usize },
	#[error("missing table with ID: {table_id}")]
	MissingTable { table_id: TableId },
	#[error("missing column with ID: {0:?}")]
	MissingColumn(ColumnId),
	#[error("missing partition with log_vals_per_row={log_vals_per_row} in table {table_id}")]
	MissingPartition {
		table_id: TableId,
		log_vals_per_row: usize,
	},
	#[error("cannot construct witness index for empty table {table_id}")]
	EmptyTable { table_id: TableId },
	#[error("failed to write element to a column with a lower tower height")]
	FieldElementTooBig,
	#[error("structured column error: {0}")]
	Structured(#[from] StructuredError),
	#[error("table {table_id} index has already been initialized")]
	TableIndexAlreadyInitialized { table_id: TableId },
	#[error("column is not in table; column table ID: {column_table_id}, witness table ID: {witness_table_id}")]
	TableMismatch {
		column_table_id: TableId,
		witness_table_id: TableId,
	},
	#[error("table {table_id} is required to have a power-of-two size, instead got {size}")]
	TableSizePowerOfTwoRequired { table_id: TableId, size: usize },
	// TODO: These should have column IDs
	#[error("witness borrow error: {0}. Note that packed columns are aliases for the unpacked column when accessing witness data")]
	WitnessBorrow(#[source] BorrowError),
	#[error("witness borrow error: {0}. Note that packed columns are aliases for the unpacked column when accessing witness data")]
	WitnessBorrowMut(#[source] BorrowMutError),
	#[error(
		"the table index was initialized for {expected} events; attempted to fill with {actual}"
	)]
	IncorrectNumberOfTableEvents { expected: usize, actual: usize },
	#[error("table fill error: {0}")]
	TableFill(anyhow::Error),
	#[error("math error: {0}")]
	Math(#[from] MathError),
	#[error("oracle error: {0}")]
	Oracle(#[from] OracleError),
	#[error("polynomial error: {0}")]
	Polynomial(#[from] PolynomialError),
}
