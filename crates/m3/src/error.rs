// Copyright 2024 Irreducible Inc.

use std::cell::{BorrowError, BorrowMutError};

use crate::constraint_system::{ColumnId, TableId};

#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error("statement table sizes does not match the number of tables; expected {expected}, got {actual}")]
	StatementMissingTableSize { expected: usize, actual: usize },
	#[error("missing table with ID: {table_id}")]
	MissingTable { table_id: TableId },
	#[error("missing column with ID: {0:?}")]
	MissingColumn(ColumnId),
	#[error("column is not in table; column table ID: {column_table_id}, witness table ID: {witness_table_id}")]
	TableMismatch {
		column_table_id: TableId,
		witness_table_id: TableId,
	},
	// TODO: These should have column IDs
	#[error("witness borrow error: {0}")]
	WitnessBorrow(#[source] BorrowError),
	#[error("witness borrow error: {0}")]
	WitnessBorrowMut(#[source] BorrowMutError),
	#[error("table fill error: {0}")]
	TableFill(Box<dyn std::error::Error + Send + Sync>),
}
