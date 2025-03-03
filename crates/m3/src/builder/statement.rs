// Copyright 2025 Irreducible Inc.

use binius_core::constraint_system::channel::Boundary;
use binius_field::TowerField;

use super::types::B128;

/// A statement of values claimed to satisfy a constraint system.
pub struct Statement<F: TowerField = B128> {
	pub boundaries: Vec<Boundary<F>>,
	// TODO: This doesn't belong in `Statement`. We should split this struct somehow. Perhaps table
	// sizes go into a separate `Advice` struct.
	/// Direct index mapping table IDs to the count of rows per table.
	pub table_sizes: Vec<usize>,
}
