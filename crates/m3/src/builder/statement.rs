use binius_core::constraint_system::channel::Boundary;
use binius_field::TowerField;

use super::types::B128;

pub struct Statement<F: TowerField = B128> {
	pub boundaries: Vec<Boundary<F>>,
	/// Direct index mapping table IDs to the count of rows per table.
	///
	/// The table sizes seem like advice values that don't affect the semantic meaning of the
	/// statement, but we include them in the statement directly. This makes sense because
	///
	/// 1. These values affect the control flow of the verification routine.
	/// 2. These values are necessarily made public.
	/// 3. For some constraint systems, the verifier does care about the values. For example, the
	///    statement could be that a VM execution state is reachable within a certain number of
	///    cycles.
	pub table_sizes: Vec<usize>,
}
