// Copyright 2025 Irreducible Inc.

use binius_field::{PackedExtension, PackedField, PackedFieldIndexable, PackedSubfield};

use crate::builder::{B32, B128, column::Col, error::Error, witness::TableWitnessSegment};

/// Fills a structured [`crate::builder::structured::StructuredDynSize::Incrementing`] B32 column
/// with values.
///
/// This is specialized for B32 because that is a common case, which can be implemented
/// efficiently.
pub fn fill_incrementing_b32<P>(
	witness: &mut TableWitnessSegment<P>,
	col: Col<B32>,
) -> Result<(), Error>
where
	P: PackedField<Scalar = B128> + PackedExtension<B32>,
	PackedSubfield<P, B32>: PackedFieldIndexable,
{
	let mut col_data = witness.get_scalars_mut(col)?;
	let start_index = witness.index() << witness.log_size();
	for (i, col_data_i) in col_data.iter_mut().enumerate() {
		*col_data_i = B32::new((start_index + i) as u32);
	}
	Ok(())
}
