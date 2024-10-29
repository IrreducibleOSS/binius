// Copyright 2024 Irreducible Inc.

use crate::polynomial::{Error, MultivariatePoly};
use binius_field::{packed::set_packed_slice, BinaryField1b, Field, PackedField};
use binius_math::MultilinearExtension;
use binius_utils::bail;

/// Represents a multilinear F2-polynomial whose evaluations over the hypercube is 1 at
/// a specific hypercube index, and 0 everywhere else.
///
/// ```txt
///     (1 << n_vars)
/// <-------------------->
/// 0,0 .. 0,0,1,0, .. 0,0
///            ^
///            index of 1
/// ```
///
/// This is useful for defining boundary constraints
#[derive(Debug, Clone)]
pub struct SelectRow {
	n_vars: usize,
	index: usize,
}

impl SelectRow {
	pub fn new(n_vars: usize, index: usize) -> Result<Self, Error> {
		if index >= (1 << n_vars) {
			bail!(Error::ArgumentRangeError {
				arg: "index".into(),
				range: 0..(1 << n_vars),
			})
		} else {
			Ok(Self { n_vars, index })
		}
	}

	pub fn multilinear_extension<P: PackedField<Scalar = BinaryField1b>>(
		&self,
	) -> Result<MultilinearExtension<P>, Error> {
		if self.n_vars < P::LOG_WIDTH {
			bail!(Error::PackedFieldNotFilled {
				length: 1 << self.n_vars,
				packed_width: 1 << P::LOG_WIDTH,
			});
		}
		let mut result = vec![P::zero(); 1 << (self.n_vars - P::LOG_WIDTH)];
		set_packed_slice(&mut result, self.index, P::Scalar::ONE);
		Ok(MultilinearExtension::from_values(result)?)
	}
}

impl<F: Field> MultivariatePoly<F> for SelectRow {
	fn degree(&self) -> usize {
		self.n_vars
	}

	fn n_vars(&self) -> usize {
		self.n_vars
	}

	fn evaluate(&self, query: &[F]) -> Result<F, Error> {
		let n_vars = MultivariatePoly::<F>::n_vars(self);
		if query.len() != n_vars {
			bail!(Error::IncorrectQuerySize { expected: n_vars });
		}
		let mut k = self.index;
		let mut result = F::ONE;
		for q in query {
			if k & 1 == 1 {
				// interpolate a line that is 0 at 0 and `result` at 1, at the point q
				result *= q;
			} else {
				// interpolate a line that is `result` at 0 and 0 at 1, at the point q
				result *= F::ONE - q;
			}
			k >>= 1;
		}
		Ok(result)
	}

	fn binary_tower_level(&self) -> usize {
		0
	}
}

#[cfg(test)]
mod tests {
	use super::SelectRow;
	use crate::polynomial::test_utils::{hypercube_evals_from_oracle, packed_slice};
	use binius_field::{
		BinaryField1b, PackedBinaryField128x1b, PackedBinaryField256x1b, PackedField,
	};
	use binius_utils::felts;

	#[test]
	fn test_select_row_evals_without_packing_simple_cases() {
		assert_eq!(select_row_evals::<BinaryField1b>(2, 0), felts!(BinaryField1b[1, 0, 0, 0]));
		assert_eq!(select_row_evals::<BinaryField1b>(2, 1), felts!(BinaryField1b[0, 1, 0, 0]));
		assert_eq!(select_row_evals::<BinaryField1b>(2, 2), felts!(BinaryField1b[0, 0, 1, 0]));
		assert_eq!(select_row_evals::<BinaryField1b>(2, 3), felts!(BinaryField1b[0, 0, 0, 1]));
		assert_eq!(
			select_row_evals::<BinaryField1b>(3, 0),
			felts!(BinaryField1b[1, 0, 0, 0, 0, 0, 0, 0])
		);
		assert_eq!(
			select_row_evals::<BinaryField1b>(3, 1),
			felts!(BinaryField1b[0, 1, 0, 0, 0, 0, 0, 0])
		);
		assert_eq!(
			select_row_evals::<BinaryField1b>(3, 2),
			felts!(BinaryField1b[0, 0, 1, 0, 0, 0, 0, 0])
		);
		assert_eq!(
			select_row_evals::<BinaryField1b>(3, 3),
			felts!(BinaryField1b[0, 0, 0, 1, 0, 0, 0, 0])
		);
		assert_eq!(
			select_row_evals::<BinaryField1b>(3, 4),
			felts!(BinaryField1b[0, 0, 0, 0, 1, 0, 0, 0])
		);
		assert_eq!(
			select_row_evals::<BinaryField1b>(3, 5),
			felts!(BinaryField1b[0, 0, 0, 0, 0, 1, 0, 0])
		);
		assert_eq!(
			select_row_evals::<BinaryField1b>(3, 6),
			felts!(BinaryField1b[0, 0, 0, 0, 0, 0, 1, 0])
		);
		assert_eq!(
			select_row_evals::<BinaryField1b>(3, 7),
			felts!(BinaryField1b[0, 0, 0, 0, 0, 0, 0, 1])
		);
	}

	#[test]
	fn test_select_row_evals_without_packing() {
		assert_eq!(
			select_row_evals::<BinaryField1b>(9, 314),
			packed_slice::<BinaryField1b>(&[(0..314, 0), (314..315, 1), (315..512, 0)])
		);
		assert_eq!(
			select_row_evals::<BinaryField1b>(10, 555),
			packed_slice::<BinaryField1b>(&[(0..555, 0), (555..556, 1), (556..1024, 0)])
		);
		assert_eq!(
			select_row_evals::<BinaryField1b>(11, 1),
			packed_slice::<BinaryField1b>(&[(0..1, 0), (1..2, 1), (2..2048, 0)])
		);
	}

	#[test]
	fn test_select_row_evals_packing_128() {
		assert_eq!(
			select_row_evals::<PackedBinaryField128x1b>(9, 314),
			packed_slice::<PackedBinaryField128x1b>(&[(0..314, 0), (314..315, 1), (315..512, 0)])
		);
		assert_eq!(
			select_row_evals::<PackedBinaryField128x1b>(10, 555),
			packed_slice::<PackedBinaryField128x1b>(&[(0..555, 0), (555..556, 1), (556..1024, 0)])
		);
		assert_eq!(
			select_row_evals::<PackedBinaryField128x1b>(11, 1),
			packed_slice::<PackedBinaryField128x1b>(&[(0..1, 0), (1..2, 1), (2..2048, 0)])
		);
	}

	#[test]
	fn test_select_row_evals_packing_256() {
		assert_eq!(
			select_row_evals::<PackedBinaryField256x1b>(9, 314),
			packed_slice::<PackedBinaryField256x1b>(&[(0..314, 0), (314..315, 1), (315..512, 0)])
		);
		assert_eq!(
			select_row_evals::<PackedBinaryField256x1b>(10, 555),
			packed_slice::<PackedBinaryField256x1b>(&[(0..555, 0), (555..556, 1), (556..1024, 0)])
		);
		assert_eq!(
			select_row_evals::<PackedBinaryField256x1b>(11, 1),
			packed_slice::<PackedBinaryField256x1b>(&[(0..1, 0), (1..2, 1), (2..2048, 0)])
		);
	}

	#[test]
	fn test_consistency_between_multilinear_extension_and_multilinear_poly_oracle() {
		for n_vars in 1..5 {
			for index in 0..(1 << n_vars) {
				let select_row = SelectRow::new(n_vars, index).unwrap();
				assert_eq!(
					hypercube_evals_from_oracle::<BinaryField1b>(&select_row),
					select_row
						.multilinear_extension::<BinaryField1b>()
						.unwrap()
						.evals()
				);
			}
		}
	}

	fn select_row_evals<P>(n_vars: usize, index: usize) -> Vec<P>
	where
		P: PackedField<Scalar = BinaryField1b>,
	{
		SelectRow::new(n_vars, index)
			.unwrap()
			.multilinear_extension::<P>()
			.unwrap()
			.evals()
			.to_vec()
	}
}
