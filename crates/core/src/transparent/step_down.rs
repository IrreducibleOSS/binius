// Copyright 2024-2025 Irreducible Inc.

use binius_field::{Field, PackedField};
use binius_math::MultilinearExtension;
use binius_utils::bail;

use crate::polynomial::{Error, MultivariatePoly};

/// Represents a multilinear F2-polynomial whose evaluations over the hypercube are
/// 1 until a specified index where they change to 0.
///
/// If the index is the length of the multilinear, then all coefficients are 1.
///
/// ```txt
///     (1 << n_vars)
/// <-------------------->
/// 1,1 .. 1,1,0,0, .. 0,0
///            ^
///            index of first 0
/// ```
///
/// This is useful for making constraints that are not enforced at the last rows of the trace
#[derive(Debug, Clone)]
pub struct StepDown {
	n_vars: usize,
	index: usize,
}

impl StepDown {
	pub fn new(n_vars: usize, index: usize) -> Result<Self, Error> {
		if index > 1 << n_vars {
			bail!(Error::ArgumentRangeError {
				arg: "index".into(),
				range: 0..(1 << n_vars) + 1,
			})
		}
		Ok(Self { n_vars, index })
	}

	pub const fn n_vars(&self) -> usize {
		self.n_vars
	}

	pub fn multilinear_extension<P: PackedField>(&self) -> Result<MultilinearExtension<P>, Error> {
		if self.n_vars < P::LOG_WIDTH {
			bail!(Error::PackedFieldNotFilled {
				length: 1 << self.n_vars,
				packed_width: 1 << P::LOG_WIDTH,
			});
		}
		let log_packed_length = self.n_vars - P::LOG_WIDTH;
		let mut data = vec![P::one(); 1 << log_packed_length];
		self.populate(&mut data);
		Ok(MultilinearExtension::from_values(data)?)
	}

	pub fn populate<P: PackedField>(&self, data: &mut [P]) {
		let packed_index = self.index / P::WIDTH;
		data[..packed_index].fill(P::one());
		data[packed_index..].fill(P::zero());
		for i in 0..(self.index % P::WIDTH) {
			data[packed_index].set(i, P::Scalar::ONE);
		}
	}
}

impl<F: Field> MultivariatePoly<F> for StepDown {
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

		if k == 1 << n_vars {
			return Ok(F::ONE);
		}
		let mut result = F::ZERO;
		for q in query {
			if k & 1 == 1 {
				// interpolate a line that is 1 at 0 and `result` at 1, at the point q
				result = (F::ONE - q) + result * q;
			} else {
				// interpolate a line that is `result` at 0 and 0 at 1, and evaluate at q
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
	use binius_field::{
		BinaryField1b, PackedBinaryField128x1b, PackedBinaryField256x1b, PackedField,
	};
	use binius_utils::felts;

	use super::StepDown;
	use crate::polynomial::test_utils::{hypercube_evals_from_oracle, packed_slice};

	#[test]
	fn test_step_down_trace_without_packing_simple_cases() {
		assert_eq!(stepdown_evals::<BinaryField1b>(2, 0), felts!(BinaryField1b[0, 0, 0, 0]));
		assert_eq!(stepdown_evals::<BinaryField1b>(2, 1), felts!(BinaryField1b[1, 0, 0, 0]));
		assert_eq!(stepdown_evals::<BinaryField1b>(2, 2), felts!(BinaryField1b[1, 1, 0, 0]));
		assert_eq!(stepdown_evals::<BinaryField1b>(2, 3), felts!(BinaryField1b[1, 1, 1, 0]));
		assert_eq!(stepdown_evals::<BinaryField1b>(2, 4), felts!(BinaryField1b[1, 1, 1, 1]));
		assert_eq!(
			stepdown_evals::<BinaryField1b>(3, 0),
			felts!(BinaryField1b[0, 0, 0, 0, 0, 0, 0, 0])
		);
		assert_eq!(
			stepdown_evals::<BinaryField1b>(3, 1),
			felts!(BinaryField1b[1, 0, 0, 0, 0, 0, 0, 0])
		);
		assert_eq!(
			stepdown_evals::<BinaryField1b>(3, 2),
			felts!(BinaryField1b[1, 1, 0, 0, 0, 0, 0, 0])
		);
		assert_eq!(
			stepdown_evals::<BinaryField1b>(3, 3),
			felts!(BinaryField1b[1, 1, 1, 0, 0, 0, 0, 0])
		);
		assert_eq!(
			stepdown_evals::<BinaryField1b>(3, 4),
			felts!(BinaryField1b[1, 1, 1, 1, 0, 0, 0, 0])
		);
		assert_eq!(
			stepdown_evals::<BinaryField1b>(3, 5),
			felts!(BinaryField1b[1, 1, 1, 1, 1, 0, 0, 0])
		);
		assert_eq!(
			stepdown_evals::<BinaryField1b>(3, 6),
			felts!(BinaryField1b[1, 1, 1, 1, 1, 1, 0, 0])
		);
		assert_eq!(
			stepdown_evals::<BinaryField1b>(3, 7),
			felts!(BinaryField1b[1, 1, 1, 1, 1, 1, 1, 0])
		);
		assert_eq!(
			stepdown_evals::<BinaryField1b>(3, 8),
			felts!(BinaryField1b[1, 1, 1, 1, 1, 1, 1, 1])
		);
	}

	#[test]
	fn test_step_down_trace_without_packing() {
		assert_eq!(
			stepdown_evals::<BinaryField1b>(9, 314),
			packed_slice::<BinaryField1b>(&[(0..314, 1), (314..512, 0)])
		);
		assert_eq!(
			stepdown_evals::<BinaryField1b>(10, 555),
			packed_slice::<BinaryField1b>(&[(0..555, 1), (555..1024, 0)])
		);
		assert_eq!(
			stepdown_evals::<BinaryField1b>(11, 0),
			packed_slice::<BinaryField1b>(&[(0..2048, 0)])
		);
		assert_eq!(
			stepdown_evals::<BinaryField1b>(11, 1),
			packed_slice::<BinaryField1b>(&[(0..1, 1), (1..2048, 0)])
		);
		assert_eq!(
			stepdown_evals::<BinaryField1b>(11, 2048),
			packed_slice::<BinaryField1b>(&[(0..2048, 1)])
		);
	}

	#[test]
	fn test_step_down_trace_with_packing_128() {
		assert_eq!(
			stepdown_evals::<PackedBinaryField128x1b>(9, 314),
			packed_slice::<PackedBinaryField128x1b>(&[(0..314, 1), (314..512, 0)])
		);
		assert_eq!(
			stepdown_evals::<PackedBinaryField128x1b>(10, 555),
			packed_slice::<PackedBinaryField128x1b>(&[(0..555, 1), (555..1024, 0)])
		);
		assert_eq!(
			stepdown_evals::<PackedBinaryField128x1b>(11, 0),
			packed_slice::<PackedBinaryField128x1b>(&[(0..2048, 0)])
		);
		assert_eq!(
			stepdown_evals::<PackedBinaryField128x1b>(11, 1),
			packed_slice::<PackedBinaryField128x1b>(&[(0..1, 1), (1..2048, 0)])
		);
		assert_eq!(
			stepdown_evals::<PackedBinaryField128x1b>(11, 2048),
			packed_slice::<PackedBinaryField128x1b>(&[(0..2048, 1)])
		);
	}

	#[test]
	fn test_step_down_trace_with_packing_256() {
		assert_eq!(
			stepdown_evals::<PackedBinaryField256x1b>(9, 314),
			packed_slice::<PackedBinaryField256x1b>(&[(0..314, 1), (314..512, 0)])
		);
		assert_eq!(
			stepdown_evals::<PackedBinaryField256x1b>(10, 555),
			packed_slice::<PackedBinaryField256x1b>(&[(0..555, 1), (555..1024, 0)])
		);
		assert_eq!(
			stepdown_evals::<PackedBinaryField256x1b>(11, 0),
			packed_slice::<PackedBinaryField256x1b>(&[(0..2048, 0)])
		);
		assert_eq!(
			stepdown_evals::<PackedBinaryField256x1b>(11, 1),
			packed_slice::<PackedBinaryField256x1b>(&[(0..1, 1), (1..2048, 0)])
		);
		assert_eq!(
			stepdown_evals::<PackedBinaryField256x1b>(11, 2048),
			packed_slice::<PackedBinaryField256x1b>(&[(0..2048, 1)])
		);
	}

	#[test]
	fn test_consistency_between_multilinear_extension_and_multilinear_poly_oracle() {
		for n_vars in 1..6 {
			for index in 0..=(1 << n_vars) {
				let step_down = StepDown::new(n_vars, index).unwrap();
				assert_eq!(
					hypercube_evals_from_oracle::<BinaryField1b>(&step_down),
					step_down
						.multilinear_extension::<BinaryField1b>()
						.unwrap()
						.evals()
				);
			}
		}
	}

	fn stepdown_evals<P>(n_vars: usize, index: usize) -> Vec<P>
	where
		P: PackedField<Scalar = BinaryField1b>,
	{
		StepDown::new(n_vars, index)
			.unwrap()
			.multilinear_extension::<P>()
			.unwrap()
			.evals()
			.to_vec()
	}
}
