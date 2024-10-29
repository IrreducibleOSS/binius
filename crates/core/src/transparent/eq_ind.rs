// Copyright 2024 Irreducible Inc.

use crate::polynomial::{Error, MultivariatePoly};
use binius_field::{Field, PackedField, TowerField};
use binius_hal::ComputationBackend;
use binius_math::MultilinearExtension;
use binius_utils::bail;

/// Represents the MLE of the eq(X, Y) polynomial on 2*n_vars variables partially evaluated at Y = r
///
/// Recall that the multilinear polynomial eq(X, Y) is defined s.t. $\forall x, y \in \{0, 1\}^{\mu}$,
/// eq(x, y) = 1 iff x = y and eq(x, y) = 0 otherwise.
/// Specifically, the function is defined like so $\prod_{i=0}^{\mu - 1} (X_i * Y_i + (1 - X_i)(1-Y_i))$
#[derive(Debug)]
pub struct EqIndPartialEval<F: Field> {
	n_vars: usize,
	r: Vec<F>,
}

impl<F: Field> EqIndPartialEval<F> {
	// TODO: n_vars param here is unnecessary
	pub fn new(n_vars: usize, r: Vec<F>) -> Result<Self, Error> {
		if r.len() != n_vars {
			bail!(Error::IncorrectQuerySize { expected: n_vars });
		}
		Ok(Self { n_vars, r })
	}

	pub fn multilinear_extension<P: PackedField<Scalar = F>, Backend: ComputationBackend>(
		&self,
		backend: &Backend,
	) -> Result<MultilinearExtension<P, Backend::Vec<P>>, Error> {
		let multilin_query = backend.tensor_product_full_query(&self.r)?;
		Ok(MultilinearExtension::from_values_generic(multilin_query)?)
	}
}

impl<F: TowerField, P: PackedField<Scalar = F>> MultivariatePoly<P> for EqIndPartialEval<F> {
	fn n_vars(&self) -> usize {
		self.n_vars
	}

	fn degree(&self) -> usize {
		self.n_vars
	}

	fn evaluate(&self, query: &[P]) -> Result<P, Error> {
		let n_vars = MultivariatePoly::<P>::n_vars(self);
		if query.len() != n_vars {
			bail!(Error::IncorrectQuerySize { expected: n_vars });
		}

		let mut result = P::one();
		for (&q_i, &r_i) in query.iter().zip(self.r.iter()) {
			let term_one = q_i * r_i;
			let term_two = (P::one() - q_i) * (P::one() - r_i);
			let factor = term_one + term_two;
			result *= factor;
		}
		Ok(result)
	}

	fn binary_tower_level(&self) -> usize {
		F::TOWER_LEVEL
	}
}

#[cfg(test)]
mod tests {
	use super::EqIndPartialEval;
	use crate::polynomial::MultivariatePoly;
	use binius_field::{BinaryField32b, PackedBinaryField4x32b, PackedField};
	use binius_hal::{make_portable_backend, ComputationBackendExt};
	use rand::{rngs::StdRng, SeedableRng};
	use std::iter::repeat_with;

	fn test_eq_consistency_help(n_vars: usize) {
		type F = BinaryField32b;
		type P = PackedBinaryField4x32b;

		let mut rng = StdRng::seed_from_u64(0);
		let r = repeat_with(|| F::random(&mut rng))
			.take(n_vars)
			.collect::<Vec<_>>();
		let eval_point = &repeat_with(|| F::random(&mut rng))
			.take(n_vars)
			.collect::<Vec<_>>();
		let backend = make_portable_backend();

		// Get Multivariate Poly version of eq_r
		let eq_r_mvp = EqIndPartialEval::new(n_vars, r).unwrap();
		let eval_mvp = eq_r_mvp.evaluate(eval_point).unwrap();

		// Get MultilinearExtension version of eq_r
		let eq_r_mle = eq_r_mvp.multilinear_extension::<P, _>(&backend).unwrap();
		let multilin_query = backend.multilinear_query::<P>(eval_point).unwrap();
		let eval_mle = eq_r_mle.evaluate(&multilin_query).unwrap();

		// Assert equality
		assert_eq!(eval_mle, eval_mvp);
	}

	#[test]
	fn test_eq_consistency_schwartz_zippel() {
		for n_vars in 2..=10 {
			test_eq_consistency_help(n_vars);
		}
	}
}
