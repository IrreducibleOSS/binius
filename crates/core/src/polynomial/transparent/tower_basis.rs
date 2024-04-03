// Copyright 2024 Ulvetanna Inc.

use crate::polynomial::{Error, MultilinearExtension, MultivariatePoly};
use binius_field::{PackedField, TowerField};
use std::marker::PhantomData;

/// Represents the $\mathcal{T}_{\iota}$-basis of $\mathcal{T}_{\iota+k}$
///
/// Recall that $\mathcal{T}_{\iota}$ is defined as
/// * Let \mathbb{F} := \mathbb{F}_2[X_0, \ldots, X_{\iota-1}]
/// * Let \mathcal{J} := (X_0^2 + X_0 + 1, \ldots, X_{\iota-1}^2 + X_{\iota-1}X_{\iota-2} + 1)
/// * $\mathcal{T}_{\iota} := \mathbb{F} / J $
/// and $\mathcal{T}_{\iota}$ has the following $\mathbb{F}_2$-basis:
/// * $1, X_0, X_1, X_0X_1, X_2, \ldots, X_0 X_1 \ldots X_{\iota-1}$
///
/// Thus, $\mathcal{T}_{\iota+k}$ has a $\mathcal{T}_{\iota}$-basis of size $2^k$:
/// * $1, X_{\iota}, X_{\iota+1}, X_{\iota}X_{\iota+1}, X_{\iota+2}, \ldots, X_{\iota} X_{\iota+1} \ldots X_{\iota+k-1}$
#[derive(Debug, Copy, Clone)]
pub struct TowerBasis<F: TowerField> {
	k: usize,
	iota: usize,
	_marker: PhantomData<F>,
}

impl<F: TowerField> TowerBasis<F> {
	pub fn new(k: usize, iota: usize) -> Result<Self, Error> {
		if iota + k > F::TOWER_LEVEL {
			return Err(Error::ArgumentRangeError {
				arg: "iota + k".into(),
				range: 0..F::TOWER_LEVEL + 1,
			});
		}
		Ok(Self {
			k,
			iota,
			_marker: Default::default(),
		})
	}

	pub fn multilinear_extension<P: PackedField<Scalar = F>>(
		&self,
	) -> Result<MultilinearExtension<P>, Error> {
		let n_values = (1 << self.k) / P::WIDTH;
		let values = (0..n_values)
			.map(|i| {
				let mut packed_value = P::default();
				for j in 0..P::WIDTH {
					let basis_idx = i * P::WIDTH + j;
					let value = TowerField::basis(self.iota, basis_idx)?;
					packed_value.set(j, value);
				}
				Ok(packed_value)
			})
			.collect::<Result<Vec<_>, Error>>()?;

		MultilinearExtension::from_values(values)
	}

	/*
	pub fn multilinear_poly_oracle(&self) -> MultilinearPolyOracle<F> {
		MultilinearPolyOracle::Transparent(TransparentPolyOracle::new(
			Arc::new(*self),
			F::TOWER_LEVEL,
		))
	}
	 */
}

impl<F> MultivariatePoly<F> for TowerBasis<F>
where
	F: TowerField,
{
	fn n_vars(&self) -> usize {
		self.k
	}

	fn degree(&self) -> usize {
		self.k
	}

	fn evaluate(&self, query: &[F]) -> Result<F, Error> {
		if query.len() != self.k {
			return Err(Error::IncorrectQuerySize { expected: self.k });
		}

		let mut result = F::ONE;
		for (i, query_i) in query.iter().enumerate() {
			let r_comp = F::ONE - query_i;
			let basis_elt = <F as TowerField>::basis(self.iota + i, 1)?;
			result *= r_comp + *query_i * basis_elt;
		}
		Ok(result)
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::polynomial::multilinear_query::MultilinearQuery;
	use binius_field::{BinaryField128b, BinaryField32b, PackedBinaryField4x32b};
	use rand::{rngs::StdRng, SeedableRng};
	use std::iter::repeat_with;

	fn test_consistency(iota: usize, k: usize) {
		type F = BinaryField128b;
		let mut rng = StdRng::seed_from_u64(0);

		let basis = TowerBasis::<F>::new(k, iota).unwrap();
		let challenge = repeat_with(|| F::random(&mut rng))
			.take(k)
			.collect::<Vec<_>>();

		let eval1 = basis.evaluate(&challenge).unwrap();
		let multilin_query = MultilinearQuery::<F>::with_full_query(&challenge).unwrap();
		let mle = basis.multilinear_extension::<F>().unwrap();
		let eval2 = mle.evaluate(&multilin_query).unwrap();

		assert_eq!(eval1, eval2);
	}

	#[test]
	fn test_consistency_packing() {
		let iota = 2;
		let kappa = 3;
		type F = BinaryField32b;
		type P = PackedBinaryField4x32b;
		let mut rng = StdRng::seed_from_u64(0);

		let basis = TowerBasis::<F>::new(kappa, iota).unwrap();
		let challenge = repeat_with(|| F::random(&mut rng))
			.take(kappa)
			.collect::<Vec<_>>();
		let eval1 = basis.evaluate(&challenge).unwrap();
		let multilin_query = MultilinearQuery::<F>::with_full_query(&challenge).unwrap();
		let mle = basis.multilinear_extension::<P>().unwrap();
		let eval2 = mle.evaluate(&multilin_query).unwrap();
		assert_eq!(eval1, eval2);
	}

	#[test]
	fn test_consistency_all() {
		for iota in 0..=7 {
			for k in 0..=(7 - iota) {
				test_consistency(iota, k);
			}
		}
	}
}
