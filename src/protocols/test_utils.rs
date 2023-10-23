// Copyright 2023 Ulvetanna Inc.

use std::sync::Arc;

use crate::{
	field::{ExtensionField, Field, PackedField},
	polynomial::{
		CompositionPoly, Error as PolynomialError, MultilinearComposite, MultilinearPoly,
	},
};

use super::evalcheck::evalcheck::EvalcheckClaim;

// A verifier has oracles to multilinears, but not the composite function
pub fn verify_evalcheck_claim<P: PackedField, F: ExtensionField<P::Scalar>>(
	evalcheck: EvalcheckClaim<F>,
	multilinears: &[MultilinearPoly<'static, P>],
) {
	assert_eq!(evalcheck.multilinear_composition.n_vars(), multilinears.len());
	// evaluate each multilinear on the eval point

	let evaluations = multilinears
		.iter()
		.map(|multilin| multilin.evaluate(&evalcheck.eval_point).unwrap())
		.collect::<Vec<_>>();

	let eval = evalcheck
		.multilinear_composition
		.evaluate(&evaluations)
		.unwrap();
	assert_eq!(eval, evalcheck.eval);
}

pub struct ProductMultivariate {
	arity: usize,
}

impl ProductMultivariate {
	pub fn new(arity: usize) -> Self {
		Self { arity }
	}
}

impl<F: Field> CompositionPoly<F, F> for ProductMultivariate {
	fn n_vars(&self) -> usize {
		self.arity
	}

	fn degree(&self) -> usize {
		self.arity
	}

	fn evaluate(&self, query: &[F]) -> Result<F, PolynomialError> {
		let n_vars = CompositionPoly::<F, F>::n_vars(self);
		assert_eq!(query.len(), n_vars);
		Ok(query.iter().product())
	}

	fn evaluate_ext(&self, query: &[F]) -> Result<F, PolynomialError> {
		let n_vars = CompositionPoly::<F, F>::n_vars(self);
		assert_eq!(query.len(), n_vars);
		Ok(query.iter().product())
	}
}

pub fn transform_poly<F, OF>(
	poly: &MultilinearComposite<F, F>,
	composition: Arc<dyn CompositionPoly<OF, OF>>,
) -> Result<MultilinearComposite<'static, OF, OF>, PolynomialError>
where
	F: Field,
	OF: Field + From<F> + Into<F>,
{
	let multilinears = poly
		.iter_multilinear_polys()
		.map(|multilin| {
			let values = multilin
				.evals()
				.iter()
				.cloned()
				.map(OF::from)
				.collect::<Vec<_>>();
			MultilinearPoly::from_values(values)
		})
		.collect::<Result<Vec<_>, _>>()?;
	let ret = MultilinearComposite::new(poly.n_vars(), composition, multilinears)?;
	Ok(ret)
}
