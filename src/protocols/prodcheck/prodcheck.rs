// Copyright 2024 Ulvetanna Inc.

use std::sync::Arc;

use super::error::Error;
use crate::{
	field::{Field, PackedField},
	iopoly::{
		CompositePolyOracle, MultilinearPolyOracle, MultivariatePolyOracle, Projected,
		ProjectionVariant,
	},
	polynomial::{CompositionPoly, Error as PolynomialError, MultilinearPoly},
	protocols::{
		evalcheck::evalcheck::{EvalcheckClaim, EvalcheckWitness},
		zerocheck::zerocheck::{ZerocheckClaim, ZerocheckWitness},
	},
};

#[derive(Debug)]
pub struct ReducedProductCheckClaims<F: Field> {
	pub t_prime_claim: ZerocheckClaim<F>,
	pub grand_product_poly_claim: EvalcheckClaim<F>,
}

#[derive(Debug)]
pub struct ReducedProductCheckWitnesses<'a, F: Field> {
	pub t_prime_witness: ZerocheckWitness<'a, F>,
	pub grand_product_poly_witness: EvalcheckWitness<
		F,
		dyn MultilinearPoly<F> + Send + Sync + 'a,
		Arc<dyn MultilinearPoly<F> + Send + Sync + 'a>,
	>,
}

#[derive(Debug)]
pub struct ProdcheckProveOutput<'a, F: Field> {
	pub reduced_product_check_claims: ReducedProductCheckClaims<F>,
	pub reduced_product_check_witnesses: ReducedProductCheckWitnesses<'a, F>,
}

#[derive(Debug, Clone)]
pub struct ProdcheckClaim<F: Field> {
	/// Oracle to the polynomial T
	pub t_oracle: MultilinearPolyOracle<F>,
	/// Oracle to the polynomial U
	pub u_oracle: MultilinearPolyOracle<F>,
	/// Number of variables in T and U
	pub n_vars: usize,
}

#[derive(Debug, Clone)]
pub struct ProdcheckWitness<F: Field> {
	pub t_polynomial: Arc<dyn MultilinearPoly<F> + Sync>,
	pub u_polynomial: Arc<dyn MultilinearPoly<F> + Sync>,
}

/// Composition for Simple Multiplication Gate: f(X, Y, Z) := X - Y*Z
///
/// Expects three variables, ordered as follows:
/// 1) Output (X)
/// 2) First Input (Y)
/// 3) Second Input (Z)
#[derive(Debug)]
pub struct SimpleMultGateComposition;

impl<P: PackedField> CompositionPoly<P> for SimpleMultGateComposition {
	fn n_vars(&self) -> usize {
		3
	}

	fn degree(&self) -> usize {
		2
	}

	fn evaluate(&self, query: &[P::Scalar]) -> Result<P::Scalar, PolynomialError> {
		self.evaluate_packed(query)
	}

	fn evaluate_packed(&self, query: &[P]) -> Result<P, PolynomialError> {
		if query.len() != 3 {
			return Err(PolynomialError::IncorrectQuerySize { expected: 3 });
		}

		Ok(query[0] - query[1] * query[2])
	}
}

pub fn reduce_prodcheck_claim<F: Field>(
	prodcheck_claim: &ProdcheckClaim<F>,
	grand_prod_oracle: MultilinearPolyOracle<F>,
) -> Result<ReducedProductCheckClaims<F>, Error> {
	let n_vars = prodcheck_claim.n_vars;
	let f_prime_oracle = grand_prod_oracle.clone();

	// Construct f' partially evaluated oracles
	// [f'](x, 0)
	let projected_zero_last =
		Projected::new(f_prime_oracle.clone(), vec![F::ZERO], ProjectionVariant::LastVars)?;
	let f_prime_x_zero_oracle = MultilinearPolyOracle::Projected(projected_zero_last);

	// [f'](x, 1)
	let projected_one_last =
		Projected::new(f_prime_oracle.clone(), vec![F::ONE], ProjectionVariant::LastVars)?;
	let f_prime_x_one_oracle = MultilinearPolyOracle::Projected(projected_one_last);

	// [f'](0, x)
	let projected_zero_first =
		Projected::new(f_prime_oracle.clone(), vec![F::ZERO], ProjectionVariant::FirstVars)?;
	let f_prime_zero_x_oracle = MultilinearPolyOracle::Projected(projected_zero_first);

	// [f'](1, x)
	let projected_one_first =
		Projected::new(f_prime_oracle.clone(), vec![F::ONE], ProjectionVariant::FirstVars)?;
	let f_prime_one_x_oracle = MultilinearPolyOracle::Projected(projected_one_first);

	// merge([T], [f'](x, 1))
	// Note: What the paper calls "merge" is called "interleave" in the code
	// merge is similar to interleave, but the new selector variables are introduced
	// as the highest indices rather than the lowest
	let out_oracle = MultilinearPolyOracle::Merged(
		Box::new(prodcheck_claim.t_oracle.clone()),
		Box::new(f_prime_x_one_oracle),
	);

	// merge([U], [f'](0, x))
	let in1_oracle = MultilinearPolyOracle::Merged(
		Box::new(prodcheck_claim.u_oracle.clone()),
		Box::new(f_prime_zero_x_oracle),
	);

	// merge([f'](x, 0), [f'](1, x))
	let in2_oracle = MultilinearPolyOracle::Merged(
		Box::new(f_prime_x_zero_oracle),
		Box::new(f_prime_one_x_oracle),
	);

	// Construct T' polynomial oracle
	let composite_poly = CompositePolyOracle::new(
		n_vars + 1,
		vec![out_oracle, in1_oracle, in2_oracle],
		Arc::new(SimpleMultGateComposition),
	)?;
	let t_prime_oracle = MultivariatePolyOracle::Composite(composite_poly);

	// Construct ReducedProductCheckClaims
	let t_prime_claim = ZerocheckClaim {
		poly: t_prime_oracle,
	};
	let mut grand_prod_eval_point = vec![F::ONE; n_vars + 1];
	grand_prod_eval_point[0] = F::ZERO;
	let grand_product_poly_claim = EvalcheckClaim {
		poly: grand_prod_oracle.into(),
		eval: F::ONE,
		eval_point: grand_prod_eval_point,
		is_random_point: false,
	};

	Ok(ReducedProductCheckClaims {
		t_prime_claim,
		grand_product_poly_claim,
	})
}
