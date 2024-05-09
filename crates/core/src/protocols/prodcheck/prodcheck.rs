// Copyright 2024 Ulvetanna Inc.

use super::error::Error;
use crate::{
	oracle::{CompositePolyOracle, MultilinearOracleSet, MultilinearPolyOracle, ProjectionVariant},
	polynomial::{CompositionPoly, Error as PolynomialError},
	protocols::{
		evalcheck::EvalcheckClaim,
		zerocheck::{ZerocheckClaim, ZerocheckWitness},
	},
	witness::MultilinearWitness,
};
use binius_field::{Field, PackedField, TowerField};

#[derive(Debug)]
pub struct ReducedProductCheckClaims<F: Field> {
	pub t_prime_claim: ZerocheckClaim<F>,
	pub grand_product_poly_claim: EvalcheckClaim<F>,
}

#[derive(Debug)]
pub struct ProdcheckProveOutput<'a, F: Field, FW: Field> {
	pub reduced_product_check_claims: ReducedProductCheckClaims<F>,
	pub t_prime_witness: ZerocheckWitness<'a, FW, SimpleMultGateComposition>,
}

#[derive(Debug, Clone)]
pub struct ProdcheckClaim<F: Field> {
	/// Oracle to the polynomial T
	pub t_oracle: MultilinearPolyOracle<F>,
	/// Oracle to the polynomial U
	pub u_oracle: MultilinearPolyOracle<F>,
}

impl<F: Field> ProdcheckClaim<F> {
	pub fn n_vars(&self) -> Option<usize> {
		if self.t_oracle.n_vars() != self.u_oracle.n_vars() {
			return None;
		}

		Some(self.t_oracle.n_vars())
	}
}

#[derive(Debug, Clone)]
pub struct ProdcheckWitness<'a, FW: Field> {
	pub t_polynomial: MultilinearWitness<'a, FW>,
	pub u_polynomial: MultilinearWitness<'a, FW>,
}

/// Composition for Simple Multiplication Gate: f(X, Y, Z) := X - Y*Z
///
/// Expects three variables, ordered as follows:
/// 1) Output (X)
/// 2) First Input (Y)
/// 3) Second Input (Z)
#[derive(Clone, Debug)]
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

	fn binary_tower_level(&self) -> usize {
		0
	}
}

pub fn reduce_prodcheck_claim<F: TowerField>(
	oracles: &mut MultilinearOracleSet<F>,
	prodcheck_claim: &ProdcheckClaim<F>,
	grand_prod_oracle: MultilinearPolyOracle<F>,
) -> Result<ReducedProductCheckClaims<F>, Error> {
	let n_vars = prodcheck_claim
		.n_vars()
		.ok_or(Error::NumeratorDenominatorSizeMismatch)?;
	let f_prime_oracle = grand_prod_oracle.clone();

	// Construct f' partially evaluated oracles
	// [f'](x, 0)
	let f_prime_x_zero_oracle_id =
		oracles.add_projected(f_prime_oracle.id(), vec![F::ZERO], ProjectionVariant::LastVars)?;
	let f_prime_x_zero_oracle = oracles.oracle(f_prime_x_zero_oracle_id);

	// [f'](x, 1)
	let f_prime_x_one_oracle_id =
		oracles.add_projected(f_prime_oracle.id(), vec![F::ONE], ProjectionVariant::LastVars)?;
	let f_prime_x_one_oracle = oracles.oracle(f_prime_x_one_oracle_id);

	// [f'](0, x)
	let f_prime_zero_x_oracle_id =
		oracles.add_projected(f_prime_oracle.id(), vec![F::ZERO], ProjectionVariant::FirstVars)?;
	let f_prime_zero_x_oracle = oracles.oracle(f_prime_zero_x_oracle_id);

	// [f'](1, x)
	let f_prime_one_x_oracle =
		oracles.add_projected(f_prime_oracle.id(), vec![F::ONE], ProjectionVariant::FirstVars)?;
	let f_prime_one_x_oracle = oracles.oracle(f_prime_one_x_oracle);

	// merge([T], [f'](x, 1))
	// Note: What the paper calls "merge" is called "interleave" in the code
	// merge is similar to interleave, but the new selector variables are introduced
	// as the highest indices rather than the lowest
	let out_oracle =
		oracles.add_merged(prodcheck_claim.t_oracle.id(), f_prime_x_one_oracle.id())?;

	// merge([U], [f'](0, x))
	let in1_oracle =
		oracles.add_merged(prodcheck_claim.u_oracle.id(), f_prime_zero_x_oracle.id())?;

	// merge([f'](x, 0), [f'](1, x))
	let in2_oracle = oracles.add_merged(f_prime_x_zero_oracle.id(), f_prime_one_x_oracle.id())?;

	// Construct T' polynomial oracle
	let t_prime_oracle = CompositePolyOracle::new(
		n_vars + 1,
		vec![
			oracles.oracle(out_oracle),
			oracles.oracle(in1_oracle),
			oracles.oracle(in2_oracle),
		],
		SimpleMultGateComposition,
	)?;

	// Construct ReducedProductCheckClaims
	let t_prime_claim = ZerocheckClaim {
		poly: t_prime_oracle,
	};
	let mut grand_prod_eval_point = vec![F::ONE; n_vars + 1];
	grand_prod_eval_point[0] = F::ZERO;
	let grand_product_poly_claim = EvalcheckClaim {
		poly: grand_prod_oracle.into_composite(),
		eval: F::ONE,
		eval_point: grand_prod_eval_point,
		is_random_point: false,
	};

	Ok(ReducedProductCheckClaims {
		t_prime_claim,
		grand_product_poly_claim,
	})
}
