// Copyright 2024 Ulvetanna Inc.
use std::sync::Arc;

use crate::field::Field;

use crate::{
	iopoly::MultilinearPolyOracle,
	polynomial::{MultilinearComposite, MultilinearPoly},
	protocols::{
		evalcheck::evalcheck::EvalcheckWitness, prodcheck::error::Error,
		zerocheck::zerocheck::ZerocheckWitness,
	},
};

use super::prodcheck::{
	reduce_prodcheck_claim, ProdcheckClaim, ProdcheckProveOutput, ProdcheckWitness,
	ReducedProductCheckWitnesses, SimpleMultGateComposition,
};

/// Returns merge(x, y) where x, y are multilinear polynomials
fn construct_merge_polynomial<F: Field>(
	x: MultilinearPoly<F>,
	y: MultilinearPoly<F>,
) -> Result<MultilinearPoly<'static, F>, Error> {
	if x.n_vars() != y.n_vars() {
		return Err(Error::ImproperInput(format!(
			"x and y must have same number of variables, but x has {} and y has {}",
			x.n_vars(),
			y.n_vars()
		)));
	}
	let n_vars = x.n_vars() + 1;

	// TODO: Find a way to avoid these copies
	let mut values = Vec::with_capacity(1 << n_vars);
	values.extend(x.evals());
	values.extend(y.evals());
	let merge_poly = MultilinearPoly::from_values(values)?;
	Ok(merge_poly)
}

/// Prove a prodcheck instance reduction, step one of two.
///
/// Given as input $\nu$-variate multilins $T, U$, we define $f := T/U$
/// We output (\nu+1)-variate f' such that for all $v \in \{0, 1\}^{\nu}$
/// 1) $f'(v, 0) = f(v)$
/// 2) $f'(v, 1) = f'(0, v) * f'(1, v)$
pub fn prove_step_one<F: Field>(
	prodcheck_witness: ProdcheckWitness<'_, F>,
) -> Result<MultilinearPoly<'_, F>, Error> {
	if prodcheck_witness.t_polynomial.n_vars() != prodcheck_witness.u_polynomial.n_vars() {
		return Err(Error::NumVariablesMismatch);
	}
	let n_vars = prodcheck_witness.t_polynomial.n_vars();

	let t_evals = prodcheck_witness.t_polynomial.evals();
	let u_evals = prodcheck_witness.u_polynomial.evals();
	if t_evals.len() != u_evals.len() {
		return Err(Error::NumVariablesMismatch);
	}
	if t_evals.len() != (1 << n_vars) {
		return Err(Error::NumVariablesMismatch);
	}

	// Step 1: Prover constructs f' polynomial, and sends oracle to verifier
	let n_values = 1 << (n_vars + 1);
	// TODO: Preallocate values and parallelize initialization
	let mut values = vec![F::ZERO; n_values];

	// for each v in B_{n_vars}, set values[v] = f(v) := T(v)/U(v)
	t_evals
		.iter()
		.zip(u_evals.iter())
		.enumerate()
		.for_each(|(i, (t_i, u_i))| {
			values[i] = u_i.invert().map(|u_i_inv| *t_i * u_i_inv).unwrap_or(F::ONE)
		});

	// for each v in B_{n_vars}, set values[2^n_vars + v] = values[2v] * values[2v+1]
	for i in 0..(1 << n_vars) {
		values[(1 << n_vars) | i] = values[i << 1] * values[i << 1 | 1];
	}
	let f_prime_poly = MultilinearPoly::from_values(values)?;
	Ok(f_prime_poly)
}

/// Prove a prodcheck instance reduction, step two of two.
///
/// Given f' polynomial and its oracle, we construct multivariate polynomial
/// T'(x) := merge(T(x), f'(x, 1)) - merge(U(x), f'(0, x)) * merge(f'(x, 0), f'(1, x))
/// such that prodcheck reduces to a zerocheck instance on T'(x) as well as
/// a Hypercube Evalcheck instance on f'
///
/// # Notes
///
/// 1) What we refer to as merge in this codebase is different from the
/// paper's definition of merge. What the paper defines as merge is
/// what the codebase names as interleaved. We will use the
/// codebase version of the terminology in all documentation.
/// The difference is that new selector variables in merge are added
/// to higher-index variables as opposed to interleaved, where the
/// selectors are the lower-indexed variables.
///
/// Formally, merge in this codebase is defined as follows:
/// Let $\mu = 2^{\alpha}$ for some $\alpha \in \mathbb{N}$.
/// Let $t_0, \ldots, t_{\mu-1}$ be $\nu$-variate multilinear polynomials.
/// Then, $\textit{merge}(t_0, \ldots, t_{\mu-1})$ is
/// a $\nu+\alpha$-variate multilinear polynomial such that
/// $\forall v \in \{0, 1\}^{\nu}$ and $\forall u \in \{0, 1\}^{\alpha}$,
/// $(v ~ || ~ u) \rightarrow t_{\{u\}}(v)$
///
/// 2) This functionality deviates from the Succinct Arguments over
/// Towers of Binary Fields paper in that we use the merge virtual
/// polynomial instead of the interleave virtual polynomial. This is an
/// optimization, and does not affect the soundness of prodcheck.
pub fn prove_step_two<'a, F: Field>(
	prodcheck_witness: ProdcheckWitness<'a, F>,
	prodcheck_claim: &'a ProdcheckClaim<F>,
	f_prime_oracle: MultilinearPolyOracle<F>,
	f_prime_poly: MultilinearPoly<'a, F>,
) -> Result<ProdcheckProveOutput<'a, F>, Error> {
	let n_vars = prodcheck_witness.t_polynomial.n_vars();
	if n_vars != prodcheck_witness.u_polynomial.n_vars() {
		return Err(Error::NumVariablesMismatch);
	}
	let t_poly = prodcheck_witness.t_polynomial;
	let u_poly = prodcheck_witness.u_polynomial;

	// Construct the claims
	let reduced_product_check_claims =
		reduce_prodcheck_claim(prodcheck_claim, f_prime_oracle.clone())?;

	// Construct the witnesses
	let values = f_prime_poly.evals();
	let n_values = values.len();
	if n_values != (1 << (t_poly.n_vars() + 1)) {
		return Err(Error::NumVariablesMismatch);
	}

	let first_values = &values[0..n_values / 2];
	let f_prime_x_zero = MultilinearPoly::from_values_slice(first_values)?;
	let second_values = &values[n_values / 2..];
	let f_prime_x_one = MultilinearPoly::from_values_slice(second_values)?;
	let even_values = values.iter().copied().step_by(2).collect::<Vec<_>>();
	let f_prime_zero_x = MultilinearPoly::from_values(even_values)?;
	let odd_values = values
		.iter()
		.copied()
		.skip(1)
		.step_by(2)
		.collect::<Vec<_>>();
	let f_prime_one_x = MultilinearPoly::from_values(odd_values)?;

	// Construct merge primed polynomials
	let out_poly = construct_merge_polynomial(t_poly, f_prime_x_one)?;
	let in1_poly = construct_merge_polynomial(u_poly, f_prime_zero_x)?;
	let in2_poly = construct_merge_polynomial(f_prime_x_zero, f_prime_one_x)?;

	// Construct T' polynomial
	let t_prime_multilinears = vec![out_poly, in1_poly, in2_poly];

	let t_prime_poly = MultilinearComposite::new(
		n_vars + 1,
		Arc::new(SimpleMultGateComposition),
		t_prime_multilinears,
	)?;

	// Package return values
	let t_prime_witness = ZerocheckWitness {
		polynomial: t_prime_poly,
	};

	let grand_product_poly_witness = EvalcheckWitness {
		polynomial: f_prime_poly.into_multilinear_composite()?,
	};

	let reduced_product_check_witnesses = ReducedProductCheckWitnesses {
		t_prime_witness,
		grand_product_poly_witness,
	};

	let prodcheck_proof = ProdcheckProveOutput {
		reduced_product_check_witnesses,
		reduced_product_check_claims,
	};
	Ok(prodcheck_proof)
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::{
		field::{BinaryField, BinaryField32b},
		iopoly::MultilinearPolyOracle,
		polynomial::MultilinearPoly,
		protocols::prodcheck::verify::verify,
	};

	// Creates T(x), a multilinear with evaluations {1, 2, 3, 4} over the boolean hypercube on 2 vars
	fn create_numerator() -> MultilinearPoly<'static, BinaryField32b> {
		type F = BinaryField32b;
		let n_vars = 2;
		let values: Vec<F> = (0..1 << n_vars).map(|i| F::new(i + 1)).collect::<Vec<_>>();

		MultilinearPoly::from_values(values).unwrap()
	}

	fn create_numerator_oracle() -> MultilinearPolyOracle<BinaryField32b> {
		let n_vars = 2;
		MultilinearPolyOracle::Committed {
			id: 0,
			n_vars,
			tower_level: BinaryField32b::TOWER_LEVEL,
		}
	}

	// Creates U(x), a multilinear with evaluations {3, 2, 4, 1} over the boolean hypercube on 2 vars
	fn create_denominator() -> MultilinearPoly<'static, BinaryField32b> {
		type F = BinaryField32b;
		let n_vars = 2;
		let values = vec![F::new(3), F::new(2), F::new(4), F::new(1)];
		assert_eq!(values.len(), 1 << n_vars);

		MultilinearPoly::from_values(values).unwrap()
	}

	fn create_denominator_oracle() -> MultilinearPolyOracle<BinaryField32b> {
		let n_vars = 2;
		MultilinearPolyOracle::Committed {
			id: 1,
			n_vars,
			tower_level: BinaryField32b::TOWER_LEVEL,
		}
	}

	#[test]
	fn test_prove_verify_interaction() {
		type F = BinaryField32b;
		let n_vars = 2;

		// Setup witness
		let numerator = create_numerator();
		let denominator = create_denominator();
		let prodcheck_witness = ProdcheckWitness {
			t_polynomial: numerator,
			u_polynomial: denominator,
		};

		// Setup claim
		let numerator_oracle = create_numerator_oracle();
		let denominator_oracle = create_denominator_oracle();
		let prodcheck_claim = ProdcheckClaim {
			t_oracle: numerator_oracle,
			u_oracle: denominator_oracle,
			n_vars,
		};

		// PROVER
		let f_prime_poly = prove_step_one(prodcheck_witness.clone()).unwrap();
		let f_prime_oracle = MultilinearPolyOracle::Committed {
			id: 99,
			n_vars: n_vars + 1,
			tower_level: F::TOWER_LEVEL,
		};
		assert_eq!(f_prime_poly.evals()[(1 << (n_vars + 1)) - 2], F::ONE);
		assert_eq!(f_prime_poly.evals()[(1 << (n_vars + 1)) - 1], F::ZERO);

		let prove_output = prove_step_two(
			prodcheck_witness,
			&prodcheck_claim,
			f_prime_oracle.clone(),
			f_prime_poly,
		)
		.unwrap();
		let reduced_claims = prove_output.reduced_product_check_claims;

		// VERIFIER
		let verified_reduced_claims = verify(&prodcheck_claim, f_prime_oracle).unwrap();

		// Check consistency
		assert_eq!(reduced_claims.grand_product_poly_claim.eval, F::ONE);
		assert_eq!(verified_reduced_claims.grand_product_poly_claim.eval, F::ONE);
		assert!(
			!verified_reduced_claims
				.grand_product_poly_claim
				.is_random_point
		);
		let mut expected_eval_point = vec![F::ONE; n_vars + 1];
		expected_eval_point[0] = F::ZERO;
		assert_eq!(reduced_claims.grand_product_poly_claim.eval_point, expected_eval_point);
		assert_eq!(
			verified_reduced_claims.grand_product_poly_claim.eval_point,
			expected_eval_point
		);
		assert_eq!(reduced_claims.grand_product_poly_claim.poly.n_vars(), n_vars + 1);
		assert_eq!(
			verified_reduced_claims
				.grand_product_poly_claim
				.poly
				.n_vars(),
			n_vars + 1
		);

		assert_eq!(reduced_claims.t_prime_claim.poly.n_vars(), n_vars + 1);
		assert_eq!(verified_reduced_claims.t_prime_claim.poly.n_vars(), n_vars + 1);
	}
}
