// Copyright 2024 Ulvetanna Inc.

use super::prodcheck::{
	reduce_prodcheck_claim, ProdcheckClaim, ProdcheckProveOutput, ProdcheckWitness,
	ReducedProductCheckWitnesses, SimpleMultGateComposition,
};
use crate::{
	field::{Field, TowerField},
	oracle::{MultilinearOracleSet, MultilinearPolyOracle},
	polynomial::{MultilinearComposite, MultilinearExtension, MultilinearPoly},
	protocols::{evalcheck::EvalcheckWitness, prodcheck::error::Error},
};
use std::{borrow::Borrow, sync::Arc};
use tracing::instrument;

/// Returns merge(x, y) where x, y are multilinear polynomials
fn construct_merge_polynomial<F, MX, MY>(
	x: MX,
	y: MY,
) -> Result<impl MultilinearPoly<F> + Send + Sync, Error>
where
	F: Field,
	MX: MultilinearPoly<F>,
	MY: MultilinearPoly<F>,
{
	let x = x.borrow();
	let y = y.borrow();

	let n_vars = x.n_vars();
	if y.n_vars() != n_vars {
		return Err(Error::ImproperInput(format!(
			"x and y must have same number of variables, but x has {} and y has {}",
			x.n_vars(),
			y.n_vars()
		)));
	}

	// TODO: Find a way to avoid these copies
	let mut values = vec![F::ZERO; 1 << (n_vars + 1)];
	let (x_values, y_values) = values.split_at_mut(1 << n_vars);
	x.subcube_evals(x.n_vars(), 0, x_values)?;
	y.subcube_evals(y.n_vars(), 0, y_values)?;
	let merge_poly = MultilinearExtension::from_values(values)?;
	Ok(merge_poly.specialize())
}

/// Prove a prodcheck instance reduction, step one of two.
///
/// Given as input $\nu$-variate multilins $T, U$, we define $f := T/U$
/// We output (\nu+1)-variate f' such that for all $v \in \{0, 1\}^{\nu}$
/// 1) $f'(v, 0) = f(v)$
/// 2) $f'(v, 1) = f'(0, v) * f'(1, v)$
#[instrument(skip_all, name = "prodcheck::prove_step_one")]
pub fn prove_step_one<F: Field>(
	prodcheck_witness: ProdcheckWitness<F>,
) -> Result<MultilinearExtension<'static, F>, Error> {
	// TODO: This is inefficient because it cannot construct the new witness polynomial in the
	// subfield.

	let ProdcheckWitness {
		t_polynomial,
		u_polynomial,
	} = prodcheck_witness;

	if t_polynomial.n_vars() != u_polynomial.n_vars() {
		return Err(Error::NumVariablesMismatch);
	}
	let n_vars = t_polynomial.n_vars();

	// Step 1: Prover constructs f' polynomial, and sends oracle to verifier
	let n_values = 1 << (n_vars + 1);
	// TODO: Preallocate values and parallelize initialization
	let mut values = vec![F::ZERO; n_values];

	// for each v in B_{n_vars}, set values[v] = f(v) := T(v)/U(v)
	for (i, values_i) in values[..(1 << n_vars)].iter_mut().enumerate() {
		let t_i = t_polynomial.evaluate_on_hypercube(i)?;
		let u_i = u_polynomial.evaluate_on_hypercube(i)?;
		*values_i = u_i.invert().map(|u_i_inv| t_i * u_i_inv).unwrap_or(F::ONE);
	}

	// for each v in B_{n_vars}, set values[2^n_vars + v] = values[2v] * values[2v+1]
	for i in 0..(1 << n_vars) {
		values[(1 << n_vars) | i] = values[i << 1] * values[i << 1 | 1];
	}
	let f_prime_poly = MultilinearExtension::from_values(values)?;
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
#[instrument(skip_all, name = "prodcheck::prove_step_two")]
pub fn prove_step_two<'a, F: TowerField>(
	oracles: &mut MultilinearOracleSet<F>,
	prodcheck_witness: ProdcheckWitness<F>,
	prodcheck_claim: &ProdcheckClaim<F>,
	f_prime_oracle: MultilinearPolyOracle<F>,
	f_prime_poly: &'a MultilinearExtension<'a, F>,
) -> Result<ProdcheckProveOutput<'a, F>, Error> {
	let n_vars = prodcheck_witness.t_polynomial.n_vars();
	if n_vars != prodcheck_witness.u_polynomial.n_vars() {
		return Err(Error::NumVariablesMismatch);
	}
	let t_poly = prodcheck_witness.t_polynomial;
	let u_poly = prodcheck_witness.u_polynomial;

	// Construct the claims
	let reduced_product_check_claims =
		reduce_prodcheck_claim(oracles, prodcheck_claim, f_prime_oracle.clone())?;

	// Construct the witnesses
	let values = f_prime_poly.evals();
	let n_values = values.len();
	if n_values != (1 << (t_poly.n_vars() + 1)) {
		return Err(Error::NumVariablesMismatch);
	}

	let first_values = &values[0..n_values / 2];
	let f_prime_x_zero = MultilinearExtension::from_values_slice(first_values)?;
	let second_values = &values[n_values / 2..];
	let f_prime_x_one = MultilinearExtension::from_values_slice(second_values)?;
	let even_values = values.iter().copied().step_by(2).collect::<Vec<_>>();
	let f_prime_zero_x = MultilinearExtension::from_values(even_values)?;
	let odd_values = values
		.iter()
		.copied()
		.skip(1)
		.step_by(2)
		.collect::<Vec<_>>();
	let f_prime_one_x = MultilinearExtension::from_values(odd_values)?;

	// Construct merge primed polynomials
	let out_poly = construct_merge_polynomial(t_poly, f_prime_x_one.specialize())?;
	let in1_poly = construct_merge_polynomial(u_poly, f_prime_zero_x.specialize())?;
	let in2_poly =
		construct_merge_polynomial(f_prime_x_zero.specialize(), f_prime_one_x.specialize())?;

	// Construct T' polynomial
	let t_prime_multilinears: Vec<Arc<dyn MultilinearPoly<F> + Send + Sync>> =
		vec![Arc::new(out_poly), Arc::new(in1_poly), Arc::new(in2_poly)];

	let t_prime_witness =
		MultilinearComposite::new(n_vars + 1, SimpleMultGateComposition, t_prime_multilinears)?;

	// Package return values
	let grand_product_poly_witness = EvalcheckWitness::new(vec![(
		f_prime_oracle.id(),
		f_prime_poly.to_ref().specialize_arc_dyn(),
	)]);

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
		field::BinaryField32b,
		oracle::{CommittedBatchSpec, CommittedId},
		protocols::prodcheck::verify::verify,
	};

	// Creates T(x), a multilinear with evaluations {1, 2, 3, 4} over the boolean hypercube on 2 vars
	fn create_numerator() -> MultilinearExtension<'static, BinaryField32b> {
		type F = BinaryField32b;
		let n_vars = 2;
		let values: Vec<F> = (0..1 << n_vars).map(|i| F::new(i + 1)).collect::<Vec<_>>();

		MultilinearExtension::from_values(values).unwrap()
	}

	// Creates U(x), a multilinear with evaluations {3, 2, 4, 1} over the boolean hypercube on 2 vars
	fn create_denominator() -> MultilinearExtension<'static, BinaryField32b> {
		type F = BinaryField32b;
		let n_vars = 2;
		let values = vec![F::new(3), F::new(2), F::new(4), F::new(1)];
		assert_eq!(values.len(), 1 << n_vars);

		MultilinearExtension::from_values(values).unwrap()
	}

	#[test]
	fn test_prove_verify_interaction() {
		type F = BinaryField32b;
		let n_vars = 2;

		// Setup witness
		let numerator = create_numerator();
		let denominator = create_denominator();
		let prodcheck_witness = ProdcheckWitness::<F> {
			t_polynomial: numerator.specialize_arc_dyn(),
			u_polynomial: denominator.specialize_arc_dyn(),
		};

		// Setup claim
		let mut oracles = MultilinearOracleSet::new();
		let round_1_batch_id = oracles.add_committed_batch(CommittedBatchSpec {
			round_id: 1,
			n_vars,
			n_polys: 2,
			tower_level: F::TOWER_LEVEL,
		});

		let numerator_oracle = oracles.committed_oracle(CommittedId {
			batch_id: round_1_batch_id,
			index: 0,
		});
		let denominator_oracle = oracles.committed_oracle(CommittedId {
			batch_id: round_1_batch_id,
			index: 1,
		});
		let prodcheck_claim = ProdcheckClaim {
			t_oracle: numerator_oracle,
			u_oracle: denominator_oracle,
			n_vars,
		};

		let round_2_batch_id = oracles.add_committed_batch(CommittedBatchSpec {
			round_id: 2,
			n_vars: n_vars + 1,
			n_polys: 1,
			tower_level: F::TOWER_LEVEL,
		});

		let f_prime_oracle = oracles.committed_oracle(CommittedId {
			batch_id: round_2_batch_id,
			index: 0,
		});

		// PROVER
		let f_prime_poly = prove_step_one(prodcheck_witness.clone()).unwrap();
		assert_eq!(f_prime_poly.evals()[(1 << (n_vars + 1)) - 2], F::ONE);
		assert_eq!(f_prime_poly.evals()[(1 << (n_vars + 1)) - 1], F::ZERO);

		let prove_output = prove_step_two(
			&mut oracles.clone(),
			prodcheck_witness,
			&prodcheck_claim,
			f_prime_oracle.clone(),
			&f_prime_poly,
		)
		.unwrap();
		let reduced_claims = prove_output.reduced_product_check_claims;

		// VERIFIER
		let verified_reduced_claims =
			verify(&mut oracles.clone(), &prodcheck_claim, f_prime_oracle).unwrap();

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
