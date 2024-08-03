// Copyright 2024 Ulvetanna Inc.

use super::{
	error::Error,
	prodcheck::{
		reduce_prodcheck_claim, ProdcheckClaim, ProdcheckProveOutput,
		ProdcheckReducedClaimOracleIds, ProdcheckWitness, SimpleMultGateComposition,
	},
};
use crate::{
	oracle::{MultilinearOracleSet, OracleId},
	polynomial::{Error as PolynomialError, MultilinearComposite, MultilinearPoly},
	witness::MultilinearExtensionIndex,
};
use binius_field::{
	as_packed_field::{PackScalar, PackedType},
	underlier::{UnderlierType, WithUnderlier},
	Field, PackedField, PackedFieldIndexable, TowerField,
};
use rayon::prelude::*;
use tracing::instrument;

fn multilin_poly_to_underliers_vec<U: UnderlierType + PackScalar<F>, F: Field>(
	poly: impl MultilinearPoly<PackedType<U, F>>,
) -> Result<Vec<U>, Error> {
	debug_assert!(poly.n_vars() >= PackedType::<U, F>::LOG_WIDTH);
	let mut underliers = vec![U::default(); 1 << (poly.n_vars() - PackedType::<U, F>::LOG_WIDTH)];
	let packed = PackedType::<U, F>::from_underliers_ref_mut(underliers.as_mut_slice());
	poly.subcube_evals(poly.n_vars(), 0, packed)?;
	Ok(underliers)
}

fn underliers_unpack_scalars_mut<U: UnderlierType + PackScalar<F>, F: Field>(
	underliers: &mut [U],
) -> &mut [F]
where
	PackedType<U, F>: PackedFieldIndexable,
{
	PackedType::<U, F>::unpack_scalars_mut(PackedType::<U, F>::from_underliers_ref_mut(underliers))
}

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
#[instrument(skip_all, name = "prodcheck::prove")]
pub fn prove<'a, U, F>(
	oracles: &mut MultilinearOracleSet<F>,
	witness_index: MultilinearExtensionIndex<'a, U, F>,
	prodcheck_claim: &ProdcheckClaim<F>,
	prodcheck_witness: ProdcheckWitness<'a, PackedType<U, F>>,
	f_prime: OracleId,
) -> Result<ProdcheckProveOutput<'a, U, F>, Error>
where
	U: UnderlierType + PackScalar<F>,
	F: TowerField,
	PackedType<U, F>: PackedFieldIndexable,
{
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
	let packing_log_width = PackedType::<U, F>::LOG_WIDTH;
	if n_vars < packing_log_width {
		return Err(Error::WitnessSmallerThanUnderlier);
	}

	let f_prime_oracle = oracles.oracle(f_prime);
	if f_prime_oracle.n_vars() != n_vars + 1 {
		return Err(Error::NumGrandProductVariablesIncorrect);
	}

	// Step 1: Prover constructs f' polynomial, and sends oracle to verifier
	// TODO: Preallocate values
	let n_underliers = 1 << (n_vars + 1 - packing_log_width);
	let mut f_prime_underliers = vec![U::default(); n_underliers];
	let f_prime_scalars = underliers_unpack_scalars_mut(f_prime_underliers.as_mut_slice());
	let n_scalars = f_prime_scalars.len();
	assert_eq!(n_scalars, 1 << (n_vars + 1));

	// for each v in B_{n_vars}, set values[v] = f(v) := T(v)/U(v)
	f_prime_scalars[0..(1 << n_vars)]
		.par_iter_mut()
		.enumerate()
		.try_for_each(|(i, values_i)| -> Result<_, PolynomialError> {
			let t_i = t_polynomial.evaluate_on_hypercube(i)?;
			let u_i = u_polynomial.evaluate_on_hypercube(i)?;
			*values_i = u_i.invert().map(|u_i_inv| t_i * u_i_inv).unwrap_or(F::ONE);
			Ok(())
		})?;

	// for each v in B_{n_vars}, set values[2^n_vars + v] = values[2v] * values[2v+1]
	for i in 0..(1 << n_vars) {
		f_prime_scalars[(1 << n_vars) | i] = f_prime_scalars[i << 1] * f_prime_scalars[i << 1 | 1];
	}

	// Construct the claims
	let (reduced_product_check_claims, reduced_claim_oracle_ids) =
		reduce_prodcheck_claim(oracles, prodcheck_claim, f_prime_oracle.clone())?;

	let ProdcheckReducedClaimOracleIds {
		in1_oracle_id,
		in2_oracle_id,
		out_oracle_id,
		f_prime_x_zero_oracle_id,
		f_prime_x_one_oracle_id,
		f_prime_zero_x_oracle_id,
		f_prime_one_x_oracle_id,
	} = reduced_claim_oracle_ids;

	// Construct the even/odd witnesses
	// TODO: try to find right lifetimes to eliminate duplicate witnesses within index

	let mut even_underliers = vec![U::default(); n_underliers / 2];
	let even_scalars = underliers_unpack_scalars_mut(even_underliers.as_mut_slice());

	for (i, even_scalar) in even_scalars.iter_mut().enumerate() {
		*even_scalar = f_prime_scalars[i << 1];
	}

	let mut odd_underliers = vec![U::default(); n_underliers / 2];
	let odd_scalars = underliers_unpack_scalars_mut(odd_underliers.as_mut_slice());

	for (i, odd_scalar) in odd_scalars.iter_mut().enumerate() {
		*odd_scalar = f_prime_scalars[i << 1 | 1];
	}

	let first_half_underliers = f_prime_underliers[0..n_underliers / 2].to_vec();
	let second_half_underliers = f_prime_underliers[n_underliers / 2..].to_vec();

	// Construct merge primed polynomials
	let mut out_poly = multilin_poly_to_underliers_vec(t_polynomial)?;
	out_poly.extend(second_half_underliers.as_slice());

	let mut in1_poly = multilin_poly_to_underliers_vec(u_polynomial)?;
	in1_poly.extend(even_underliers.as_slice());

	let mut in2_poly = first_half_underliers.to_vec();
	in2_poly.extend(odd_underliers.as_slice());

	let witness_index = witness_index.update_owned::<F, _>([
		(f_prime_x_zero_oracle_id, first_half_underliers),
		(f_prime_x_one_oracle_id, second_half_underliers),
		(f_prime_zero_x_oracle_id, even_underliers),
		(f_prime_one_x_oracle_id, odd_underliers),
		(f_prime_oracle.id(), f_prime_underliers),
		(in1_oracle_id, in1_poly),
		(in2_oracle_id, in2_poly),
		(out_oracle_id, out_poly),
	])?;

	// Construct T' polynomial
	let t_prime_multilinears = vec![
		witness_index.get_multilin_poly(out_oracle_id)?,
		witness_index.get_multilin_poly(in1_oracle_id)?,
		witness_index.get_multilin_poly(in2_oracle_id)?,
	];

	let t_prime_witness =
		MultilinearComposite::new(n_vars + 1, SimpleMultGateComposition, t_prime_multilinears)?;

	// Package return values
	let prodcheck_proof = ProdcheckProveOutput {
		reduced_product_check_claims,
		t_prime_witness,
		witness_index,
	};

	Ok(prodcheck_proof)
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::{polynomial::MultilinearExtension, protocols::prodcheck::verify::verify};
	use binius_field::{BinaryField32b, PackedBinaryField4x32b};

	// Creates T(x), a multilinear with evaluations {1, 2, 3, 4} over the boolean hypercube on 2 vars
	fn create_numerator() -> MultilinearExtension<BinaryField32b> {
		type F = BinaryField32b;
		let n_vars = 2;
		let values: Vec<F> = (0..1 << n_vars).map(|i| F::new(i + 1)).collect::<Vec<_>>();

		MultilinearExtension::from_values(values).unwrap()
	}

	// Creates U(x), a multilinear with evaluations {3, 2, 4, 1} over the boolean hypercube on 2 vars
	fn create_denominator() -> MultilinearExtension<BinaryField32b> {
		type F = BinaryField32b;
		let n_vars = 2;
		let values = vec![F::new(3), F::new(2), F::new(4), F::new(1)];
		assert_eq!(values.len(), 1 << n_vars);

		MultilinearExtension::from_values(values).unwrap()
	}

	#[test]
	fn test_prove_verify_interaction() {
		type F = BinaryField32b;
		type P = PackedBinaryField4x32b;
		type U = <P as WithUnderlier>::Underlier;
		let n_vars = 2;

		// Setup witness
		let numerator = create_numerator();
		let denominator = create_denominator();
		let prodcheck_witness = ProdcheckWitness::<P> {
			t_polynomial: numerator.specialize_arc_dyn(),
			u_polynomial: denominator.specialize_arc_dyn(),
		};

		// Setup claim
		let mut oracles = MultilinearOracleSet::<F>::new();
		let round_1_batch_id = oracles.add_committed_batch(n_vars, F::TOWER_LEVEL);
		let numerator = oracles.add_committed(round_1_batch_id);
		let denominator = oracles.add_committed(round_1_batch_id);
		let prodcheck_claim = ProdcheckClaim {
			t_oracle: oracles.oracle(numerator),
			u_oracle: oracles.oracle(denominator),
		};

		let round_2_batch_id = oracles.add_committed_batch(n_vars + 1, F::TOWER_LEVEL);
		let f_prime_oracle_id = oracles.add_committed(round_2_batch_id);

		let f_prime_oracle = oracles.oracle(f_prime_oracle_id);

		// PROVER
		let witness_index = MultilinearExtensionIndex::<U, F>::new();

		let prove_output = prove(
			&mut oracles.clone(),
			witness_index,
			&prodcheck_claim,
			prodcheck_witness,
			f_prime_oracle_id,
		)
		.unwrap();
		let reduced_claims = prove_output.reduced_product_check_claims;

		let f_prime_poly = prove_output
			.witness_index
			.get_multilin_poly(f_prime_oracle.id())
			.unwrap();

		assert_eq!(
			f_prime_poly
				.evaluate_on_hypercube((1 << (n_vars + 1)) - 2)
				.unwrap(),
			F::ONE
		);
		assert_eq!(
			f_prime_poly
				.evaluate_on_hypercube((1 << (n_vars + 1)) - 1)
				.unwrap(),
			F::ZERO
		);

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
