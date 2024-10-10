// Copyright 2024 Ulvetanna Inc.

use super::{
	error::Error,
	msetcheck::{
		reduce_msetcheck_claim, MsetcheckClaim, MsetcheckProof, MsetcheckProveOutput,
		MsetcheckWitness,
	},
};
use crate::{
	oracle::MultilinearOracleSet,
	polynomial::Error as PolynomialError,
	protocols::gkr_gpa::{
		construct_grand_product_claims, construct_grand_product_witnesses,
		get_grand_products_from_witnesses,
	},
	witness::{MultilinearExtensionIndex, MultilinearWitness},
};
use binius_field::{
	as_packed_field::{PackScalar, PackedType},
	underlier::{UnderlierType, WithUnderlier},
	PackedField, PackedFieldIndexable, TowerField,
};
use binius_utils::bail;
use rayon::prelude::*;
use std::sync::Arc;
use tracing::instrument;

/// Prove a multiset check instance reduction.
///
/// Given two $n$-arity tuples $(T_1, \ldots, T_n)$  and $(U_1, \ldots, U_n)$
/// of $\nu$-variate multilins, each representing an $n$-dimensional relation
/// of cardinality $2^{\nu}$ (when treated as a multiset), this protocol reduces
/// multiset equality predicate to a grand product check of the two polynomials of the
/// form:
///
/// 1) $T'(x) = \gamma + T_1(x) + \alpha * T_2(x) + \ldots + \alpha^{n-1} * T_n(x)$
/// 2) $U'(x) = \gamma + U_1(x) + \alpha * U_2(x) + \ldots + \alpha^{n-1} * U_n(x)$
///
/// where $\gamma$ and $\alpha$ are some large field challenges sampled via Fiat-Shamir
/// (`alpha` is non-`None` if $n \ge 2$).
#[instrument(skip_all, name = "msetcheck::prove", level = "debug")]
pub fn prove<'a, U, F>(
	oracles: &mut MultilinearOracleSet<F>,
	witness_index: MultilinearExtensionIndex<'a, U, F>,
	msetcheck_claim: &MsetcheckClaim<F>,
	msetcheck_witness: MsetcheckWitness<'a, PackedType<U, F>>,
	gamma: F,
	alpha: Option<F>,
) -> Result<MsetcheckProveOutput<'a, U, F>, Error>
where
	U: UnderlierType + PackScalar<F>,
	F: TowerField,
	PackedType<U, F>: PackedFieldIndexable,
{
	let gpa_claim_oracle_ids = reduce_msetcheck_claim(oracles, msetcheck_claim, gamma, alpha)?;

	let dimensions = msetcheck_claim.dimensions();
	let n_vars = msetcheck_claim.n_vars();

	if msetcheck_witness.dimensions() != dimensions {
		bail!(Error::WitnessDimensionalityMismatch);
	}

	if msetcheck_witness.n_vars() != n_vars {
		bail!(Error::WitnessNumVariablesMismatch);
	}

	let packing_log_width = PackedType::<U, F>::LOG_WIDTH;

	let lincom_witness = |relation_witnesses: &[MultilinearWitness<'a, PackedType<U, F>>]| -> Result<Arc<[U]>, Error> {
		// TODO: preallocate values
		// Populate the accumulator vector with additive challenge term
        let mut underliers = vec![U::default(); 1 << (n_vars - packing_log_width)];
        let values = PackedType::<U, F>::unpack_scalars_mut(
            PackedType::<U, F>::from_underliers_ref_mut(underliers.as_mut_slice()));

        values.fill(gamma);

		let (first_witness, rest_witnesses) = relation_witnesses
			.split_first()
			.expect("dimensionality checked above");

		// first dimension of the relation is not weighted
		values.par_iter_mut().enumerate().try_for_each(
			|(i, values_i)| -> Result<_, PolynomialError> {
				*values_i += first_witness.evaluate_on_hypercube(i)?;
				Ok(())
			},
		)?;

		let alpha = alpha.expect("dimensionality checked on reduction");
		let mut coeff = alpha;

		// the rest are weighted - apply small field optimization where possible
		for rest_witness in rest_witnesses {
			values.par_iter_mut().enumerate().try_for_each(
				|(i, values_i)| -> Result<_, PolynomialError> {
					*values_i += rest_witness.evaluate_on_hypercube_and_scale(i, coeff)?;
					Ok(())
				},
			)?;

			coeff *= alpha;
		}

		Ok(underliers.into())
	};

	let [t_oracle_id, u_oracle_id] = gpa_claim_oracle_ids;

	let t_polynomial = lincom_witness(msetcheck_witness.t_polynomials())?;
	let u_polynomial = lincom_witness(msetcheck_witness.u_polynomials())?;

	let witness_index = witness_index
		.update_owned::<F, _>([(t_oracle_id, t_polynomial), (u_oracle_id, u_polynomial)])?;

	let reduced_gpa_witnesses: [_; 2] =
		construct_grand_product_witnesses(&gpa_claim_oracle_ids, &witness_index)?
			.try_into()
			.expect("The length must be 2");
	let grand_products: [_; 2] = get_grand_products_from_witnesses(&reduced_gpa_witnesses)
		.try_into()
		.expect("The length must be 2");

	let claims = construct_grand_product_claims(&gpa_claim_oracle_ids, oracles, &grand_products)?;

	let reduced_gpa_claims = claims.try_into().expect("The length must be 2");

	if grand_products[0] != grand_products[1] {
		bail!(Error::ProductsDiffer);
	}

	Ok(MsetcheckProveOutput {
		reduced_gpa_witnesses,
		reduced_gpa_claims,
		gpa_metas: gpa_claim_oracle_ids,
		msetcheck_proof: MsetcheckProof { grand_products },
		witness_index,
	})
}
