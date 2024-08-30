// Copyright 2024 Ulvetanna Inc.

use super::{
	error::Error,
	msetcheck::{
		reduce_msetcheck_claim, MsetcheckClaim, MsetcheckProof, MsetcheckProveOutput,
		MsetcheckWitness,
	},
};
use crate::{
	oracle::{MultilinearOracleSet, OracleId},
	polynomial::Error as PolynomialError,
	protocols::gkr_gpa::{GrandProductClaim, GrandProductWitness},
	witness::{MultilinearExtensionIndex, MultilinearWitness},
};
use binius_field::{
	as_packed_field::{PackScalar, PackedType},
	underlier::{UnderlierType, WithUnderlier},
	Field, PackedField, PackedFieldIndexable, TowerField,
};
use binius_utils::bail;
use itertools::multiunzip;
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
pub fn prove<'a, U, F, FW>(
	oracles: &mut MultilinearOracleSet<F>,
	witness_index: MultilinearExtensionIndex<'a, U, FW>,
	msetcheck_claim: &MsetcheckClaim<F>,
	msetcheck_witness: MsetcheckWitness<'a, PackedType<U, FW>>,
	gamma: F,
	alpha: Option<F>,
) -> Result<MsetcheckProveOutput<'a, U, F, FW>, Error>
where
	U: UnderlierType + PackScalar<FW>,
	F: TowerField + From<FW>,
	FW: TowerField + From<F>,
	PackedType<U, FW>: PackedFieldIndexable,
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

	let packing_log_width = PackedType::<U, FW>::LOG_WIDTH;

	let lincom_witness = |relation_witnesses: &[MultilinearWitness<'a, PackedType<U, FW>>]| -> Result<Arc<[U]>, Error> {
		// TODO: preallocate values
		// Populate the accumulator vector with additive challenge term
        let mut underliers = vec![U::default(); 1 << (n_vars - packing_log_width)];
        let values = PackedType::<U, FW>::unpack_scalars_mut(
            PackedType::<U, FW>::from_underliers_ref_mut(underliers.as_mut_slice()));

        values.fill(FW::from(gamma));

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

		let fw_alpha = FW::from(alpha.expect("dimensionality checked on reduction"));
		let mut fw_coeff = fw_alpha;

		// the rest are weighted - apply small field optimization where possible
		for rest_witness in rest_witnesses {
			values.par_iter_mut().enumerate().try_for_each(
				|(i, values_i)| -> Result<_, PolynomialError> {
					*values_i += rest_witness.evaluate_on_hypercube_and_scale(i, fw_coeff)?;
					Ok(())
				},
			)?;

			fw_coeff *= fw_alpha;
		}

		Ok(underliers.into())
	};

	let [t_oracle_id, u_oracle_id] = gpa_claim_oracle_ids;

	let t_polynomial = lincom_witness(msetcheck_witness.t_polynomials())?;
	let u_polynomial = lincom_witness(msetcheck_witness.u_polynomials())?;

	let witness_index = witness_index
		.update_owned::<FW, _>([(t_oracle_id, t_polynomial), (u_oracle_id, u_polynomial)])?;

	let (grand_products, reduced_gpa_witnesses, reduced_gpa_claims): (Vec<_>, Vec<_>, Vec<_>) =
		multiunzip(gpa_claim_oracle_ids.map(|id| {
			grand_product_witness_claim(id, &witness_index, oracles)
				.expect("all elements have been added above")
		}));

	let [t_product, u_product] = grand_products[0..2]
		.try_into()
		.expect("The length must be 2");

	if t_product != u_product {
		bail!(Error::ProductsDiffer);
	}

	let reduced_gpa_witnesses = reduced_gpa_witnesses
		.try_into()
		.expect("The length must be 2 because the prodcheck_claims have a length of 2.");

	let reduced_gpa_claims = reduced_gpa_claims
		.try_into()
		.expect("The length must be 2 because the prodcheck_claims have a length of 2.");

	let grand_products = grand_products.try_into().expect("The length must be 2");

	Ok(MsetcheckProveOutput {
		reduced_gpa_witnesses,
		reduced_gpa_claims,
		msetcheck_proof: MsetcheckProof { grand_products },
		witness_index,
	})
}

type GrandProductWitnessClaimResult<'a, U, FW, F> =
	Result<(F, GrandProductWitness<'a, PackedType<U, FW>>, GrandProductClaim<F>), Error>;

fn grand_product_witness_claim<
	'a,
	U: UnderlierType + PackScalar<FW>,
	F: TowerField + From<FW>,
	FW: Field,
>(
	id: OracleId,
	witness_index: &MultilinearExtensionIndex<'a, U, FW>,
	oracles: &MultilinearOracleSet<F>,
) -> GrandProductWitnessClaimResult<'a, U, FW, F> {
	let poly = witness_index.get_multilin_poly(id)?;

	let oracle = oracles.oracle(id);

	let gpa_witness = GrandProductWitness::new(poly)?;

	let product = gpa_witness.grand_product_evaluation().into();
	let gpa_claim = GrandProductClaim {
		poly: oracle,
		product,
	};

	Ok((product, gpa_witness, gpa_claim))
}
