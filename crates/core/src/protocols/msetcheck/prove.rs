// Copyright 2024 Ulvetanna Inc.

use super::{
	error::Error,
	msetcheck::{reduce_msetcheck_claim, MsetcheckClaim, MsetcheckProveOutput, MsetcheckWitness},
};
use crate::{
	oracle::MultilinearOracleSet,
	polynomial::{Error as PolynomialError, MultilinearExtension},
	protocols::prodcheck::ProdcheckWitness,
	witness::{MultilinearWitness, MultilinearWitnessIndex},
};
use binius_field::TowerField;
use rayon::prelude::*;
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
#[instrument(skip_all, name = "msetcheck::prove")]
pub fn prove<'a, F, FW>(
	oracles: &mut MultilinearOracleSet<F>,
	witness_index: &mut MultilinearWitnessIndex<'a, FW>,
	msetcheck_claim: &MsetcheckClaim<F>,
	msetcheck_witness: MsetcheckWitness<'a, FW>,
	gamma: F,
	alpha: Option<F>,
) -> Result<MsetcheckProveOutput<'a, F, FW>, Error>
where
	F: TowerField + From<FW>,
	FW: TowerField + From<F>,
{
	let prodcheck_claim = reduce_msetcheck_claim(oracles, msetcheck_claim, gamma, alpha)?;

	let dimensions = msetcheck_claim.dimensions();
	let n_vars = msetcheck_claim.n_vars();

	if msetcheck_witness.dimensions() != dimensions {
		return Err(Error::WitnessDimensionalityMismatch);
	}

	if msetcheck_witness.n_vars() != n_vars {
		return Err(Error::WitnessNumVariablesMismatch);
	}

	let lincom_witness = |relation_witnesses: &[MultilinearWitness<'a, FW>]| -> Result<_, Error> {
		// TODO: preallocate values
		// Populate the accumulator vector with additive challenge term
		let mut values = vec![FW::from(gamma); 1 << n_vars];

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

		Ok(MultilinearExtension::from_values(values)?.specialize_arc_dyn())
	};

	let t_polynomial = lincom_witness(msetcheck_witness.t_polynomials())?;
	let u_polynomial = lincom_witness(msetcheck_witness.u_polynomials())?;

	witness_index.set(prodcheck_claim.t_oracle.id(), t_polynomial.clone());
	witness_index.set(prodcheck_claim.u_oracle.id(), u_polynomial.clone());

	let prodcheck_witness = ProdcheckWitness {
		t_polynomial,
		u_polynomial,
	};

	Ok(MsetcheckProveOutput {
		prodcheck_claim,
		prodcheck_witness,
	})
}
