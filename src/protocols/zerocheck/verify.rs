// Copyright 2023 Ulvetanna Inc.

use std::sync::Arc;

use crate::{
	field::Field,
	polynomial::{eq_ind_partial_eval, EvaluationDomain},
	protocols::{
		evalcheck::evalcheck::EvalcheckClaim,
		sumcheck::{self, SumcheckClaim},
	},
};
use p3_challenger::{CanObserve, CanSample};

use super::{
	error::Error,
	zerocheck::{CompositeMultilinearProductComposition, ZerocheckClaim, ZerocheckProof},
};

/// Verifies a zerocheck reduction proof.
///
/// Returns the evaluation point and the claimed evaluation.
pub fn verify<F, CH>(
	zerocheck_claim: ZerocheckClaim<F>,
	domain: &EvaluationDomain<F>,
	proof: &ZerocheckProof<F>,
	challenger: &mut CH,
) -> Result<EvalcheckClaim<F>, Error>
where
	F: Field,
	CH: CanSample<F> + CanObserve<F>,
{
	// Step 1: Sample a random vector r \in F^n_vars
	let r: Vec<F> = challenger.sample_vec(zerocheck_claim.n_vars);

	// Step 2: Construct a multilinear polynomial eq(X, Y) on 2*n_vars variables, partially evaluated at r
	let eq_r = eq_ind_partial_eval(zerocheck_claim.n_vars, &r)?;

	// Step 3: Verify Sumcheck Proof related to the new multivariate composition eq_r(X) * poly
	let new_composition = CompositeMultilinearProductComposition::new(
		zerocheck_claim.multilinear_composition.clone(),
	);

	let sumcheck_claim = SumcheckClaim {
		multilinear_composition: Arc::new(new_composition),
		sum: F::ZERO,
		n_vars: zerocheck_claim.n_vars,
	};
	let evalcheck_claim_composition =
		sumcheck::verify::verify(sumcheck_claim, domain, &proof.sumcheck_proof, challenger)?;

	// Step 4: This verification gave us an evalcheck_claim on the new multi-linear composition
	// We need to divide by eq_r(eval_point) to get the evalcheck_claim on the original multilinear composition
	let eq_r_eval = eq_r.evaluate(&evalcheck_claim_composition.eval_point)?;

	let evalcheck_claim = EvalcheckClaim {
		multilinear_composition: zerocheck_claim.multilinear_composition,
		eval_point: evalcheck_claim_composition.eval_point,
		eval: evalcheck_claim_composition.eval
			* eq_r_eval.invert().expect(
				"eq_r_eval is invertible except with negligible probability in the size of F",
			),
	};

	// Step 5: Return the appropriate evalcheck claim for the original multilinear composition
	Ok(evalcheck_claim)
}
