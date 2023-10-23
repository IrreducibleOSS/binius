// Copyright 2023 Ulvetanna Inc.

use std::iter::repeat_with;

use crate::{
	field::Field,
	polynomial::{eq_ind_partial_eval, EvaluationDomain, MultilinearPoly},
	protocols::sumcheck::{self, SumcheckProof, SumcheckWitness},
};
use p3_challenger::{CanObserve, CanSample};

use super::{
	error::Error,
	zerocheck::{multiply_multilinear_composite, ZerocheckProof, ZerocheckWitness},
};

/// Prove a zerocheck instance.
///
/// The input polynomial is a composition of multilinear polynomials over a field F. The routine is
/// also parameterized by an operating field OF, which is isomorphic to F and over which the
/// majority of field operations are to be performed.
pub fn prove<F, OF, CH>(
	zerocheck_witness: ZerocheckWitness<'_, OF>,
	domain: &EvaluationDomain<F>,
	challenger: &mut CH,
) -> Result<ZerocheckProof<F>, Error>
where
	F: Field,
	OF: Field + From<F> + Into<F>,
	CH: CanSample<F> + CanObserve<F>,
{
	let poly = zerocheck_witness.polynomial;
	let n_vars: usize = poly.n_vars();

	let r: Vec<OF> = repeat_with(|| OF::from(challenger.sample()))
		.take(n_vars)
		.collect();

	// Step 2: Construct a multilinear polynomial eq(X, Y) on 2*n_vars variables
	// partially evaluated at r, will refer to this multilinear polynomial
	// as eq_r(X) on n_vars variables
	let eq_r: MultilinearPoly<OF> = eq_ind_partial_eval(n_vars, &r)?;

	// Step 3: Multiply eq_r(X) by poly to get a new multivariate polynomial
	// and represent it as a Multilinear composite
	let new_poly = multiply_multilinear_composite(poly.clone(), eq_r)?;

	// Step 4: Run the SumCheck Protocol on the new multivariate polynomial
	let sumcheck_witness = SumcheckWitness {
		polynomial: &new_poly,
	};
	let sumcheck_proof: SumcheckProof<F> =
		sumcheck::prove::prove(sumcheck_witness, domain, challenger)?;

	Ok(ZerocheckProof { sumcheck_proof })
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::{
		challenger::HashChallenger,
		field::{BinaryField128b, BinaryField128bPolyval, BinaryField32b},
		hash::GroestlHasher,
		polynomial::{CompositionPoly, MultilinearComposite, MultilinearPoly},
		protocols::{
			test_utils::{transform_poly, verify_evalcheck_claim, ProductMultivariate},
			zerocheck::{verify::verify, zerocheck::ZerocheckClaim},
		},
	};
	use std::sync::Arc;

	// f(x) = (x_1, x_2, x_3)) = g(h_0(x), ..., h_7(x)) where
	// g is the product composition
	// h_i(x) = 0 if x= <i>, 1 otw, defined first on boolean hypercube, then take multilinear extension
	// Specifically, f should vanish on the boolean hypercube, because some h_i will be 0.
	#[test]
	fn test_prove_verify_interaction() {
		type F = BinaryField32b; //field and operating field are both BinaryField32b

		let n_vars: usize = 3;
		let composition: Arc<dyn CompositionPoly<F, F>> =
			Arc::new(ProductMultivariate::new(1 << n_vars));
		let multilinears: Vec<MultilinearPoly<'_, F>> = (0..1 << n_vars)
			.map(|i| {
				let values: Vec<F> = (0..1 << n_vars)
					.map(|j| if i == j { F::ZERO } else { F::ONE })
					.collect::<Vec<_>>();
				MultilinearPoly::from_values(values).unwrap()
			})
			.collect::<Vec<_>>();
		let poly = MultilinearComposite::new(n_vars, composition, multilinears.clone()).unwrap();

		// CLAIM
		let zerocheck_claim: ZerocheckClaim<F> = ZerocheckClaim {
			n_vars,
			multilinear_composition: poly.clone().composition,
		};

		// SETUP
		let domain_points: Vec<F> = (0..=(1 << n_vars) + 1).map(F::new).collect::<Vec<_>>();
		let domain: EvaluationDomain<F> = EvaluationDomain::new(domain_points).unwrap();
		let mut prove_challenger = <HashChallenger<_, GroestlHasher<_>>>::new();
		let mut verify_challenger = prove_challenger.clone();

		// PROVER
		let zerocheck_witness: ZerocheckWitness<F> = ZerocheckWitness { polynomial: &poly };
		let proof: ZerocheckProof<F> =
			prove(zerocheck_witness, &domain, &mut prove_challenger).unwrap();

		// VERIFIER
		let evalcheck_claim =
			verify(zerocheck_claim, &domain, &proof, &mut verify_challenger).unwrap();

		// TESTING: Assert validity of evalcheck claim
		verify_evalcheck_claim(evalcheck_claim, &multilinears);
	}

	#[test]
	fn test_prove_verify_interaction_with_monomial_basis_conversion() {
		type F = BinaryField128b;
		type OF = BinaryField128bPolyval;

		let n_vars: usize = 3;
		let composition: Arc<dyn CompositionPoly<F, F>> =
			Arc::new(ProductMultivariate::new(1 << n_vars));
		let multilinears: Vec<MultilinearPoly<'_, F>> = (0..1 << n_vars)
			.map(|i| {
				let values: Vec<F> = (0..1 << n_vars)
					.map(|j| if i == j { F::ZERO } else { F::ONE })
					.collect::<Vec<_>>();
				MultilinearPoly::from_values(values).unwrap()
			})
			.collect::<Vec<_>>();
		let poly = MultilinearComposite::new(n_vars, composition, multilinears.clone()).unwrap();

		// CLAIM
		let zerocheck_claim: ZerocheckClaim<F> = ZerocheckClaim {
			n_vars,
			multilinear_composition: poly.clone().composition,
		};

		// SETUP
		let domain_points: Vec<F> = (0..=(1 << n_vars) + 1).map(F::new).collect::<Vec<_>>();
		let domain: EvaluationDomain<F> = EvaluationDomain::new(domain_points).unwrap();
		let mut prove_challenger = <HashChallenger<_, GroestlHasher<_>>>::new();
		let mut verify_challenger = prove_challenger.clone();

		// PROVER
		let prover_composition: Arc<dyn CompositionPoly<OF, OF>> =
			Arc::new(ProductMultivariate::new(1 << n_vars));
		let prover_poly = transform_poly(&poly, prover_composition).unwrap();
		let zerocheck_witness: ZerocheckWitness<OF> = ZerocheckWitness {
			polynomial: &prover_poly,
		};
		let proof: ZerocheckProof<F> =
			prove(zerocheck_witness, &domain, &mut prove_challenger).unwrap();

		// VERIFIER
		let evalcheck_claim =
			verify(zerocheck_claim, &domain, &proof, &mut verify_challenger).unwrap();

		// TESTING: Assert validity of evalcheck claim
		verify_evalcheck_claim(evalcheck_claim, &multilinears);
	}
}
