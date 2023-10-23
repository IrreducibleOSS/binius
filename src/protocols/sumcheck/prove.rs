// Copyright 2023 Ulvetanna Inc.

use crate::{
	field::{Field, PackedField},
	polynomial::{EvaluationDomain, MultilinearComposite, MultivariatePoly},
};
use p3_challenger::{CanObserve, CanSample};
use std::{borrow::Cow, slice};

use super::{
	error::Error,
	sumcheck::{SumcheckProof, SumcheckRound},
	SumcheckWitness,
};

/// Prove a sumcheck instance.
///
/// The input polynomial is a composition of multilinear polynomials over a field F. The routine is
/// also parameterized by an operating field OF, which is isomorphic to F and over which the
/// majority of field operations are to be performed.
pub fn prove<F, OF, CH>(
	sumcheck_witness: SumcheckWitness<'_, OF>,
	domain: &EvaluationDomain<F>,
	challenger: &mut CH,
) -> Result<SumcheckProof<F>, Error>
where
	F: Field,
	OF: Field + From<F> + Into<F>,
	CH: CanSample<F> + CanObserve<F>,
{
	let poly: &MultilinearComposite<OF> = sumcheck_witness.polynomial;
	let degree = poly.degree();
	if degree == 0 {
		return Err(Error::PolynomialDegreeIsZero);
	}
	if domain.size() != poly.degree() + 1 {
		return Err(Error::EvaluationDomainMismatch);
	}
	if domain.points()[0] != F::ZERO {
		return Err(Error::EvaluationDomainMismatch);
	}
	if domain.points()[1] != F::ONE {
		return Err(Error::EvaluationDomainMismatch);
	}

	let operating_domain = domain
		.points()
		.iter()
		.cloned()
		.map(OF::from)
		.collect::<Vec<_>>();

	let mut proof = SumcheckProof {
		rounds: Vec::with_capacity(poly.n_vars()),
	};

	let mut poly = Cow::Borrowed(poly);

	let n_multilinears = poly.composition.n_vars();

	let mut evals_0 = vec![OF::ZERO; n_multilinears];
	let mut evals_1 = vec![OF::ZERO; n_multilinears];
	let mut evals_z = vec![OF::ZERO; n_multilinears];
	let mut round_evals = vec![OF::ZERO; degree];

	while poly.n_vars() > 0 {
		round_evals.fill(OF::ZERO);
		for i in 0..1 << (poly.n_vars() - 1) {
			for (j, multilin) in poly.iter_multilinear_polys().enumerate() {
				evals_0[j] = multilin.evaluate_on_hypercube(i)?;
				evals_1[j] = multilin.evaluate_on_hypercube(i | 1 << (poly.n_vars() - 1))?;
			}

			round_evals[0] += poly.composition.evaluate(&evals_1)?;
			for d in 2..degree + 1 {
				evals_0
					.iter()
					.zip(evals_1.iter())
					.zip(evals_z.iter_mut())
					.for_each(|((&evals_0_j, &evals_1_j), evals_z_j)| {
						*evals_z_j = interpolate_line(evals_0_j, evals_1_j, operating_domain[d]);
					});
				round_evals[d - 1] += poly.composition.evaluate(&evals_z)?;
			}
		}

		let coeffs = round_evals
			.iter()
			.map(|&elem| elem.into())
			.collect::<Vec<_>>();
		challenger.observe_slice(&coeffs);
		proof.rounds.push(SumcheckRound { coeffs });

		let challenge = OF::from(challenger.sample());
		poly = Cow::Owned(poly.evaluate_partial(slice::from_ref(&challenge))?);
	}

	Ok(proof)
}

#[inline]
fn interpolate_line<P: PackedField>(x_0: P, x_1: P, z: P::Scalar) -> P {
	x_0 + (x_1 - x_0) * z
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::{
		challenger::HashChallenger,
		field::{BinaryField128b, BinaryField128bPolyval, BinaryField32b},
		hash::GroestlHasher,
		polynomial::MultilinearPoly,
		protocols::{
			sumcheck::{verify::verify, SumcheckClaim},
			test_utils::{transform_poly, verify_evalcheck_claim, ProductMultivariate},
		},
	};
	use rand::{rngs::StdRng, SeedableRng};
	use std::{iter::repeat_with, sync::Arc};

	#[test]
	fn test_prove_verify_interaction() {
		type F = BinaryField32b;

		let mut rng = StdRng::seed_from_u64(0);

		let n_vars = 8;
		let composition: Arc<dyn MultivariatePoly<F, F>> = Arc::new(ProductMultivariate::new(3));
		let multilinears = repeat_with(|| {
			let values = repeat_with(|| Field::random(&mut rng))
				.take(1 << n_vars)
				.collect::<Vec<F>>();
			MultilinearPoly::from_values(values).unwrap()
		})
		.take(composition.n_vars())
		.collect::<Vec<_>>();
		let poly = MultilinearComposite::new(n_vars, composition, multilinears.clone()).unwrap();

		let sum = (0..1 << n_vars)
			.map(|i| poly.evaluate_on_hypercube(i).unwrap())
			.sum();

		// CLAIM
		let sumcheck_claim = SumcheckClaim {
			n_vars,
			sum,
			multilinear_composition: poly.clone().composition,
		};

		// SETUP
		let domain: EvaluationDomain<BinaryField32b> =
			EvaluationDomain::new(vec![F::ZERO, F::ONE, F::new(2), F::new(3)]).unwrap();
		let mut prove_challenger = <HashChallenger<_, GroestlHasher<_>>>::new();
		let mut verify_challenger = prove_challenger.clone();

		// PROVER
		let sumcheck_witness = SumcheckWitness { polynomial: &poly };
		let proof = prove(sumcheck_witness, &domain, &mut prove_challenger).unwrap();

		// VERIFIER
		let evalcheck_claim =
			verify(sumcheck_claim, &domain, &proof, &mut verify_challenger).unwrap();

		// TESTING: Assert validity of evalcheck claim
		verify_evalcheck_claim(evalcheck_claim, &multilinears);
	}

	#[test]
	fn test_prove_verify_interaction_with_monomial_basis_conversion() {
		type F = BinaryField128b;
		type OF = BinaryField128bPolyval;

		let mut rng = StdRng::seed_from_u64(0);

		let n_vars = 8;
		let composition: Arc<dyn MultivariatePoly<F, F>> = Arc::new(ProductMultivariate::new(3));
		let multilinears = repeat_with(|| {
			let values = repeat_with(|| Field::random(&mut rng))
				.take(1 << n_vars)
				.collect::<Vec<F>>();
			MultilinearPoly::from_values(values).unwrap()
		})
		.take(composition.n_vars())
		.collect::<Vec<_>>();
		let poly = MultilinearComposite::new(n_vars, composition, multilinears.clone()).unwrap();

		let sum = (0..1 << n_vars)
			.map(|i| poly.evaluate_on_hypercube(i).unwrap())
			.sum();

		// CLAIM
		let sumcheck_claim: SumcheckClaim<F> = SumcheckClaim {
			n_vars,
			sum,
			multilinear_composition: poly.clone().composition,
		};

		// SETUP
		let domain = EvaluationDomain::new(vec![F::ZERO, F::ONE, F::new(2), F::new(3)]).unwrap();
		let mut prove_challenger = <HashChallenger<_, GroestlHasher<_>>>::new();
		let mut verify_challenger = prove_challenger.clone();

		// PROVER
		let prover_composition: Arc<dyn MultivariatePoly<OF, OF>> =
			Arc::new(ProductMultivariate::new(3));
		let prover_poly = transform_poly(&poly, prover_composition).unwrap();
		let sumcheck_witness = SumcheckWitness {
			polynomial: &prover_poly,
		};
		let proof = prove(sumcheck_witness, &domain, &mut prove_challenger).unwrap();

		// VERIFIER
		let evalcheck_claim =
			verify(sumcheck_claim, &domain, &proof, &mut verify_challenger).unwrap();

		// TESTING: Assert validity of evalcheck claim
		verify_evalcheck_claim(evalcheck_claim, &multilinears);
	}
}
