// Copyright 2023 Ulvetanna Inc.

use rayon::{iter::IntoParallelIterator, prelude::*};

use crate::{
	field::Field,
	polynomial::{extrapolate_line, EvaluationDomain, MultilinearComposite},
	protocols::{evalcheck::evalcheck::EvalcheckWitness, sumcheck::reduce_sumcheck_claim_round},
};
use std::slice;

use super::{
	check_evaluation_domain,
	error::Error,
	reduce_sumcheck_claim_final,
	sumcheck::{SumcheckProof, SumcheckRound},
	SumcheckClaim, SumcheckProveOutput, SumcheckRoundClaim, SumcheckWitness,
};

#[derive(Clone)]
pub struct ProveRoundOutput<'a, F: Field, OF: Field + From<F> + Into<F>> {
	pub sumcheck_proof: SumcheckProof<F>,
	pub sumcheck_witness: SumcheckWitness<'a, OF>,
	pub sumcheck_reduced_claim: SumcheckRoundClaim<F>,
}

impl<'a, F: Field, OF: Field + From<F> + Into<F>> ProveRoundOutput<'a, F, OF> {
	pub fn finished(&self) -> bool {
		self.sumcheck_witness.polynomial.n_vars() == 0
	}
}

fn validate_input<'a, F: Field, OF: Field + From<F> + Into<F>>(
	sumcheck_claim: SumcheckClaim<'a, F>,
	sumcheck_witness: SumcheckWitness<'a, OF>,
	domain: &EvaluationDomain<F>,
) -> Result<(), Error> {
	let degree = sumcheck_witness.polynomial.composition.degree();
	if degree == 0 {
		return Err(Error::PolynomialDegreeIsZero);
	}
	check_evaluation_domain(degree, domain)?;
	if sumcheck_claim.poly.n_vars() != sumcheck_witness.polynomial.n_vars() {
		println!(
			"Claim and Witness n_vars mismatch. Claim: {}, Witness: {}",
			sumcheck_claim.poly.n_vars(),
			sumcheck_witness.polynomial.n_vars()
		);
		return Err(Error::ImproperInput);
	}
	Ok(())
}

fn prove_round<'a, F: Field, OF: Field + From<F> + Into<F>>(
	poly: MultilinearComposite<'a, OF, OF>,
	domain: &EvaluationDomain<F>,
	current_proof: SumcheckProof<F>,
	round_claim: SumcheckRoundClaim<F>,
) -> Result<ProveRoundOutput<'a, F, OF>, Error> {
	let degree = poly.degree();
	let operating_domain = domain
		.points()
		.iter()
		.cloned()
		.map(OF::from)
		.collect::<Vec<_>>();

	let mut updated_proof = current_proof;

	let n_multilinears = poly.composition.n_vars();

	let fold_result = (0..1 << (poly.n_vars() - 1)).into_par_iter().fold(
		|| {
			(
				vec![OF::ZERO; n_multilinears],
				vec![OF::ZERO; n_multilinears],
				vec![OF::ZERO; n_multilinears],
				vec![OF::ZERO; degree],
			)
		},
		|(mut evals_0, mut evals_1, mut evals_z, mut round_evals), i| {
			for (j, multilin) in poly.iter_multilinear_polys().enumerate() {
				evals_0[j] = multilin.evaluate_on_hypercube(2 * i).expect(
					"all multilinear polynomials in a composite have n_vars variables by
						an invariant on the struct; i is in the range [0, 2^{n_vars - 1}) by
						iteration range; thus 2 * i is in the range [0, 2^{n_vars}) and indexes
						a valid hypercube vertex",
				);
				evals_1[j] = multilin.evaluate_on_hypercube(2 * i + 1).expect(
					"all multilinear polynomials in a composite have n_vars variables by
						an invariant on the struct; i is in the range [0, 2^{n_vars - 1}) by
						iteration range; thus 2 * i + 1 is in the range [0, 2^{n_vars}) and
						indexes a valid hypercube vertex",
				);
			}
			round_evals[0] = poly
				.composition
				.evaluate(&evals_1)
				.expect("evals_1 is initialized with a length of poly.composition.n_vars()");
			for d in 2..degree + 1 {
				evals_0
					.iter()
					.zip(evals_1.iter())
					.zip(evals_z.iter_mut())
					.for_each(|((&evals_0_j, &evals_1_j), evals_z_j)| {
						*evals_z_j = extrapolate_line(evals_0_j, evals_1_j, operating_domain[d]);
					});
				round_evals[d - 1] = poly
					.composition
					.evaluate(&evals_z)
					.expect("evals_z is initialized with a length of poly.composition.n_vars()");
			}
			(evals_0, evals_1, evals_z, round_evals)
		},
	);

	let round_evals = fold_result
		.map(|(_, _, _, round_evals)| round_evals)
		.reduce(
			|| vec![OF::ZERO; degree],
			|mut overall_round_evals, partial_round_evals| {
				overall_round_evals
					.iter_mut()
					.zip(partial_round_evals.iter())
					.for_each(|(f, s)| *f += s);
				overall_round_evals
			},
		);

	// round_evals and round_claim, if honest, gives verifier enough information to
	// determine r(X) := \sum_{i \in B_{n-1}} poly(X, i)
	let coeffs = round_evals
		.iter()
		.map(|&elem| elem.into())
		.collect::<Vec<_>>();
	updated_proof.rounds.push(SumcheckRound { coeffs });
	let updated_witness = SumcheckWitness { polynomial: poly };
	let result = ProveRoundOutput {
		sumcheck_proof: updated_proof,
		sumcheck_witness: updated_witness,
		sumcheck_reduced_claim: round_claim,
	};
	Ok(result)
}

pub fn prove_first_round<'a, F: Field, OF: Field + From<F> + Into<F>>(
	claim: SumcheckClaim<'a, F>,
	witness: SumcheckWitness<'a, OF>,
	domain: &EvaluationDomain<F>,
) -> Result<ProveRoundOutput<'a, F, OF>, Error> {
	validate_input(claim.clone(), witness.clone(), domain)?;

	let current_proof = SumcheckProof { rounds: vec![] };
	let curr_rd_reduced_claim = SumcheckRoundClaim {
		partial_point: vec![],
		current_round_sum: claim.sum,
	};

	prove_round(witness.polynomial, domain, current_proof, curr_rd_reduced_claim)
}

/// Prove a sumcheck instance reduction, single round
///
/// The input polynomial is a composition of multilinear polynomials over a field F. The routine is
/// also parameterized by an operating field OF, which is isomorphic to F and over which the
/// majority of field operations are to be performed.
pub fn prove_later_rounds<'a, F, OF>(
	original_claim: &SumcheckClaim<F>,
	current_witness: SumcheckWitness<'a, OF>,
	domain: &EvaluationDomain<F>,
	current_proof: SumcheckProof<F>,
	prev_rd_challenge: F,
	prev_rd_reduced_claim: SumcheckRoundClaim<F>,
) -> Result<ProveRoundOutput<'a, F, OF>, Error>
where
	F: Field,
	OF: Field + From<F> + Into<F>,
{
	// Input Validations
	let completed_rounds = current_proof.rounds.len();
	if completed_rounds == 0 {
		return Err(Error::ImproperInput);
	}

	let mut poly = current_witness.polynomial.clone();

	// Previous Round Housekeeping
	if poly.n_vars() == 0 {
		return Err(Error::ImproperInput);
	}
	// Update poly from end of last round
	poly = poly.evaluate_partial_low(slice::from_ref(&OF::from(prev_rd_challenge)))?;
	// Perform reduce sumcheck claim round from result of last round
	let round = current_proof.rounds[current_proof.rounds.len() - 1].clone();
	let curr_rd_reduced_claim = reduce_sumcheck_claim_round(
		&original_claim.poly,
		domain,
		round,
		prev_rd_reduced_claim.current_round_sum,
		prev_rd_reduced_claim.partial_point,
		prev_rd_challenge,
	)?;
	if poly.n_vars() == 0 {
		let result = ProveRoundOutput {
			sumcheck_proof: current_proof,
			sumcheck_witness: current_witness,
			sumcheck_reduced_claim: curr_rd_reduced_claim,
		};
		return Ok(result);
	}

	// Current Round Begins
	prove_round(poly, domain, current_proof, curr_rd_reduced_claim)
}

/// Prove a sumcheck instance reduction, final step, after all rounds are completed.
///
/// The input polynomial is a composition of multilinear polynomials over a field F. The routine is
/// also parameterized by an operating field OF, which is isomorphic to F and over which the
/// majority of field operations are to be performed.
pub fn prove_final<'a, F, OF>(
	sumcheck_claim: &'a SumcheckClaim<F>,
	sumcheck_witness: SumcheckWitness<'a, OF>,
	sumcheck_proof: SumcheckProof<F>,
	final_rd_reduced_claim_output: &SumcheckRoundClaim<F>,
) -> Result<SumcheckProveOutput<'a, F, OF>, Error>
where
	F: Field,
	OF: Field + From<F> + Into<F>,
{
	let evalcheck_claim =
		reduce_sumcheck_claim_final(sumcheck_claim, final_rd_reduced_claim_output)?;
	let evalcheck_witness = EvalcheckWitness {
		polynomial: sumcheck_witness.polynomial,
	};
	Ok(SumcheckProveOutput {
		sumcheck_proof,
		evalcheck_claim,
		evalcheck_witness,
	})
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::{
		field::{BinaryField128b, BinaryField128bPolyval, BinaryField32b},
		iopoly::{CompositePoly, MultilinearPolyOracle, MultivariatePolyOracle},
		polynomial::{CompositionPoly, MultilinearComposite, MultilinearPoly},
		protocols::{
			sumcheck::{
				verify::{verify_final, verify_round},
				SumcheckClaim,
			},
			test_utils::{transform_poly, TestProductComposition},
		},
	};
	use rand::{rngs::StdRng, SeedableRng};
	use std::{iter::repeat_with, sync::Arc};

	fn full_prove_verify_checks<'a, F: Field, OF: Field + Into<F> + From<F>>(
		witness: SumcheckWitness<'a, OF>,
		claim: SumcheckClaim<'a, F>,
		domain: &EvaluationDomain<F>,
		challenges: Vec<F>,
	) {
		let mut current_witness = witness.clone();

		let n_vars = claim.poly.n_vars();
		assert_eq!(witness.polynomial.n_vars(), n_vars);
		assert_eq!(challenges.len(), n_vars);
		assert!(n_vars > 0);
		let prove_round_output = prove_first_round(claim.clone(), current_witness, domain).unwrap();
		current_witness = prove_round_output.sumcheck_witness;
		let mut current_partial_proof = prove_round_output.sumcheck_proof;
		let mut prev_rd_reduced_claim = prove_round_output.sumcheck_reduced_claim;

		#[allow(clippy::needless_range_loop)]
		for i in 0..n_vars {
			let verified_rd_claim = verify_round(
				&claim.poly,
				domain,
				current_partial_proof.rounds[i].clone(),
				prev_rd_reduced_claim.current_round_sum,
				prev_rd_reduced_claim.partial_point.clone(),
				challenges[i],
			)
			.unwrap();
			let partial_prove_round_output = prove_later_rounds(
				&claim,
				current_witness,
				domain,
				current_partial_proof,
				challenges[i],
				prev_rd_reduced_claim,
			)
			.unwrap();
			current_witness = partial_prove_round_output.sumcheck_witness;
			current_partial_proof = partial_prove_round_output.sumcheck_proof;
			prev_rd_reduced_claim = partial_prove_round_output.sumcheck_reduced_claim;
			assert_eq!(prev_rd_reduced_claim, verified_rd_claim);
		}
		let final_prove_output =
			prove_final(&claim, witness, current_partial_proof, &prev_rd_reduced_claim).unwrap();
		let final_verify_output = verify_final(&claim, &prev_rd_reduced_claim).unwrap();

		assert_eq!(final_prove_output.evalcheck_claim.eval, final_verify_output.eval);
		assert_eq!(final_prove_output.evalcheck_claim.eval_point, final_verify_output.eval_point);
		assert_eq!(
			final_prove_output
				.evalcheck_claim
				.poly
				.into_composite()
				.n_vars(),
			claim.poly.n_vars()
		);
		assert_eq!(final_verify_output.poly.into_composite().n_vars(), claim.poly.n_vars());
	}

	#[test]
	fn test_prove_verify_interaction() {
		type F = BinaryField32b;

		let mut rng = StdRng::seed_from_u64(0);

		// Setup Witness
		let n_vars = 8;
		let n_multilinears = 3;
		let composition: Arc<dyn CompositionPoly<F>> =
			Arc::new(TestProductComposition::new(n_multilinears));
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
			.map(|i| {
				let mut prod = F::ONE;
				(0..n_multilinears).for_each(|j| {
					prod *= multilinears[j].evaluate_on_hypercube(i).unwrap();
				});
				prod
			})
			.sum::<F>();
		let sumcheck_witness = SumcheckWitness { polynomial: poly };

		// Setup Claim
		let h = (0..n_multilinears)
			.map(|i| MultilinearPolyOracle::Committed { id: i, n_vars })
			.collect();
		let composite_poly =
			CompositePoly::new(n_vars, h, Arc::new(TestProductComposition::new(n_multilinears)))
				.unwrap();
		let poly_oracle = MultivariatePolyOracle::Composite(composite_poly);
		let sumcheck_claim = SumcheckClaim {
			sum,
			poly: poly_oracle,
		};

		// Setup evaluation domain
		let domain = EvaluationDomain::new(n_multilinears + 1).unwrap();
		let challenges = repeat_with(|| Field::random(&mut rng))
			.take(n_vars)
			.collect::<Vec<_>>();

		full_prove_verify_checks(sumcheck_witness, sumcheck_claim, &domain, challenges);
	}

	#[test]
	fn test_prove_verify_interaction_with_monomial_basis_conversion() {
		type F = BinaryField128b;
		type OF = BinaryField128bPolyval;

		let mut rng = StdRng::seed_from_u64(0);

		let n_vars = 8;
		let n_multilinears = 3;
		let composition: Arc<dyn CompositionPoly<F>> =
			Arc::new(TestProductComposition::new(n_multilinears));
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
			.map(|i| {
				let mut prod = F::ONE;
				(0..n_multilinears).for_each(|j| {
					prod *= multilinears[j].evaluate_on_hypercube(i).unwrap();
				});
				prod
			})
			.sum::<F>();

		let prover_composition: Arc<dyn CompositionPoly<OF>> =
			Arc::new(TestProductComposition::new(n_multilinears));
		let prover_poly = transform_poly(&poly, prover_composition).unwrap();
		let sumcheck_witness = SumcheckWitness {
			polynomial: prover_poly,
		};

		// CLAIM
		let h = (0..n_multilinears)
			.map(|i| MultilinearPolyOracle::Committed { id: i, n_vars })
			.collect();
		let composite_poly =
			CompositePoly::new(n_vars, h, Arc::new(TestProductComposition::new(n_multilinears)))
				.unwrap();
		let poly_oracle = MultivariatePolyOracle::Composite(composite_poly);
		let sumcheck_claim = SumcheckClaim {
			sum,
			poly: poly_oracle,
		};

		// Setup evaluation domain
		let domain = EvaluationDomain::new(n_multilinears + 1).unwrap();
		let challenges = repeat_with(|| Field::random(&mut rng))
			.take(n_vars)
			.collect::<Vec<_>>();

		full_prove_verify_checks(sumcheck_witness, sumcheck_claim, &domain, challenges)
	}
}
