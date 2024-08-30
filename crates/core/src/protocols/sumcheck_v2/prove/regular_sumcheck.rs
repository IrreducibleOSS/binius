// Copyright 2024 Ulvetanna Inc.

use super::{
	batch_prove::SumcheckProver,
	prover_state::{ProverState, SumcheckEvaluator},
};
use crate::{
	polynomial::{
		CompositionPoly, Error as PolynomialError, MultilinearComposite, MultilinearPoly,
	},
	protocols::sumcheck_v2::{
		common::{CompositeSumClaim, RoundCoeffs},
		error::Error,
	},
};
use binius_field::{ExtensionField, Field, PackedExtension, PackedField};
use binius_hal::ComputationBackend;
use binius_math::{extrapolate_line, EvaluationDomain, EvaluationDomainFactory};
use binius_utils::bail;
use itertools::izip;
use rayon::prelude::*;
use std::marker::PhantomData;

pub fn validate_witness<F, P, M, Composition>(
	multilinears: &[M],
	sum_claims: impl IntoIterator<Item = CompositeSumClaim<F, Composition>>,
) -> Result<(), Error>
where
	F: Field,
	P: PackedField<Scalar = F>,
	M: MultilinearPoly<P> + Send + Sync,
	Composition: CompositionPoly<P>,
{
	let n_vars = multilinears
		.first()
		.map(|multilinear| multilinear.n_vars())
		.unwrap_or_default();
	for multilinear in multilinears.iter() {
		if multilinear.n_vars() != n_vars {
			bail!(Error::NumberOfVariablesMismatch);
		}
	}

	let multilinears = multilinears.iter().collect::<Vec<_>>();

	for (i, claim) in sum_claims.into_iter().enumerate() {
		let CompositeSumClaim {
			composition,
			sum: expected_sum,
		} = claim;
		let witness = MultilinearComposite::new(n_vars, composition, multilinears.clone())?;
		let sum = (0..(1 << n_vars))
			.into_par_iter()
			.map(|j| witness.evaluate_on_hypercube(j))
			.try_reduce(|| F::ZERO, |a, b| Ok(a + b))?;

		if sum != expected_sum {
			bail!(Error::SumcheckNaiveValidationFailure {
				composition_index: i,
			});
		}
	}
	Ok(())
}

pub struct RegularSumcheckProver<FDomain, P, Composition, M, Backend>
where
	FDomain: Field,
	P: PackedField,
	M: MultilinearPoly<P> + Send + Sync,
	Backend: ComputationBackend,
{
	n_vars: usize,
	state: ProverState<P, M, Backend>,
	compositions: Vec<Composition>,
	domains: Vec<EvaluationDomain<FDomain>>,
}

impl<F, FDomain, P, Composition, M, Backend>
	RegularSumcheckProver<FDomain, P, Composition, M, Backend>
where
	F: Field + ExtensionField<FDomain>,
	FDomain: Field,
	P: PackedField<Scalar = F>,
	Composition: CompositionPoly<P>,
	M: MultilinearPoly<P> + Send + Sync,
	Backend: ComputationBackend,
{
	pub fn new(
		multilinears: Vec<M>,
		composite_claims: impl IntoIterator<Item = CompositeSumClaim<F, Composition>>,
		evaluation_domain_factory: impl EvaluationDomainFactory<FDomain>,
		switchover_fn: impl Fn(usize) -> usize,
		backend: Backend,
	) -> Result<Self, Error> {
		let composite_claims = composite_claims.into_iter().collect::<Vec<_>>();
		for claim in composite_claims.iter() {
			if claim.composition.n_vars() != multilinears.len() {
				bail!(Error::InvalidComposition {
					expected_n_vars: multilinears.len(),
				});
			}
		}

		let claimed_sums = composite_claims
			.iter()
			.map(|composite_claim| composite_claim.sum)
			.collect();
		let state = ProverState::new(multilinears, claimed_sums, switchover_fn, backend)?;
		let n_vars = state.n_vars();

		let domains = composite_claims
			.iter()
			.map(|composite_claim| {
				let degree = composite_claim.composition.degree();
				evaluation_domain_factory.create(degree + 1)
			})
			.collect::<Result<_, _>>()
			.map_err(Error::MathError)?;

		let compositions = composite_claims
			.into_iter()
			.map(|claim| claim.composition)
			.collect();

		Ok(Self {
			n_vars,
			state,
			compositions,
			domains,
		})
	}
}

impl<F, FDomain, P, Composition, M, Backend> SumcheckProver<F>
	for RegularSumcheckProver<FDomain, P, Composition, M, Backend>
where
	F: Field + ExtensionField<FDomain>,
	FDomain: Field,
	P: PackedField<Scalar = F> + PackedExtension<FDomain>,
	Composition: CompositionPoly<P>,
	M: MultilinearPoly<P> + Send + Sync,
	Backend: ComputationBackend,
{
	fn n_vars(&self) -> usize {
		self.n_vars
	}

	fn fold(&mut self, challenge: F) -> Result<(), Error> {
		self.state.fold(challenge)?;
		Ok(())
	}

	fn execute(&mut self, batch_coeff: F) -> Result<RoundCoeffs<F>, Error> {
		let evaluators = izip!(&self.compositions, &self.domains)
			.map(|(composition, evaluation_domain)| RegularSumcheckEvaluator {
				composition,
				evaluation_domain,
				domain_points: evaluation_domain.points(),
				_marker: PhantomData,
			})
			.collect::<Vec<_>>();

		self.state.calculate_round_coeffs(&evaluators, batch_coeff)
	}

	fn finish(self) -> Result<Vec<F>, Error> {
		self.state.finish()
	}
}

struct RegularSumcheckEvaluator<'a, P, FDomain, Composition>
where
	P: PackedField,
	FDomain: Field,
{
	composition: &'a Composition,
	evaluation_domain: &'a EvaluationDomain<FDomain>,
	domain_points: &'a [FDomain],
	_marker: PhantomData<P>,
}

impl<'a, F, P, FDomain, Composition> SumcheckEvaluator<P>
	for RegularSumcheckEvaluator<'a, P, FDomain, Composition>
where
	F: Field + ExtensionField<FDomain>,
	P: PackedField<Scalar = F> + PackedExtension<FDomain>,
	FDomain: Field,
	Composition: CompositionPoly<P>,
{
	fn n_round_evals(&self) -> usize {
		// NB: We skip evaluation of $r(X)$ at $X = 0$ as it is derivable from the
		// current_round_sum - $r(1)$.
		self.composition.degree()
	}

	fn process_vertex(
		&self,
		_i: usize,
		evals_0: &[P],
		evals_1: &[P],
		evals_z: &mut [P],
		round_evals: &mut [P],
	) {
		// Sumcheck evaluation at a specific point - given an array of 0 & 1 evaluations at some
		// index, use them to linearly interpolate each MLE value at domain point, and then
		// evaluate multivariate composite over those.

		round_evals[0] += self
			.composition
			.evaluate(evals_1)
			.expect("evals_1 is initialized with a length of poly.composition.n_vars()");

		// The rest require interpolation.
		for d in 2..=self.composition.degree() {
			evals_0
				.iter()
				.zip(evals_1.iter())
				.zip(evals_z.iter_mut())
				.for_each(|((&evals_0_j, &evals_1_j), evals_z_j)| {
					*evals_z_j = extrapolate_line(evals_0_j, evals_1_j, self.domain_points[d]);
				});

			round_evals[d - 1] += self
				.composition
				.evaluate(evals_z)
				.expect("evals_z is initialized with a length of poly.composition.n_vars()");
		}
	}

	fn round_evals_to_coeffs(
		&self,
		last_round_sum: F,
		mut round_evals: Vec<F>,
	) -> Result<Vec<F>, PolynomialError> {
		// Given $r(1), \ldots, r(d+1)$, letting $s$ be the current round's claimed sum,
		// we can compute $r(0)$ using the identity $r(0) = s - r(1)$
		round_evals.insert(0, last_round_sum - round_evals[0]);

		let coeffs = self.evaluation_domain.interpolate(&round_evals)?;
		Ok(coeffs)
	}
}
