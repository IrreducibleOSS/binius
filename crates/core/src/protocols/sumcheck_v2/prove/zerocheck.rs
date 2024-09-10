// Copyright 2024 Ulvetanna Inc.

use crate::{
	polynomial::{
		CompositionPoly, Error as PolynomialError, MultilinearComposite, MultilinearPoly,
		MultilinearQuery,
	},
	protocols::{
		sumcheck_v2::{
			prove::{
				prover_state::{ProverState, SumcheckEvaluator},
				SumcheckProver,
			},
			Error, RoundCoeffs,
		},
		utils::packed_from_fn_with_offset,
	},
};
use binius_field::{
	packed::get_packed_slice, ExtensionField, Field, PackedExtension, PackedField,
	PackedFieldIndexable,
};
use binius_hal::ComputationBackend;
use binius_math::{EvaluationDomain, EvaluationDomainFactory};
use binius_utils::bail;
use itertools::izip;
use rayon::prelude::*;
use stackalloc::stackalloc_with_default;
use std::ops::Range;

pub fn validate_witness<F, P, M, Composition>(
	multilinears: &[M],
	zero_claims: impl IntoIterator<Item = Composition>,
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

	for (i, composition) in zero_claims.into_iter().enumerate() {
		let witness = MultilinearComposite::new(n_vars, composition, multilinears.clone())?;
		(0..(1 << n_vars)).into_par_iter().try_for_each(|j| {
			if witness.evaluate_on_hypercube(j)? != F::ZERO {
				return Err(Error::ZerocheckNaiveValidationFailure {
					composition_index: i,
					vertex_index: j,
				});
			}
			Ok(())
		})?;
	}
	Ok(())
}

#[derive(Debug)]
pub struct ZerocheckProver<FDomain, P, Composition, M, Backend>
where
	FDomain: Field,
	P: PackedField,
	M: MultilinearPoly<P> + Send + Sync,
	Backend: ComputationBackend,
{
	n_vars: usize,
	state: ProverState<FDomain, P, M, Backend>,
	eq_ind_eval: P::Scalar,
	partial_eq_ind_evals: Backend::Vec<P>,
	zerocheck_challenges: Vec<P::Scalar>,
	compositions: Vec<Composition>,
	domains: Vec<EvaluationDomain<FDomain>>,
}

impl<F, FDomain, P, Composition, M, Backend> ZerocheckProver<FDomain, P, Composition, M, Backend>
where
	F: Field + ExtensionField<FDomain>,
	FDomain: Field,
	P: PackedFieldIndexable<Scalar = F> + PackedExtension<FDomain>,
	Composition: CompositionPoly<P>,
	M: MultilinearPoly<P> + Send + Sync,
	Backend: ComputationBackend,
{
	pub fn new(
		multilinears: Vec<M>,
		zero_claims: impl IntoIterator<Item = Composition>,
		challenges: &[F],
		evaluation_domain_factory: impl EvaluationDomainFactory<FDomain>,
		switchover_fn: impl Fn(usize) -> usize,
		backend: Backend,
	) -> Result<Self, Error> {
		let compositions = zero_claims.into_iter().collect::<Vec<_>>();
		for composition in compositions.iter() {
			if composition.n_vars() != multilinears.len() {
				bail!(Error::InvalidComposition {
					expected_n_vars: multilinears.len(),
				});
			}
		}

		let claimed_sums = vec![F::ZERO; compositions.len()];

		let domains = compositions
			.iter()
			.map(|composition| {
				let degree = composition.degree();
				evaluation_domain_factory.create(degree + 1)
			})
			.collect::<Result<Vec<_>, _>>()?;

		let evaluation_points = domains
			.iter()
			.max_by_key(|domain| domain.points().len())
			.map_or_else(|| Vec::new(), |domain| domain.points().to_vec());

		let state = ProverState::new(
			multilinears,
			claimed_sums,
			evaluation_points,
			switchover_fn,
			backend.clone(),
		)?;
		let n_vars = state.n_vars();

		if challenges.len() != n_vars {
			return Err(Error::IncorrectZerocheckChallengesLength);
		}

		let partial_eq_ind_evals =
			MultilinearQuery::with_full_query(&challenges[1..], backend)?.into_expansion();

		Ok(Self {
			n_vars,
			state,
			eq_ind_eval: F::ONE,
			partial_eq_ind_evals,
			zerocheck_challenges: challenges.to_vec(),
			compositions,
			domains,
		})
	}

	fn round(&self) -> usize {
		self.n_vars - self.n_rounds_remaining()
	}

	fn n_rounds_remaining(&self) -> usize {
		self.state.n_vars()
	}

	fn update_eq_ind_eval(&mut self, challenge: F) {
		// Update the running eq ind evaluation.
		let alpha = self.zerocheck_challenges[self.round()];
		// NB: In binary fields, this expression can be simplified to 1 + α + challenge. However,
		// we opt to keep this prover generic over all fields. These two multiplications per round
		// have negligible performance impact.
		self.eq_ind_eval *= alpha * challenge + (F::ONE - alpha) * (F::ONE - challenge);
	}

	fn fold_partial_eq_ind(&mut self) {
		let n_rounds_remaining = self.n_rounds_remaining();
		if n_rounds_remaining == 0 {
			return;
		}
		if self.partial_eq_ind_evals.len() == 1 {
			let unpacked = P::unpack_scalars_mut(&mut self.partial_eq_ind_evals);
			for i in 0..(1 << (n_rounds_remaining - 1)) {
				unpacked[i] = unpacked[2 * i] + unpacked[2 * i + 1];
			}
		} else {
			let current_evals = &self.partial_eq_ind_evals;
			let updated_evals = (0..current_evals.len() / 2)
				.into_par_iter()
				.map(|i| {
					packed_from_fn_with_offset(i, |index| {
						let eval0 = get_packed_slice(current_evals, index << 1);
						let eval1 = get_packed_slice(current_evals, (index << 1) + 1);
						eval0 + eval1
					})
				})
				.collect();
			self.partial_eq_ind_evals = Backend::to_hal_slice(updated_evals);
		}
	}
}

impl<F, FDomain, P, Composition, M, Backend> SumcheckProver<F>
	for ZerocheckProver<FDomain, P, Composition, M, Backend>
where
	F: Field + ExtensionField<FDomain>,
	FDomain: Field,
	P: PackedFieldIndexable<Scalar = F> + PackedExtension<FDomain>,
	Composition: CompositionPoly<P>,
	M: MultilinearPoly<P> + Send + Sync,
	Backend: ComputationBackend,
{
	fn n_vars(&self) -> usize {
		self.n_vars
	}

	fn fold(&mut self, challenge: F) -> Result<(), Error> {
		self.update_eq_ind_eval(challenge);
		self.state.fold(challenge)?;

		// This must happen after state fold, which decrements n_rounds_remaining.
		self.fold_partial_eq_ind();

		Ok(())
	}

	fn execute(&mut self, batch_coeff: F) -> Result<RoundCoeffs<F>, Error> {
		let round = self.round();
		let coeffs = if round == 0 {
			let evaluators = izip!(&self.compositions, &self.domains)
				.map(|(composition, evaluation_domain)| ZerocheckFirstRoundEvaluator {
					composition,
					evaluation_domain,
					partial_eq_ind_evals: &self.partial_eq_ind_evals,
				})
				.collect::<Vec<_>>();
			self.state
				.calculate_round_coeffs(&evaluators, batch_coeff)?
		} else {
			let evaluators = izip!(&self.compositions, &self.domains)
				.map(|(composition, evaluation_domain)| ZerocheckLaterRoundEvaluator {
					composition,
					evaluation_domain,
					partial_eq_ind_evals: &self.partial_eq_ind_evals,
					round_zerocheck_challenge: self.zerocheck_challenges[round],
				})
				.collect::<Vec<_>>();
			self.state
				.calculate_round_coeffs(&evaluators, batch_coeff)?
		};

		// Convert v' polynomial into v polynomial
		let alpha = self.zerocheck_challenges[round];

		// eq(X, α) = (1 − α) + (2 α − 1) X
		// NB: In binary fields, this expression is simply  eq(X, α) = 1 + α + X
		// However, we opt to keep this prover generic over all fields.
		let constant_scalar = F::ONE - alpha;
		let linear_scalar = alpha.double() - F::ONE;

		let coeffs_scaled_by_constant_term = coeffs.clone() * constant_scalar;
		let mut coeffs_scaled_by_linear_term = coeffs * linear_scalar;
		coeffs_scaled_by_linear_term.0.insert(0, F::ZERO); // Multiply polynomial by X

		let sumcheck_coeffs = coeffs_scaled_by_constant_term + &coeffs_scaled_by_linear_term;
		Ok(sumcheck_coeffs * self.eq_ind_eval)
	}

	fn finish(self) -> Result<Vec<F>, Error> {
		let mut evals = self.state.finish()?;
		evals.push(self.eq_ind_eval);
		Ok(evals)
	}
}

struct ZerocheckFirstRoundEvaluator<'a, P, FDomain, Composition>
where
	P: PackedField,
	FDomain: Field,
{
	composition: &'a Composition,
	evaluation_domain: &'a EvaluationDomain<FDomain>,
	partial_eq_ind_evals: &'a [P],
}

impl<'a, F, P, FDomain, Composition> SumcheckEvaluator<P>
	for ZerocheckFirstRoundEvaluator<'a, P, FDomain, Composition>
where
	F: Field + ExtensionField<FDomain>,
	P: PackedField<Scalar = F> + PackedExtension<FDomain>,
	FDomain: Field,
	Composition: CompositionPoly<P>,
{
	fn eval_point_indices(&self) -> Range<usize> {
		// In the first round of zerocheck we can uniquely determine the degree d
		// univariate round polynomial $R(X)$ with evaluations at X = 2, ..., d
		// because we know r(0) = r(1) = 0
		2..self.composition.degree() + 1
	}

	fn process_subcube_at_eval_point(
		&self,
		subcube_vars: usize,
		subcube_index: usize,
		sparse_batch_query: &[&[P]],
	) -> P {
		let row_len = sparse_batch_query.first().map_or(0, |row| row.len());

		stackalloc_with_default(row_len, |evals| {
			self.composition
				.sparse_batch_evaluate(sparse_batch_query, evals)
				.expect("correct by query construction invariant");

			let subcube_start = subcube_index << subcube_vars.saturating_sub(P::LOG_WIDTH);
			for (i, eval) in evals.iter_mut().enumerate() {
				*eval *= self.partial_eq_ind_evals[subcube_start + i];
			}

			evals.iter().copied().sum::<P>()
		})
	}

	fn round_evals_to_coeffs(
		&self,
		last_round_sum: F,
		mut round_evals: Vec<F>,
	) -> Result<Vec<F>, PolynomialError> {
		assert_eq!(last_round_sum, F::ZERO);

		// We are given $r(2), \ldots, r(d)$.
		// From context, we infer that $r(0) = r(1) = 0$.
		round_evals.insert(0, P::Scalar::ZERO);
		round_evals.insert(0, P::Scalar::ZERO);

		let coeffs = self.evaluation_domain.interpolate(&round_evals)?;
		Ok(coeffs)
	}
}

struct ZerocheckLaterRoundEvaluator<'a, P, FDomain, Composition>
where
	P: PackedField,
	FDomain: Field,
{
	composition: &'a Composition,
	evaluation_domain: &'a EvaluationDomain<FDomain>,
	partial_eq_ind_evals: &'a [P],
	round_zerocheck_challenge: P::Scalar,
}

impl<'a, F, P, FDomain, Composition> SumcheckEvaluator<P>
	for ZerocheckLaterRoundEvaluator<'a, P, FDomain, Composition>
where
	F: Field + ExtensionField<FDomain>,
	P: PackedField<Scalar = F> + PackedExtension<FDomain>,
	FDomain: Field,
	Composition: CompositionPoly<P>,
{
	fn eval_point_indices(&self) -> Range<usize> {
		// We can uniquely derive the degree d univariate round polynomial r from evaluations at
		// X = 1, ..., d because we have an identity that relates r(0), r(1), and the current
		// round's claimed sum
		1..self.composition.degree() + 1
	}

	fn process_subcube_at_eval_point(
		&self,
		subcube_vars: usize,
		subcube_index: usize,
		sparse_batch_query: &[&[P]],
	) -> P {
		let row_len = sparse_batch_query.first().map_or(0, |row| row.len());

		stackalloc_with_default(row_len, |evals| {
			self.composition
				.sparse_batch_evaluate(sparse_batch_query, evals)
				.expect("correct by query construction invariant");

			let subcube_start = subcube_index << subcube_vars.saturating_sub(P::LOG_WIDTH);
			for (i, eval) in evals.iter_mut().enumerate() {
				*eval *= self.partial_eq_ind_evals[subcube_start + i];
			}

			evals.iter().copied().sum::<P>()
		})
	}

	fn round_evals_to_coeffs(
		&self,
		last_round_sum: F,
		mut round_evals: Vec<F>,
	) -> Result<Vec<F>, PolynomialError> {
		// This is a subsequent round of a sumcheck that came from zerocheck, given $r(1), \ldots, r(d)$
		// Letting $s$ be the current round's claimed sum, and $\alpha_i$ the ith zerocheck challenge
		// we have the identity $r(0) = \frac{1}{1 - \alpha_i} * (s - \alpha_i * r(1))$
		// which allows us to compute the value of $r(0)$

		let alpha = self.round_zerocheck_challenge;
		let one_evaluation = round_evals[0]; // r(1)
		let zero_evaluation_numerator = last_round_sum - one_evaluation * alpha;
		let zero_evaluation_denominator_inv = (F::ONE - alpha).invert_or_zero();
		let zero_evaluation = zero_evaluation_numerator * zero_evaluation_denominator_inv;

		round_evals.insert(0, zero_evaluation);

		let coeffs = self.evaluation_domain.interpolate(&round_evals)?;
		Ok(coeffs)
	}
}
