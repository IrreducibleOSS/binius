// Copyright 2024-2025 Irreducible Inc.

use std::{marker::PhantomData, ops::Range};

use binius_fast_compute::arith_circuit::ArithCircuitPoly;
use binius_field::{ExtensionField, Field, PackedExtension, PackedField, TowerField};
use binius_hal::{ComputationBackend, SumcheckEvaluator, SumcheckMultilinear};
use binius_math::{
	CompositionPoly, EvaluationDomainFactory, EvaluationOrder, InterpolationDomain,
	MultilinearPoly, RowsBatchRef,
};
use binius_maybe_rayon::prelude::*;
use binius_utils::bail;
use itertools::izip;
use stackalloc::stackalloc_with_default;
use tracing::instrument;

use crate::{
	polynomial::{Error as PolynomialError, MultilinearComposite},
	protocols::sumcheck::{
		common::{
			equal_n_vars_check, get_nontrivial_evaluation_points,
			interpolation_domains_for_composition_degrees, CompositeSumClaim, RoundCoeffs,
		},
		error::Error,
		prove::{ProverState, SumcheckInterpolator, SumcheckProver},
	},
};

pub fn validate_witness<'a, F, P, M, Composition>(
	multilinears: &[M],
	sum_claims: impl IntoIterator<Item = CompositeSumClaim<F, &'a Composition>>,
) -> Result<(), Error>
where
	F: Field,
	P: PackedField<Scalar = F>,
	M: MultilinearPoly<P> + Send + Sync,
	Composition: CompositionPoly<P> + 'a,
{
	let n_vars = multilinears
		.first()
		.map(|multilinear| multilinear.n_vars())
		.unwrap_or_default();
	for multilinear in multilinears {
		if multilinear.n_vars() != n_vars {
			bail!(Error::NumberOfVariablesMismatch);
		}
	}

	let multilinears = multilinears.iter().collect::<Vec<_>>();

	for (i, claim) in sum_claims.into_iter().enumerate() {
		let CompositeSumClaim {
			composition,
			sum: expected_sum,
			..
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

pub struct RegularSumcheckProver<'a, FDomain, P, Composition, M, Backend>
where
	FDomain: Field,
	P: PackedField,
	M: MultilinearPoly<P> + Send + Sync,
	Backend: ComputationBackend,
{
	n_vars: usize,
	state: ProverState<'a, FDomain, P, M, Backend>,
	compositions: Vec<Composition>,
	domains: Vec<InterpolationDomain<FDomain>>,
}

impl<'a, F, FDomain, P, Composition, M, Backend>
	RegularSumcheckProver<'a, FDomain, P, Composition, M, Backend>
where
	F: Field,
	FDomain: Field,
	P: PackedField<Scalar = F> + PackedExtension<FDomain>,
	Composition: CompositionPoly<P>,
	M: MultilinearPoly<P> + Send + Sync,
	Backend: ComputationBackend,
{
	#[instrument(skip_all, level = "debug", name = "RegularSumcheckProver::new")]
	pub fn new(
		evaluation_order: EvaluationOrder,
		multilinears: Vec<M>,
		composite_claims: impl IntoIterator<Item = CompositeSumClaim<F, Composition>>,
		evaluation_domain_factory: impl EvaluationDomainFactory<FDomain>,
		switchover_fn: impl Fn(usize) -> usize,
		backend: &'a Backend,
	) -> Result<Self, Error> {
		let n_vars = equal_n_vars_check(&multilinears)?;
		let composite_claims = composite_claims.into_iter().collect::<Vec<_>>();

		#[cfg(feature = "debug_validate_sumcheck")]
		{
			let composite_claims = composite_claims
				.iter()
				.map(|x| CompositeSumClaim {
					sum: x.sum,
					composition: &x.composition,
				})
				.collect::<Vec<_>>();
			validate_witness(&multilinears, composite_claims)?;
		}

		for claim in &composite_claims {
			if claim.composition.n_vars() != multilinears.len() {
				bail!(Error::InvalidComposition {
					actual: claim.composition.n_vars(),
					expected: multilinears.len(),
				});
			}
		}

		let claimed_sums = composite_claims
			.iter()
			.map(|composite_claim| composite_claim.sum)
			.collect();

		let domains = interpolation_domains_for_composition_degrees(
			evaluation_domain_factory,
			composite_claims
				.iter()
				.map(|composite_claim| composite_claim.composition.degree()),
		)?;

		let compositions = composite_claims
			.into_iter()
			.map(|claim| claim.composition)
			.collect();

		let nontrivial_evaluation_points = get_nontrivial_evaluation_points(&domains)?;

		let multilinears = multilinears
			.into_iter()
			.map(|multilinear| SumcheckMultilinear::transparent(multilinear, &switchover_fn))
			.collect();

		let state = ProverState::new(
			evaluation_order,
			n_vars,
			multilinears,
			claimed_sums,
			nontrivial_evaluation_points,
			backend,
		)?;

		Ok(Self {
			n_vars,
			state,
			compositions,
			domains,
		})
	}
}

impl<F, FDomain, P, Composition, M, Backend> SumcheckProver<F>
	for RegularSumcheckProver<'_, FDomain, P, Composition, M, Backend>
where
	F: TowerField + ExtensionField<FDomain>,
	FDomain: Field,
	P: PackedField<Scalar = F> + PackedExtension<F, PackedSubfield = P> + PackedExtension<FDomain>,
	Composition: CompositionPoly<P>,
	M: MultilinearPoly<P> + Send + Sync,
	Backend: ComputationBackend,
{
	fn n_vars(&self) -> usize {
		self.n_vars
	}

	fn evaluation_order(&self) -> EvaluationOrder {
		self.state.evaluation_order()
	}

	#[instrument("RegularSumcheckProver::fold", skip_all, level = "debug")]
	fn fold(&mut self, challenge: F) -> Result<(), Error> {
		self.state.fold(challenge)?;
		Ok(())
	}

	#[instrument("RegularSumcheckProver::execute", skip_all, level = "debug")]
	fn execute(&mut self, batch_coeff: F) -> Result<RoundCoeffs<F>, Error> {
		let evaluators = izip!(&self.compositions, &self.domains)
			.map(|(composition, interpolation_domain)| {
				let composition_at_infinity =
					ArithCircuitPoly::new(composition.expression().leading_term());

				RegularSumcheckEvaluator {
					composition,
					composition_at_infinity,
					interpolation_domain,
					_marker: PhantomData,
				}
			})
			.collect::<Vec<_>>();

		let round_evals = self.state.calculate_round_evals(&evaluators)?;
		self.state
			.calculate_round_coeffs_from_evals(&evaluators, batch_coeff, round_evals)
	}

	fn finish(self: Box<Self>) -> Result<Vec<F>, Error> {
		self.state.finish()
	}
}

struct RegularSumcheckEvaluator<'a, P, FDomain, Composition>
where
	P: PackedField,
	FDomain: Field,
{
	composition: &'a Composition,
	composition_at_infinity: ArithCircuitPoly<P::Scalar>,
	interpolation_domain: &'a InterpolationDomain<FDomain>,
	_marker: PhantomData<P>,
}

impl<F, P, FDomain, Composition> SumcheckEvaluator<P, Composition>
	for RegularSumcheckEvaluator<'_, P, FDomain, Composition>
where
	F: TowerField + ExtensionField<FDomain>,
	P: PackedField<Scalar = F> + PackedExtension<F, PackedSubfield = P> + PackedExtension<FDomain>,
	FDomain: Field,
	Composition: CompositionPoly<P>,
{
	fn eval_point_indices(&self) -> Range<usize> {
		// NB: We skip evaluation of $r(X)$ at $X = 0$ as it is derivable from the
		// current_round_sum - $r(1)$.
		1..self.composition.degree() + 1
	}

	fn process_subcube_at_eval_point(
		&self,
		_subcube_vars: usize,
		_subcube_index: usize,
		is_infinity_point: bool,
		batch_query: &RowsBatchRef<P>,
	) -> P {
		let row_len = batch_query.row_len();

		stackalloc_with_default(row_len, |evals| {
			if is_infinity_point {
				self.composition_at_infinity
					.batch_evaluate(batch_query, evals)
					.expect("correct by query construction invariant");
			} else {
				self.composition
					.batch_evaluate(batch_query, evals)
					.expect("correct by query construction invariant");
			}

			evals.iter().copied().sum()
		})
	}

	fn composition(&self) -> &Composition {
		self.composition
	}

	fn eq_ind_partial_eval(&self) -> Option<&[P]> {
		None
	}
}

impl<F, P, FDomain, Composition> SumcheckInterpolator<F>
	for RegularSumcheckEvaluator<'_, P, FDomain, Composition>
where
	F: Field,
	P: PackedField<Scalar = F> + PackedExtension<FDomain>,
	FDomain: Field,
{
	fn round_evals_to_coeffs(
		&self,
		last_round_sum: F,
		mut round_evals: Vec<F>,
	) -> Result<Vec<F>, PolynomialError> {
		// Given $r(1), \ldots, r(d+1)$, letting $s$ be the current round's claimed sum,
		// we can compute $r(0)$ using the identity $r(0) = s - r(1)$
		round_evals.insert(0, last_round_sum - round_evals[0]);

		if round_evals.len() > 3 {
			// SumcheckRoundCalculator orders interpolation points as 0, 1, "infinity", then
			// subspace points. InterpolationDomain expects "infinity" at the last position, thus
			// reordering is needed. Putting "special" evaluation points at the beginning of
			// domain allows benefitting from faster/skipped interpolation even in case of mixed
			// degree compositions .
			let infinity_round_eval = round_evals.remove(2);
			round_evals.push(infinity_round_eval);
		}

		let coeffs = self.interpolation_domain.interpolate(&round_evals)?;
		Ok(coeffs)
	}
}
