// Copyright 2024-2025 Irreducible Inc.

use binius_field::{ExtensionField, Field, PackedExtension, PackedField};
use binius_hal::ComputationBackend;
use binius_math::{
	CompositionPolyOS, EvaluationDomainFactory, InterpolationDomain, MultilinearPoly,
};
use binius_utils::bail;
use itertools::izip;
use tracing::instrument;

use super::{
	batch_prove::SumcheckProver, prover_state::ProverState,
	regular_sumcheck::RegularSumcheckEvaluator,
};
use crate::protocols::sumcheck::{
	common::{CompositeSumClaim, RoundCoeffs},
	error::Error,
};

/// This prover processes the polynomial from the highest to the lowest variable,  
/// which allows minimizing interaction with Scalars instead of PackedFields  
/// compared to [super::RegularSumcheckProver].  
pub struct HighToLowSumcheckProver<'a, FDomain, P, Composition, M, Backend>
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
	HighToLowSumcheckProver<'a, FDomain, P, Composition, M, Backend>
where
	F: Field + ExtensionField<FDomain>,
	FDomain: Field,
	P: PackedField<Scalar = F> + PackedExtension<F, PackedSubfield = P> + PackedExtension<FDomain>,
	Composition: CompositionPolyOS<P>,
	M: MultilinearPoly<P> + Send + Sync,
	Backend: ComputationBackend,
{
	#[instrument(skip_all, level = "debug", name = "HighToLowSumcheckProver::new")]
	pub fn new(
		multilinears: Vec<M>,
		composite_claims: impl IntoIterator<Item = CompositeSumClaim<F, Composition>>,
		evaluation_domain_factory: impl EvaluationDomainFactory<FDomain>,
		backend: &'a Backend,
	) -> Result<Self, Error> {
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
			super::regular_sumcheck::validate_witness(&multilinears, composite_claims)?;
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

		let domains = composite_claims
			.iter()
			.map(|composite_claim| {
				let degree = composite_claim.composition.degree();
				let domain = evaluation_domain_factory.create(degree + 1)?;
				Ok(domain.into())
			})
			.collect::<Result<Vec<InterpolationDomain<FDomain>>, _>>()
			.map_err(Error::MathError)?;

		let compositions = composite_claims
			.into_iter()
			.map(|claim| claim.composition)
			.collect();

		let evaluation_points = domains
			.iter()
			.max_by_key(|domain| domain.size())
			.map_or_else(|| Vec::new(), |domain| domain.finite_points().to_vec());

		let state = ProverState::new_with_big_field(
			multilinears,
			claimed_sums,
			evaluation_points,
			backend,
		)?;
		let n_vars = state.n_vars();

		Ok(Self {
			n_vars,
			state,
			compositions,
			domains,
		})
	}
}

impl<F, FDomain, P, Composition, M, Backend> SumcheckProver<F>
	for HighToLowSumcheckProver<'_, FDomain, P, Composition, M, Backend>
where
	F: Field + ExtensionField<FDomain>,
	FDomain: Field,
	P: PackedField<Scalar = F> + PackedExtension<F, PackedSubfield = P> + PackedExtension<FDomain>,
	Composition: CompositionPolyOS<P>,
	M: MultilinearPoly<P> + Send + Sync,
	Backend: ComputationBackend,
{
	fn n_vars(&self) -> usize {
		self.n_vars
	}

	#[instrument("HighToLowSumcheckProver::fold", skip_all, level = "debug")]
	fn fold(&mut self, challenge: F) -> Result<(), Error> {
		self.state.high_to_low_fold(challenge)?;
		Ok(())
	}

	#[instrument("HighToLowSumcheckProver::execute", skip_all, level = "debug")]
	fn execute(&mut self, batch_coeff: F) -> Result<RoundCoeffs<F>, Error> {
		let evaluators = izip!(&self.compositions, &self.domains)
			.map(|(composition, interpolation_domain)| {
				RegularSumcheckEvaluator::new(composition, interpolation_domain)
			})
			.collect::<Vec<_>>();

		let evals = self.state.high_to_low_calculate_round_evals(&evaluators)?;
		self.state
			.calculate_round_coeffs_from_evals(&evaluators, batch_coeff, evals)
	}

	fn finish(self: Box<Self>) -> Result<Vec<F>, Error> {
		self.state.finish()
	}
}
