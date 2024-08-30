// Copyright 2024 Ulvetanna Inc.

use crate::{
	oracle::OracleId,
	protocols::{
		abstract_sumcheck::AbstractSumcheckWitness,
		zerocheck::{
			evaluator::{
				ZerocheckFirstRoundEvaluator, ZerocheckLaterRoundEvaluator,
				ZerocheckSimpleEvaluator,
			},
			Error, SmallerDomainOptimization, ZerocheckClaim, ZerocheckProversState,
		},
	},
};
use binius_field::{ExtensionField, Field, PackedExtension};
use binius_hal::{
	zerocheck::{ZerocheckCpuBackendHelper, ZerocheckRoundInput, ZerocheckRoundParameters},
	ComputationBackend,
};
use binius_math::{EvaluationDomain, EvaluationDomainFactory};
use rayon::prelude::*;
use tracing::instrument;

/// Wrapper that handles Zerocheck round computations using CPU.
pub(crate) struct ZerocheckProverBackendWrapper<'a, F, PW, DomainField, W, EDF, Backend>
where
	F: Field,
	PW: PackedExtension<DomainField, Scalar: From<F> + Into<F> + ExtensionField<DomainField>>,
	DomainField: Field,
	W: AbstractSumcheckWitness<PW>,
	EDF: EvaluationDomainFactory<DomainField>,
	Backend: ComputationBackend,
{
	pub(crate) claim: &'a ZerocheckClaim<F>,
	pub(crate) smaller_domain_optimization: Option<SmallerDomainOptimization<PW, DomainField>>,
	pub(crate) provers_state: &'a ZerocheckProversState<'a, F, PW, DomainField, EDF, W, Backend>,
	pub(crate) domain: &'a EvaluationDomain<DomainField>,
	pub(crate) oracle_ids: &'a [OracleId],
	pub(crate) witness: &'a W,
}

impl<'a, F, PW, DomainField, W, EDF, Backend> ZerocheckCpuBackendHelper<F, PW>
	for ZerocheckProverBackendWrapper<'a, F, PW, DomainField, W, EDF, Backend>
where
	F: Field,
	PW: PackedExtension<DomainField, Scalar: From<F> + Into<F> + ExtensionField<DomainField>>,
	DomainField: Field,
	W: AbstractSumcheckWitness<PW, MultilinearId = OracleId>,
	EDF: EvaluationDomainFactory<DomainField>,
	Backend: ComputationBackend,
{
	#[instrument(skip_all)]
	fn handle_zerocheck_round(
		&mut self,
		params: &ZerocheckRoundParameters,
		input: &ZerocheckRoundInput<F, PW>,
	) -> Result<Vec<PW::Scalar>, binius_hal::Error> {
		let round_coeffs = match (
			self.claim.poly.max_individual_degree(),
			&self.smaller_domain_optimization,
			params.round,
		) {
			(1, _, _) => Ok(vec![PW::Scalar::default()]),
			(_, None, _) => self.compute_round_coeffs_no_smaller_domain(params, input),
			(_, Some(_), 0) => self.compute_round_coeffs_first_round(input),
			(_, Some(_), _) => self.compute_round_coeffs_later_round(params, input),
		};
		round_coeffs.map_err(|err| binius_hal::Error::ZerocheckCpuHandlerError(err.to_string()))
	}
}

impl<'a, F, PW, DomainField, W, EDF, Backend>
	ZerocheckProverBackendWrapper<'a, F, PW, DomainField, W, EDF, Backend>
where
	F: Field,
	PW: PackedExtension<DomainField, Scalar: From<F> + Into<F> + ExtensionField<DomainField>>,
	DomainField: Field,
	W: AbstractSumcheckWitness<PW, MultilinearId = OracleId>,
	EDF: EvaluationDomainFactory<DomainField>,
	Backend: ComputationBackend,
{
	#[instrument(skip_all)]
	fn compute_round_coeffs_no_smaller_domain(
		&mut self,
		params: &ZerocheckRoundParameters,
		input: &ZerocheckRoundInput<F, PW>,
	) -> Result<Vec<PW::Scalar>, Error> {
		let rd_vars = self.claim.n_vars() - params.round;
		let vertex_state_iterator = (0..1 << (rd_vars - 1)).into_par_iter().map(|_i| ());
		let degree = self.claim.poly.max_individual_degree();
		let evaluator = ZerocheckSimpleEvaluator::<PW, DomainField, _> {
			degree,
			eq_ind: self.provers_state.round_eq_ind.to_ref(),
			evaluation_domain: self.domain,
			domain_points: self.domain.points(),
			composition: self.witness.composition(),
			round_zerocheck_challenge: Some(PW::Scalar::from(
				input.zc_challenges[params.round - 1],
			)),
		};
		let round_coeffs = self.provers_state.common.calculate_round_coeffs(
			self.oracle_ids,
			evaluator,
			input.current_round_sum.into(),
			vertex_state_iterator,
		)?;

		Ok(round_coeffs)
	}

	#[instrument(skip_all)]
	fn compute_round_coeffs_first_round(
		&mut self,
		input: &ZerocheckRoundInput<F, PW>,
	) -> Result<Vec<PW::Scalar>, Error> {
		let degree = self.claim.poly.max_individual_degree();
		let smaller_domain_optimization = self.smaller_domain_optimization.as_mut().unwrap();
		let vertex_state_iterator = smaller_domain_optimization
			.round_q
			.par_chunks_exact_mut(degree - 1);
		let evaluator = ZerocheckFirstRoundEvaluator {
			composition: self.witness.composition(),
			domain_points: self.domain.points(),
			evaluation_domain: self.domain,
			degree,
			eq_ind: self.provers_state.round_eq_ind.to_ref(),
			denom_inv: &smaller_domain_optimization.smaller_denom_inv,
		};
		Ok(self.provers_state.common.calculate_round_coeffs(
			self.oracle_ids,
			evaluator,
			input.current_round_sum.into(),
			vertex_state_iterator,
		)?)
	}

	#[instrument(skip_all)]
	fn compute_round_coeffs_later_round(
		&mut self,
		params: &ZerocheckRoundParameters,
		input: &ZerocheckRoundInput<F, PW>,
	) -> Result<Vec<PW::Scalar>, Error> {
		let degree = self.claim.poly.max_individual_degree();
		let smaller_domain_optimization = self.smaller_domain_optimization.as_mut().unwrap();
		let vertex_state_iterator = smaller_domain_optimization
			.round_q
			.par_chunks_exact_mut(degree - 1);
		assert_eq!(input.zc_challenges.len(), params.n_vars - 1);
		let evaluator = ZerocheckLaterRoundEvaluator::<PW, DomainField, _> {
			composition: self.witness.composition(),
			domain_points: self.domain.points(),
			evaluation_domain: self.domain,
			degree,
			eq_ind: self.provers_state.round_eq_ind.to_ref(),
			round_zerocheck_challenge: input.zc_challenges[params.round - 1].into(),
			denom_inv: &smaller_domain_optimization.smaller_denom_inv,
			round_q_bar: smaller_domain_optimization
				.round_q_bar
				.as_ref()
				.expect("round_q_bar is Some after round 0")
				.to_ref(),
		};
		Ok(self.provers_state.common.calculate_round_coeffs(
			self.oracle_ids,
			evaluator,
			input.current_round_sum.into(),
			vertex_state_iterator,
		)?)
	}
}
