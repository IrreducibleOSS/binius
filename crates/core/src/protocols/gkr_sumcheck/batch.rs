// Copyright 2024 Ulvetanna Inc.

use binius_field::{ExtensionField, Field, PackedField};
use p3_challenger::{CanObserve, CanSample};
use tracing::instrument;

use crate::{
	polynomial::{CompositionPoly, EvaluationDomainFactory, MultilinearPoly},
	protocols::abstract_sumcheck::{
		self, AbstractSumcheckBatchProof, AbstractSumcheckBatchProveOutput, AbstractSumcheckClaim,
		ReducedClaim,
	},
};

use super::{
	gkr_sumcheck::{GkrSumcheckClaim, GkrSumcheckReductor, GkrSumcheckWitness},
	prove::GkrSumcheckProversState,
	Error,
};

pub type GkrSumcheckBatchProof<F> = AbstractSumcheckBatchProof<F>;
pub type GkrSumcheckBatchProveOutput<F> = AbstractSumcheckBatchProveOutput<F>;

/// Prove a batched GkrSumcheck instance.
///
/// See module documentation for details.
#[instrument(skip_all, name = "gkr_sumcheck::batch_prove", level = "debug")]
pub fn batch_prove<F, PW, DomainField, CW, M, CH>(
	gkr_sumchecks: impl IntoIterator<Item = (GkrSumcheckClaim<F>, GkrSumcheckWitness<PW, CW, M>)>,
	evaluation_domain_factory: impl EvaluationDomainFactory<DomainField>,
	switchover_fn: impl Fn(usize) -> usize + 'static,
	challenger: CH,
) -> Result<GkrSumcheckBatchProveOutput<F>, Error>
where
	F: Field,
	DomainField: Field,
	PW: PackedField<Scalar: From<F> + Into<F> + ExtensionField<DomainField>>,
	CW: CompositionPoly<PW>,
	M: MultilinearPoly<PW> + Clone + Send + Sync,
	CH: CanObserve<F> + CanSample<F>,
{
	let gkr_sumchecks = gkr_sumchecks.into_iter().collect::<Vec<_>>();
	let n_vars = gkr_sumchecks
		.iter()
		.map(|(claim, _)| claim.n_vars())
		.max()
		.unwrap_or(0);

	let gkr_round_challenge = gkr_sumchecks
		.first()
		.map(|(claim, _)| claim.r.clone())
		.ok_or(Error::EmptyClaimsArray)?;

	let mut provers_state = GkrSumcheckProversState::<F, PW, DomainField, _, _, _>::new(
		n_vars,
		evaluation_domain_factory,
		gkr_round_challenge.as_slice(),
		switchover_fn,
	)?;

	abstract_sumcheck::batch_prove(gkr_sumchecks, &mut provers_state, challenger)
}

/// Verify a batched GkrSumcheck instance.
///
/// See module documentation for details.
#[instrument(skip_all, name = "gkr_sumcheck::batch_verify", level = "debug")]
pub fn batch_verify<F, CH>(
	claims: impl IntoIterator<Item = GkrSumcheckClaim<F>>,
	proof: GkrSumcheckBatchProof<F>,
	challenger: CH,
) -> Result<Vec<ReducedClaim<F>>, Error>
where
	F: Field,
	CH: CanSample<F> + CanObserve<F>,
{
	let claims_vec = claims.into_iter().collect::<Vec<_>>();
	if claims_vec.is_empty() {
		return Err(Error::EmptyClaimsArray);
	}

	let gkr_challenge_point = claims_vec[0].r.clone();

	// Ensure all claims have the same gkr_challenge
	if !claims_vec
		.iter()
		.all(|claim| claim.r == gkr_challenge_point)
	{
		return Err(Error::MismatchedGkrChallengeInClaimsBatch);
	}

	let max_individual_degree = claims_vec
		.iter()
		.map(|claim| claim.max_individual_degree())
		.max()
		.unwrap_or(0);

	let reductor = GkrSumcheckReductor {
		max_individual_degree,
		gkr_challenge_point: &gkr_challenge_point,
	};

	abstract_sumcheck::batch_verify(claims_vec, proof, reductor, challenger)
}
