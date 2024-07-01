// Copyright 2024 Ulvetanna Inc.

use binius_field::{ExtensionField, Field, PackedField};
use p3_challenger::{CanObserve, CanSample};

use crate::{
	polynomial::CompositionPoly,
	protocols::abstract_sumcheck::{
		self, AbstractSumcheckBatchProof, AbstractSumcheckBatchProveOutput, ReducedClaim,
	},
};

use super::{
	gkr_sumcheck::{GkrSumcheckClaim, GkrSumcheckReductor},
	Error, GkrSumcheckProver,
};

pub type GkrSumcheckBatchProof<F> = AbstractSumcheckBatchProof<F>;
pub type GkrSumcheckBatchProveOutput<F> = AbstractSumcheckBatchProveOutput<F>;

/// Prove a batched GkrSumcheck instance.
///
/// See module documentation for details.
pub fn batch_prove<'a, F, PW, DomainField, CW, CH>(
	provers: impl IntoIterator<Item = GkrSumcheckProver<'a, F, PW, DomainField, CW>>,
	challenger: CH,
) -> Result<GkrSumcheckBatchProveOutput<F>, Error>
where
	F: Field + From<PW::Scalar>,
	PW: PackedField,
	PW::Scalar: From<F> + ExtensionField<DomainField>,
	DomainField: Field,
	CW: CompositionPoly<PW::Scalar>,
	CH: CanObserve<F> + CanSample<F>,
{
	abstract_sumcheck::batch_prove(provers, challenger)
}

/// Verify a batched GkrSumcheck instance.
///
/// See module documentation for details.
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

	let reductor = GkrSumcheckReductor {
		gkr_challenge_point: &gkr_challenge_point,
	};

	abstract_sumcheck::batch_verify(
		claims_vec.into_iter().map(|c| c.into()),
		proof,
		reductor,
		challenger,
	)
}
