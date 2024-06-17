// Copyright 2024 Ulvetanna Inc.

//! Batch proving and verification of the zerocheck protocol.
//!
//! The zerocheck protocol over can be batched over multiple instances by taking random linear
//! combinations over the claimed sums and polynomials. When the zerocheck instances are not all
//! over polynomials with the same number of variables, we can still batch them together, sharing
//! later round challenges. Importantly, the verifier samples mixing challenges "just-in-time".
//! That is, the verifier samples mixing challenges for new zerocheck claims over n variables only
//! after the last zerocheck round message has been sent by the prover.

use super::{error::Error, prove::ZerocheckProver, zerocheck::ZerocheckReductor, ZerocheckClaim};
use crate::{
	challenger::{CanObserve, CanSample},
	polynomial::CompositionPoly,
	protocols::{
		abstract_sumcheck::{self, AbstractSumcheckBatchProof, AbstractSumcheckBatchProveOutput},
		evalcheck::EvalcheckClaim,
	},
};
use binius_field::{ExtensionField, Field, PackedField};

pub type ZerocheckBatchProof<F> = AbstractSumcheckBatchProof<F>;
pub type ZerocheckBatchProveOutput<F> = AbstractSumcheckBatchProveOutput<F>;

/// Prove a batched zerocheck instance.
///
/// See module documentation for details.
pub fn batch_prove<'a, F, PW, DomainField, CW, CH>(
	provers: impl IntoIterator<Item = ZerocheckProver<'a, F, PW, DomainField, CW>>,
	challenger: CH,
) -> Result<ZerocheckBatchProveOutput<F>, Error>
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

/// Verify a batched zerocheck instance.
///
/// See module documentation for details.
pub fn batch_verify<F, CH>(
	claims: impl IntoIterator<Item = ZerocheckClaim<F>>,
	proof: ZerocheckBatchProof<F>,
	mut challenger: CH,
) -> Result<Vec<EvalcheckClaim<F>>, Error>
where
	F: Field,
	CH: CanSample<F> + CanObserve<F>,
{
	let claims_vec = claims.into_iter().collect::<Vec<_>>();

	// Ensure all claims have at least one variable
	claims_vec
		.iter()
		.all(|claim| claim.poly.n_vars() > 0)
		.then_some(())
		.ok_or(Error::ZeroVariableClaim)?;

	// Find the maximum number of variables in any claim,
	// while also ensuring there is at least one claim
	let max_n_vars = claims_vec
		.iter()
		.map(|claim| claim.n_vars())
		.max()
		.ok_or(Error::EmptyClaimsArray)?;

	let alphas = challenger.sample_vec(max_n_vars - 1);

	let reductor = ZerocheckReductor { alphas: &alphas };
	abstract_sumcheck::batch_verify(
		claims_vec.into_iter().map(|c| c.into()),
		proof,
		reductor,
		challenger,
	)
}
