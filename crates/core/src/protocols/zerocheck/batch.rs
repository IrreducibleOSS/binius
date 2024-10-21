// Copyright 2024 Ulvetanna Inc.

//! Batch proving and verification of the zerocheck protocol.
//!
//! The zerocheck protocol over can be batched over multiple instances by taking random linear
//! combinations over the claimed sums and polynomials. When the zerocheck instances are not all
//! over polynomials with the same number of variables, we can still batch them together, sharing
//! later round challenges. Importantly, the verifier samples mixing challenges "just-in-time".
//! That is, the verifier samples mixing challenges for new zerocheck claims over n variables only
//! after the last zerocheck round message has been sent by the prover.

use super::{
	error::Error, prove::ZerocheckProversState, zerocheck::ZerocheckReductor, ZerocheckClaim,
};
use crate::{
	challenger::{CanObserve, CanSample},
	oracle::OracleId,
	protocols::{
		abstract_sumcheck::{
			self, finalize_evalcheck_claim, AbstractSumcheckBatchProof,
			AbstractSumcheckBatchProveOutput, AbstractSumcheckClaim, AbstractSumcheckWitness,
		},
		evalcheck::EvalcheckClaim,
	},
};
use binius_field::{ExtensionField, Field, PackedExtension};
use binius_hal::ComputationBackend;
use binius_math::EvaluationDomainFactory;
use std::cmp;

pub type ZerocheckBatchProof<F> = AbstractSumcheckBatchProof<F>;

#[derive(Debug)]
pub struct ZerocheckBatchProveOutput<F: Field> {
	pub evalcheck_claims: Vec<EvalcheckClaim<F>>,
	pub proof: ZerocheckBatchProof<F>,
}

/// Prove a batched zerocheck instance.
///
/// See module documentation for details.
pub fn batch_prove<F, PW, DomainField, CH, Backend>(
	zerochecks: impl IntoIterator<
		Item = (ZerocheckClaim<F>, impl AbstractSumcheckWitness<PW, MultilinearId = OracleId>),
	>,
	evaluation_domain_factory: impl EvaluationDomainFactory<DomainField>,
	switchover_fn: impl Fn(usize) -> usize + 'static,
	mixing_challenge: F,
	mut challenger: CH,
	backend: &Backend,
) -> Result<ZerocheckBatchProveOutput<F>, Error>
where
	F: Field,
	DomainField: Field,
	PW: PackedExtension<DomainField, Scalar: From<F> + Into<F> + ExtensionField<DomainField>>,
	CH: CanSample<F> + CanObserve<F>,
	Backend: ComputationBackend,
{
	let zerochecks = zerochecks.into_iter().collect::<Vec<_>>();
	let n_vars = zerochecks
		.iter()
		.map(|(claim, _)| claim.n_vars())
		.max()
		.unwrap_or(0);

	let zerocheck_challenges = challenger.sample_vec(cmp::max(n_vars, 1) - 1);

	let mut provers_state = ZerocheckProversState::<F, PW, DomainField, _, _, _>::new(
		n_vars,
		evaluation_domain_factory,
		zerocheck_challenges.as_slice(),
		switchover_fn,
		mixing_challenge,
		backend,
	)?;

	let oracles = zerochecks
		.iter()
		.map(|(claim, _)| claim.poly.clone())
		.collect::<Vec<_>>();

	let AbstractSumcheckBatchProveOutput {
		proof,
		reduced_claims,
	} = abstract_sumcheck::batch_prove(zerochecks, &mut provers_state, challenger)?;

	let evalcheck_claims = reduced_claims
		.into_iter()
		.zip(oracles.into_iter())
		.map(|(rc, o)| finalize_evalcheck_claim(&o, rc))
		.collect::<Result<_, _>>()?;

	Ok(ZerocheckBatchProveOutput {
		evalcheck_claims,
		proof,
	})
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
	let poly_oracles = claims_vec
		.iter()
		.map(|c| c.poly.clone())
		.collect::<Vec<_>>();

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

	let max_individual_degree = claims_vec
		.iter()
		.map(|claim| claim.max_individual_degree())
		.max()
		.unwrap_or(0);

	let alphas = challenger.sample_vec(max_n_vars - 1);

	let reductor = ZerocheckReductor {
		max_individual_degree,
		alphas: &alphas,
	};
	let reduced_claims = abstract_sumcheck::batch_verify(claims_vec, proof, reductor, challenger)?;

	let evalcheck_claims = reduced_claims
		.into_iter()
		.zip(poly_oracles)
		.map(|(reduced_claim, poly_oracle)| finalize_evalcheck_claim(&poly_oracle, reduced_claim))
		.collect::<Result<_, _>>()?;

	Ok(evalcheck_claims)
}
