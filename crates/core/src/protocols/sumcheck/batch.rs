// Copyright 2024 Ulvetanna Inc.

//! Batch proving and verification of the sumcheck protocol.
//!
//! The sumcheck protocol over can be batched over multiple instances by taking random linear
//! combinations over the claimed sums and polynomials. When the sumcheck instances are not all
//! over polynomials with the same number of variables, we can still batch them together, sharing
//! later round challenges. Importantly, the verifier samples mixing challenges "just-in-time".
//! That is, the verifier samples mixing challenges for new sumcheck claims over n variables only
//! after the last sumcheck round message has been sent by the prover.

use super::{error::Error, prove::SumcheckProversState, sumcheck::SumcheckReductor, SumcheckClaim};
use crate::{
	challenger::{CanObserve, CanSample},
	oracle::OracleId,
	polynomial::EvaluationDomainFactory,
	protocols::{
		abstract_sumcheck::{
			self, finalize_evalcheck_claim, AbstractSumcheckBatchProof,
			AbstractSumcheckBatchProveOutput, AbstractSumcheckClaim, AbstractSumcheckWitness,
		},
		evalcheck::EvalcheckClaim,
	},
};
use binius_field::{ExtensionField, Field, PackedExtension};

pub type SumcheckBatchProof<F> = AbstractSumcheckBatchProof<F>;

#[derive(Debug)]
pub struct SumcheckBatchProveOutput<F: Field> {
	pub evalcheck_claims: Vec<EvalcheckClaim<F>>,
	pub proof: SumcheckBatchProof<F>,
}

/// Prove a batched sumcheck instance.
///
/// See module documentation for details.
pub fn batch_prove<F, PW, DomainField, CH>(
	sumchecks: impl IntoIterator<
		Item = (SumcheckClaim<F>, impl AbstractSumcheckWitness<PW, MultilinearId = OracleId>),
	>,
	evaluation_domain_factory: impl EvaluationDomainFactory<DomainField>,
	switchover_fn: impl Fn(usize) -> usize + 'static,
	challenger: CH,
) -> Result<SumcheckBatchProveOutput<F>, Error>
where
	F: Field,
	DomainField: Field,
	PW: PackedExtension<DomainField, Scalar: From<F> + Into<F> + ExtensionField<DomainField>>,

	CH: CanSample<F> + CanObserve<F>,
{
	let sumchecks = sumchecks.into_iter().collect::<Vec<_>>();
	let n_vars = sumchecks
		.iter()
		.map(|(claim, _)| claim.n_vars())
		.max()
		.unwrap_or(0);

	let mut provers_state = SumcheckProversState::<F, PW, DomainField, _, _>::new(
		n_vars,
		evaluation_domain_factory,
		switchover_fn,
	);

	let oracles = sumchecks
		.iter()
		.map(|(claim, _)| claim.poly.clone())
		.collect::<Vec<_>>();

	let AbstractSumcheckBatchProveOutput {
		proof,
		reduced_claims,
	} = abstract_sumcheck::batch_prove(sumchecks, &mut provers_state, challenger)?;

	let evalcheck_claims = reduced_claims
		.into_iter()
		.zip(oracles)
		.map(|(rc, o)| finalize_evalcheck_claim(&o, rc))
		.collect::<Result<_, _>>()?;

	Ok(SumcheckBatchProveOutput {
		evalcheck_claims,
		proof,
	})
}

/// Verify a batched sumcheck instance.
///
/// See module documentation for details.
pub fn batch_verify<F, CH>(
	claims: impl IntoIterator<Item = SumcheckClaim<F>>,
	proof: SumcheckBatchProof<F>,
	challenger: CH,
) -> Result<Vec<EvalcheckClaim<F>>, Error>
where
	F: Field,
	CH: CanSample<F> + CanObserve<F>,
{
	let claims_vec = claims.into_iter().collect::<Vec<_>>();
	let max_individual_degree = claims_vec
		.iter()
		.map(|claim| claim.max_individual_degree())
		.max()
		.unwrap_or(0);

	let sumcheck_reductor = SumcheckReductor {
		max_individual_degree,
	};

	let reduced_claims =
		abstract_sumcheck::batch_verify(claims_vec.clone(), proof, sumcheck_reductor, challenger)?;

	let evalcheck_claims = reduced_claims
		.into_iter()
		.zip(claims_vec)
		.map(|(rc, c)| finalize_evalcheck_claim(&c.poly, rc))
		.collect::<Result<_, _>>()?;

	Ok(evalcheck_claims)
}
