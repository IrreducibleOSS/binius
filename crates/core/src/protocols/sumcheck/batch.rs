// Copyright 2024 Ulvetanna Inc.

//! Batch proving and verification of the sumcheck protocol.
//!
//! The sumcheck protocol over can be batched over multiple instances by taking random linear
//! combinations over the claimed sums and polynomials. When the sumcheck instances are not all
//! over polynomials with the same number of variables, we can still batch them together, sharing
//! later round challenges. Importantly, the verifier samples mixing challenges "just-in-time".
//! That is, the verifier samples mixing challenges for new sumcheck claims over n variables only
//! after the last sumcheck round message has been sent by the prover.

use super::{error::Error, prove::SumcheckProver, sumcheck::SumcheckReductor, SumcheckClaim};
use crate::{
	challenger::{CanObserve, CanSample},
	polynomial::{CompositionPoly, MultilinearPoly},
	protocols::{
		abstract_sumcheck::{self, AbstractSumcheckBatchProof, AbstractSumcheckBatchProveOutput},
		evalcheck::EvalcheckClaim,
	},
};
use binius_field::{ExtensionField, Field, PackedField};

pub type SumcheckBatchProof<F> = AbstractSumcheckBatchProof<F>;
pub type SumcheckBatchProveOutput<F> = AbstractSumcheckBatchProveOutput<F>;

/// Prove a batched sumcheck instance.
///
/// See module documentation for details.
pub fn batch_prove<'a, F, PW, DomainField, CW, M, CH>(
	provers: impl IntoIterator<Item = SumcheckProver<'a, F, PW, DomainField, CW, M>>,
	challenger: CH,
) -> Result<SumcheckBatchProveOutput<F>, Error>
where
	F: Field + From<PW::Scalar>,
	PW: PackedField,
	PW::Scalar: From<F> + ExtensionField<DomainField>,
	DomainField: Field,
	CW: CompositionPoly<PW>,
	M: MultilinearPoly<PW> + Sync + Send,
	CH: CanObserve<F> + CanSample<F>,
{
	abstract_sumcheck::batch_prove(provers, challenger)
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
	let sumcheck_reductor = SumcheckReductor;
	abstract_sumcheck::batch_verify(claims, proof, sumcheck_reductor, challenger)
}
