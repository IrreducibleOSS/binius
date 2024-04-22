// Copyright 2023 Ulvetanna Inc.

use super::error::VerificationError;
use crate::{
	oracle::CompositePolyOracle,
	polynomial::MultilinearComposite,
	protocols::sumcheck::{SumcheckClaim, SumcheckWitness},
	witness::MultilinearWitness,
};
use binius_field::{Field, PackedField, TowerField};
use std::fmt::Debug;

#[derive(Debug)]
pub struct ZerocheckProof;

#[derive(Debug)]

pub struct ZerocheckProveOutput<'a, F: Field, PW: PackedField, CW> {
	pub sumcheck_claim: SumcheckClaim<F>,
	pub sumcheck_witness: SumcheckWitness<PW, CW, MultilinearWitness<'a, PW>>,
	pub zerocheck_proof: ZerocheckProof,
}

#[derive(Debug, Clone)]
pub struct ZerocheckClaim<F: Field> {
	/// Virtual Polynomial Oracle of the function claimed to be zero on hypercube
	pub poly: CompositePolyOracle<F>,
}

/// Polynomial must be representable as a composition of multilinear polynomials
pub type ZerocheckWitness<'a, P, C> = MultilinearComposite<P, C, MultilinearWitness<'a, P>>;

pub fn reduce_zerocheck_claim<F: TowerField>(
	claim: &ZerocheckClaim<F>,
	challenge: Vec<F>,
) -> Result<SumcheckClaim<F>, VerificationError> {
	if claim.poly.n_vars() != challenge.len() + 1 {
		return Err(VerificationError::ChallengeVectorMismatch);
	}

	let sumcheck_claim = SumcheckClaim {
		poly: claim.poly.clone(),
		sum: F::ZERO,
		zerocheck_challenges: Some(challenge),
	};
	Ok(sumcheck_claim)
}
