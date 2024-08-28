// Copyright 2023 Ulvetanna Inc.

use crate::challenger::{CanObserve, CanSample, CanSampleBits};
use binius_field::{ExtensionField, PackedField};
use binius_math::polynomial::MultilinearExtension;
use std::ops::Deref;

pub trait PolyCommitScheme<P, FE>
where
	P: PackedField,
	FE: ExtensionField<P::Scalar>,
{
	type Commitment: Clone;
	type Committed;
	type Proof;
	type Error: std::error::Error + Send + Sync + 'static;

	fn n_vars(&self) -> usize;

	/// Commit to a batch of polynomials
	fn commit<Data>(
		&self,
		polys: &[MultilinearExtension<P, Data>],
	) -> Result<(Self::Commitment, Self::Committed), Self::Error>
	where
		Data: Deref<Target = [P]> + Send + Sync;

	/// Generate an evaluation proof at a *random* challenge point.
	fn prove_evaluation<Data, CH>(
		&self,
		challenger: &mut CH,
		committed: &Self::Committed,
		polys: &[MultilinearExtension<P, Data>],
		query: &[FE],
	) -> Result<Self::Proof, Self::Error>
	where
		Data: Deref<Target = [P]> + Send + Sync,
		CH: CanObserve<FE> + CanObserve<Self::Commitment> + CanSample<FE> + CanSampleBits<usize>;

	/// Verify an evaluation proof at a *random* challenge point.
	fn verify_evaluation<CH>(
		&self,
		challenger: &mut CH,
		commitment: &Self::Commitment,
		query: &[FE],
		proof: Self::Proof,
		values: &[FE],
	) -> Result<(), Self::Error>
	where
		CH: CanObserve<FE> + CanObserve<Self::Commitment> + CanSample<FE> + CanSampleBits<usize>;

	/// Return the byte-size of a proof.
	fn proof_size(&self, n_polys: usize) -> usize;
}
