// Copyright 2023 Ulvetanna Inc.

use crate::{
	field::{ExtensionField, PackedField},
	polynomial::MultilinearExtension,
};
use p3_challenger::{CanObserve, CanSample, CanSampleBits};

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
	fn commit(
		&self,
		polys: &[MultilinearExtension<P>],
	) -> Result<(Self::Commitment, Self::Committed), Self::Error>;

	/// Generate an evaluation proof at a *random* challenge point.
	fn prove_evaluation<CH>(
		&self,
		challenger: &mut CH,
		committed: &Self::Committed,
		polys: &[MultilinearExtension<P>],
		query: &[FE],
	) -> Result<Self::Proof, Self::Error>
	where
		CH: CanObserve<FE> + CanSample<FE> + CanSampleBits<usize>;

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
		CH: CanObserve<FE> + CanSample<FE> + CanSampleBits<usize>;

	/// Return the byte-size of a proof.
	fn proof_size(&self, n_polys: usize) -> usize;
}
