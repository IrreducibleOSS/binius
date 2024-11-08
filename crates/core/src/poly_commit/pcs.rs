// Copyright 2023-2024 Irreducible Inc.

use crate::{
	challenger::{CanObserve, CanSample, CanSampleBits},
	transcript::{CanRead, CanWrite},
};
use binius_field::{ExtensionField, PackedField, TowerField};
use binius_hal::ComputationBackend;
use binius_math::MultilinearExtension;
use std::ops::Deref;

pub trait PolyCommitScheme<P, FE>
where
	P: PackedField,
	FE: ExtensionField<P::Scalar> + TowerField,
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
	fn prove_evaluation<Data, Transcript, Backend>(
		&self,
		transcript: &mut Transcript,
		// TODO: this should probably consume committed
		committed: &Self::Committed,
		polys: &[MultilinearExtension<P, Data>],
		query: &[FE],
		backend: &Backend,
	) -> Result<Self::Proof, Self::Error>
	where
		Data: Deref<Target = [P]> + Send + Sync,
		Transcript: CanObserve<FE>
			+ CanObserve<Self::Commitment>
			+ CanSample<FE>
			+ CanSampleBits<usize>
			+ CanWrite,
		Backend: ComputationBackend;

	/// Verify an evaluation proof at a *random* challenge point.
	fn verify_evaluation<Transcript, Backend>(
		&self,
		transcript: &mut Transcript,
		commitment: &Self::Commitment,
		query: &[FE],
		proof: Self::Proof,
		values: &[FE],
		backend: &Backend,
	) -> Result<(), Self::Error>
	where
		Transcript: CanObserve<FE>
			+ CanObserve<Self::Commitment>
			+ CanSample<FE>
			+ CanSampleBits<usize>
			+ CanRead,
		Backend: ComputationBackend;

	/// Return the byte-size of a proof.
	fn proof_size(&self, n_polys: usize) -> usize;
}
