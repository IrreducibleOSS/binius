// Copyright 2024 Irreducible Inc.

use crate::{
	poly_commit::{batch_pcs, batch_pcs::BatchPCS, PolyCommitScheme, FRIPCS},
	tower::{PackedTop, TowerFamily, TowerUnderlier},
	transcript::{AdviceReader, CanRead},
};
use binius_field::{as_packed_field::PackedType, PackedFieldIndexable};
use binius_hal::ComputationBackend;
use binius_math::EvaluationDomainFactory;
use p3_challenger::{CanObserve, CanSample, CanSampleBits};
use std::{fmt::Debug, marker::PhantomData};

/// A trait that groups a family of PCSs for different fields in a tower as associated types.
pub trait TowerPCSFamily<Tower: TowerFamily, U: TowerUnderlier<Tower>> {
	type Committed;
	type Commitment: Clone;
	type Error: std::error::Error + Send + Sync + 'static;

	type PCS1: PolyCommitScheme<
		PackedType<U, Tower::B1>,
		Tower::B128,
		Commitment = Self::Commitment,
		Committed = Self::Committed,
		Error = Self::Error,
	>;
	type PCS8: PolyCommitScheme<
		PackedType<U, Tower::B8>,
		Tower::B128,
		Commitment = Self::Commitment,
		Committed = Self::Committed,
		Error = Self::Error,
	>;
	type PCS16: PolyCommitScheme<
		PackedType<U, Tower::B16>,
		Tower::B128,
		Commitment = Self::Commitment,
		Committed = Self::Committed,
		Error = Self::Error,
	>;
	type PCS32: PolyCommitScheme<
		PackedType<U, Tower::B32>,
		Tower::B128,
		Commitment = Self::Commitment,
		Committed = Self::Committed,
		Error = Self::Error,
	>;
	type PCS64: PolyCommitScheme<
		PackedType<U, Tower::B64>,
		Tower::B128,
		Commitment = Self::Commitment,
		Committed = Self::Committed,
		Error = Self::Error,
	>;
	type PCS128: PolyCommitScheme<
		PackedType<U, Tower::B128>,
		Tower::B128,
		Commitment = Self::Commitment,
		Committed = Self::Committed,
		Error = Self::Error,
	>;
}

/// An enum with variants for each PCS in a [`TowerPCSFamily`].
#[derive(Debug)]
pub enum TowerPCS<Tower, U, PCSFamily>
where
	Tower: TowerFamily,
	U: TowerUnderlier<Tower>,
	PCSFamily: TowerPCSFamily<Tower, U>,
{
	B1(PCSFamily::PCS1),
	B8(PCSFamily::PCS8),
	B16(PCSFamily::PCS16),
	B32(PCSFamily::PCS32),
	B64(PCSFamily::PCS64),
	B128(PCSFamily::PCS128),
}

impl<Tower, U, PCSFamily> TowerPCS<Tower, U, PCSFamily>
where
	U: TowerUnderlier<Tower>,
	Tower: TowerFamily,
	PCSFamily: TowerPCSFamily<Tower, U>,
{
	pub fn verify_evaluation<CH, Backend>(
		&self,
		advice: &mut AdviceReader,
		challenger: &mut CH,
		commitment: &PCSFamily::Commitment,
		query: &[FExt<Tower>],
		values: &[FExt<Tower>],
		backend: &Backend,
	) -> Result<(), PCSFamily::Error>
	where
		CH: CanObserve<FExt<Tower>>
			+ CanObserve<PCSFamily::Commitment>
			+ CanSample<FExt<Tower>>
			+ CanSampleBits<usize>
			+ CanRead,
		Backend: ComputationBackend,
	{
		match self {
			Self::B1(pcs) => {
				pcs.verify_evaluation(advice, challenger, commitment, query, values, backend)
			}
			Self::B8(pcs) => {
				pcs.verify_evaluation(advice, challenger, commitment, query, values, backend)
			}
			Self::B16(pcs) => {
				pcs.verify_evaluation(advice, challenger, commitment, query, values, backend)
			}
			Self::B32(pcs) => {
				pcs.verify_evaluation(advice, challenger, commitment, query, values, backend)
			}
			Self::B64(pcs) => {
				pcs.verify_evaluation(advice, challenger, commitment, query, values, backend)
			}
			Self::B128(pcs) => {
				pcs.verify_evaluation(advice, challenger, commitment, query, values, backend)
			}
		}
	}
}

/// The cryptographic extension field that the constraint system protocol is defined over.
pub type FExt<Tower> = <Tower as TowerFamily>::B128;

pub mod standard_pcs {
	use super::*;
	use crate::merkle_tree_vcs::{
		BinaryMerkleTreeProver, BinaryMerkleTreeScheme, MerkleTreeProver, MerkleTreeScheme,
	};
	use binius_field::{PackedExtension, PackedField, TowerField};

	/// The evaluation domain used in sumcheck protocols.
	///
	/// This is fixed to be 8-bits, which is large enough to handle all reasonable sumcheck
	/// constraint degrees, even with a moderate number of skipped rounds using the univariate skip
	/// technique.
	pub type FDomain<Tower> = <Tower as TowerFamily>::B8;

	/// The Reed–Solomon alphabet used for FRI encoding.
	///
	/// This is fixed to be 32-bits, which is large enough to handle trace sizes up to 64 GiB
	/// of committed data.
	pub type FEncode<Tower> = <Tower as TowerFamily>::B32;

	pub type BatchFRIPCS<Tower, U, F, DomainFactory, MerkleProver, VCS> = BatchPCS<
		<PackedType<U, FExt<Tower>> as PackedExtension<F>>::PackedSubfield,
		FExt<Tower>,
		FRIPCS<
			F,
			FDomain<Tower>,
			FEncode<Tower>,
			PackedType<U, FExt<Tower>>,
			DomainFactory,
			MerkleProver,
			VCS,
		>,
	>;

	#[derive(Debug)]
	pub struct FRITowerPCSFamily<Tower, U, DomainFactory, MerkleProver, VCS> {
		_marker: PhantomData<(Tower, U, DomainFactory, MerkleProver, VCS)>,
	}

	impl<Tower, U, DomainFactory, MerkleProver, DigestType, VCS> TowerPCSFamily<Tower, U>
		for FRITowerPCSFamily<Tower, U, DomainFactory, MerkleProver, VCS>
	where
		Tower: TowerFamily,
		U: TowerUnderlier<Tower>,
		Tower::B128: PackedTop<Tower>,
		DomainFactory: EvaluationDomainFactory<Tower::B8>,
		MerkleProver: MerkleTreeProver<Tower::B128, Committed: Send + Sync, Scheme = VCS> + Sync,
		DigestType: PackedField<Scalar: TowerField>,
		VCS: MerkleTreeScheme<Tower::B128, Digest = DigestType, Proof = Vec<DigestType>>,
		PackedType<U, Tower::B128>: PackedFieldIndexable,
	{
		type Commitment = VCS::Digest;
		type Committed = (Vec<PackedType<U, Tower::B128>>, MerkleProver::Committed);
		type Error = batch_pcs::Error;

		type PCS1 = BatchFRIPCS<Tower, U, Tower::B1, DomainFactory, MerkleProver, VCS>;
		type PCS8 = BatchFRIPCS<Tower, U, Tower::B8, DomainFactory, MerkleProver, VCS>;
		type PCS16 = BatchFRIPCS<Tower, U, Tower::B16, DomainFactory, MerkleProver, VCS>;
		type PCS32 = BatchFRIPCS<Tower, U, Tower::B32, DomainFactory, MerkleProver, VCS>;
		type PCS64 = BatchFRIPCS<Tower, U, Tower::B64, DomainFactory, MerkleProver, VCS>;
		type PCS128 = BatchFRIPCS<Tower, U, Tower::B128, DomainFactory, MerkleProver, VCS>;
	}

	pub type FRIMerklePCS<Tower, U, F, Digest, DomainFactory, Hash, Compress> = BatchFRIPCS<
		Tower,
		U,
		F,
		DomainFactory,
		BinaryMerkleTreeProver<Digest, Hash, Compress>,
		BinaryMerkleTreeScheme<Digest, Hash, Compress>,
	>;

	pub type FRIMerkleTowerPCS<Tower, U, Digest, DomainFactory, Hash, Compress> = TowerPCS<
		Tower,
		U,
		FRITowerPCSFamily<
			Tower,
			U,
			DomainFactory,
			BinaryMerkleTreeProver<Digest, Hash, Compress>,
			BinaryMerkleTreeScheme<Digest, Hash, Compress>,
		>,
	>;
}
