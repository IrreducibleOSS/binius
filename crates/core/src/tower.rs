// Copyright 2024-2025 Irreducible Inc.

//! Traits for working with field towers.

use std::marker::PhantomData;

use binius_field::{
	arch::OptimalUnderlier,
	as_packed_field::{PackScalar, PackedType},
	linear_transformation::{PackedTransformationFactory, Transformation},
	polyval::{
		AES_TO_POLYVAL_TRANSFORMATION, BINARY_TO_POLYVAL_TRANSFORMATION,
		POLYVAL_TO_AES_TRANSFORMARION, POLYVAL_TO_BINARY_TRANSFORMATION,
	},
	underlier::UnderlierType,
	AESTowerField128b, AESTowerField16b, AESTowerField32b, AESTowerField64b, AESTowerField8b,
	BinaryField128b, BinaryField128bPolyval, BinaryField16b, BinaryField1b, BinaryField32b,
	BinaryField64b, BinaryField8b, ExtensionField, PackedExtension, PackedField, RepackedExtension,
	TowerField,
};
use trait_set::trait_set;

/// A trait that groups a family of related [`TowerField`]s as associated types.
pub trait TowerFamily: Sized {
	type B1: TowerField + TryFrom<Self::B128>;
	type B8: TowerField + TryFrom<Self::B128> + ExtensionField<Self::B1>;
	type B16: TowerField + TryFrom<Self::B128> + ExtensionField<Self::B1> + ExtensionField<Self::B8>;
	type B32: TowerField
		+ TryFrom<Self::B128>
		+ ExtensionField<Self::B1>
		+ ExtensionField<Self::B8>
		+ ExtensionField<Self::B16>;
	type B64: TowerField
		+ TryFrom<Self::B128>
		+ ExtensionField<Self::B1>
		+ ExtensionField<Self::B8>
		+ ExtensionField<Self::B16>
		+ ExtensionField<Self::B32>;
	type B128: TowerField
		+ ExtensionField<Self::B1>
		+ ExtensionField<Self::B8>
		+ ExtensionField<Self::B16>
		+ ExtensionField<Self::B32>
		+ ExtensionField<Self::B64>;
}

pub trait ProverTowerFamily: TowerFamily {
	type FastB128: TowerField + From<Self::B128> + Into<Self::B128> + ExtensionField<Self::B1>;

	fn packed_transformation_to_fast<Top, FastTop>() -> impl Transformation<Top, FastTop>
	where
		Top: PackedTop<Self> + PackedTransformationFactory<FastTop>,
		FastTop: PackedField<Scalar = Self::FastB128>;

	fn packed_transformation_from_fast<FastTop, Top>() -> impl Transformation<FastTop, Top>
	where
		FastTop: PackedTransformationFactory<Top>,
		Top: PackedField<Scalar = Self::B128>;
}

pub trait PackedTowerFamily {
	type Tower: TowerFamily;

	type PackedB1: PackedField<Scalar = <Self::Tower as TowerFamily>::B1>
		+ PackedExtension<<Self::Tower as TowerFamily>::B1, PackedSubfield = Self::PackedB1>;
	type PackedB8: PackedField<Scalar = <Self::Tower as TowerFamily>::B8>
		+ PackedExtension<<Self::Tower as TowerFamily>::B8, PackedSubfield = Self::PackedB8>
		+ RepackedExtension<Self::PackedB1>;
	type PackedB16: PackedField<Scalar = <Self::Tower as TowerFamily>::B16>
		+ PackedExtension<<Self::Tower as TowerFamily>::B16, PackedSubfield = Self::PackedB16>
		+ RepackedExtension<Self::PackedB8>
		+ RepackedExtension<Self::PackedB1>;
	type PackedB32: PackedField<Scalar = <Self::Tower as TowerFamily>::B32>
		+ PackedExtension<<Self::Tower as TowerFamily>::B32, PackedSubfield = Self::PackedB32>
		+ RepackedExtension<Self::PackedB16>
		+ RepackedExtension<Self::PackedB8>
		+ RepackedExtension<Self::PackedB1>;
	type PackedB64: PackedField<Scalar = <Self::Tower as TowerFamily>::B64>
		+ PackedExtension<<Self::Tower as TowerFamily>::B64, PackedSubfield = Self::PackedB64>
		+ RepackedExtension<Self::PackedB32>
		+ RepackedExtension<Self::PackedB16>
		+ RepackedExtension<Self::PackedB8>
		+ RepackedExtension<Self::PackedB1>;
	type PackedB128: PackedField<Scalar = <Self::Tower as TowerFamily>::B128>
		+ PackedExtension<<Self::Tower as TowerFamily>::B128, PackedSubfield = Self::PackedB128>
		+ RepackedExtension<Self::PackedB64>
		+ RepackedExtension<Self::PackedB32>
		+ RepackedExtension<Self::PackedB16>
		+ RepackedExtension<Self::PackedB8>
		+ RepackedExtension<Self::PackedB1>;
}
/// The canonical Fan-Paar tower family.
#[derive(Debug)]
pub struct CanonicalTowerFamily;

impl TowerFamily for CanonicalTowerFamily {
	type B1 = BinaryField1b;
	type B8 = BinaryField8b;
	type B16 = BinaryField16b;
	type B32 = BinaryField32b;
	type B64 = BinaryField64b;
	type B128 = BinaryField128b;
}

impl ProverTowerFamily for CanonicalTowerFamily {
	type FastB128 = BinaryField128bPolyval;

	fn packed_transformation_to_fast<Top, FastTop>() -> impl Transformation<Top, FastTop>
	where
		Top: PackedTop<Self> + PackedTransformationFactory<FastTop>,
		FastTop: PackedField<Scalar = Self::FastB128>,
	{
		Top::make_packed_transformation(BINARY_TO_POLYVAL_TRANSFORMATION)
	}

	fn packed_transformation_from_fast<FastTop, Top>() -> impl Transformation<FastTop, Top>
	where
		FastTop: PackedTransformationFactory<Top>,
		Top: PackedField<Scalar = Self::B128>,
	{
		FastTop::make_packed_transformation(POLYVAL_TO_BINARY_TRANSFORMATION)
	}
}

/// The tower defined by Fan-Paar extensions built on top of the Rijndael field.
#[derive(Debug)]
pub struct AESTowerFamily;

impl TowerFamily for AESTowerFamily {
	type B1 = BinaryField1b;
	type B8 = AESTowerField8b;
	type B16 = AESTowerField16b;
	type B32 = AESTowerField32b;
	type B64 = AESTowerField64b;
	type B128 = AESTowerField128b;
}

impl ProverTowerFamily for AESTowerFamily {
	type FastB128 = BinaryField128bPolyval;

	fn packed_transformation_to_fast<Top, FastTop>() -> impl Transformation<Top, FastTop>
	where
		Top: PackedTop<Self> + PackedTransformationFactory<FastTop>,
		FastTop: PackedField<Scalar = Self::FastB128>,
	{
		Top::make_packed_transformation(AES_TO_POLYVAL_TRANSFORMATION)
	}

	fn packed_transformation_from_fast<FastTop, Top>() -> impl Transformation<FastTop, Top>
	where
		FastTop: PackedTransformationFactory<Top>,
		Top: PackedField<Scalar = Self::B128>,
	{
		FastTop::make_packed_transformation(POLYVAL_TO_AES_TRANSFORMARION)
	}
}

trait_set! {
	/// An underlier with associated packed types for fields in a tower.
	pub trait TowerUnderlier<Tower: TowerFamily> =
		UnderlierType
		+ PackScalar<Tower::B1>
		+ PackScalar<Tower::B8>
		+ PackScalar<Tower::B16>
		+ PackScalar<Tower::B32>
		+ PackScalar<Tower::B64>
		+ PackScalar<Tower::B128>;

	pub trait ProverTowerUnderlier<Tower: ProverTowerFamily> =
		TowerUnderlier<Tower> + PackScalar<Tower::FastB128>;

	/// A packed field type that is the top packed field in a tower.
	pub trait PackedTop<Tower: TowerFamily> =
		PackedField<Scalar=Tower::B128>
		+ PackedExtension<Tower::B1>
		+ PackedExtension<Tower::B8>
		+ PackedExtension<Tower::B16>
		+ PackedExtension<Tower::B32>
		+ PackedExtension<Tower::B64>
		+ PackedExtension<Tower::B128>;
}

pub struct PackUnderlierFamily<Tower: TowerFamily, Underlier: TowerUnderlier<Tower>> {
	_pd: PhantomData<(Tower, Underlier)>,
}

impl<Tower: TowerFamily, Underlier: TowerUnderlier<Tower>> PackedTowerFamily
	for PackUnderlierFamily<Tower, Underlier>
{
	type Tower = Tower;

	type PackedB1 = PackedType<Underlier, Tower::B1>;
	type PackedB8 = PackedType<Underlier, Tower::B8>;
	type PackedB16 = PackedType<Underlier, Tower::B16>;
	type PackedB32 = PackedType<Underlier, Tower::B32>;
	type PackedB64 = PackedType<Underlier, Tower::B64>;
	type PackedB128 = PackedType<Underlier, Tower::B128>;
}

pub type CanonicalOptimalPackedTowerFamily =
	PackUnderlierFamily<CanonicalTowerFamily, OptimalUnderlier>;
pub type AESOptimalPackedTowerFamily = PackUnderlierFamily<AESTowerFamily, OptimalUnderlier>;
