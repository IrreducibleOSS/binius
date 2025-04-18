// Copyright 2024-2025 Irreducible Inc.

//! Traits for working with field towers.

use std::marker::PhantomData;

use binius_field::{
	as_packed_field::PackScalar,
	linear_transformation::{PackedTransformationFactory, Transformation},
	polyval::{
		AES_TO_POLYVAL_TRANSFORMATION, BINARY_TO_POLYVAL_TRANSFORMATION,
		POLYVAL_TO_AES_TRANSFORMARION, POLYVAL_TO_BINARY_TRANSFORMATION,
	},
	underlier::UnderlierType,
	AESTowerField128b, AESTowerField16b, AESTowerField32b, AESTowerField64b, AESTowerField8b,
	BinaryField128b, BinaryField128bPolyval, BinaryField16b, BinaryField1b, BinaryField32b,
	BinaryField64b, BinaryField8b, ExtensionField, PackedExtension, PackedField, PackedSubfield,
	TowerField,
};
use getset::Getters;
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

pub trait TowerFamilyTransform {
	type FromTower: TowerFamily;
	type FromTop: PackedTop<Self::FromTower>;
	type ToTower: TowerFamily;
	type ToTop: PackedTop<Self::ToTower>;

	// type B1: Transformation<
	// 	PackedSubfield<Self::FromTop, <Self::FromTower as TowerFamily>::B1>,
	// 	PackedSubfield<Self::ToTop, <Self::ToTower as TowerFamily>::B1>,
	// >;

	//type B1: Transformation<Self::FromTop, Self::ToTop>;
	type B8: Transformation<Self::FromTop, Self::ToTop>;
	type B16: Transformation<Self::FromTop, Self::ToTop>;
	// type B16: Transformation<
	// 	PackedSubfield<Self::FromTop, <Self::FromTower as TowerFamily>::B16>,
	// 	PackedSubfield<Self::ToTop, <Self::ToTower as TowerFamily>::B16>,
	// >;

	//fn new_b1_transformation() -> Self::B1;

	// fn new_b1_transformation() -> impl Transformation<
	// 	PackedSubfield<Self::FromTop, <Self::FromTower as TowerFamily>::B1>,
	// 	PackedSubfield<Self::ToTop, <Self::ToTower as TowerFamily>::B1>,
	// >;

	fn new_b8_transformation() -> Self::B8;
	// fn new_b8_transformation() -> impl Transformation<
	// 	PackedSubfield<Self::FromTop, <Self::FromTower as TowerFamily>::B8>,
	// 	PackedSubfield<Self::ToTop, <Self::ToTower as TowerFamily>::B8>,
	// >;

	fn new_b16_transformation() -> Self::B16;
	// fn new_b16_transformation() -> impl Transformation<
	// 	PackedSubfield<Self::FromTop, <Self::FromTower as TowerFamily>::B16>,
	// 	PackedSubfield<Self::ToTop, <Self::ToTower as TowerFamily>::B16>,
	// >;

	// fn new_b32_transformation() -> impl Transformation<
	// 	PackedSubfield<Self::FromTop, Self::FromTower::B32>,
	// 	PackedSubfield<Self::ToTop, Self::ToTower::B32>,
	// >;
	//
	// fn new_b64_transformation() -> impl Transformation<
	// 	PackedSubfield<Self::FromTop, Self::FromTower::B64>,
	// 	PackedSubfield<Self::ToTop, Self::ToTower::B64>,
	// >;
	//
	// fn new_b128_transformation() -> impl Transformation<
	// 	PackedSubfield<Self::FromTop, Self::FromTower::B128>,
	// 	PackedSubfield<Self::ToTop, Self::ToTower::B128>,
	// >;
}

// #[derive(Getters)]
// struct TowerFamilyFactoryTransform<TFT: TowerFamilyTransform> {
// 	// FromTower, ToTower, FromTop, ToTop> {
// 	#[getset(get = "pub")]
// 	b1_transformation: TFT::B1,
// 	#[getset(get = "pub")]
// 	b8_transformation: TFT::B8,
// 	#[getset(get = "pub")]
// 	b16_transformation: TFT::B16,
// }

#[derive(Getters)]
#[allow(dead_code)]
struct CanonicalToAESFamilyTransform<FromTop, ToTop>
// where
// 	FromTop: PackedTop<CanonicalTowerFamily>,
// 	ToTop: PackedTop<AESTowerFamily>,
{
	_marker: PhantomData<(FromTop, ToTop)>,
}

impl<FromTop, ToTop> CanonicalToAESFamilyTransform<FromTop, ToTop>
where
	FromTop: PackedTop<CanonicalTowerFamily>,
	ToTop: PackedTop<AESTowerFamily>,
{
	#[allow(dead_code)]
	pub fn new() -> Self {
		Self {
			_marker: PhantomData,
		}
	}
}

impl<FromTop, ToTop> CanonicalToAESFamilyTransform<FromTop, ToTop>
where
	FromTop: PackedExtension<<CanonicalTowerFamily as TowerFamily>::B8>, //PackedTop<CanonicalTowerFamily>
	<FromTop as PackedField>::Scalar: ExtensionField<BinaryField8b>,
	// FromTop: PackedField //PackedTop<CanonicalTowerFamily>
	// 	+ PackedExtension<<CanonicalTowerFamily as TowerFamily>::B8>,
	//ToTop: PackedTop<AESTowerFamily>,
	PackedSubfield<FromTop, <CanonicalTowerFamily as TowerFamily>::B8>: PackedField,
	//PackedTransformationFactory<PackedSubfield<ToTop, <AESTowerFamily as TowerFamily>::B8>>,
{
	// type FromTower = CanonicalTowerFamily;
	// type FromTop = FromTop;
	// type ToTower = AESTowerFamily;
	// type ToTop = ToTop;
	//
	// //type B1 = SubfieldTransformer<BinaryField1b, BinaryField1b, IDTransformation>;
	// type B8 = BinaryToAesTransformation<FromTop, ToTop>;
	// type B16 = BinaryToAesTransformation<FromTop, ToTop>;
	//
	// // fn new_b1_transformation() -> Self::B1 {
	// // 	SubfieldTransformer::new(IDTransformation)
	// // }
	//
	// fn new_b8_transformation() -> Self::B8 {
	// 	make_binary_to_aes_packed_transformer()
	// }
	//
	// fn new_b16_transformation() -> Self::B16 {
	// 	make_binary_to_aes_packed_transformer()
	// }
}
