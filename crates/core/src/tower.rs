// Copyright 2024-2025 Irreducible Inc.

//! Traits for working with field towers.

use std::marker::PhantomData;

use binius_field::{
	as_packed_field::{PackScalar, PackedType},
	linear_transformation::{IDTransformation, PackedTransformationFactory, Transformation},
	make_binary_to_aes_packed_transformer,
	polyval::{
		AES_TO_POLYVAL_TRANSFORMATION, BINARY_TO_POLYVAL_TRANSFORMATION,
		POLYVAL_TO_AES_TRANSFORMARION, POLYVAL_TO_BINARY_TRANSFORMATION,
	},
	underlier::UnderlierType,
	AESTowerField128b, AESTowerField16b, AESTowerField32b, AESTowerField64b, AESTowerField8b,
	BinaryField128b, BinaryField128bPolyval, BinaryField16b, BinaryField1b, BinaryField32b,
	BinaryField64b, BinaryField8b, BinaryToAesTransformation, ExtensionField, PackedExtension,
	PackedField, PackedSubfield, SubfieldTransformer, TowerField,
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

pub trait TowerTransformFactory {
	type FromTower: TowerFamily;
	type FromTop: PackedTop<Self::FromTower>;
	type ToTower: TowerFamily;
	type ToTop: PackedTop<Self::ToTower>;

	// type B1: Transformation<
	// 	PackedSubfield<Self::FromTop, <Self::FromTower as TowerFamily>::B1>,
	// 	PackedSubfield<Self::ToTop, <Self::ToTower as TowerFamily>::B1>,
	// >;

	type B1: Transformation<Self::FromTop, Self::ToTop>;
	type B8: Transformation<Self::FromTop, Self::ToTop>;
	type B16: Transformation<Self::FromTop, Self::ToTop>;
	// type B16: Transformation<
	// 	PackedSubfield<Self::FromTop, <Self::FromTower as TowerFamily>::B16>,
	// 	PackedSubfield<Self::ToTop, <Self::ToTower as TowerFamily>::B16>,
	// >;

	fn new_b1_transformation() -> Self::B1;

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

#[derive(Getters, Default)]
#[allow(dead_code)]
struct DenseCanonicalToAESFamilyTransform<U> {
	_marker: PhantomData<U>,
}

impl<U> TowerTransformFactory for DenseCanonicalToAESFamilyTransform<U>
where
	U: TowerUnderlier<CanonicalTowerFamily>
		+ TowerUnderlier<AESTowerFamily>
		+ PackScalar<BinaryField8b>
		+ PackScalar<AESTowerField8b>,
	PackedType<U, BinaryField8b>: PackedTransformationFactory<PackedType<U, AESTowerField8b>>,
{
	type FromTower = CanonicalTowerFamily;
	type FromTop = PackedType<U, BinaryField128b>;
	type ToTower = AESTowerFamily;
	type ToTop = PackedType<U, AESTowerField128b>;

	type B1 = SubfieldTransformer<BinaryField1b, BinaryField1b, IDTransformation>;
	type B8 = BinaryToAesTransformation<Self::FromTop, Self::ToTop>;
	type B16 = BinaryToAesTransformation<Self::FromTop, Self::ToTop>;

	fn new_b1_transformation() -> Self::B1 {
		SubfieldTransformer::new(IDTransformation)
	}

	fn new_b8_transformation() -> Self::B8 {
		make_binary_to_aes_packed_transformer::<Self::FromTop, Self::ToTop>()
	}

	fn new_b16_transformation() -> Self::B16 {
		make_binary_to_aes_packed_transformer::<Self::FromTop, Self::ToTop>()
	}
}

#[derive(Default)]
#[allow(dead_code)]
struct CanonicalToAESFamilyTransform<FromTop, ToTop> {
	_marker: PhantomData<(FromTop, ToTop)>,
}

impl<FromTop, ToTop, Packed1b> TowerTransformFactory
	for CanonicalToAESFamilyTransform<FromTop, ToTop>
where
	Packed1b: PackedField<Scalar = BinaryField1b>,
	FromTop: PackedTop<CanonicalTowerFamily>
		+ PackedExtension<BinaryField8b>
		+ PackedExtension<BinaryField1b, PackedSubfield = Packed1b>,
	ToTop: PackedTop<AESTowerFamily>
		+ PackedExtension<AESTowerField8b>
		+ PackedExtension<BinaryField1b>
		+ PackedExtension<BinaryField1b, PackedSubfield = Packed1b>,
	PackedSubfield<FromTop, BinaryField1b>:
		PackedTransformationFactory<PackedSubfield<ToTop, BinaryField1b>>,
	PackedSubfield<FromTop, BinaryField8b>:
		PackedTransformationFactory<PackedSubfield<ToTop, AESTowerField8b>>,
{
	type FromTower = CanonicalTowerFamily;
	type FromTop = FromTop;
	type ToTower = AESTowerFamily;
	type ToTop = ToTop;

	type B1 = SubfieldTransformer<BinaryField1b, BinaryField1b, IDTransformation>;
	type B8 = BinaryToAesTransformation<FromTop, ToTop>;
	type B16 = BinaryToAesTransformation<FromTop, ToTop>;

	fn new_b1_transformation() -> Self::B1 {
		SubfieldTransformer::new(IDTransformation)
	}

	fn new_b8_transformation() -> Self::B8 {
		make_binary_to_aes_packed_transformer::<Self::FromTop, Self::ToTop>()
	}

	fn new_b16_transformation() -> Self::B16 {
		make_binary_to_aes_packed_transformer::<Self::FromTop, Self::ToTop>()
	}
}

#[derive(Getters)]
pub struct TowerTransform<TFT: TowerTransformFactory> {
	#[getset(get = "pub")]
	b1_transformation: TFT::B1,
	#[getset(get = "pub")]
	b8_transformation: TFT::B8,
	#[getset(get = "pub")]
	b16_transformation: TFT::B16,
}

impl<TFT: TowerTransformFactory> Default for TowerTransform<TFT> {
	fn default() -> Self {
		Self {
			b1_transformation: TFT::new_b1_transformation(),
			b8_transformation: TFT::new_b8_transformation(),
			b16_transformation: TFT::new_b16_transformation(),
		}
	}
}
