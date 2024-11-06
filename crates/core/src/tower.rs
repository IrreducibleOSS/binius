// Copyright 2024 Irreducible Inc.

//! Traits for working with field towers.

use binius_field::{
	as_packed_field::PackScalar, underlier::UnderlierType, AESTowerField128b, AESTowerField16b,
	AESTowerField32b, AESTowerField64b, AESTowerField8b, BinaryField128b, BinaryField16b,
	BinaryField1b, BinaryField32b, BinaryField64b, BinaryField8b, ExtensionField, PackedExtension,
	PackedField, TowerField,
};
use trait_set::trait_set;

/// A trait that groups a family of related [`TowerField`]s as associated types.
pub trait TowerFamily {
	type B1: TowerField;
	type B8: TowerField;
	type B16: TowerField;
	type B32: TowerField;
	type B64: TowerField;
	type B128: TowerField
		+ ExtensionField<Self::B1>
		+ ExtensionField<Self::B8>
		+ ExtensionField<Self::B16>
		+ ExtensionField<Self::B32>
		+ ExtensionField<Self::B64>;
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
