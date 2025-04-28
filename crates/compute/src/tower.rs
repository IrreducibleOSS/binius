// Copyright 2024-2025 Irreducible Inc.

//! Traits for working with field towers.

use binius_field::{
	BinaryField128b, BinaryField16b, BinaryField1b, BinaryField32b, BinaryField64b, BinaryField8b,
	ExtensionField, TowerField,
};

/// A trait that groups a family of related [`TowerField`]s as associated types.
pub trait TowerFamily: Sized + 'static {
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

/// The canonical Fan-Paar tower family.
#[derive(Debug, Default)]
pub struct CanonicalTowerFamily;

impl TowerFamily for CanonicalTowerFamily {
	type B1 = BinaryField1b;
	type B8 = BinaryField8b;
	type B16 = BinaryField16b;
	type B32 = BinaryField32b;
	type B64 = BinaryField64b;
	type B128 = BinaryField128b;
}
