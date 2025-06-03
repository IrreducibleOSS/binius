// Copyright 2025 Irreducible Inc.

// TODO: Introduce "aes-tower" feature that exports AES tower instead. This has to after the
// TowerFamily is fully removed.
pub use binius_field::{
	BinaryField1b as B1, BinaryField8b as B8, BinaryField16b as B16, BinaryField32b as B32,
	BinaryField64b as B64, BinaryField128b as B128,
};
use binius_field::{ExtensionField, PackedExtension, PackedField, TowerField};

trait_set::trait_set! {
	/// The top packed field in a tower.
	pub trait TowerTop =
		TowerField
		+ ExtensionField<B1>
		+ ExtensionField<B8>
		+ ExtensionField<B16>
		+ ExtensionField<B32>
		+ ExtensionField<B64>
		+ ExtensionField<B128>;

	/// A packed field type that is the top packed field in a tower.
	pub trait PackedTop =
		PackedField
		+ PackedExtension<B1>
		+ PackedExtension<B8>
		+ PackedExtension<B16>
		+ PackedExtension<B32>
		+ PackedExtension<B64>
		+ PackedExtension<B128>;
}
