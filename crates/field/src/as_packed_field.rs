// Copyright 2024 Ulvetanna Inc.

use crate::{
	aes_field::*,
	arch::{
		packed_1::*, packed_128::*, packed_16::*, packed_2::*, packed_32::*, packed_4::*,
		packed_64::*, packed_8::*, packed_aes_128::*, packed_aes_16::*, packed_aes_32::*,
		packed_aes_64::*, packed_aes_8::*, packed_polyval_128::PackedBinaryPolyval1x128b,
	},
	binary_field::*,
	polyval::BinaryField128bPolyval,
	underlier::WithUnderlier,
	Field, PackedField,
};

/// A trait to convert field to a same bit size packed field with some smaller scalar.
pub(crate) trait AsPackedField<Scalar: Field>: Field + WithUnderlier {
	type Packed: PackedField<Scalar = Scalar>
		+ WithUnderlier<Underlier: From<Self::Underlier> + Into<Self::Underlier>>;

	fn to_packed(self) -> Self::Packed {
		Self::Packed::from(self.to_underlier().into())
	}

	fn from_packed(value: Self::Packed) -> Self {
		Self::from(value.to_underlier().into())
	}
}

macro_rules! impl_as_packed_field {
	($field:ty, $packed_field:ty) => {
		impl AsPackedField<<$packed_field as PackedField>::Scalar> for $field {
			type Packed = $packed_field;
		}
	};
}

impl_as_packed_field!(BinaryField1b, PackedBinaryField1x1b);
impl_as_packed_field!(BinaryField2b, PackedBinaryField1x2b);
impl_as_packed_field!(BinaryField2b, PackedBinaryField2x1b);
impl_as_packed_field!(BinaryField4b, PackedBinaryField1x4b);
impl_as_packed_field!(BinaryField4b, PackedBinaryField2x2b);
impl_as_packed_field!(BinaryField4b, PackedBinaryField4x1b);
impl_as_packed_field!(BinaryField8b, PackedBinaryField1x8b);
impl_as_packed_field!(BinaryField8b, PackedBinaryField2x4b);
impl_as_packed_field!(BinaryField8b, PackedBinaryField4x2b);
impl_as_packed_field!(BinaryField8b, PackedBinaryField8x1b);
impl_as_packed_field!(BinaryField16b, PackedBinaryField16x1b);
impl_as_packed_field!(BinaryField16b, PackedBinaryField8x2b);
impl_as_packed_field!(BinaryField16b, PackedBinaryField4x4b);
impl_as_packed_field!(BinaryField16b, PackedBinaryField2x8b);
impl_as_packed_field!(BinaryField16b, PackedBinaryField1x16b);
impl_as_packed_field!(BinaryField32b, PackedBinaryField32x1b);
impl_as_packed_field!(BinaryField32b, PackedBinaryField16x2b);
impl_as_packed_field!(BinaryField32b, PackedBinaryField8x4b);
impl_as_packed_field!(BinaryField32b, PackedBinaryField4x8b);
impl_as_packed_field!(BinaryField32b, PackedBinaryField2x16b);
impl_as_packed_field!(BinaryField32b, PackedBinaryField1x32b);
impl_as_packed_field!(BinaryField64b, PackedBinaryField64x1b);
impl_as_packed_field!(BinaryField64b, PackedBinaryField32x2b);
impl_as_packed_field!(BinaryField64b, PackedBinaryField16x4b);
impl_as_packed_field!(BinaryField64b, PackedBinaryField8x8b);
impl_as_packed_field!(BinaryField64b, PackedBinaryField4x16b);
impl_as_packed_field!(BinaryField64b, PackedBinaryField2x32b);
impl_as_packed_field!(BinaryField64b, PackedBinaryField1x64b);
impl_as_packed_field!(BinaryField128b, PackedBinaryField128x1b);
impl_as_packed_field!(BinaryField128b, PackedBinaryField64x2b);
impl_as_packed_field!(BinaryField128b, PackedBinaryField32x4b);
impl_as_packed_field!(BinaryField128b, PackedBinaryField16x8b);
impl_as_packed_field!(BinaryField128b, PackedBinaryField8x16b);
impl_as_packed_field!(BinaryField128b, PackedBinaryField4x32b);
impl_as_packed_field!(BinaryField128b, PackedBinaryField2x64b);
impl_as_packed_field!(BinaryField128b, PackedBinaryField1x128b);

impl_as_packed_field!(AESTowerField8b, PackedAESBinaryField1x8b);
impl_as_packed_field!(AESTowerField16b, PackedAESBinaryField2x8b);
impl_as_packed_field!(AESTowerField16b, PackedAESBinaryField1x16b);
impl_as_packed_field!(AESTowerField32b, PackedAESBinaryField4x8b);
impl_as_packed_field!(AESTowerField32b, PackedAESBinaryField2x16b);
impl_as_packed_field!(AESTowerField32b, PackedAESBinaryField1x32b);
impl_as_packed_field!(AESTowerField64b, PackedAESBinaryField8x8b);
impl_as_packed_field!(AESTowerField64b, PackedAESBinaryField4x16b);
impl_as_packed_field!(AESTowerField64b, PackedAESBinaryField2x32b);
impl_as_packed_field!(AESTowerField64b, PackedAESBinaryField1x64b);
impl_as_packed_field!(AESTowerField128b, PackedAESBinaryField16x8b);
impl_as_packed_field!(AESTowerField128b, PackedAESBinaryField8x16b);
impl_as_packed_field!(AESTowerField128b, PackedAESBinaryField4x32b);
impl_as_packed_field!(AESTowerField128b, PackedAESBinaryField2x64b);
impl_as_packed_field!(AESTowerField128b, PackedAESBinaryField1x128b);

impl_as_packed_field!(BinaryField128bPolyval, PackedBinaryPolyval1x128b);
