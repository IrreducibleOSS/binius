use crate::{
	aes_field::*,
	arch::{
		packed_1::PackedBinaryField1x1b, packed_128::PackedBinaryField1x128b,
		packed_16::PackedBinaryField1x16b, packed_2::PackedBinaryField1x2b,
		packed_32::PackedBinaryField1x32b, packed_4::PackedBinaryField1x4b,
		packed_64::PackedBinaryField1x64b, packed_8::PackedBinaryField1x8b,
		packed_aes_128::PackedAESBinaryField1x128b, packed_aes_16::PackedAESBinaryField1x16b,
		packed_aes_32::PackedAESBinaryField1x32b, packed_aes_64::PackedAESBinaryField1x64b,
		packed_aes_8::PackedAESBinaryField1x8b, packed_polyval_128::PackedBinaryPolyval1x128b,
	},
	binary_field::*,
	packed::PackedField,
	polyval::BinaryField128bPolyval,
	Field,
};

/// Trait to specify packed type that contains single item
pub trait AsSinglePacked: Field {
	type SingleElementPacked: PackedField<Scalar = Self>;

	fn to_single_packed(self) -> Self::SingleElementPacked {
		assert_eq!(Self::SingleElementPacked::WIDTH, 1);

		Self::SingleElementPacked::set_single(self)
	}

	fn from_single_packed(value: Self::SingleElementPacked) -> Self {
		assert_eq!(Self::SingleElementPacked::WIDTH, 1);

		value.get(0)
	}
}

macro_rules! impl_single_packed {
	($field:ty, $packed:ty) => {
		impl AsSinglePacked for $field {
			type SingleElementPacked = $packed;
		}
	};
}

impl_single_packed!(BinaryField1b, PackedBinaryField1x1b);
impl_single_packed!(BinaryField2b, PackedBinaryField1x2b);
impl_single_packed!(BinaryField4b, PackedBinaryField1x4b);
impl_single_packed!(BinaryField8b, PackedBinaryField1x8b);
impl_single_packed!(BinaryField16b, PackedBinaryField1x16b);
impl_single_packed!(BinaryField32b, PackedBinaryField1x32b);
impl_single_packed!(BinaryField64b, PackedBinaryField1x64b);
impl_single_packed!(BinaryField128b, PackedBinaryField1x128b);

impl_single_packed!(AESTowerField8b, PackedAESBinaryField1x8b);
impl_single_packed!(AESTowerField16b, PackedAESBinaryField1x16b);
impl_single_packed!(AESTowerField32b, PackedAESBinaryField1x32b);
impl_single_packed!(AESTowerField64b, PackedAESBinaryField1x64b);
impl_single_packed!(AESTowerField128b, PackedAESBinaryField1x128b);

impl_single_packed!(BinaryField128bPolyval, PackedBinaryPolyval1x128b);
