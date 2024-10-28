// Copyright 2024 Irreducible Inc.

use crate::{
	underlier::WithUnderlier, AESTowerField8b, BinaryField128b, BinaryField128bPolyval,
	BinaryField1b, BinaryField32b, Field, PackedBinaryField1x128b, PackedBinaryField4x32b,
	PackedField,
};

#[test]
fn test_field_text_debug() {
	assert_eq!(format!("{:?}", BinaryField1b::ONE), "BinaryField1b(0x1)");
	assert_eq!(format!("{:?}", AESTowerField8b::from_underlier(127)), "AESTowerField8b(0x7f)");
	assert_eq!(
		format!("{:?}", BinaryField128bPolyval::from_underlier(162259276829213363391578010288127)),
		"BinaryField128bPolyval(0xcffc05f0000000000000000000000000)"
	);
	assert_eq!(
		format!(
			"{:?}",
			PackedBinaryField1x128b::broadcast(BinaryField128b::from_underlier(
				162259276829213363391578010288127
			))
		),
		"Packed1x128([0x000007ffffffffffffffffffffffffff])"
	);
	assert_eq!(
		format!("{:?}", PackedBinaryField4x32b::broadcast(BinaryField32b::from_underlier(123))),
		"Packed4x32([0x0000007b,0x0000007b,0x0000007b,0x0000007b])"
	)
}
