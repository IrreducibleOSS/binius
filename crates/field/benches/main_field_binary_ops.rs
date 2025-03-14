// Copyright 2025 Irreducible Inc.

mod packed_field_utils;

use std::ops::Mul;

use binius_field::{
	arch::OptimalUnderlier, as_packed_field::PackedType, AESTowerField128b, AESTowerField16b,
	AESTowerField32b, AESTowerField64b, AESTowerField8b, BinaryField128b, BinaryField128bPolyval,
	BinaryField16b, BinaryField32b, BinaryField64b, BinaryField8b, ByteSlicedAES32x128b,
	ByteSlicedAES32x16b, ByteSlicedAES32x32b, ByteSlicedAES32x64b, ByteSlicedAES32x8b,
};
use criterion::criterion_main;
use packed_field_utils::benchmark_packed_operation;

type OptimalPackedFP8b = PackedType<OptimalUnderlier, BinaryField8b>;
type OptimalPackedFP16b = PackedType<OptimalUnderlier, BinaryField16b>;
type OptimalPackedFP32b = PackedType<OptimalUnderlier, BinaryField32b>;
type OptimalPackedFP64b = PackedType<OptimalUnderlier, BinaryField64b>;
type OptimalPackedFP128b = PackedType<OptimalUnderlier, BinaryField128b>;

type OptimalPackedAES8b = PackedType<OptimalUnderlier, AESTowerField8b>;
type OptimalPackedAES16b = PackedType<OptimalUnderlier, AESTowerField16b>;
type OptimalPackedAES32b = PackedType<OptimalUnderlier, AESTowerField32b>;
type OptimalPackedAES64b = PackedType<OptimalUnderlier, AESTowerField64b>;
type OptimalPackedAES128b = PackedType<OptimalUnderlier, AESTowerField128b>;

type OptimalPackedPolyval128b = PackedType<OptimalUnderlier, BinaryField128bPolyval>;

trait SelfMul: Mul<Self, Output = Self> + Sized {}

impl<T: Mul<Self, Output = Self> + Sized> SelfMul for T {}

fn mul_main<T: SelfMul>(lhs: T, rhs: T) -> T {
	lhs * rhs
}

benchmark_packed_operation!(
	op_name @ main_binary_ops,
	bench_type @ binary_op,
	strategies @ (
		(mul_main, SelfMul, mul_main),
	),
	packed_fields @ [
		// Fan-Paar Packed Fields
		OptimalPackedFP8b
		OptimalPackedFP16b
		OptimalPackedFP32b
		OptimalPackedFP64b
		OptimalPackedFP128b

		// AES-Tower Packed Fields
		OptimalPackedAES8b
		OptimalPackedAES16b
		OptimalPackedAES32b
		OptimalPackedAES64b
		OptimalPackedAES128b

		// Polyval field
		OptimalPackedPolyval128b

		// Byte-sliced fields
		ByteSlicedAES32x8b
		ByteSlicedAES32x16b
		ByteSlicedAES32x32b
		ByteSlicedAES32x64b
		ByteSlicedAES32x128b
	]
);

criterion_main!(main_binary_ops);
