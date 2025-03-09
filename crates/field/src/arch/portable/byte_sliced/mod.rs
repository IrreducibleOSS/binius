// Copyright 2024-2025 Irreducible Inc.

mod invert;
mod multiply;
mod packed_byte_sliced;
mod square;

pub use packed_byte_sliced::*;

#[cfg(test)]
pub mod tests {
	use super::*;

	macro_rules! define_byte_sliced_test {
		($module_name:ident, $name:ident, $scalar_type:ty) => {
			mod $module_name{
				use super::*;

				use proptest::prelude::*;
				use crate::{$scalar_type, underlier::WithUnderlier, packed::PackedField};

				fn scalar_array_strategy() -> impl Strategy<Value = [$scalar_type; <$name>::WIDTH]> {
					any::<[<$scalar_type as WithUnderlier>::Underlier; <$name>::WIDTH]>().prop_map(|arr| arr.map(<$scalar_type>::from_underlier))
				}

				proptest! {
					#[test]
					fn check_add(scalar_elems_a in scalar_array_strategy(), scalar_elems_b in scalar_array_strategy()) {
						let bytesliced_a = <$name>::from_scalars(scalar_elems_a);
						let bytesliced_b = <$name>::from_scalars(scalar_elems_b);

						let bytesliced_result = bytesliced_a + bytesliced_b;

						for i in 0..<$name>::WIDTH {
							assert_eq!(scalar_elems_a[i] + scalar_elems_b[i], bytesliced_result.get(i));
						}
					}

					#[test]
					fn check_add_assign(scalar_elems_a in scalar_array_strategy(), scalar_elems_b in scalar_array_strategy()) {
						let mut bytesliced_a = <$name>::from_scalars(scalar_elems_a);
						let bytesliced_b = <$name>::from_scalars(scalar_elems_b);

						bytesliced_a += bytesliced_b;

						for i in 0..<$name>::WIDTH {
							assert_eq!(scalar_elems_a[i] + scalar_elems_b[i], bytesliced_a.get(i));
						}
					}

					#[test]
					fn check_sub(scalar_elems_a in scalar_array_strategy(), scalar_elems_b in scalar_array_strategy()) {
						let bytesliced_a = <$name>::from_scalars(scalar_elems_a);
						let bytesliced_b = <$name>::from_scalars(scalar_elems_b);

						let bytesliced_result = bytesliced_a - bytesliced_b;

						for i in 0..<$name>::WIDTH {
							assert_eq!(scalar_elems_a[i] - scalar_elems_b[i], bytesliced_result.get(i));
						}
					}

					#[test]
					fn check_sub_assign(scalar_elems_a in scalar_array_strategy(), scalar_elems_b in scalar_array_strategy()) {
						let mut bytesliced_a = <$name>::from_scalars(scalar_elems_a);
						let bytesliced_b = <$name>::from_scalars(scalar_elems_b);

						bytesliced_a -= bytesliced_b;

						for i in 0..<$name>::WIDTH {
							assert_eq!(scalar_elems_a[i] - scalar_elems_b[i], bytesliced_a.get(i));
						}
					}

					#[test]
					fn check_mul(scalar_elems_a in scalar_array_strategy(), scalar_elems_b in scalar_array_strategy()) {
						let bytesliced_a = <$name>::from_scalars(scalar_elems_a);
						let bytesliced_b = <$name>::from_scalars(scalar_elems_b);

						let bytesliced_result = bytesliced_a * bytesliced_b;

						for i in 0..<$name>::WIDTH {
							assert_eq!(scalar_elems_a[i] * scalar_elems_b[i], bytesliced_result.get(i));
						}
					}

					#[test]
					fn check_mul_assign(scalar_elems_a in scalar_array_strategy(), scalar_elems_b in scalar_array_strategy()) {
						let mut bytesliced_a = <$name>::from_scalars(scalar_elems_a);
						let bytesliced_b = <$name>::from_scalars(scalar_elems_b);

						bytesliced_a *= bytesliced_b;

						for i in 0..<$name>::WIDTH {
							assert_eq!(scalar_elems_a[i] * scalar_elems_b[i], bytesliced_a.get(i));
						}
					}

					#[test]
					fn check_inv(scalar_elems in scalar_array_strategy()) {
						let bytesliced = <$name>::from_scalars(scalar_elems);

						let bytesliced_result = bytesliced.invert_or_zero();

						for (i, scalar_elem) in scalar_elems.iter().enumerate() {
							assert_eq!(scalar_elem.invert_or_zero(), bytesliced_result.get(i));
						}
					}

					#[test]
					fn check_square(scalar_elems in scalar_array_strategy()) {
						let bytesliced = <$name>::from_scalars(scalar_elems);

						let bytesliced_result = bytesliced.square();

						for (i, scalar_elem) in scalar_elems.iter().enumerate() {
							assert_eq!(scalar_elem.square(), bytesliced_result.get(i));
						}
					}

					#[test]
					fn check_linear_transformation(scalar_elems in scalar_array_strategy()) {
						use crate::linear_transformation::{PackedTransformationFactory, FieldLinearTransformation, Transformation};
						use rand::{rngs::StdRng, SeedableRng};

						let bytesliced = <$name>::from_scalars(scalar_elems);

						let linear_transformation = FieldLinearTransformation::random(StdRng::seed_from_u64(0));
						let packed_transformation = <$name>::make_packed_transformation(linear_transformation.clone());

						let bytesliced_result = packed_transformation.transform(&bytesliced);

						for i in 0..<$name>::WIDTH {
							assert_eq!(linear_transformation.transform(&scalar_elems[i]), bytesliced_result.get(i));
						}
					}

					#[test]
					fn check_interleave(scalar_elems_a in scalar_array_strategy(), scalar_elems_b in scalar_array_strategy()) {
						let bytesliced_a = <$name>::from_scalars(scalar_elems_a);
						let bytesliced_b = <$name>::from_scalars(scalar_elems_b);

						for log_block_len in 0..<$name>::LOG_WIDTH {
							let (bytesliced_c, bytesliced_d) = bytesliced_a.interleave(bytesliced_b, log_block_len);

							let block_len = 1 << log_block_len;
							for offset in (0..<$name>::WIDTH).step_by(2 * block_len) {
								for i in 0..block_len {
									assert_eq!(bytesliced_c.get(offset + i), scalar_elems_a[offset + i]);
									assert_eq!(bytesliced_c.get(offset + block_len + i), scalar_elems_b[offset + i]);

									assert_eq!(bytesliced_d.get(offset + i), scalar_elems_a[offset + block_len + i]);
									assert_eq!(bytesliced_d.get(offset + block_len + i), scalar_elems_b[offset + block_len + i]);
								}
							}
						}
					}
				}
			}
		};
	}

	// 128-bit byte-sliced
	define_byte_sliced_test!(tests_3d_16x128, ByteSliced3DAES16x128b, AESTowerField128b);
	define_byte_sliced_test!(tests_3d_32x64, ByteSliced3DAES32x64b, AESTowerField64b);
	define_byte_sliced_test!(tests_3d_64x32, ByteSliced3DAES64x32b, AESTowerField32b);
	define_byte_sliced_test!(tests_3d_128x16, ByteSliced3DAES128x16b, AESTowerField16b);
	define_byte_sliced_test!(tests_3d_256x8, ByteSliced3DAES256x8b, AESTowerField8b);

	// 256-bit byte-sliced

	define_byte_sliced_test!(tests_3d_32x128, ByteSliced3DAES32x128b, AESTowerField128b);
	define_byte_sliced_test!(tests_3d_64x64, ByteSliced3DAES64x64b, AESTowerField64b);
	define_byte_sliced_test!(tests_3d_128x32, ByteSliced3DAES128x32b, AESTowerField32b);
	define_byte_sliced_test!(tests_3d_256x16, ByteSliced3DAES256x16b, AESTowerField16b);
	define_byte_sliced_test!(tests_3d_512x8, ByteSliced3DAES512x8b, AESTowerField8b);

	// 512-bit byte-sliced
	define_byte_sliced_test!(tests_3d_64x128, ByteSliced3DAES64x128b, AESTowerField128b);
	define_byte_sliced_test!(tests_3d_128x64, ByteSliced3DAES128x64b, AESTowerField64b);
	define_byte_sliced_test!(tests_3d_256x32, ByteSliced3DAES256x32b, AESTowerField32b);
	define_byte_sliced_test!(tests_3d_512x16, ByteSliced3DAES512x16b, AESTowerField16b);
	define_byte_sliced_test!(tests_3d_1024x8, ByteSliced3DAES1024x8b, AESTowerField8b);
}
