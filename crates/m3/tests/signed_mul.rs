// Copyright 2025 Irreducible Inc.

use proptest::prelude::*;

proptest! {
	#[test]
	fn test_i32_mul_reduce_to_u32(a in any::<i32>(), b in any::<i32>()) {
		let expected = a as i64 * b as i64;

		let a_u32 = a as u32;
		let b_u32 = b as u32;
		let mut prod = a_u32 as u64 * b_u32 as u64;
		if a < 0 {
			prod = prod.wrapping_sub((b_u32 as u64) << 32);
		}
		if b < 0 {
			prod = prod.wrapping_sub((a_u32 as u64) << 32);
		}
		assert_eq!(prod as i64, expected);
	}
}
