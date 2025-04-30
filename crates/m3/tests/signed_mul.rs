// Copyright 2025 Irreducible Inc.

use proptest::prelude::*;

proptest! {
	#[test]
	fn test_i32_mul_reduce_to_u32(a in any::<i32>(), b in any::<i32>()) {
		let expected = a as i64 * b as i64;

		let a_u32 = a as u32;
		let b_u32 = b as u32;
		let prod = a_u32 as u64 * b_u32 as u64;
		let mut prod_hi = (prod >> 32) as u32;
		let prod_lo = (prod % (1u64 << 32)) as u32;
		if a < 0 {
			prod_hi = prod_hi.wrapping_sub(b_u32);
		}
		if b < 0 {
			prod_hi = prod_hi.wrapping_sub(a_u32);
		}
		let prod = ((prod_hi as u64) << 32) | prod_lo as u64;
		assert_eq!(prod as i64, expected);
	}
}
