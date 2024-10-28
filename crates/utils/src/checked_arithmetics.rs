// Copyright 2024 Irreducible Inc.

/// Division implementation that fails in case when `a`` isn't divisible by `b`
pub const fn checked_int_div(a: usize, b: usize) -> usize {
	let result = a / b;
	assert!(b * result == a);

	result
}

/// log2 implementation that fails when `val` is not a power of 2.
pub const fn checked_log_2(val: usize) -> usize {
	let result = val.ilog2();
	assert!(2usize.pow(result) == val);

	result as _
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_checked_int_div_success() {
		assert_eq!(checked_int_div(6, 1), 6);
		assert_eq!(checked_int_div(6, 2), 3);
		assert_eq!(checked_int_div(6, 6), 1);
	}

	#[test]
	#[should_panic]
	fn test_checked_int_div_fail() {
		_ = checked_int_div(5, 2);
	}

	#[test]
	fn test_checked_log2_success() {
		assert_eq!(checked_log_2(1), 0);
		assert_eq!(checked_log_2(2), 1);
		assert_eq!(checked_log_2(4), 2);
		assert_eq!(checked_log_2(64), 6);
	}

	#[test]
	#[should_panic]
	fn test_checked_log2_fail() {
		_ = checked_log_2(6)
	}
}
