// Copyright 2024-2025 Irreducible Inc.
// Copyright (c) 2024 The Plonky3 Authors

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

/// Computes the binary logarithm of $n$ rounded up to the nearest integer.
///
/// When $n$ is 0, this function returns 0. Otherwise, it returns $\lceil \log_2 n \rceil$.
#[must_use]
pub const fn log2_ceil_usize(n: usize) -> usize {
	min_bits(n.saturating_sub(1))
}

/// Returns the number of bits needed to represent $n$.
///
/// When $n$ is 0, this function returns 0. Otherwise, it returns $\lfloor \log_2 n \rfloor + 1$.
#[must_use]
pub const fn min_bits(n: usize) -> usize {
	(usize::BITS - n.leading_zeros()) as usize
}

/// Computes `log_2(n)`
///
/// # Panics
/// Panics if `n` is not a power of two.
#[must_use]
#[inline]
pub fn log2_strict_usize(n: usize) -> usize {
	let res = n.trailing_zeros();
	assert_eq!(n.wrapping_shr(res), 1, "Not a power of two: {n}");
	res as usize
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
	const fn test_checked_int_div_fail() {
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
	const fn test_checked_log2_fail() {
		_ = checked_log_2(6)
	}
}
