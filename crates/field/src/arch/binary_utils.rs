// Copyright 2024 Ulvetanna Inc.

use bytemuck::{must_cast_mut, must_cast_ref, AnyBitPattern, NoUninit};

/// Execute function `f` with a reference to an array of length `N` casted from `val`.
#[allow(unused)]
pub(super) fn as_array_ref<T, U, const N: usize, R>(val: &T, f: impl FnOnce(&[U; N]) -> R) -> R
where
	T: NoUninit + AnyBitPattern,
	[U; N]: NoUninit + AnyBitPattern,
{
	let array = must_cast_ref(val);
	f(array)
}

/// Execute function `f` with a mutable reference to an array of length `N` casted from `val`.
#[allow(unused)]
pub(super) fn as_array_mut<T, U, const N: usize>(val: &mut T, f: impl FnOnce(&mut [U; N]))
where
	T: NoUninit + AnyBitPattern,
	[U; N]: NoUninit + AnyBitPattern,
{
	let array = must_cast_mut(val);
	f(array);
}
