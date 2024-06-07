use super::{Random, UnderlierType};
use binius_utils::checked_arithmetics::checked_log_2;
use bytemuck::{Pod, Zeroable};
use rand::RngCore;
use std::array;
use subtle::{Choice, ConstantTimeEq};

/// A type that represents a pair of elements of the same underlier type.
/// We use it as an underlier for the `ScaledPAckedField` type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct ScaledUnderlier<U, const N: usize>(pub [U; N]);

impl<U: Default, const N: usize> Default for ScaledUnderlier<U, N> {
	fn default() -> Self {
		ScaledUnderlier(array::from_fn(|_| U::default()))
	}
}

impl<U: Random, const N: usize> Random for ScaledUnderlier<U, N> {
	fn random(mut rng: impl RngCore) -> Self {
		ScaledUnderlier(array::from_fn(|_| U::random(&mut rng)))
	}
}

impl<U, const N: usize> From<ScaledUnderlier<U, N>> for [U; N] {
	fn from(val: ScaledUnderlier<U, N>) -> Self {
		val.0
	}
}

impl<U: ConstantTimeEq, const N: usize> ConstantTimeEq for ScaledUnderlier<U, N> {
	fn ct_eq(&self, other: &Self) -> Choice {
		self.0.ct_eq(&other.0)
	}
}

unsafe impl<U: Zeroable, const N: usize> Zeroable for ScaledUnderlier<U, N> {}

unsafe impl<U: Pod, const N: usize> Pod for ScaledUnderlier<U, N> {}

impl<U: UnderlierType + Pod, const N: usize> UnderlierType for ScaledUnderlier<U, N> {
	const LOG_BITS: usize = U::LOG_BITS + checked_log_2(N);
}
