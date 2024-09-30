// Copyright 2023 Ulvetanna Inc.

use super::{error::Error, Field};
use std::{
	iter,
	ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign},
};

pub trait ExtensionField<F: Field>:
	Field
	+ From<F>
	+ TryInto<F>
	+ Add<F, Output = Self>
	+ Sub<F, Output = Self>
	+ Mul<F, Output = Self>
	+ AddAssign<F>
	+ SubAssign<F>
	+ MulAssign<F>
{
	type Iterator: Iterator<Item = F>;

	const DEGREE: usize;

	fn basis(i: usize) -> Result<Self, Error>;

	fn from_bases(base_elems: &[F]) -> Result<Self, Error> {
		Self::from_bases_sparse(base_elems, 0)
	}

	fn from_bases_sparse(base_elems: &[F], log_stride: usize) -> Result<Self, Error>;

	fn iter_bases(&self) -> Self::Iterator;
}

impl<F: Field> ExtensionField<F> for F {
	type Iterator = iter::Once<F>;

	const DEGREE: usize = 1;

	fn basis(i: usize) -> Result<Self, Error> {
		if i != 0 {
			return Err(Error::ExtensionDegreeMismatch);
		}
		Ok(Self::ONE)
	}

	fn from_bases_sparse(base_elems: &[F], log_stride: usize) -> Result<Self, Error> {
		if log_stride != 0 {
			return Err(Error::ExtensionDegreeMismatch);
		}

		match base_elems.len() {
			0 => Ok(F::ZERO),
			1 => Ok(base_elems[0]),
			_ => Err(Error::ExtensionDegreeMismatch),
		}
	}

	fn iter_bases(&self) -> Self::Iterator {
		iter::once(*self)
	}
}
