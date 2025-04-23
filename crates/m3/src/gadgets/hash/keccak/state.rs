// Copyright 2025 Irreducible Inc.

use std::{array, fmt};

/// The state of the permutation. Consists of 5x5 lanes.
///
/// Indexing is defined as `mod 5`.
///
/// See section 1.6 of the [keccak specification] for more details.
///
/// [keccak specification]: https://keccak.team/files/Keccak-reference-3.0.pdf
#[derive(Default, Clone, PartialEq, Eq)]
pub struct StateMatrix<T> {
	v: [T; 25],
}

impl<T> StateMatrix<T> {
	/// Creates a new state matrix from an array of values stored in row-major order.
	pub fn from_values(v: [T; 25]) -> Self {
		Self { v }
	}

	/// Creates a new state matrix from a function that takes a coordinate and returns a value.
	pub fn from_fn<F>(mut f: F) -> Self
	where
		F: FnMut((usize, usize)) -> T,
	{
		Self {
			v: array::from_fn(|i| f((i % 5, i / 5))),
		}
	}

	/// Creates a new state matrix from a function that takes a coordinate and returns a value or
	/// an error.
	pub fn try_from_fn<F, E>(mut f: F) -> Result<Self, E>
	where
		F: FnMut((usize, usize)) -> Result<T, E>,
	{
		Ok(Self {
			v: array_util::try_from_fn(|i| f((i % 5, i / 5)))?,
		})
	}

	/// Returns the raw array of values stored in row-major order.
	pub fn as_inner(&self) -> &[T; 25] {
		&self.v
	}

	/// Returns the raw array of values stored in row-major order.
	#[cfg(test)]
	pub fn into_inner(self) -> [T; 25] {
		self.v
	}
}

impl<T> std::ops::Index<(usize, usize)> for StateMatrix<T> {
	type Output = T;
	fn index(&self, (x, y): (usize, usize)) -> &T {
		let x = x % 5;
		let y = y % 5;
		&self.v[x + y * 5]
	}
}

impl<T> std::ops::IndexMut<(usize, usize)> for StateMatrix<T> {
	fn index_mut(&mut self, (x, y): (usize, usize)) -> &mut T {
		let x = x % 5;
		let y = y % 5;
		&mut self.v[x + y * 5]
	}
}

impl<T: fmt::Debug + fmt::UpperHex> fmt::Debug for StateMatrix<T> {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		write!(f, "StateMatrix {{")?;
		if f.alternate() {
			writeln!(f)?;
		}
		// generate a 5x5 matrix of hex values.
		for y in 0..5 {
			write!(f, "[")?;
			if f.alternate() {
				write!(f, "  ")?;
			}
			for x in 0..5 {
				write!(f, "0x{:02X} ", self[(x, y)])?;
			}
			write!(f, "]")?;

			if f.alternate() {
				writeln!(f)?;
			}
		}
		write!(f, "}}")?;
		Ok(())
	}
}

/// A row of the state. This is basically a slice of the state matrix along the x coordinate.
///
/// Indexing is defined as `mod 5`.
///
/// See section 1.6 of the [keccak specification] for more details.
///
/// [keccak specification]: https://keccak.team/files/Keccak-reference-3.0.pdf
#[derive(Clone)]
pub struct StateRow<T>([T; 5]);

impl<T> std::ops::Index<usize> for StateRow<T> {
	type Output = T;
	fn index(&self, i: usize) -> &T {
		&self.0[i % 5]
	}
}

impl<T> std::ops::IndexMut<usize> for StateRow<T> {
	fn index_mut(&mut self, i: usize) -> &mut T {
		&mut self.0[i % 5]
	}
}

impl<T> StateRow<T> {
	/// Creates a new state row from a function that takes a coordinate and returns a value.
	pub fn from_fn<F>(f: F) -> Self
	where
		F: FnMut(usize) -> T,
	{
		Self(array::from_fn(f))
	}
}
