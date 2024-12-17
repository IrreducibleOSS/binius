// Copyright 2024 Irreducible Inc.

use super::error::Error;
use binius_field::Field;
use std::{
	cmp::max,
	fmt::{self, Display},
	iter::{Product, Sum},
	ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign},
};

/// Represents an arithmetic expression that can be evaluated symbolically.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArithExpr<F: Field> {
	Const(F),
	Var(usize),
	Add(Box<ArithExpr<F>>, Box<ArithExpr<F>>),
	Mul(Box<ArithExpr<F>>, Box<ArithExpr<F>>),
	Pow(Box<ArithExpr<F>>, u64),
}

impl<F: Field + Display> Display for ArithExpr<F> {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		match self {
			Self::Const(v) => write!(f, "{v}"),
			Self::Var(i) => write!(f, "x{i}"),
			Self::Add(x, y) => write!(f, "({} + {})", &**x, &**y),
			Self::Mul(x, y) => write!(f, "({} * {})", &**x, &**y),
			Self::Pow(x, p) => write!(f, "({})^{p}", &**x),
		}
	}
}

impl<F: Field> ArithExpr<F> {
	pub fn n_vars(&self) -> usize {
		match self {
			ArithExpr::Const(_) => 0,
			ArithExpr::Var(index) => *index + 1,
			ArithExpr::Add(left, right) | ArithExpr::Mul(left, right) => {
				max(left.n_vars(), right.n_vars())
			}
			ArithExpr::Pow(id, _) => id.n_vars(),
		}
	}

	pub fn degree(&self) -> usize {
		match self {
			ArithExpr::Const(_) => 0,
			ArithExpr::Var(_) => 1,
			ArithExpr::Add(left, right) => max(left.degree(), right.degree()),
			ArithExpr::Mul(left, right) => left.degree() + right.degree(),
			ArithExpr::Pow(base, exp) => base.degree() * *exp as usize,
		}
	}

	pub fn pow(self, exp: u64) -> Self {
		ArithExpr::Pow(Box::new(self), exp)
	}

	pub const fn zero() -> Self {
		ArithExpr::Const(F::ZERO)
	}

	pub const fn one() -> Self {
		ArithExpr::Const(F::ONE)
	}

	/// Creates a new expression with the variable indices remapped.
	///
	/// This recursively replaces the variable sub-expressions with an index `i` with the variable
	/// `indices[i]`.
	///
	/// ## Throws
	///
	/// * [`Error::IncorrectArgumentLength`] if indices has length less than the current number of
	///   variables
	pub fn remap_vars(self, indices: &[usize]) -> Result<Self, Error> {
		let expr = match self {
			ArithExpr::Const(_) => self,
			ArithExpr::Var(index) => {
				let new_index =
					indices
						.get(index)
						.ok_or_else(|| Error::IncorrectArgumentLength {
							arg: "subset".to_string(),
							expected: index,
						})?;
				ArithExpr::Var(*new_index)
			}
			ArithExpr::Add(left, right) => {
				let new_left = left.remap_vars(indices)?;
				let new_right = right.remap_vars(indices)?;
				ArithExpr::Add(Box::new(new_left), Box::new(new_right))
			}
			ArithExpr::Mul(left, right) => {
				let new_left = left.remap_vars(indices)?;
				let new_right = right.remap_vars(indices)?;
				ArithExpr::Mul(Box::new(new_left), Box::new(new_right))
			}
			ArithExpr::Pow(base, exp) => {
				let new_base = base.remap_vars(indices)?;
				ArithExpr::Pow(Box::new(new_base), exp)
			}
		};
		Ok(expr)
	}

	pub fn convert_field<FTgt: Field + From<F>>(&self) -> ArithExpr<FTgt> {
		match self {
			ArithExpr::Const(val) => ArithExpr::Const((*val).into()),
			ArithExpr::Var(index) => ArithExpr::Var(*index),
			ArithExpr::Add(left, right) => {
				let new_left = left.convert_field();
				let new_right = right.convert_field();
				ArithExpr::Add(Box::new(new_left), Box::new(new_right))
			}
			ArithExpr::Mul(left, right) => {
				let new_left = left.convert_field();
				let new_right = right.convert_field();
				ArithExpr::Mul(Box::new(new_left), Box::new(new_right))
			}
			ArithExpr::Pow(base, exp) => {
				let new_base = base.convert_field();
				ArithExpr::Pow(Box::new(new_base), *exp)
			}
		}
	}

	pub fn try_convert_field<FTgt: Field + TryFrom<F>>(
		&self,
	) -> Result<ArithExpr<FTgt>, <FTgt as TryFrom<F>>::Error> {
		Ok(match self {
			ArithExpr::Const(val) => ArithExpr::Const((*val).try_into()?),
			ArithExpr::Var(index) => ArithExpr::Var(*index),
			ArithExpr::Add(left, right) => {
				let new_left = left.try_convert_field()?;
				let new_right = right.try_convert_field()?;
				ArithExpr::Add(Box::new(new_left), Box::new(new_right))
			}
			ArithExpr::Mul(left, right) => {
				let new_left = left.try_convert_field()?;
				let new_right = right.try_convert_field()?;
				ArithExpr::Mul(Box::new(new_left), Box::new(new_right))
			}
			ArithExpr::Pow(base, exp) => {
				let new_base = base.try_convert_field()?;
				ArithExpr::Pow(Box::new(new_base), *exp)
			}
		})
	}
}

impl<F> Default for ArithExpr<F>
where
	F: Field,
{
	fn default() -> Self {
		Self::zero()
	}
}

impl<F> Add for ArithExpr<F>
where
	F: Field,
{
	type Output = Self;

	fn add(self, rhs: Self) -> Self {
		ArithExpr::Add(Box::new(self), Box::new(rhs))
	}
}

impl<F> AddAssign for ArithExpr<F>
where
	F: Field,
{
	fn add_assign(&mut self, rhs: Self) {
		*self = std::mem::take(self) + rhs;
	}
}

impl<F> Sub for ArithExpr<F>
where
	F: Field,
{
	type Output = Self;

	fn sub(self, rhs: Self) -> Self {
		ArithExpr::Add(Box::new(self), Box::new(rhs))
	}
}

impl<F> SubAssign for ArithExpr<F>
where
	F: Field,
{
	fn sub_assign(&mut self, rhs: Self) {
		*self = std::mem::take(self) - rhs;
	}
}

impl<F> Mul for ArithExpr<F>
where
	F: Field,
{
	type Output = Self;

	fn mul(self, rhs: Self) -> Self {
		ArithExpr::Mul(Box::new(self), Box::new(rhs))
	}
}

impl<F> MulAssign for ArithExpr<F>
where
	F: Field,
{
	fn mul_assign(&mut self, rhs: Self) {
		*self = std::mem::take(self) * rhs;
	}
}

impl<F: Field> Sum for ArithExpr<F> {
	fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
		iter.reduce(|acc, item| acc + item).unwrap_or(Self::zero())
	}
}

impl<F: Field> Product for ArithExpr<F> {
	fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
		iter.reduce(|acc, item| acc * item).unwrap_or(Self::one())
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use assert_matches::assert_matches;
	use binius_field::{BinaryField128b, BinaryField1b, BinaryField8b};

	#[test]
	fn test_degree_with_pow() {
		let expr = ArithExpr::Const(BinaryField8b::new(6)).pow(7);
		assert_eq!(expr.degree(), 0);

		let expr: ArithExpr<BinaryField8b> = ArithExpr::Var(0).pow(7);
		assert_eq!(expr.degree(), 7);

		let expr: ArithExpr<BinaryField8b> = (ArithExpr::Var(0) * ArithExpr::Var(1)).pow(7);
		assert_eq!(expr.degree(), 14);
	}

	#[test]
	fn test_remap_vars_with_too_few_vars() {
		type F = BinaryField8b;
		let expr = ((ArithExpr::Var(0) + ArithExpr::Const(F::ONE)) * ArithExpr::Var(1)).pow(3);
		assert_matches!(expr.remap_vars(&[5]), Err(Error::IncorrectArgumentLength { .. }));
	}

	#[test]
	fn test_remap_vars_works() {
		type F = BinaryField8b;
		let expr = ((ArithExpr::Var(0) + ArithExpr::Const(F::ONE)) * ArithExpr::Var(1)).pow(3);
		let new_expr = expr.remap_vars(&[5, 3]);

		let expected = ((ArithExpr::Var(5) + ArithExpr::Const(F::ONE)) * ArithExpr::Var(3)).pow(3);
		assert_eq!(new_expr.unwrap(), expected);
	}

	#[test]
	fn test_expression_upcast() {
		type F8 = BinaryField8b;
		type F = BinaryField128b;

		let expr = ((ArithExpr::Var(0) + ArithExpr::Const(F8::ONE))
			* ArithExpr::Const(F8::new(222)))
		.pow(3);

		let expected =
			((ArithExpr::Var(0) + ArithExpr::Const(F::ONE)) * ArithExpr::Const(F::new(222))).pow(3);
		assert_eq!(expr.convert_field::<F>(), expected);
	}

	#[test]
	fn test_expression_downcast() {
		type F8 = BinaryField8b;
		type F = BinaryField128b;

		let expr =
			((ArithExpr::Var(0) + ArithExpr::Const(F::ONE)) * ArithExpr::Const(F::new(222))).pow(3);

		assert!(expr.clone().try_convert_field::<BinaryField1b>().is_err());

		let expected = ((ArithExpr::Var(0) + ArithExpr::Const(F8::ONE))
			* ArithExpr::Const(F8::new(222)))
		.pow(3);
		assert_eq!(expr.try_convert_field::<BinaryField8b>().unwrap(), expected);
	}
}
