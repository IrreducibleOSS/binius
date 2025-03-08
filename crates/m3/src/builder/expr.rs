// Copyright 2025 Irreducible Inc.

use binius_field::{Field, TowerField};
use binius_math::ArithExpr;
use getset::{CopyGetters, Getters};

use super::{column::Col, table::TableId};

/// A constraint that the evaluation of an expression over a table is zero at every row.
#[derive(Debug)]
pub struct ZeroConstraint<F: Field> {
	pub name: String,
	pub expr: ArithExpr<F>,
}

/// A type representing an arithmetic expression composed over some table columns.
///
/// If the expression degree is 1, then it is a linear expression.
#[derive(Debug, Getters, CopyGetters)]
pub struct Expr<F: TowerField, const VALUES_PER_ROW: usize> {
	#[get_copy = "pub"]
	table_id: TableId,
	#[get_copy = "pub"]
	partition_id: usize,
	#[get = "pub"]
	expr: ArithExpr<F>,
}

impl<F: TowerField, const VALUES_PER_ROW: usize> Expr<F, VALUES_PER_ROW> {
	/// Polynomial degree of the arithmetic expression.
	pub fn degree(&self) -> usize {
		self.expr.degree()
	}
}

impl<F: TowerField, const VALUES_PER_ROW: usize> From<Col<F, VALUES_PER_ROW>>
	for Expr<F, VALUES_PER_ROW>
{
	fn from(value: Col<F, VALUES_PER_ROW>) -> Self {
		Expr {
			table_id: value.id.table_id,
			partition_id: value.id.partition_id,
			expr: ArithExpr::Var(value.id.partition_index),
		}
	}
}

impl<F: TowerField, const VALUES_PER_ROW: usize> std::ops::Add<Self> for Col<F, VALUES_PER_ROW> {
	type Output = Expr<F, VALUES_PER_ROW>;

	fn add(self, rhs: Self) -> Self::Output {
		assert_eq!(self.id.table_id, rhs.id.table_id);
		assert_eq!(self.id.partition_id, rhs.id.partition_id);

		let lhs_expr = ArithExpr::Var(self.id.partition_index);
		let rhs_expr = ArithExpr::Var(rhs.id.partition_index);

		Expr {
			table_id: self.id.table_id,
			partition_id: self.id.partition_id,
			expr: lhs_expr + rhs_expr,
		}
	}
}

impl<F: TowerField, const VALUES_PER_ROW: usize> std::ops::Add<Col<F, VALUES_PER_ROW>>
	for Expr<F, VALUES_PER_ROW>
{
	type Output = Expr<F, VALUES_PER_ROW>;

	fn add(self, rhs: Col<F, VALUES_PER_ROW>) -> Self::Output {
		assert_eq!(self.table_id, rhs.id.table_id);
		assert_eq!(self.partition_id, rhs.id.partition_id);

		let rhs_expr = ArithExpr::Var(rhs.id.partition_index);

		Expr {
			table_id: self.table_id,
			partition_id: self.partition_id,
			expr: self.expr + rhs_expr,
		}
	}
}

impl<F: TowerField, const VALUES_PER_ROW: usize> std::ops::Add<Expr<F, VALUES_PER_ROW>>
	for Expr<F, VALUES_PER_ROW>
{
	type Output = Expr<F, VALUES_PER_ROW>;

	fn add(self, rhs: Expr<F, VALUES_PER_ROW>) -> Self::Output {
		assert_eq!(self.table_id, rhs.table_id);
		assert_eq!(self.partition_id, rhs.partition_id);
		Expr {
			table_id: self.table_id,
			partition_id: self.partition_id,
			expr: self.expr + rhs.expr,
		}
	}
}

impl<F: TowerField, const VALUES_PER_ROW: usize> std::ops::Add<F> for Expr<F, VALUES_PER_ROW> {
	type Output = Expr<F, VALUES_PER_ROW>;

	fn add(self, rhs: F) -> Self::Output {
		Expr {
			table_id: self.table_id,
			partition_id: self.partition_id,
			expr: self.expr + ArithExpr::Const(rhs),
		}
	}
}

impl<F: TowerField, const VALUES_PER_ROW: usize> std::ops::Add<F> for Col<F, VALUES_PER_ROW> {
	type Output = Expr<F, VALUES_PER_ROW>;

	fn add(self, rhs: F) -> Self::Output {
		Expr::from(self) + rhs
	}
}

impl<F: TowerField, const VALUES_PER_ROW: usize> std::ops::Sub<Self> for Col<F, VALUES_PER_ROW> {
	type Output = Expr<F, VALUES_PER_ROW>;

	fn sub(self, rhs: Self) -> Self::Output {
		assert_eq!(self.id.table_id, rhs.id.table_id);
		assert_eq!(self.id.partition_id, rhs.id.partition_id);
		let lhs_expr = ArithExpr::Var(self.id.partition_index);
		let rhs_expr = ArithExpr::Var(rhs.id.partition_index);

		Expr {
			table_id: self.id.table_id,
			partition_id: self.id.partition_id,
			expr: lhs_expr - rhs_expr,
		}
	}
}

impl<F: TowerField, const VALUES_PER_ROW: usize> std::ops::Sub<Col<F, VALUES_PER_ROW>>
	for Expr<F, VALUES_PER_ROW>
{
	type Output = Expr<F, VALUES_PER_ROW>;

	fn sub(self, rhs: Col<F, VALUES_PER_ROW>) -> Self::Output {
		self - Expr::from(rhs)
	}
}

impl<F: TowerField, const VALUES_PER_ROW: usize> std::ops::Sub<Expr<F, VALUES_PER_ROW>>
	for Expr<F, VALUES_PER_ROW>
{
	type Output = Expr<F, VALUES_PER_ROW>;

	fn sub(self, rhs: Expr<F, VALUES_PER_ROW>) -> Self::Output {
		assert_eq!(self.table_id, rhs.table_id);
		assert_eq!(self.partition_id, rhs.partition_id);
		Expr {
			table_id: self.table_id,
			partition_id: self.partition_id,
			expr: self.expr - rhs.expr,
		}
	}
}

impl<F: TowerField, const VALUES_PER_ROW: usize> std::ops::Sub<F> for Expr<F, VALUES_PER_ROW> {
	type Output = Expr<F, VALUES_PER_ROW>;

	fn sub(self, rhs: F) -> Self::Output {
		Expr {
			table_id: self.table_id,
			partition_id: self.partition_id,
			expr: self.expr - ArithExpr::Const(rhs),
		}
	}
}

impl<F: TowerField, const VALUES_PER_ROW: usize> std::ops::Sub<F> for Col<F, VALUES_PER_ROW> {
	type Output = Expr<F, VALUES_PER_ROW>;

	fn sub(self, rhs: F) -> Self::Output {
		Expr::from(self) - rhs
	}
}

impl<F: TowerField, const VALUES_PER_ROW: usize> std::ops::Mul<Self> for Col<F, VALUES_PER_ROW> {
	type Output = Expr<F, VALUES_PER_ROW>;

	fn mul(self, rhs: Self) -> Self::Output {
		assert_eq!(self.id.table_id, rhs.id.table_id);
		assert_eq!(self.id.partition_id, rhs.id.partition_id);
		let lhs_expr = ArithExpr::Var(self.id.partition_index);
		let rhs_expr = ArithExpr::Var(rhs.id.partition_index);

		Expr {
			table_id: self.id.table_id,
			partition_id: self.id.partition_id,
			expr: lhs_expr * rhs_expr,
		}
	}
}

impl<F: TowerField, const VALUES_PER_ROW: usize> std::ops::Mul<Col<F, VALUES_PER_ROW>>
	for Expr<F, VALUES_PER_ROW>
{
	type Output = Expr<F, VALUES_PER_ROW>;

	fn mul(self, rhs: Col<F, VALUES_PER_ROW>) -> Self::Output {
		self * Expr::from(rhs)
	}
}

impl<F: TowerField, const VALUES_PER_ROW: usize> std::ops::Mul<Expr<F, VALUES_PER_ROW>>
	for Expr<F, VALUES_PER_ROW>
{
	type Output = Expr<F, VALUES_PER_ROW>;

	fn mul(self, rhs: Expr<F, VALUES_PER_ROW>) -> Self::Output {
		assert_eq!(self.table_id, rhs.table_id);
		assert_eq!(self.partition_id, rhs.partition_id);
		Expr {
			table_id: self.table_id,
			partition_id: self.partition_id,
			expr: self.expr * rhs.expr,
		}
	}
}

impl<F: TowerField, const VALUES_PER_ROW: usize> std::ops::Mul<F> for Expr<F, VALUES_PER_ROW> {
	type Output = Expr<F, VALUES_PER_ROW>;

	fn mul(self, rhs: F) -> Self::Output {
		Expr {
			table_id: self.table_id,
			partition_id: self.partition_id,
			expr: self.expr * ArithExpr::Const(rhs),
		}
	}
}

impl<F: TowerField, const VALUES_PER_ROW: usize> std::ops::Mul<F> for Col<F, VALUES_PER_ROW> {
	type Output = Expr<F, VALUES_PER_ROW>;

	fn mul(self, rhs: F) -> Self::Output {
		Expr::from(self) * rhs
	}
}

/// This exists only to implement Display for ArithExpr with named variables.
pub struct ArithExprNamedVars<'a, F: TowerField>(pub &'a ArithExpr<F>, pub &'a [String]);

impl<F: TowerField> std::fmt::Display for ArithExprNamedVars<'_, F> {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		let Self(expr, names) = self;
		match expr {
			ArithExpr::Const(v) => write!(f, "{v}"),
			ArithExpr::Var(i) => write!(f, "{}", names[*i]),
			ArithExpr::Add(x, y) => {
				write!(f, "{} + {}", self.expr(x), self.expr(y))
			}
			ArithExpr::Mul(x, y) => {
				write!(f, "({}) * ({})", self.expr(x), self.expr(y))
			}
			ArithExpr::Pow(x, p) => {
				write!(f, "({})^{p}", self.expr(x))
			}
		}
	}
}

impl<'a, F: TowerField> ArithExprNamedVars<'a, F> {
	fn expr(&self, expr: &'a ArithExpr<F>) -> Self {
		Self(expr, self.1)
	}
}
