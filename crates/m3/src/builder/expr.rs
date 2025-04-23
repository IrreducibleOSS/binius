// Copyright 2025 Irreducible Inc.

use binius_field::{ExtensionField, Field, TowerField};
use binius_math::{ArithCircuitStep, ArithExpr};
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
#[derive(Debug, Clone, Getters, CopyGetters)]
pub struct Expr<F: TowerField, const V: usize> {
	#[get_copy = "pub"]
	table_id: TableId,
	#[get = "pub"]
	expr: ArithExpr<F>,
}

impl<F: TowerField, const V: usize> Expr<F, V> {
	/// Polynomial degree of the arithmetic expression.
	pub fn degree(&self) -> usize {
		self.expr.degree()
	}

	/// Exponentiate the expression by a constant power.
	pub fn pow(self, exp: u64) -> Self {
		Self {
			table_id: self.table_id,
			expr: self.expr.pow(exp),
		}
	}
}

impl<F: TowerField, const V: usize> From<Col<F, V>> for Expr<F, V> {
	fn from(value: Col<F, V>) -> Self {
		Expr {
			table_id: value.table_id,
			expr: ArithExpr::var(value.partition_index),
		}
	}
}

impl<F: TowerField, const V: usize> std::ops::Add<Self> for Col<F, V> {
	type Output = Expr<F, V>;

	fn add(self, rhs: Self) -> Self::Output {
		assert_eq!(self.table_id, rhs.table_id);

		let lhs_expr = ArithExpr::var(self.partition_index);
		let rhs_expr = ArithExpr::var(rhs.partition_index);

		Expr {
			table_id: self.table_id,
			expr: lhs_expr + rhs_expr,
		}
	}
}

impl<F: TowerField, const V: usize> std::ops::Add<Col<F, V>> for Expr<F, V> {
	type Output = Expr<F, V>;

	fn add(self, rhs: Col<F, V>) -> Self::Output {
		assert_eq!(self.table_id, rhs.table_id);

		let rhs_expr = ArithExpr::var(rhs.partition_index);
		Expr {
			table_id: self.table_id,
			expr: self.expr + rhs_expr,
		}
	}
}

impl<F: TowerField, const V: usize> std::ops::Add<Expr<F, V>> for Expr<F, V> {
	type Output = Expr<F, V>;

	fn add(self, rhs: Expr<F, V>) -> Self::Output {
		assert_eq!(self.table_id, rhs.table_id);
		Expr {
			table_id: self.table_id,
			expr: self.expr + rhs.expr,
		}
	}
}

impl<F: TowerField, const V: usize> std::ops::Add<F> for Expr<F, V> {
	type Output = Expr<F, V>;

	fn add(self, rhs: F) -> Self::Output {
		Expr {
			table_id: self.table_id,
			expr: self.expr + ArithExpr::constant(rhs),
		}
	}
}

impl<F: TowerField, const V: usize> std::ops::Add<Expr<F, V>> for Col<F, V> {
	type Output = Expr<F, V>;

	fn add(self, rhs: Expr<F, V>) -> Self::Output {
		Expr::from(self) + rhs
	}
}

impl<F: TowerField, const V: usize> std::ops::Add<F> for Col<F, V> {
	type Output = Expr<F, V>;

	fn add(self, rhs: F) -> Self::Output {
		Expr::from(self) + rhs
	}
}

impl<F: TowerField, const V: usize> std::ops::Sub<Self> for Col<F, V> {
	type Output = Expr<F, V>;

	fn sub(self, rhs: Self) -> Self::Output {
		assert_eq!(self.table_id, rhs.table_id);
		let lhs_expr = ArithExpr::var(self.partition_index);
		let rhs_expr = ArithExpr::var(rhs.partition_index);

		Expr {
			table_id: self.table_id,
			expr: lhs_expr - rhs_expr,
		}
	}
}

impl<F: TowerField, const V: usize> std::ops::Sub<Col<F, V>> for Expr<F, V> {
	type Output = Expr<F, V>;

	fn sub(self, rhs: Col<F, V>) -> Self::Output {
		self - Expr::from(rhs)
	}
}

impl<F: TowerField, const V: usize> std::ops::Sub<Expr<F, V>> for Expr<F, V> {
	type Output = Expr<F, V>;

	fn sub(self, rhs: Expr<F, V>) -> Self::Output {
		assert_eq!(self.table_id, rhs.table_id);
		Expr {
			table_id: self.table_id,
			expr: self.expr - rhs.expr,
		}
	}
}

impl<F: TowerField, const V: usize> std::ops::Sub<F> for Expr<F, V> {
	type Output = Expr<F, V>;

	fn sub(self, rhs: F) -> Self::Output {
		Expr {
			table_id: self.table_id,
			expr: self.expr - ArithExpr::constant(rhs),
		}
	}
}

impl<F: TowerField, const V: usize> std::ops::Sub<Expr<F, V>> for Col<F, V> {
	type Output = Expr<F, V>;

	fn sub(self, rhs: Expr<F, V>) -> Self::Output {
		Expr::from(self) - rhs
	}
}

impl<F: TowerField, const V: usize> std::ops::Sub<F> for Col<F, V> {
	type Output = Expr<F, V>;

	fn sub(self, rhs: F) -> Self::Output {
		Expr::from(self) - rhs
	}
}

impl<F: TowerField, const V: usize> std::ops::Mul<Self> for Col<F, V> {
	type Output = Expr<F, V>;

	fn mul(self, rhs: Self) -> Self::Output {
		Expr::from(self) * Expr::from(rhs)
	}
}

impl<F: TowerField, const V: usize> std::ops::Mul<Col<F, V>> for Expr<F, V> {
	type Output = Expr<F, V>;

	fn mul(self, rhs: Col<F, V>) -> Self::Output {
		self * Expr::from(rhs)
	}
}

impl<F: TowerField, const V: usize> std::ops::Mul<Expr<F, V>> for Expr<F, V> {
	type Output = Expr<F, V>;

	fn mul(self, rhs: Expr<F, V>) -> Self::Output {
		assert_eq!(self.table_id, rhs.table_id);
		Expr {
			table_id: self.table_id,
			expr: self.expr * rhs.expr,
		}
	}
}

impl<F: TowerField, const V: usize> std::ops::Mul<F> for Expr<F, V> {
	type Output = Expr<F, V>;

	fn mul(self, rhs: F) -> Self::Output {
		Expr {
			table_id: self.table_id,
			expr: self.expr * ArithExpr::constant(rhs),
		}
	}
}

impl<F: TowerField, const V: usize> std::ops::Mul<Expr<F, V>> for Col<F, V> {
	type Output = Expr<F, V>;

	fn mul(self, rhs: Expr<F, V>) -> Self::Output {
		Expr::from(self) * rhs
	}
}

impl<F: TowerField, const V: usize> std::ops::Mul<F> for Col<F, V> {
	type Output = Expr<F, V>;

	fn mul(self, rhs: F) -> Self::Output {
		Expr::from(self) * rhs
	}
}

/// Upcast an expression from a subfield to an extension field.
pub fn upcast_expr<F, FSub, const V: usize>(expr: Expr<FSub, V>) -> Expr<F, V>
where
	FSub: TowerField,
	F: TowerField + ExtensionField<FSub>,
{
	let Expr { table_id, expr } = expr;
	Expr {
		table_id,
		expr: expr.convert_field(),
	}
}

/// This exists only to implement Display for ArithExpr with named variables.
pub struct ArithExprNamedVars<'a, F: TowerField>(pub &'a ArithExpr<F>, pub &'a [String]);

impl<F: TowerField> std::fmt::Display for ArithExprNamedVars<'_, F> {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		fn write_step<F: TowerField>(
			f: &mut std::fmt::Formatter<'_>,
			step: usize,
			steps: &[ArithCircuitStep<F>],
			names: &[String],
		) -> std::fmt::Result {
			match &steps[step] {
				ArithCircuitStep::Const(v) => write!(f, "{v}"),
				ArithCircuitStep::Var(i) => write!(f, "{}", names[*i]),
				ArithCircuitStep::Add(x, y) => {
					write_step(f, *x, steps, names)?;
					write!(f, " + ")?;
					write_step(f, *y, steps, names)
				}
				ArithCircuitStep::Mul(x, y) => {
					write!(f, "(")?;
					write_step(f, *x, steps, names)?;
					write!(f, ") * (")?;
					write_step(f, *y, steps, names)?;
					write!(f, ")")
				}
				ArithCircuitStep::Pow(x, p) => {
					write!(f, "(")?;
					write_step(f, *x, steps, names)?;
					write!(f, ")^{p}")
				}
			}
		}

		write_step(f, 0, self.0.steps(), self.1)
	}
}
