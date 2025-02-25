// TODO: Do we need upcast_expr too?

use binius_field::{Field, TowerField};
use binius_math::ArithExpr;
use getset::{CopyGetters, Getters};

use super::{column::Col, table::TableId};

#[derive(Debug)]
pub struct ZeroConstraint<F: Field> {
	pub name: String,
	pub expr: ArithExpr<F>,
}

/// A type representing an arithmetic expression composed over some table columns.
///
/// If the expression degree is 1, then it is a linear expression.
#[derive(Debug, Getters, CopyGetters)]
pub struct Expr<F: TowerField, const V: usize> {
	#[get_copy = "pub"]
	table_id: TableId,
	#[get = "pub"]
	expr: ArithExpr<F>,
}

impl<F: TowerField, const V: usize> Expr<F, V> {
	pub fn degree(&self) -> usize {
		self.expr.degree()
	}
}

impl<F: TowerField, const V: usize> std::ops::Add<Self> for Col<F, V> {
	type Output = Expr<F, V>;

	fn add(self, rhs: Self) -> Self::Output {
		assert_eq!(self.table_id, rhs.table_id);

		let lhs_expr = ArithExpr::Var(self.index);
		let rhs_expr = ArithExpr::Var(rhs.index);

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

		let rhs_expr = ArithExpr::Var(rhs.index);

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

impl<F: TowerField, const V: usize> std::ops::Sub<Self> for Col<F, V> {
	type Output = Expr<F, V>;

	fn sub(self, rhs: Self) -> Self::Output {
		assert_eq!(self.table_id, rhs.table_id);

		let lhs_expr = ArithExpr::Var(self.index);
		let rhs_expr = ArithExpr::Var(rhs.index);

		Expr {
			table_id: self.table_id,
			expr: lhs_expr - rhs_expr,
		}
	}
}

impl<F: TowerField, const V: usize> std::ops::Sub<Col<F, V>> for Expr<F, V> {
	type Output = Expr<F, V>;

	fn sub(self, rhs: Col<F, V>) -> Self::Output {
		assert_eq!(self.table_id, rhs.table_id);

		let rhs_expr = ArithExpr::Var(rhs.index);

		Expr {
			table_id: self.table_id,
			expr: self.expr - rhs_expr,
		}
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

impl<F: TowerField, const V: usize> std::ops::Mul<Self> for Col<F, V> {
	type Output = Expr<F, V>;

	fn mul(self, rhs: Self) -> Self::Output {
		assert_eq!(self.table_id, rhs.table_id);

		let lhs_expr = ArithExpr::Var(self.index);
		let rhs_expr = ArithExpr::Var(rhs.index);

		Expr {
			table_id: self.table_id,
			expr: lhs_expr * rhs_expr,
		}
	}
}

impl<F: TowerField, const V: usize> std::ops::Mul<Col<F, V>> for Expr<F, V> {
	type Output = Expr<F, V>;

	fn mul(self, rhs: Col<F, V>) -> Self::Output {
		assert_eq!(self.table_id, rhs.table_id);

		let rhs_expr = ArithExpr::Var(rhs.index);

		Expr {
			table_id: self.table_id,
			expr: self.expr * rhs_expr,
		}
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
