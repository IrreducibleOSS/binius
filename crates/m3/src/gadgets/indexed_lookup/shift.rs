// Copyright 2025 Irreducible Inc.

use binius_field::Field;
use binius_math::{ArithCircuit, ArithExpr};

use crate::builder::B128;

/// Arithmetic expression for eq on 2 k bit queries
pub fn eq(k: usize) -> ArithExpr<B128> {
	let mut expr = ArithExpr::one();
	for i in 0..k {
		expr *= ArithExpr::Const(B128::ONE) + ArithExpr::Var(i) + ArithExpr::Var(i + 8);
	}
	expr
}

/// Arithmetic expression for LTU on 2 8 bit queries
pub fn ltu(k: usize) -> ArithExpr<B128> {
	let mut expr = (ArithExpr::Const(B128::ONE) + ArithExpr::Var(8)) * ArithExpr::Var(0);
	for i in 1..k {
		expr +=
			(ArithExpr::Const(B128::ONE) + ArithExpr::Var(i + 8)) * ArithExpr::Var(i) * eq(i - 1);
	}
	expr
}

