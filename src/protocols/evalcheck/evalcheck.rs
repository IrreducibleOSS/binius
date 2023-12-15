// Copyright 2023 Ulvetanna Inc.

use crate::{
	field::{ExtensionField, Field},
	iopoly::MultivariatePolyOracle,
	polynomial::MultilinearComposite,
};

#[derive(Debug)]
pub struct EvalcheckClaim<'a, F: Field, FE: ExtensionField<F>> {
	/// Virtual Polynomial Oracle for which the evaluation is claimed
	pub poly: MultivariatePolyOracle<'a, F>,
	/// Evaluation Point
	pub eval_point: Vec<FE>,
	/// Claimed Evaluation
	pub eval: FE,
}

#[derive(Debug)]
pub struct EvalcheckWitness<'a, F: Field, FE: ExtensionField<F>> {
	/// Polynomial must be representable as a composition of multilinear polynomials
	pub polynomial: MultilinearComposite<'a, F, FE>,
}
