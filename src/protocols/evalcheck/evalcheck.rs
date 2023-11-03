// Copyright 2023 Ulvetanna Inc.

use crate::{field::Field, iopoly::MultivariatePolyOracle, polynomial::MultilinearComposite};

#[derive(Debug)]
pub struct EvalcheckClaim<'a, F: Field> {
	/// Virtual Polynomial Oracle for which the evaluation is claimed
	pub poly: MultivariatePolyOracle<'a, F>,
	/// Evaluation Point
	pub eval_point: Vec<F>,
	/// Claimed Evaluation
	pub eval: F,
}

#[derive(Debug)]
pub struct EvalcheckWitness<'a, OF: Field> {
	/// Polynomial must be representable as a composition of multilinear polynomials
	pub polynomial: MultilinearComposite<'a, OF, OF>,
}
