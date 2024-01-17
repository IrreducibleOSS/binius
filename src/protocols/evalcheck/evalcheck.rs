// Copyright 2023 Ulvetanna Inc.

use crate::{field::Field, iopoly::MultivariatePolyOracle, polynomial::MultilinearComposite};

#[derive(Debug)]
pub struct EvalcheckClaim<F: Field> {
	/// Virtual Polynomial Oracle for which the evaluation is claimed
	pub poly: MultivariatePolyOracle<F>,
	/// Evaluation Point
	pub eval_point: Vec<F>,
	/// Claimed Evaluation
	pub eval: F,
	/// Whether the evaluation point is random
	pub is_random_point: bool,
}

/// Polynomial must be representable as a composition of multilinear polynomials
pub type EvalcheckWitness<F, M, BM> = MultilinearComposite<F, M, BM>;
