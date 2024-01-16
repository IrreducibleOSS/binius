// Copyright 2023 Ulvetanna Inc.

use crate::{
	field::Field,
	iopoly::MultivariatePolyOracle,
	polynomial::{MultilinearComposite, MultilinearPoly},
};
use std::borrow::Borrow;

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

#[derive(Debug)]
pub struct EvalcheckWitness<F, M, BM>
where
	F: Field,
	M: MultilinearPoly<F> + ?Sized,
	BM: Borrow<M>,
{
	/// Polynomial must be representable as a composition of multilinear polynomials
	pub polynomial: MultilinearComposite<F, M, BM>,
}
