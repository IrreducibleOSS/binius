// Copyright 2023 Ulvetanna Inc.

use std::sync::Arc;

use crate::{
	field::Field,
	polynomial::{MultilinearComposite, MultivariatePoly},
};

/// EvalcheckClaim Struct
pub struct EvalcheckClaim<F> {
	/// Virtual Polynomial Oracle is derivable from (Multilinear) Polynomial Oracles
	/// compositions may be nested
	pub multilinear_composition: Arc<dyn MultivariatePoly<F, F>>,
	/// Evaluation Point
	pub eval_point: Vec<F>,
	/// Claimed Evaluation
	pub eval: F,
}

/// EvalCheckWitness Struct
pub struct EvalcheckWitness<'a, OF: Field> {
	/// Polynomial must be representable as a composition of multilinear polynomials
	pub polynomial: &'a MultilinearComposite<'a, OF>,
}
