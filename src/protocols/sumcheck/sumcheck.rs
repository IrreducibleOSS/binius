// Copyright 2023 Ulvetanna Inc.

use std::sync::Arc;

use crate::{
	field::Field,
	polynomial::{CompositionPoly, MultilinearComposite},
};

pub struct SumcheckRound<F> {
	pub coeffs: Vec<F>,
}

pub struct SumcheckProof<F> {
	pub rounds: Vec<SumcheckRound<F>>,
}

pub struct SumcheckClaim<F> {
	/// Virtual Polynomial Oracle is derivable from (Multilinear) Polynomial Oracles
	/// compositions may be nested
	pub multilinear_composition: Arc<dyn CompositionPoly<F, F>>,
	/// Claimed Sum over the Boolean Hypercube
	pub sum: F,
	/// Number of variables
	pub n_vars: usize,
}

/// SumCheckWitness Struct
#[derive(Clone, Copy)]
pub struct SumcheckWitness<'a, OF: Field> {
	/// Polynomial must be representable as a composition of multilinear polynomials
	pub polynomial: &'a MultilinearComposite<'a, OF, OF>,
}
