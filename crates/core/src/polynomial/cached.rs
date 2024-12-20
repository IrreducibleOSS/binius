// Copyright 2024 Irreducible Inc.

use std::{
	any::{Any, TypeId},
	collections::HashMap,
	fmt::Debug,
	marker::PhantomData,
};

use binius_field::{ExtensionField, Field, PackedField};
use binius_math::{ArithExpr, CompositionPoly, CompositionPolyOS, Error};

/// Cached composition poly wrapper.
///
/// It stores the efficient implementations of the composition poly for some known set of packed field types.
/// We are usually able to use this when the inner poly is constructed with a macro for the known field and packed field types.
#[derive(Default, Debug)]
pub struct CachedPoly<F: Field, Inner: CompositionPoly<F>> {
	inner: Inner,
	cache: PackedFieldCache<F>,
}

impl<F: Field, Inner: CompositionPoly<F>> CachedPoly<F, Inner> {
	/// Create a new cached polynomial with the given inner polynomial.
	pub fn new(inner: Inner) -> Self {
		Self {
			inner,
			cache: Default::default(),
		}
	}

	/// Register efficient implementations for the `P` packed field type in the cache.
	pub fn register<P: PackedField<Scalar: ExtensionField<F>>>(
		&mut self,
		composition: impl CompositionPolyOS<P> + 'static,
	) {
		self.cache.register(composition);
	}
}

impl<F: Field, Inner: CompositionPoly<F>> CompositionPoly<F> for CachedPoly<F, Inner> {
	fn n_vars(&self) -> usize {
		self.inner.n_vars()
	}

	fn degree(&self) -> usize {
		self.inner.degree()
	}

	fn binary_tower_level(&self) -> usize {
		self.inner.binary_tower_level()
	}

	fn expression<FE: ExtensionField<F>>(&self) -> ArithExpr<FE> {
		self.inner.expression()
	}

	fn evaluate<P: PackedField<Scalar: ExtensionField<F>>>(&self, query: &[P]) -> Result<P, Error> {
		if let Some(result) = self.cache.try_evaluate(query) {
			result
		} else {
			self.inner.evaluate(query)
		}
	}

	fn batch_evaluate<P: PackedField<Scalar: ExtensionField<F>>>(
		&self,
		batch_query: &[&[P]],
		evals: &mut [P],
	) -> Result<(), Error> {
		if let Some(result) = self.cache.try_batch_evaluate(batch_query, evals) {
			result
		} else {
			self.inner.batch_evaluate(batch_query, evals)
		}
	}
}

impl<F: Field, Inner: CompositionPoly<F>, P: PackedField<Scalar: ExtensionField<F>>>
	CompositionPolyOS<P> for CachedPoly<F, Inner>
{
	fn binary_tower_level(&self) -> usize {
		CompositionPoly::binary_tower_level(&self)
	}

	fn n_vars(&self) -> usize {
		CompositionPoly::n_vars(&self)
	}

	fn degree(&self) -> usize {
		CompositionPoly::degree(&self)
	}

	fn expression(&self) -> ArithExpr<P::Scalar> {
		CompositionPoly::expression(&self)
	}

	fn evaluate(&self, query: &[P]) -> Result<P, Error> {
		CompositionPoly::evaluate(&self, query)
	}

	fn batch_evaluate(&self, batch_query: &[&[P]], evals: &mut [P]) -> Result<(), Error> {
		CompositionPoly::batch_evaluate(&self, batch_query, evals)
	}
}

#[derive(Default)]
struct PackedFieldCache<F> {
	/// Map from the packed field type 'P to the efficient implementation of the composition polynomial
	/// with actual type `Box<dyn CompositionPolyOS<P>>`.
	entries: HashMap<TypeId, Box<dyn Any + Send + Sync>>,
	_pd: PhantomData<F>,
}

impl<F: Field> PackedFieldCache<F> {
	/// Register efficient implementations for the `P` packed field type in the cache.
	fn register<P: PackedField<Scalar: ExtensionField<F>>>(
		&mut self,
		composition: impl CompositionPolyOS<P> + 'static,
	) {
		let boxed_composition = Box::new(composition) as Box<dyn CompositionPolyOS<P>>;
		self.entries
			.insert(TypeId::of::<P>(), Box::new(boxed_composition) as Box<dyn Any + Send + Sync>);
	}

	/// Try to evaluate the expression using the efficient implementation for the `P` packed field type.
	/// If no implementation is found, return None.
	fn try_evaluate<P: PackedField<Scalar: ExtensionField<F>>>(
		&self,
		query: &[P],
	) -> Option<Result<P, Error>> {
		if let Some(entry) = self.entries.get(&TypeId::of::<P>()) {
			let entry = entry
				.downcast_ref::<Box<dyn CompositionPolyOS<P>>>()
				.expect("cast must succeed");
			Some(entry.evaluate(query))
		} else {
			None
		}
	}

	/// Try to batch evaluate the expression using the efficient implementation for the `P` packed field type.
	/// If no implementation is found, return None.
	fn try_batch_evaluate<P: PackedField<Scalar: ExtensionField<F>>>(
		&self,
		batch_query: &[&[P]],
		evals: &mut [P],
	) -> Option<Result<(), Error>> {
		if let Some(entry) = self.entries.get(&TypeId::of::<P>()) {
			let entry = entry
				.downcast_ref::<Box<dyn CompositionPolyOS<P>>>()
				.expect("cast must succeed");
			Some(entry.batch_evaluate(batch_query, evals))
		} else {
			None
		}
	}
}

impl<F: Field> Debug for PackedFieldCache<F> {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		f.debug_struct("PackedFieldCache")
			.field("cached_implementations", &self.entries.len())
			.finish()
	}
}

#[cfg(test)]
mod tests {
	use std::iter::zip;

	use binius_field::{BinaryField8b, ExtensionField, PackedBinaryField16x8b, PackedField};
	use binius_math::{ArithExpr, CompositionPolyOS};

	use super::*;
	use crate::polynomial::{cached::CachedPoly, ArithCircuitPoly};

	fn ensure_equal_batch_eval_results<P: PackedField>(
		circuit_1: &impl CompositionPolyOS<P>,
		circuit_2: &impl CompositionPolyOS<P>,
		batch_query: &[&[P]],
	) {
		for row in 0..batch_query[0].len() {
			let query = batch_query.iter().map(|q| q[row]).collect::<Vec<_>>();

			assert_eq!(circuit_1.evaluate(&query).unwrap(), circuit_2.evaluate(&query).unwrap());
		}

		let result_1 = {
			let mut uncached_evals = vec![P::zero(); batch_query[0].len()];
			circuit_1
				.batch_evaluate(batch_query, &mut uncached_evals)
				.unwrap();
			uncached_evals
		};

		let result_2 = {
			let mut cached_evals = vec![P::zero(); batch_query[0].len()];
			circuit_2
				.batch_evaluate(batch_query, &mut cached_evals)
				.unwrap();
			cached_evals
		};

		assert_eq!(result_1, result_2);
	}

	#[derive(Debug, Copy, Clone)]
	struct AddComposition;

	impl<P: PackedField<Scalar: ExtensionField<BinaryField8b>>> CompositionPolyOS<P>
		for AddComposition
	{
		fn binary_tower_level(&self) -> usize {
			0
		}

		fn n_vars(&self) -> usize {
			1
		}

		fn degree(&self) -> usize {
			1
		}

		fn expression(&self) -> ArithExpr<P::Scalar> {
			ArithExpr::Const(BinaryField8b::new(123).into()) + ArithExpr::Var(0)
		}

		fn evaluate(&self, query: &[P]) -> Result<P, Error> {
			Ok(query[0] + P::broadcast(BinaryField8b::new(123).into()))
		}

		fn batch_evaluate(&self, batch_query: &[&[P]], evals: &mut [P]) -> Result<(), Error> {
			for (input, output) in zip(batch_query[0], evals) {
				*output = *input + P::broadcast(BinaryField8b::new(123).into());
			}

			Ok(())
		}
	}

	#[test]
	fn test_cached_impl() {
		let expr = ArithExpr::Const(BinaryField8b::new(123)) + ArithExpr::Var(0);
		let circuit = ArithCircuitPoly::<BinaryField8b>::new(expr);

		let composition = AddComposition;

		let mut cached_circuit = CachedPoly::new(circuit.clone());
		cached_circuit.register::<BinaryField8b>(composition);

		let batch_query = [(0..255).map(BinaryField8b::new).collect::<Vec<_>>()];
		let batch_query = batch_query.iter().map(|q| q.as_slice()).collect::<Vec<_>>();
		ensure_equal_batch_eval_results(&circuit, &cached_circuit, &batch_query);
	}

	#[test]
	fn test_uncached_impl() {
		let expr = ArithExpr::Const(BinaryField8b::new(123)) + ArithExpr::Var(0);
		let circuit = ArithCircuitPoly::<BinaryField8b>::new(expr);

		let composition = AddComposition;

		let mut cached_circuit = CachedPoly::new(circuit.clone());
		cached_circuit.register::<PackedBinaryField16x8b>(composition);

		let batch_query = [(0..255).map(BinaryField8b::new).collect::<Vec<_>>()];
		let batch_query = batch_query.iter().map(|q| q.as_slice()).collect::<Vec<_>>();
		ensure_equal_batch_eval_results(&circuit, &cached_circuit, &batch_query);
	}
}
