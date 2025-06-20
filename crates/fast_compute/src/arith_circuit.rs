// Copyright 2024-2025 Irreducible Inc.

use std::{mem::MaybeUninit, sync::Arc};

use binius_field::{ExtensionField, Field, PackedField, TowerField};
use binius_math::{ArithCircuit, ArithCircuitStep, CompositionPoly, Error, RowsBatchRef};
use binius_utils::{
	DeserializeBytes, SerializationError, SerializationMode, SerializeBytes, bail,
	mem::{slice_assume_init_mut, slice_assume_init_ref},
};

/// Convert the expression to a sequence of arithmetic operations that can be evaluated in sequence.
fn convert_circuit_steps<F: Field>(
	expr: &ArithCircuit<F>,
) -> (Vec<CircuitStep<F>>, CircuitStepArgument<F>) {
	/// This struct is used to the steps in the original circuit and the converted one and back.
	struct StepsMapping {
		original_to_converted: Vec<Option<usize>>,
		converted_to_original: Vec<Option<usize>>,
	}

	impl StepsMapping {
		fn new(original_size: usize) -> Self {
			Self {
				original_to_converted: vec![None; original_size],
				// The size of this vector isn't known at the start, but that's a reasonable guess
				converted_to_original: vec![None; original_size],
			}
		}

		fn register(&mut self, original_step: usize, converted_step: usize) {
			self.original_to_converted[original_step] = Some(converted_step);

			if converted_step >= self.converted_to_original.len() {
				self.converted_to_original.resize(converted_step + 1, None);
			}
			self.converted_to_original[converted_step] = Some(original_step);
		}

		fn clear_step(&mut self, converted_step: usize) {
			if self.converted_to_original.len() <= converted_step {
				return;
			}

			if let Some(original_step) = self.converted_to_original[converted_step].take() {
				self.original_to_converted[original_step] = None;
			}
		}

		fn get_converted_step(&self, original_step: usize) -> Option<usize> {
			self.original_to_converted[original_step]
		}
	}

	fn convert_step<F: Field>(
		original_step: usize,
		original_steps: &[ArithCircuitStep<F>],
		result: &mut Vec<CircuitStep<F>>,
		node_to_step: &mut StepsMapping,
	) -> CircuitStepArgument<F> {
		if let Some(converted_step) = node_to_step.get_converted_step(original_step) {
			return CircuitStepArgument::Expr(CircuitNode::Slot(converted_step));
		}

		match &original_steps[original_step] {
			ArithCircuitStep::Const(constant) => CircuitStepArgument::Const(*constant),
			ArithCircuitStep::Var(var) => CircuitStepArgument::Expr(CircuitNode::Var(*var)),
			ArithCircuitStep::Add(left, right) => {
				let left = convert_step(*left, original_steps, result, node_to_step);

				if let CircuitStepArgument::Expr(CircuitNode::Slot(left)) = left {
					if let ArithCircuitStep::Mul(mleft, mright) = &original_steps[*right] {
						// Only handling e1 + (e2 * e3), not (e1 * e2) + e3, as latter was not
						// observed in practice (the former can be enforced by rewriting
						// expression)
						let mleft = convert_step(*mleft, original_steps, result, node_to_step);
						let mright = convert_step(*mright, original_steps, result, node_to_step);

						// Since we'we changed the value of `left` to a new value, we need to clear
						// the cache for it
						node_to_step.clear_step(left);
						node_to_step.register(original_step, left);
						result.push(CircuitStep::AddMul(left, mleft, mright));

						return CircuitStepArgument::Expr(CircuitNode::Slot(left));
					}
				}

				let right = convert_step(*right, original_steps, result, node_to_step);

				node_to_step.register(original_step, result.len());
				result.push(CircuitStep::Add(left, right));
				CircuitStepArgument::Expr(CircuitNode::Slot(result.len() - 1))
			}
			ArithCircuitStep::Mul(left, right) => {
				let left = convert_step(*left, original_steps, result, node_to_step);
				let right = convert_step(*right, original_steps, result, node_to_step);

				node_to_step.register(original_step, result.len());
				result.push(CircuitStep::Mul(left, right));
				CircuitStepArgument::Expr(CircuitNode::Slot(result.len() - 1))
			}
			ArithCircuitStep::Pow(base, exp) => {
				let mut acc = convert_step(*base, original_steps, result, node_to_step);
				let base_expr = acc;
				let highest_bit = exp.ilog2();

				for i in (0..highest_bit).rev() {
					if i == 0 {
						node_to_step.register(original_step, result.len());
					}
					result.push(CircuitStep::Square(acc));
					acc = CircuitStepArgument::Expr(CircuitNode::Slot(result.len() - 1));

					if (exp >> i) & 1 != 0 {
						result.push(CircuitStep::Mul(acc, base_expr));
						acc = CircuitStepArgument::Expr(CircuitNode::Slot(result.len() - 1));
					}
				}

				acc
			}
		}
	}

	let mut steps = Vec::new();
	let mut steps_mapping = StepsMapping::new(expr.steps().len());
	let ret = convert_step(expr.steps().len() - 1, expr.steps(), &mut steps, &mut steps_mapping);
	(steps, ret)
}

/// Input of the circuit calculation step
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CircuitNode {
	/// Input variable
	Var(usize),
	/// Evaluation at one of the previous steps
	Slot(usize),
}

impl CircuitNode {
	/// Return either the input column or the slice with evaluations at one of the previous steps.
	/// This method is used for batch evaluation.
	fn get_sparse_chunk<'a, P: PackedField>(
		&'a self,
		inputs: &'a RowsBatchRef<'a, P>,
		evals: &'a [P],
		row_len: usize,
	) -> &'a [P] {
		match self {
			Self::Var(index) => inputs.row(*index),
			Self::Slot(slot) => &evals[slot * row_len..(slot + 1) * row_len],
		}
	}
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CircuitStepArgument<F> {
	Expr(CircuitNode),
	Const(F),
}

/// Describes computation symbolically. This is used internally by ArithCircuitPoly.
///
/// ExprIds used by an Expr has to be less than the index of the Expr itself within the
/// ArithCircuitPoly, to ensure it represents a directed acyclic graph that can be computed in
/// sequence.
#[derive(Debug)]
enum CircuitStep<F: Field> {
	Add(CircuitStepArgument<F>, CircuitStepArgument<F>),
	Mul(CircuitStepArgument<F>, CircuitStepArgument<F>),
	Square(CircuitStepArgument<F>),
	AddMul(usize, CircuitStepArgument<F>, CircuitStepArgument<F>),
}

/// Describes polynomial evaluations using a directed acyclic graph of expressions.
///
/// This is meant as an alternative to a hard-coded CompositionPoly.
///
/// The advantage over a hard coded CompositionPoly is that this can be constructed and manipulated
/// dynamically at runtime and the object representing different polnomials can be stored in a
/// homogeneous collection.
#[derive(Debug, Clone)]
pub struct ArithCircuitPoly<F: Field> {
	expr: ArithCircuit<F>,
	steps: Arc<[CircuitStep<F>]>,
	/// The "top level expression", which depends on circuit expression evaluations
	retval: CircuitStepArgument<F>,
	degree: usize,
	n_vars: usize,
	tower_level: usize,
}

impl<F: Field> PartialEq for ArithCircuitPoly<F> {
	fn eq(&self, other: &Self) -> bool {
		self.n_vars == other.n_vars && self.expr == other.expr
	}
}

impl<F: Field> Eq for ArithCircuitPoly<F> {}

impl<F: TowerField> SerializeBytes for ArithCircuitPoly<F> {
	fn serialize(
		&self,
		mut write_buf: impl bytes::BufMut,
		mode: SerializationMode,
	) -> Result<(), SerializationError> {
		(&self.expr, self.n_vars).serialize(&mut write_buf, mode)
	}
}

impl<F: TowerField> DeserializeBytes for ArithCircuitPoly<F> {
	fn deserialize(
		read_buf: impl bytes::Buf,
		mode: SerializationMode,
	) -> Result<Self, SerializationError>
	where
		Self: Sized,
	{
		let (expr, n_vars) = <(ArithCircuit<F>, usize)>::deserialize(read_buf, mode)?;
		Self::with_n_vars(n_vars, expr).map_err(|_| SerializationError::InvalidConstruction {
			name: "ArithCircuitPoly",
		})
	}
}

impl<F: TowerField> ArithCircuitPoly<F> {
	pub fn new(mut expr: ArithCircuit<F>) -> Self {
		expr.optimize_in_place();

		let degree = expr.degree();
		let n_vars = expr.n_vars();
		let tower_level = expr.binary_tower_level();
		let (exprs, retval) = convert_circuit_steps(&expr);

		Self {
			expr,
			steps: exprs.into(),
			retval,
			degree,
			n_vars,
			tower_level,
		}
	}
	/// Constructs an [`ArithCircuitPoly`] with the given number of variables.
	///
	/// The number of variables may be greater than the number of variables actually read in the
	/// arithmetic expression.
	pub fn with_n_vars(n_vars: usize, mut expr: ArithCircuit<F>) -> Result<Self, Error> {
		expr.optimize_in_place();

		let degree = expr.degree();
		let tower_level = expr.binary_tower_level();
		if n_vars < expr.n_vars() {
			return Err(Error::IncorrectNumberOfVariables {
				expected: expr.n_vars(),
				actual: n_vars,
			});
		}
		let (steps, retval) = convert_circuit_steps(&expr);

		Ok(Self {
			expr,
			steps: steps.into(),
			retval,
			n_vars,
			degree,
			tower_level,
		})
	}

	/// Returns an underlying circuit from which this polynomial was constructed.
	pub fn expr(&self) -> &ArithCircuit<F> {
		&self.expr
	}
}

impl<F: TowerField, P: PackedField<Scalar: ExtensionField<F>>> CompositionPoly<P>
	for ArithCircuitPoly<F>
{
	fn degree(&self) -> usize {
		self.degree
	}

	fn n_vars(&self) -> usize {
		self.n_vars
	}

	fn binary_tower_level(&self) -> usize {
		self.tower_level
	}

	fn expression(&self) -> ArithCircuit<P::Scalar> {
		self.expr.convert_field()
	}

	fn evaluate(&self, query: &[P]) -> Result<P, Error> {
		if query.len() != self.n_vars {
			return Err(Error::IncorrectQuerySize {
				expected: self.n_vars,
				actual: query.len(),
			});
		}

		fn write_result<T>(target: &mut [MaybeUninit<T>], value: T) {
			// Safety: The index is guaranteed to be within bounds because
			// we initialize at least `self.steps.len()` using `stackalloc`.
			unsafe {
				target.get_unchecked_mut(0).write(value);
			}
		}

		alloc_scratch_space::<P, _, _>(self.steps.len(), |evals| {
			let get_argument_value = |input: CircuitStepArgument<F>, evals: &[P]| match input {
				// Safety: The index is guaranteed to be within bounds by the construction of the
				// circuit
				CircuitStepArgument::Expr(CircuitNode::Var(index)) => unsafe {
					*query.get_unchecked(index)
				},
				// Safety: The index is guaranteed to be within bounds by the circuit evaluation
				// order
				CircuitStepArgument::Expr(CircuitNode::Slot(slot)) => unsafe {
					*evals.get_unchecked(slot)
				},
				CircuitStepArgument::Const(value) => P::broadcast(value.into()),
			};

			for (i, expr) in self.steps.iter().enumerate() {
				// Safety: previous evaluations are initialized by the previous loop iterations (if
				// dereferenced)
				let (before, after) = unsafe { evals.split_at_mut_unchecked(i) };
				let before = unsafe { slice_assume_init_mut(before) };
				match expr {
					CircuitStep::Add(x, y) => write_result(
						after,
						get_argument_value(*x, before) + get_argument_value(*y, before),
					),
					CircuitStep::AddMul(target_slot, x, y) => {
						let intermediate =
							get_argument_value(*x, before) * get_argument_value(*y, before);
						// Safety: we know by evaluation order and construction of steps that
						// `target.slot` is initialized
						let target_slot = unsafe { before.get_unchecked_mut(*target_slot) };
						*target_slot += intermediate;
					}
					CircuitStep::Mul(x, y) => write_result(
						after,
						get_argument_value(*x, before) * get_argument_value(*y, before),
					),
					CircuitStep::Square(x) => {
						write_result(after, get_argument_value(*x, before).square())
					}
				};
			}

			// Some slots in `evals` might be empty, but we're guaranteed that
			// if `self.retval` points to a slot, that this slot is initialized.
			unsafe {
				let evals = slice_assume_init_ref(evals);
				Ok(get_argument_value(self.retval, evals))
			}
		})
	}

	fn batch_evaluate(&self, batch_query: &RowsBatchRef<P>, evals: &mut [P]) -> Result<(), Error> {
		let row_len = evals.len();
		if batch_query.row_len() != row_len {
			bail!(Error::BatchEvaluateSizeMismatch {
				expected: row_len,
				actual: batch_query.row_len(),
			});
		}

		alloc_scratch_space::<P, (), _>(self.steps.len() * row_len, |sparse_evals| {
			for (i, expr) in self.steps.iter().enumerate() {
				let (before, current) = sparse_evals.split_at_mut(i * row_len);

				// Safety: `before` is guaranteed to be initialized by the previous loop iterations
				// (if dereferenced).
				let before = unsafe { slice_assume_init_mut(before) };
				let current = &mut current[..row_len];

				match expr {
					CircuitStep::Add(left, right) => {
						apply_binary_op(
							left,
							right,
							batch_query,
							before,
							current,
							|left, right, out| {
								out.write(left + right);
							},
						);
					}
					CircuitStep::Mul(left, right) => {
						apply_binary_op(
							left,
							right,
							batch_query,
							before,
							current,
							|left, right, out| {
								out.write(left * right);
							},
						);
					}
					CircuitStep::Square(arg) => {
						match arg {
							CircuitStepArgument::Expr(node) => {
								let id_chunk = node.get_sparse_chunk(batch_query, before, row_len);
								for j in 0..row_len {
									// Safety: `current` and `id_chunk` have length equal to
									// `row_len`
									unsafe {
										current
											.get_unchecked_mut(j)
											.write(id_chunk.get_unchecked(j).square());
									}
								}
							}
							CircuitStepArgument::Const(value) => {
								let value: P = P::broadcast((*value).into());
								let result = value.square();
								for j in 0..row_len {
									// Safety: `current` has length equal to `row_len`
									unsafe {
										current.get_unchecked_mut(j).write(result);
									}
								}
							}
						}
					}
					CircuitStep::AddMul(target, left, right) => {
						let target = &mut before[row_len * target..(target + 1) * row_len];
						// Safety: by construction of steps and evaluation order we know
						// that `target` is not borrowed elsewhere.
						let target: &mut [MaybeUninit<P>] = unsafe {
							std::slice::from_raw_parts_mut(
								target.as_mut_ptr() as *mut MaybeUninit<P>,
								target.len(),
							)
						};
						apply_binary_op(
							left,
							right,
							batch_query,
							before,
							target,
							// Safety: by construction of steps and evaluation order we know
							// that `target`/`out` is initialized.
							|left, right, out| unsafe {
								let out = out.assume_init_mut();
								*out += left * right;
							},
						);
					}
				}
			}

			match self.retval {
				CircuitStepArgument::Expr(node) => {
					// Safety: `sparse_evals` is fully initialized by the previous loop iterations
					let sparse_evals = unsafe { slice_assume_init_ref(sparse_evals) };
					evals.copy_from_slice(node.get_sparse_chunk(batch_query, sparse_evals, row_len))
				}
				CircuitStepArgument::Const(val) => evals.fill(P::broadcast(val.into())),
			}
		});

		Ok(())
	}
}

/// Apply a binary operation to two arguments and store the result in `current_evals`.
/// `op` must be a function that takes two arguments and initialized the result with the third
/// argument.
fn apply_binary_op<F: Field, P: PackedField<Scalar: ExtensionField<F>>>(
	left: &CircuitStepArgument<F>,
	right: &CircuitStepArgument<F>,
	batch_query: &RowsBatchRef<P>,
	evals_before: &[P],
	current_evals: &mut [MaybeUninit<P>],
	op: impl Fn(P, P, &mut MaybeUninit<P>),
) {
	let row_len = current_evals.len();

	match (left, right) {
		(CircuitStepArgument::Expr(left), CircuitStepArgument::Expr(right)) => {
			let left = left.get_sparse_chunk(batch_query, evals_before, row_len);
			let right = right.get_sparse_chunk(batch_query, evals_before, row_len);
			for j in 0..row_len {
				// Safety: `current`, `left` and `right` have length equal to `row_len`
				unsafe {
					op(
						*left.get_unchecked(j),
						*right.get_unchecked(j),
						current_evals.get_unchecked_mut(j),
					)
				}
			}
		}
		(CircuitStepArgument::Expr(left), CircuitStepArgument::Const(right)) => {
			let left = left.get_sparse_chunk(batch_query, evals_before, row_len);
			let right = P::broadcast((*right).into());
			for j in 0..row_len {
				// Safety: `current` and `left` have length equal to `row_len`
				unsafe {
					op(*left.get_unchecked(j), right, current_evals.get_unchecked_mut(j));
				}
			}
		}
		(CircuitStepArgument::Const(left), CircuitStepArgument::Expr(right)) => {
			let left = P::broadcast((*left).into());
			let right = right.get_sparse_chunk(batch_query, evals_before, row_len);
			for j in 0..row_len {
				// Safety: `current` and `right` have length equal to `row_len`
				unsafe {
					op(left, *right.get_unchecked(j), current_evals.get_unchecked_mut(j));
				}
			}
		}
		(CircuitStepArgument::Const(left), CircuitStepArgument::Const(right)) => {
			let left = P::broadcast((*left).into());
			let right = P::broadcast((*right).into());
			let mut result = MaybeUninit::uninit();
			op(left, right, &mut result);
			for j in 0..row_len {
				// Safety:
				// - `current` has length equal to `row_len`
				// - `result` is initialized by `op`
				unsafe {
					current_evals
						.get_unchecked_mut(j)
						.write(result.assume_init());
				}
			}
		}
	}
}

fn alloc_scratch_space<T, U, F>(size: usize, callback: F) -> U
where
	F: FnOnce(&mut [MaybeUninit<T>]) -> U,
{
	use std::mem;
	// We don't want to deal with running destructors.
	assert!(!mem::needs_drop::<T>());

	#[cfg(miri)]
	{
		let mut scratch_space = Vec::<T>::with_capacity(size);
		let out = callback(scratch_space.spare_capacity_mut());
		drop(scratch_space);
		out
	}
	#[cfg(not(miri))]
	{
		// `stackalloc_uninit` throws a debug assert if `size` is 0, so set minimum of 1.
		let size = size.max(1);
		stackalloc::stackalloc_uninit(size, callback)
	}
}

#[cfg(test)]
mod tests {
	use binius_field::{
		BinaryField8b, BinaryField16b, PackedBinaryField8x16b, PackedField, TowerField,
	};
	use binius_math::{ArithExpr, CompositionPoly, RowsBatch};
	use binius_utils::felts;

	use super::*;

	#[test]
	fn test_constant() {
		type F = BinaryField8b;
		type P = PackedBinaryField8x16b;

		let expr = ArithExpr::Const(F::new(123));
		let circuit = ArithCircuitPoly::<F>::new(expr.into());

		let typed_circuit: &dyn CompositionPoly<P> = &circuit;
		assert_eq!(typed_circuit.binary_tower_level(), F::TOWER_LEVEL);
		assert_eq!(typed_circuit.degree(), 0);
		assert_eq!(typed_circuit.n_vars(), 0);

		assert_eq!(typed_circuit.evaluate(&[]).unwrap(), P::broadcast(F::new(123).into()));

		let mut evals = [P::default()];
		typed_circuit
			.batch_evaluate(&RowsBatchRef::new(&[], 1), &mut evals)
			.unwrap();
		assert_eq!(evals, [P::broadcast(F::new(123).into())]);
	}

	#[test]
	fn test_identity() {
		type F = BinaryField8b;
		type P = PackedBinaryField8x16b;

		// x0
		let expr = ArithExpr::Var(0);
		let circuit = ArithCircuitPoly::<F>::new(expr.into());

		let typed_circuit: &dyn CompositionPoly<P> = &circuit;
		assert_eq!(typed_circuit.binary_tower_level(), 0);
		assert_eq!(typed_circuit.degree(), 1);
		assert_eq!(typed_circuit.n_vars(), 1);

		assert_eq!(
			typed_circuit
				.evaluate(&[P::broadcast(F::new(123).into())])
				.unwrap(),
			P::broadcast(F::new(123).into())
		);

		let mut evals = [P::default()];
		let batch_query = [[P::broadcast(F::new(123).into())]; 1];
		let batch_query = RowsBatch::new_from_iter(batch_query.iter().map(|x| x.as_slice()), 1);
		typed_circuit
			.batch_evaluate(&batch_query.get_ref(), &mut evals)
			.unwrap();
		assert_eq!(evals, [P::broadcast(F::new(123).into())]);
	}

	#[test]
	fn test_add() {
		type F = BinaryField8b;
		type P = PackedBinaryField8x16b;

		// 123 + x0
		let expr = ArithExpr::Const(F::new(123)) + ArithExpr::Var(0);
		let circuit = ArithCircuitPoly::<F>::new(expr.into());

		let typed_circuit: &dyn CompositionPoly<P> = &circuit;
		assert_eq!(typed_circuit.binary_tower_level(), 3);
		assert_eq!(typed_circuit.degree(), 1);
		assert_eq!(typed_circuit.n_vars(), 1);

		assert_eq!(
			CompositionPoly::evaluate(&circuit, &[P::broadcast(F::new(0).into())]).unwrap(),
			P::broadcast(F::new(123).into())
		);
	}

	#[test]
	fn test_mul() {
		type F = BinaryField8b;
		type P = PackedBinaryField8x16b;

		// 123 * x0
		let expr = ArithExpr::Const(F::new(123)) * ArithExpr::Var(0);
		let circuit = ArithCircuitPoly::<F>::new(expr.into());

		let typed_circuit: &dyn CompositionPoly<P> = &circuit;
		assert_eq!(typed_circuit.binary_tower_level(), 3);
		assert_eq!(typed_circuit.degree(), 1);
		assert_eq!(typed_circuit.n_vars(), 1);

		assert_eq!(
			CompositionPoly::evaluate(
				&circuit,
				&[P::from_scalars(
					felts!(BinaryField16b[0, 1, 2, 3, 122, 123, 124, 125]),
				)]
			)
			.unwrap(),
			P::from_scalars(felts!(BinaryField16b[0, 123, 157, 230, 85, 46, 154, 225])),
		);
	}

	#[test]
	fn test_pow() {
		type F = BinaryField8b;
		type P = PackedBinaryField8x16b;

		// x0^13
		let expr = ArithExpr::Var(0).pow(13);
		let circuit = ArithCircuitPoly::<F>::new(expr.into());

		let typed_circuit: &dyn CompositionPoly<P> = &circuit;
		assert_eq!(typed_circuit.binary_tower_level(), 0);
		assert_eq!(typed_circuit.degree(), 13);
		assert_eq!(typed_circuit.n_vars(), 1);

		assert_eq!(
			CompositionPoly::evaluate(
				&circuit,
				&[P::from_scalars(
					felts!(BinaryField16b[0, 1, 2, 3, 122, 123, 124, 125]),
				)]
			)
			.unwrap(),
			P::from_scalars(felts!(BinaryField16b[0, 1, 2, 3, 200, 52, 51, 115])),
		);
	}

	#[test]
	fn test_mixed() {
		type F = BinaryField8b;
		type P = PackedBinaryField8x16b;

		// x0^2 * (x1 + 123)
		let expr = ArithExpr::Var(0).pow(2) * (ArithExpr::Var(1) + ArithExpr::Const(F::new(123)));
		let circuit = ArithCircuitPoly::<F>::new(expr.into());

		let typed_circuit: &dyn CompositionPoly<P> = &circuit;
		assert_eq!(typed_circuit.binary_tower_level(), 3);
		assert_eq!(typed_circuit.degree(), 3);
		assert_eq!(typed_circuit.n_vars(), 2);

		// test evaluate
		assert_eq!(
			CompositionPoly::evaluate(
				&circuit,
				&[
					P::from_scalars(felts!(BinaryField16b[0, 1, 2, 3, 4, 5, 6, 7])),
					P::from_scalars(felts!(BinaryField16b[100, 101, 102, 103, 104, 105, 106, 107])),
				]
			)
			.unwrap(),
			P::from_scalars(felts!(BinaryField16b[0, 30, 59, 36, 151, 140, 170, 176])),
		);

		// test batch evaluate
		let query1 = &[
			P::from_scalars(felts!(BinaryField16b[0, 0, 0, 0, 0, 0, 0, 0])),
			P::from_scalars(felts!(BinaryField16b[0, 0, 0, 0, 0, 0, 0, 0])),
		];
		let query2 = &[
			P::from_scalars(felts!(BinaryField16b[1, 1, 1, 1, 1, 1, 1, 1])),
			P::from_scalars(felts!(BinaryField16b[0, 1, 2, 3, 4, 5, 6, 7])),
		];
		let query3 = &[
			P::from_scalars(felts!(BinaryField16b[0, 1, 2, 3, 4, 5, 6, 7])),
			P::from_scalars(felts!(BinaryField16b[100, 101, 102, 103, 104, 105, 106, 107])),
		];
		let expected1 = P::from_scalars(felts!(BinaryField16b[0, 0, 0, 0, 0, 0, 0, 0]));
		let expected2 =
			P::from_scalars(felts!(BinaryField16b[123, 122, 121, 120, 127, 126, 125, 124]));
		let expected3 = P::from_scalars(felts!(BinaryField16b[0, 30, 59, 36, 151, 140, 170, 176]));

		let mut batch_result = vec![P::zero(); 3];
		let batch_query = &[
			&[query1[0], query2[0], query3[0]],
			&[query1[1], query2[1], query3[1]],
		];
		let batch_query = RowsBatch::new_from_iter(batch_query.iter().map(|x| x.as_slice()), 3);

		CompositionPoly::batch_evaluate(&circuit, &batch_query.get_ref(), &mut batch_result)
			.unwrap();
		assert_eq!(&batch_result, &[expected1, expected2, expected3]);
	}

	#[test]
	fn batch_evaluate_add_mul() {
		// This test is focused on exposing the the currently present stacked borrows violation. It
		// passes but still triggers `miri`.

		type F = BinaryField8b;
		type P = PackedBinaryField8x16b;

		let expr = (ArithExpr::<F>::Var(0) * ArithExpr::Var(0))
			+ (ArithExpr::Const(F::ONE) - ArithExpr::Var(0)) * ArithExpr::Var(0)
			- ArithExpr::Var(0);
		let circuit = ArithCircuitPoly::<F>::new(expr.into());

		let typed_circuit: &dyn CompositionPoly<P> = &circuit;
		assert_eq!(typed_circuit.binary_tower_level(), 0);
		assert_eq!(typed_circuit.degree(), 2);
		assert_eq!(typed_circuit.n_vars(), 1);

		let mut evals = [P::default(); 1];
		let batch_query = [[P::broadcast(F::new(1).into())]; 1];
		let batch_query = RowsBatch::new_from_iter(batch_query.iter().map(|x| x.as_slice()), 1);
		typed_circuit
			.batch_evaluate(&batch_query.get_ref(), &mut evals)
			.unwrap();
	}

	#[test]
	fn test_const_fold() {
		type F = BinaryField8b;
		type P = PackedBinaryField8x16b;

		// x0 * ((122 * 123) + (124 + 125)) + x1
		let expr = ArithExpr::Var(0)
			* ((ArithExpr::Const(F::new(122)) * ArithExpr::Const(F::new(123)))
				+ (ArithExpr::Const(F::new(124)) + ArithExpr::Const(F::new(125))))
			+ ArithExpr::Var(1);
		let circuit = ArithCircuitPoly::<F>::new(expr.into());
		assert_eq!(circuit.steps.len(), 2);

		let typed_circuit: &dyn CompositionPoly<P> = &circuit;
		assert_eq!(typed_circuit.binary_tower_level(), F::TOWER_LEVEL);
		assert_eq!(typed_circuit.degree(), 1);
		assert_eq!(typed_circuit.n_vars(), 2);

		// test evaluate
		assert_eq!(
			CompositionPoly::evaluate(
				&circuit,
				&[
					P::from_scalars(felts!(BinaryField16b[0, 1, 2, 3, 4, 5, 6, 7])),
					P::from_scalars(felts!(BinaryField16b[100, 101, 102, 103, 104, 105, 106, 107])),
				]
			)
			.unwrap(),
			P::from_scalars(felts!(BinaryField16b[100, 49, 206, 155, 177, 228, 27, 78])),
		);

		// test batch evaluate
		let query1 = &[
			P::from_scalars(felts!(BinaryField16b[0, 0, 0, 0, 0, 0, 0, 0])),
			P::from_scalars(felts!(BinaryField16b[0, 0, 0, 0, 0, 0, 0, 0])),
		];
		let query2 = &[
			P::from_scalars(felts!(BinaryField16b[1, 1, 1, 1, 1, 1, 1, 1])),
			P::from_scalars(felts!(BinaryField16b[0, 1, 2, 3, 4, 5, 6, 7])),
		];
		let query3 = &[
			P::from_scalars(felts!(BinaryField16b[0, 1, 2, 3, 4, 5, 6, 7])),
			P::from_scalars(felts!(BinaryField16b[100, 101, 102, 103, 104, 105, 106, 107])),
		];
		let expected1 = P::from_scalars(felts!(BinaryField16b[0, 0, 0, 0, 0, 0, 0, 0]));
		let expected2 = P::from_scalars(felts!(BinaryField16b[84, 85, 86, 87, 80, 81, 82, 83]));
		let expected3 =
			P::from_scalars(felts!(BinaryField16b[100, 49, 206, 155, 177, 228, 27, 78]));

		let mut batch_result = vec![P::zero(); 3];
		let batch_query = &[
			&[query1[0], query2[0], query3[0]],
			&[query1[1], query2[1], query3[1]],
		];
		let batch_query = RowsBatch::new_from_iter(batch_query.iter().map(|x| x.as_slice()), 3);
		CompositionPoly::batch_evaluate(&circuit, &batch_query.get_ref(), &mut batch_result)
			.unwrap();
		assert_eq!(&batch_result, &[expected1, expected2, expected3]);
	}

	#[test]
	fn test_pow_const_fold() {
		type F = BinaryField8b;
		type P = PackedBinaryField8x16b;

		// x0 + 2^5
		let expr = ArithExpr::Var(0) + ArithExpr::Const(F::from(2)).pow(4);
		let circuit = ArithCircuitPoly::<F>::new(expr.into());
		assert_eq!(circuit.steps.len(), 1);

		let typed_circuit: &dyn CompositionPoly<P> = &circuit;
		assert_eq!(typed_circuit.binary_tower_level(), 1);
		assert_eq!(typed_circuit.degree(), 1);
		assert_eq!(typed_circuit.n_vars(), 1);

		assert_eq!(
			CompositionPoly::evaluate(
				&circuit,
				&[P::from_scalars(
					felts!(BinaryField16b[0, 1, 2, 3, 122, 123, 124, 125]),
				)]
			)
			.unwrap(),
			P::from_scalars(felts!(BinaryField16b[2, 3, 0, 1, 120, 121, 126, 127])),
		);
	}

	#[test]
	fn test_pow_nested() {
		type F = BinaryField8b;
		type P = PackedBinaryField8x16b;

		// ((x0^2)^3)^4
		let expr = ArithExpr::Var(0).pow(2).pow(3).pow(4);
		let circuit = ArithCircuitPoly::<F>::new(expr.into());
		assert_eq!(circuit.steps.len(), 5);

		let typed_circuit: &dyn CompositionPoly<P> = &circuit;
		assert_eq!(typed_circuit.binary_tower_level(), 0);
		assert_eq!(typed_circuit.degree(), 24);
		assert_eq!(typed_circuit.n_vars(), 1);

		assert_eq!(
			CompositionPoly::evaluate(
				&circuit,
				&[P::from_scalars(
					felts!(BinaryField16b[0, 1, 2, 3, 122, 123, 124, 125]),
				)]
			)
			.unwrap(),
			P::from_scalars(felts!(BinaryField16b[0, 1, 1, 1, 20, 152, 41, 170])),
		);
	}

	#[test]
	fn test_circuit_steps_for_expr_constant() {
		type F = BinaryField8b;

		let expr = ArithExpr::Const(F::new(5));
		let (steps, retval) = convert_circuit_steps(&expr.into());

		assert!(steps.is_empty(), "No steps should be generated for a constant");
		assert_eq!(retval, CircuitStepArgument::Const(F::new(5)));
	}

	#[test]
	fn test_circuit_steps_for_expr_variable() {
		type F = BinaryField8b;

		let expr = ArithExpr::<F>::Var(18);
		let (steps, retval) = convert_circuit_steps(&expr.into());

		assert!(steps.is_empty(), "No steps should be generated for a variable");
		assert!(matches!(retval, CircuitStepArgument::Expr(CircuitNode::Var(18))));
	}

	#[test]
	fn test_circuit_steps_for_expr_addition() {
		type F = BinaryField8b;

		let expr = ArithExpr::<F>::Var(14) + ArithExpr::<F>::Var(56);
		let (steps, retval) = convert_circuit_steps(&expr.into());

		assert_eq!(steps.len(), 1, "One addition step should be generated");
		assert!(matches!(
			steps[0],
			CircuitStep::Add(
				CircuitStepArgument::Expr(CircuitNode::Var(14)),
				CircuitStepArgument::Expr(CircuitNode::Var(56))
			)
		));
		assert!(matches!(retval, CircuitStepArgument::Expr(CircuitNode::Slot(0))));
	}

	#[test]
	fn test_circuit_steps_for_expr_multiplication() {
		type F = BinaryField8b;

		let expr = ArithExpr::<F>::Var(36) * ArithExpr::Var(26);
		let (steps, retval) = convert_circuit_steps(&expr.into());

		assert_eq!(steps.len(), 1, "One multiplication step should be generated");
		assert!(matches!(
			steps[0],
			CircuitStep::Mul(
				CircuitStepArgument::Expr(CircuitNode::Var(36)),
				CircuitStepArgument::Expr(CircuitNode::Var(26))
			)
		));
		assert!(matches!(retval, CircuitStepArgument::Expr(CircuitNode::Slot(0))));
	}

	#[test]
	fn test_circuit_steps_for_expr_pow_1() {
		type F = BinaryField8b;

		let expr = ArithExpr::<F>::Var(12).pow(1);
		let (steps, retval) = convert_circuit_steps(&expr.into());

		// No steps should be generated for x^1
		assert_eq!(steps.len(), 0, "Pow(1) should not generate any computation steps");

		// The return value should just be the variable itself
		assert!(matches!(retval, CircuitStepArgument::Expr(CircuitNode::Var(12))));
	}

	#[test]
	fn test_circuit_steps_for_expr_pow_2() {
		type F = BinaryField8b;

		let expr = ArithExpr::<F>::Var(10).pow(2);
		let (steps, retval) = convert_circuit_steps(&expr.into());

		assert_eq!(steps.len(), 1, "Pow(2) should generate one squaring step");
		assert!(matches!(
			steps[0],
			CircuitStep::Square(CircuitStepArgument::Expr(CircuitNode::Var(10)))
		));
		assert!(matches!(retval, CircuitStepArgument::Expr(CircuitNode::Slot(0))));
	}

	#[test]
	fn test_circuit_steps_for_expr_pow_3() {
		type F = BinaryField8b;

		let expr = ArithExpr::<F>::Var(5).pow(3);
		let (steps, retval) = convert_circuit_steps(&expr.into());

		assert_eq!(
			steps.len(),
			2,
			"Pow(3) should generate one squaring and one multiplication step"
		);
		assert!(matches!(
			steps[0],
			CircuitStep::Square(CircuitStepArgument::Expr(CircuitNode::Var(5)))
		));
		assert!(matches!(
			steps[1],
			CircuitStep::Mul(
				CircuitStepArgument::Expr(CircuitNode::Slot(0)),
				CircuitStepArgument::Expr(CircuitNode::Var(5))
			)
		));
		assert!(matches!(retval, CircuitStepArgument::Expr(CircuitNode::Slot(1))));
	}

	#[test]
	fn test_circuit_steps_for_expr_pow_4() {
		type F = BinaryField8b;

		let expr = ArithExpr::<F>::Var(7).pow(4);
		let (steps, retval) = convert_circuit_steps(&expr.into());

		assert_eq!(steps.len(), 2, "Pow(4) should generate two squaring steps");
		assert!(matches!(
			steps[0],
			CircuitStep::Square(CircuitStepArgument::Expr(CircuitNode::Var(7)))
		));

		assert!(matches!(
			steps[1],
			CircuitStep::Square(CircuitStepArgument::Expr(CircuitNode::Slot(0)))
		));

		assert!(matches!(retval, CircuitStepArgument::Expr(CircuitNode::Slot(1))));
	}

	#[test]
	fn test_circuit_steps_for_expr_pow_5() {
		type F = BinaryField8b;

		let expr = ArithExpr::<F>::Var(3).pow(5);
		let (steps, retval) = convert_circuit_steps(&expr.into());

		assert_eq!(
			steps.len(),
			3,
			"Pow(5) should generate two squaring steps and one multiplication"
		);
		assert!(matches!(
			steps[0],
			CircuitStep::Square(CircuitStepArgument::Expr(CircuitNode::Var(3)))
		));
		assert!(matches!(
			steps[1],
			CircuitStep::Square(CircuitStepArgument::Expr(CircuitNode::Slot(0)))
		));
		assert!(matches!(
			steps[2],
			CircuitStep::Mul(
				CircuitStepArgument::Expr(CircuitNode::Slot(1)),
				CircuitStepArgument::Expr(CircuitNode::Var(3))
			)
		));

		assert!(matches!(retval, CircuitStepArgument::Expr(CircuitNode::Slot(2))));
	}

	#[test]
	fn test_circuit_steps_for_expr_pow_8() {
		type F = BinaryField8b;

		let expr = ArithExpr::<F>::Var(4).pow(8);
		let (steps, retval) = convert_circuit_steps(&expr.into());

		assert_eq!(steps.len(), 3, "Pow(8) should generate three squaring steps");
		assert!(matches!(
			steps[0],
			CircuitStep::Square(CircuitStepArgument::Expr(CircuitNode::Var(4)))
		));
		assert!(matches!(
			steps[1],
			CircuitStep::Square(CircuitStepArgument::Expr(CircuitNode::Slot(0)))
		));
		assert!(matches!(
			steps[2],
			CircuitStep::Square(CircuitStepArgument::Expr(CircuitNode::Slot(1)))
		));

		assert!(matches!(retval, CircuitStepArgument::Expr(CircuitNode::Slot(2))));
	}

	#[test]
	fn test_circuit_steps_for_expr_pow_9() {
		type F = BinaryField8b;

		let expr = ArithExpr::<F>::Var(8).pow(9);
		let (steps, retval) = convert_circuit_steps(&expr.into());

		assert_eq!(
			steps.len(),
			4,
			"Pow(9) should generate three squaring steps and one multiplication"
		);
		assert!(matches!(
			steps[0],
			CircuitStep::Square(CircuitStepArgument::Expr(CircuitNode::Var(8)))
		));
		assert!(matches!(
			steps[1],
			CircuitStep::Square(CircuitStepArgument::Expr(CircuitNode::Slot(0)))
		));
		assert!(matches!(
			steps[2],
			CircuitStep::Square(CircuitStepArgument::Expr(CircuitNode::Slot(1)))
		));
		assert!(matches!(
			steps[3],
			CircuitStep::Mul(
				CircuitStepArgument::Expr(CircuitNode::Slot(2)),
				CircuitStepArgument::Expr(CircuitNode::Var(8))
			)
		));

		assert!(matches!(retval, CircuitStepArgument::Expr(CircuitNode::Slot(3))));
	}

	#[test]
	fn test_circuit_steps_for_expr_pow_12() {
		type F = BinaryField8b;
		let expr = ArithExpr::<F>::Var(6).pow(12);
		let (steps, retval) = convert_circuit_steps(&expr.into());

		assert_eq!(steps.len(), 4, "Pow(12) should use 4 steps.");

		assert!(matches!(
			steps[0],
			CircuitStep::Square(CircuitStepArgument::Expr(CircuitNode::Var(6)))
		));
		assert!(matches!(
			steps[1],
			CircuitStep::Mul(
				CircuitStepArgument::Expr(CircuitNode::Slot(0)),
				CircuitStepArgument::Expr(CircuitNode::Var(6))
			)
		));
		assert!(matches!(
			steps[2],
			CircuitStep::Square(CircuitStepArgument::Expr(CircuitNode::Slot(1)))
		));
		assert!(matches!(
			steps[3],
			CircuitStep::Square(CircuitStepArgument::Expr(CircuitNode::Slot(2)))
		));

		assert!(matches!(retval, CircuitStepArgument::Expr(CircuitNode::Slot(3))));
	}

	#[test]
	fn test_circuit_steps_for_expr_pow_13() {
		type F = BinaryField8b;
		let expr = ArithExpr::<F>::Var(7).pow(13);
		let (steps, retval) = convert_circuit_steps(&expr.into());

		assert_eq!(steps.len(), 5, "Pow(13) should use 5 steps.");
		assert!(matches!(
			steps[0],
			CircuitStep::Square(CircuitStepArgument::Expr(CircuitNode::Var(7)))
		));
		assert!(matches!(
			steps[1],
			CircuitStep::Mul(
				CircuitStepArgument::Expr(CircuitNode::Slot(0)),
				CircuitStepArgument::Expr(CircuitNode::Var(7))
			)
		));
		assert!(matches!(
			steps[2],
			CircuitStep::Square(CircuitStepArgument::Expr(CircuitNode::Slot(1)))
		));
		assert!(matches!(
			steps[3],
			CircuitStep::Square(CircuitStepArgument::Expr(CircuitNode::Slot(2)))
		));
		assert!(matches!(
			steps[4],
			CircuitStep::Mul(
				CircuitStepArgument::Expr(CircuitNode::Slot(3)),
				CircuitStepArgument::Expr(CircuitNode::Var(7))
			)
		));
		assert!(matches!(retval, CircuitStepArgument::Expr(CircuitNode::Slot(4))));
	}

	#[test]
	fn test_circuit_steps_for_expr_complex() {
		type F = BinaryField8b;

		let expr = (ArithExpr::<F>::Var(0) * ArithExpr::Var(1))
			+ (ArithExpr::Const(F::ONE) - ArithExpr::Var(0)) * ArithExpr::Var(2)
			- ArithExpr::Var(3);

		let (steps, retval) = convert_circuit_steps(&expr.into());

		assert_eq!(steps.len(), 4, "Expression should generate 4 computation steps");

		assert!(
			matches!(
				steps[0],
				CircuitStep::Mul(
					CircuitStepArgument::Expr(CircuitNode::Var(0)),
					CircuitStepArgument::Expr(CircuitNode::Var(1))
				)
			),
			"First step should be multiplication x0 * x1"
		);

		assert!(
			matches!(
				steps[1],
				CircuitStep::Add(
					CircuitStepArgument::Const(F::ONE),
					CircuitStepArgument::Expr(CircuitNode::Var(0))
				)
			),
			"Second step should be (1 - x0)"
		);

		assert!(
			matches!(
				steps[2],
				CircuitStep::AddMul(
					0,
					CircuitStepArgument::Expr(CircuitNode::Slot(1)),
					CircuitStepArgument::Expr(CircuitNode::Var(2))
				)
			),
			"Third step should be (1 - x0) * x2"
		);

		assert!(
			matches!(
				steps[3],
				CircuitStep::Add(
					CircuitStepArgument::Expr(CircuitNode::Slot(0)),
					CircuitStepArgument::Expr(CircuitNode::Var(3))
				)
			),
			"Fourth step should be x0 * x1 + (1 - x0) * x2 + x3"
		);

		assert!(
			matches!(retval, CircuitStepArgument::Expr(CircuitNode::Slot(3))),
			"Final result should be stored in Slot(3)"
		);
	}

	#[test]
	fn check_deduplication_in_steps() {
		type F = BinaryField8b;

		let expr = (ArithExpr::<F>::Var(0) * ArithExpr::Var(1))
			+ (ArithExpr::<F>::Var(0) * ArithExpr::Var(1)) * ArithExpr::Var(2)
			- ArithExpr::Var(3);
		let expr = ArithCircuit::from(&expr);
		let expr = expr.optimize();

		let (steps, retval) = convert_circuit_steps(&expr);

		assert_eq!(steps.len(), 3, "Expression should generate 3 computation steps");

		assert!(
			matches!(
				steps[0],
				CircuitStep::Mul(
					CircuitStepArgument::Expr(CircuitNode::Var(0)),
					CircuitStepArgument::Expr(CircuitNode::Var(1))
				)
			),
			"First step should be multiplication x0 * x1"
		);

		assert!(
			matches!(
				steps[1],
				CircuitStep::AddMul(
					0,
					CircuitStepArgument::Expr(CircuitNode::Slot(0)),
					CircuitStepArgument::Expr(CircuitNode::Var(2))
				)
			),
			"Second step should be (x0 * x1) * x2"
		);

		assert!(
			matches!(
				steps[2],
				CircuitStep::Add(
					CircuitStepArgument::Expr(CircuitNode::Slot(0)),
					CircuitStepArgument::Expr(CircuitNode::Var(3))
				)
			),
			"Third step should be x0 * x1 + (x0 * x1) * x2 + x3"
		);

		assert!(
			matches!(retval, CircuitStepArgument::Expr(CircuitNode::Slot(2))),
			"Final result should be stored in Slot(2)"
		);
	}
}
