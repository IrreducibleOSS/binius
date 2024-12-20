// Copyright 2024 Irreducible Inc.

use std::{fmt::Debug, mem::MaybeUninit, sync::Arc};

use binius_field::{ExtensionField, Field, PackedField, TowerField};
use binius_math::{ArithExpr, CompositionPoly, CompositionPolyOS, Error};
use stackalloc::{helpers::slice_assume_init, stackalloc_uninit};

/// Convert the expression to a sequence of arithmetic operations that can be evaluated in sequence.
fn circuit_steps_for_expr<F: Field>(
	expr: &ArithExpr<F>,
) -> (Vec<CircuitStep<F>>, CircuitStepArgument<F>) {
	let mut steps = Vec::new();

	fn to_circuit_inner<F: Field>(
		expr: &ArithExpr<F>,
		result: &mut Vec<CircuitStep<F>>,
	) -> CircuitStepArgument<F> {
		match expr {
			ArithExpr::Const(value) => CircuitStepArgument::Const(*value),
			ArithExpr::Var(index) => CircuitStepArgument::Expr(CircuitNode::Var(*index)),
			ArithExpr::Add(left, right) => {
				let left = to_circuit_inner(left, result);
				let right = to_circuit_inner(right, result);
				result.push(CircuitStep::Add(left, right));
				CircuitStepArgument::Expr(CircuitNode::Slot(result.len() - 1))
			}
			ArithExpr::Mul(left, right) => {
				let left = to_circuit_inner(left, result);
				let right = to_circuit_inner(right, result);
				result.push(CircuitStep::Mul(left, right));
				CircuitStepArgument::Expr(CircuitNode::Slot(result.len() - 1))
			}
			ArithExpr::Pow(id, exp) => {
				let id = to_circuit_inner(id, result);
				result.push(CircuitStep::Pow(id, *exp));
				CircuitStepArgument::Expr(CircuitNode::Slot(result.len() - 1))
			}
		}
	}

	let ret = to_circuit_inner(expr, &mut steps);
	(steps, ret)
}

/// Input of the circuit calculation step
#[derive(Debug, Clone, Copy)]
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
		&self,
		inputs: &[&'a [P]],
		evals: &'a [P],
		row_len: usize,
	) -> &'a [P] {
		match self {
			CircuitNode::Var(index) => inputs[*index],
			CircuitNode::Slot(slot) => &evals[slot * row_len..(slot + 1) * row_len],
		}
	}
}

#[derive(Debug, Clone, Copy)]
enum CircuitStepArgument<F> {
	Expr(CircuitNode),
	Const(F),
}

/// Describes computation symbolically. This is used internally by ArithCircuitPoly.
///
/// ExprIds used by an Expr has to be less than the index of the Expr itself within the ArithCircuitPoly,
/// to ensure it represents a directed acyclic graph that can be computed in sequence.
#[derive(Debug)]
enum CircuitStep<F: Field> {
	Add(CircuitStepArgument<F>, CircuitStepArgument<F>),
	Mul(CircuitStepArgument<F>, CircuitStepArgument<F>),
	Pow(CircuitStepArgument<F>, u64),
}

/// Describes polynomial evaluations using a directed acyclic graph of expressions.
///
/// This is meant as an alternative to a hard-coded CompositionPolyOS.
///
/// The advantage over a hard coded CompositionPolyOS is that this can be constructed and manipulated dynamically at runtime
/// and the object representing different polnomials can be stored in a homogeneous collection.
#[derive(Debug, Clone)]
pub struct ArithCircuitPoly<F: Field> {
	expr: ArithExpr<F>,
	steps: Arc<[CircuitStep<F>]>,
	/// The "top level expression", which depends on circuit expression evaluations
	retval: CircuitStepArgument<F>,
	degree: usize,
	n_vars: usize,
}

impl<F: Field> ArithCircuitPoly<F> {
	pub fn new(expr: ArithExpr<F>) -> Self {
		let degree = expr.degree();
		let n_vars = expr.n_vars();
		let (exprs, retval) = circuit_steps_for_expr(&expr);

		Self {
			expr,
			steps: exprs.into(),
			retval,
			degree,
			n_vars,
		}
	}

	/// Constructs an [`ArithCircuitPoly`] with the given number of variables.
	///
	/// The number of variables may be greater than the number of variables actually read in the
	/// arithmetic expression.
	pub fn with_n_vars(n_vars: usize, expr: ArithExpr<F>) -> Result<Self, Error> {
		let degree = expr.degree();
		if n_vars < expr.n_vars() {
			return Err(Error::IncorrectNumberOfVariables {
				expected: expr.n_vars(),
				actual: n_vars,
			});
		}
		let (exprs, retval) = circuit_steps_for_expr(&expr);

		Ok(Self {
			expr,
			steps: exprs.into(),
			retval,
			n_vars,
			degree,
		})
	}
}

impl<F: TowerField> CompositionPoly<F> for ArithCircuitPoly<F> {
	fn degree(&self) -> usize {
		self.degree
	}

	fn n_vars(&self) -> usize {
		self.n_vars
	}

	fn binary_tower_level(&self) -> usize {
		F::TOWER_LEVEL
	}

	fn expression<FE: ExtensionField<F>>(&self) -> ArithExpr<FE> {
		self.expr.convert_field()
	}

	fn evaluate<P: PackedField<Scalar: ExtensionField<F>>>(&self, query: &[P]) -> Result<P, Error> {
		if query.len() != self.n_vars {
			return Err(Error::IncorrectQuerySize {
				expected: self.n_vars,
			});
		}

		// `stackalloc_uninit` throws a debug assert if `size` is 0, so set minimum of 1.
		stackalloc_uninit::<P, _, _>(self.steps.len().max(1), |evals| {
			let get_argument_value = |input: CircuitStepArgument<F>, evals: &[P]| match input {
				// Safety: The index is guaranteed to be within bounds by the construction of the circuit
				CircuitStepArgument::Expr(CircuitNode::Var(index)) => unsafe {
					*query.get_unchecked(index)
				},
				// Safety: The index is guaranteed to be within bounds by the circuit evaluation order
				CircuitStepArgument::Expr(CircuitNode::Slot(slot)) => unsafe {
					*evals.get_unchecked(slot)
				},
				CircuitStepArgument::Const(value) => P::broadcast(value.into()),
			};

			for (i, expr) in self.steps.iter().enumerate() {
				// Safety: previous evaluations are initialized by the previous loop iterations
				let (before, after) = unsafe { evals.split_at_mut_unchecked(i) };
				let before = unsafe { slice_assume_init(before) };
				let new_val = match expr {
					CircuitStep::Add(x, y) => {
						get_argument_value(*x, before) + get_argument_value(*y, before)
					}
					CircuitStep::Mul(x, y) => {
						get_argument_value(*x, before) * get_argument_value(*y, before)
					}
					CircuitStep::Pow(id, exp) => pow(get_argument_value(*id, before), *exp),
				};

				// Safety: `evals.len()` == `self.exprs.len()`, so `after` is guaranteed to have at least one element
				unsafe {
					after.get_unchecked_mut(0).write(new_val);
				}
			}

			// Safety: `evals.len()` == `self.exprs.len()`, and all expression evaluations have
			// been initialized
			unsafe {
				let evals = slice_assume_init(evals);
				Ok(get_argument_value(self.retval, evals))
			}
		})
	}

	fn batch_evaluate<P: PackedField<Scalar: ExtensionField<F>>>(
		&self,
		batch_query: &[&[P]],
		evals: &mut [P],
	) -> Result<(), Error> {
		let row_len = evals.len();
		if batch_query.iter().any(|row| row.len() != row_len) {
			return Err(Error::BatchEvaluateSizeMismatch);
		}

		// `stackalloc_uninit` throws a debug assert if `size` is 0, so set minimum of 1.
		stackalloc_uninit::<P, (), _>((self.steps.len() * row_len).max(1), |sparse_evals| {
			for (i, expr) in self.steps.iter().enumerate() {
				let (before, current) = sparse_evals.split_at_mut(i * row_len);

				// Safety: `before` is guaranteed to be initialized by the previous loop iterations.
				let before = unsafe { slice_assume_init(before) };
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
					CircuitStep::Pow(id, exp) => match id {
						CircuitStepArgument::Expr(id) => {
							let id = id.get_sparse_chunk(batch_query, before, row_len);
							for j in 0..row_len {
								// Safety: `current` and `id` have length equal to `row_len`
								unsafe {
									current
										.get_unchecked_mut(j)
										.write(pow(*id.get_unchecked(j), *exp));
								}
							}
						}
						CircuitStepArgument::Const(id) => {
							let id: P = P::broadcast((*id).into());
							let result = pow(id, *exp);
							for j in 0..row_len {
								// Safety: `current` has length equal to `row_len`
								unsafe {
									current.get_unchecked_mut(j).write(result);
								}
							}
						}
					},
				}
			}

			match self.retval {
				CircuitStepArgument::Expr(node) => {
					// Safety: `sparse_evals` is fully initialized by the previous loop iterations
					let sparse_evals = unsafe { slice_assume_init(sparse_evals) };
					evals.copy_from_slice(node.get_sparse_chunk(batch_query, sparse_evals, row_len))
				}
				CircuitStepArgument::Const(val) => evals.fill(P::broadcast(val.into())),
			}
		});

		Ok(())
	}
}

impl<F: TowerField, P: PackedField<Scalar: ExtensionField<F>>> CompositionPolyOS<P>
	for ArithCircuitPoly<F>
{
	fn degree(&self) -> usize {
		CompositionPoly::degree(self)
	}

	fn n_vars(&self) -> usize {
		CompositionPoly::n_vars(self)
	}

	fn expression(&self) -> ArithExpr<P::Scalar> {
		self.expr.convert_field()
	}

	fn binary_tower_level(&self) -> usize {
		CompositionPoly::binary_tower_level(self)
	}

	fn evaluate(&self, query: &[P]) -> Result<P, Error> {
		CompositionPoly::evaluate(self, query)
	}

	fn batch_evaluate(&self, batch_query: &[&[P]], evals: &mut [P]) -> Result<(), Error> {
		CompositionPoly::batch_evaluate(self, batch_query, evals)
	}
}

/// Apply a binary operation to two arguments and store the result in `current_evals`.
/// `op` must be a function that takes two arguments and initialized the result with the third argument.
fn apply_binary_op<F: Field, P: PackedField<Scalar: ExtensionField<F>>>(
	left: &CircuitStepArgument<F>,
	right: &CircuitStepArgument<F>,
	batch_query: &[&[P]],
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

fn pow<P: PackedField>(value: P, exp: u64) -> P {
	let mut res = P::one();
	for i in (0..64).rev() {
		res = res.square();
		if ((exp >> i) & 1) == 1 {
			res.mul_assign(value)
		}
	}
	res
}

#[cfg(test)]
mod tests {
	use binius_field::{
		BinaryField16b, BinaryField8b, PackedBinaryField8x16b, PackedField, TowerField,
	};
	use binius_math::CompositionPolyOS;
	use binius_utils::felts;

	use super::*;

	#[test]
	fn test_constant() {
		type F = BinaryField8b;
		type P = PackedBinaryField8x16b;

		let expr = ArithExpr::Const(F::new(123));
		let circuit = ArithCircuitPoly::<F>::new(expr);

		let typed_circuit: &dyn CompositionPolyOS<P> = &circuit;
		assert_eq!(typed_circuit.binary_tower_level(), F::TOWER_LEVEL);
		assert_eq!(typed_circuit.degree(), 0);
		assert_eq!(typed_circuit.n_vars(), 0);

		assert_eq!(typed_circuit.evaluate(&[]).unwrap(), P::broadcast(F::new(123).into()));

		let mut evals = [P::default()];
		typed_circuit.batch_evaluate(&[], &mut evals).unwrap();
		assert_eq!(evals, [P::broadcast(F::new(123).into())]);
	}

	#[test]
	fn test_identity() {
		type F = BinaryField8b;
		type P = PackedBinaryField8x16b;

		// x0
		let expr = ArithExpr::Var(0);
		let circuit = ArithCircuitPoly::<F>::new(expr);

		let typed_circuit: &dyn CompositionPolyOS<P> = &circuit;
		assert_eq!(typed_circuit.binary_tower_level(), F::TOWER_LEVEL);
		assert_eq!(typed_circuit.degree(), 1);
		assert_eq!(typed_circuit.n_vars(), 1);

		assert_eq!(
			typed_circuit
				.evaluate(&[P::broadcast(F::new(123).into())])
				.unwrap(),
			P::broadcast(F::new(123).into())
		);

		let mut evals = [P::default()];
		typed_circuit
			.batch_evaluate(&[&[P::broadcast(F::new(123).into())]], &mut evals)
			.unwrap();
		assert_eq!(evals, [P::broadcast(F::new(123).into())]);
	}

	#[test]
	fn test_add() {
		type F = BinaryField8b;
		type P = PackedBinaryField8x16b;

		// 123 + x0
		let expr = ArithExpr::Const(F::new(123)) + ArithExpr::Var(0);
		let circuit = ArithCircuitPoly::<F>::new(expr);

		let typed_circuit: &dyn CompositionPolyOS<P> = &circuit;
		assert_eq!(typed_circuit.binary_tower_level(), F::TOWER_LEVEL);
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
		let circuit = ArithCircuitPoly::<F>::new(expr);

		let typed_circuit: &dyn CompositionPolyOS<P> = &circuit;
		assert_eq!(typed_circuit.binary_tower_level(), F::TOWER_LEVEL);
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
		let circuit = ArithCircuitPoly::<F>::new(expr);

		let typed_circuit: &dyn CompositionPolyOS<P> = &circuit;
		assert_eq!(typed_circuit.binary_tower_level(), F::TOWER_LEVEL);
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
		let circuit = ArithCircuitPoly::<F>::new(expr);

		let typed_circuit: &dyn CompositionPolyOS<P> = &circuit;
		assert_eq!(typed_circuit.binary_tower_level(), F::TOWER_LEVEL);
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
		CompositionPoly::batch_evaluate(
			&circuit,
			&[
				&[query1[0], query2[0], query3[0]],
				&[query1[1], query2[1], query3[1]],
			],
			&mut batch_result,
		)
		.unwrap();
		assert_eq!(&batch_result, &[expected1, expected2, expected3]);
	}
}
