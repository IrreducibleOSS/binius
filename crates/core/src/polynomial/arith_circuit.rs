// Copyright 2024 Irreducible Inc.

use binius_field::{ExtensionField, Field, PackedField, TowerField};
use binius_math::{CompositionPoly, Error};
use stackalloc::{helpers::slice_assume_init, stackalloc_uninit};
use std::{
	cmp::max,
	mem::MaybeUninit,
	ops::{Add, Mul, Sub},
	sync::Arc,
};

/// Represents an arithmetic expression that can be evaluated symbolically.
pub enum Expr<F: Field> {
	Const(F),
	Var(usize),
	Add(Box<Expr<F>>, Box<Expr<F>>),
	Mul(Box<Expr<F>>, Box<Expr<F>>),
	Pow(Box<Expr<F>>, u64),
}

impl<F: Field> Expr<F> {
	pub fn n_vars(&self) -> usize {
		match self {
			Expr::Const(_) => 0,
			Expr::Var(index) => *index + 1,
			Expr::Add(left, right) | Expr::Mul(left, right) => max(left.n_vars(), right.n_vars()),
			Expr::Pow(id, _) => id.n_vars(),
		}
	}

	pub fn degree(&self) -> usize {
		match self {
			Expr::Const(_) => 0,
			Expr::Var(_) => 1,
			Expr::Add(left, right) => max(left.degree(), right.degree()),
			Expr::Mul(left, right) => left.degree() + right.degree(),
			Expr::Pow(_, exp) => *exp as usize,
		}
	}

	pub fn pow(self, exp: u64) -> Self {
		Expr::Pow(Box::new(self), exp)
	}

	/// Convert the expression to a sequence of arithmetic operations that can be evaluated in sequence.
	fn to_circuit(&self) -> Vec<CircuitStep<F>> {
		let mut result = Vec::new();

		fn to_circuit_inner<F: Field>(
			expr: &Expr<F>,
			result: &mut Vec<CircuitStep<F>>,
		) -> CircuitStepArgument<F> {
			match expr {
				Expr::Const(value) => CircuitStepArgument::Const(*value),
				Expr::Var(index) => CircuitStepArgument::Expr(CircuitNode::Var(*index)),
				Expr::Add(left, right) => {
					let left = to_circuit_inner(left, result);
					let right = to_circuit_inner(right, result);
					result.push(CircuitStep::Add(left, right));
					CircuitStepArgument::Expr(CircuitNode::Slot(result.len() - 1))
				}
				Expr::Mul(left, right) => {
					let left = to_circuit_inner(left, result);
					let right = to_circuit_inner(right, result);
					result.push(CircuitStep::Mul(left, right));
					CircuitStepArgument::Expr(CircuitNode::Slot(result.len() - 1))
				}
				Expr::Pow(id, exp) => {
					let id = to_circuit_inner(id, result);
					result.push(CircuitStep::Pow(id, *exp));
					CircuitStepArgument::Expr(CircuitNode::Slot(result.len() - 1))
				}
			}
		}

		to_circuit_inner(self, &mut result);
		result
	}
}

impl<F> Add for Expr<F>
where
	F: Field,
{
	type Output = Self;

	fn add(self, rhs: Self) -> Self {
		Expr::Add(Box::new(self), Box::new(rhs))
	}
}

impl<F> Sub for Expr<F>
where
	F: Field,
{
	type Output = Self;

	fn sub(self, rhs: Self) -> Self {
		Expr::Add(Box::new(self), Box::new(rhs))
	}
}

impl<F> Mul for Expr<F>
where
	F: Field,
{
	type Output = Self;

	fn mul(self, rhs: Self) -> Self {
		Expr::Mul(Box::new(self), Box::new(rhs))
	}
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
/// This is meant as an alternative to a hard-coded CompositionPoly.
///
/// The advantage over a hard coded CompositionPoly is that this can be constructed and manipulated dynamically at runtime.
#[derive(Debug)]
pub struct ArithCircuitPoly<F: TowerField> {
	/// The last expression is the "top level expression" which depends on previous entries
	exprs: Arc<[CircuitStep<F>]>,
	degree: usize,
	n_vars: usize,
}

impl<F: TowerField> ArithCircuitPoly<F> {
	pub fn new(expr: Expr<F>) -> Self {
		let degree = expr.degree();
		let n_vars = expr.n_vars();
		let exprs = expr.to_circuit().into();

		Self {
			exprs,
			degree,
			n_vars,
		}
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
		F::TOWER_LEVEL
	}

	fn evaluate(&self, query: &[P]) -> Result<P, Error> {
		if query.len() != self.n_vars {
			return Err(Error::IncorrectQuerySize {
				expected: self.n_vars,
			});
		}
		let result = stackalloc_uninit::<P, _, _>(self.exprs.len(), |evals| {
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

			for (i, expr) in self.exprs.iter().enumerate() {
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

			// Safety: `evals.len()` == `self.exprs.len()`, which is guaranteed to be non-empty
			unsafe { evals.last().unwrap().assume_init() }
		});
		Ok(result)
	}

	fn batch_evaluate(&self, sparse_batch_query: &[&[P]], evals: &mut [P]) -> Result<(), Error> {
		let row_len = sparse_batch_query.first().map_or(0, |row| row.len());
		if evals.len() != row_len || sparse_batch_query.iter().any(|row| row.len() != row_len) {
			return Err(Error::BatchEvaluateSizeMismatch);
		}

		stackalloc_uninit::<P, (), _>(self.exprs.len() * row_len, |sparse_evals| {
			for (i, expr) in self.exprs.iter().enumerate() {
				let (before, current) = sparse_evals.split_at_mut(i * row_len);

				// Safety: `before` is guaranteed to be initialized by the previous loop iterations.
				let before = unsafe { slice_assume_init(before) };
				let current = &mut current[..row_len];

				match expr {
					CircuitStep::Add(left, right) => {
						apply_binary_op(
							left,
							right,
							sparse_batch_query,
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
							sparse_batch_query,
							before,
							current,
							|left, right, out| {
								out.write(left * right);
							},
						);
					}
					CircuitStep::Pow(id, exp) => match id {
						CircuitStepArgument::Expr(id) => {
							let id = id.get_sparse_chunk(sparse_batch_query, before, row_len);
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

			// Safety: `sparse_evals` is fully initialized by the previous loop iterations
			let sparse_evals = unsafe { slice_assume_init(sparse_evals) };

			evals.copy_from_slice(&sparse_evals[row_len * (self.exprs.len() - 1)..]);
		});

		Ok(())
	}
}

/// Apply a binary operation to two arguments and store the result in `current_evals`.
/// `op` must be a function that takes two arguments and initialized the result with the third argument.
fn apply_binary_op<F: Field, P: PackedField<Scalar: ExtensionField<F>>>(
	left: &CircuitStepArgument<F>,
	right: &CircuitStepArgument<F>,
	sparse_batch_query: &[&[P]],
	evals_before: &[P],
	current_evals: &mut [MaybeUninit<P>],
	op: impl Fn(P, P, &mut MaybeUninit<P>),
) {
	let row_len = current_evals.len();

	match (left, right) {
		(CircuitStepArgument::Expr(left), CircuitStepArgument::Expr(right)) => {
			let left = left.get_sparse_chunk(sparse_batch_query, evals_before, row_len);
			let right = right.get_sparse_chunk(sparse_batch_query, evals_before, row_len);
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
			let left = left.get_sparse_chunk(sparse_batch_query, evals_before, row_len);
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
			let right = right.get_sparse_chunk(sparse_batch_query, evals_before, row_len);
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
	use super::*;
	use binius_field::{
		BinaryField16b, BinaryField8b, PackedBinaryField8x16b, PackedField, TowerField,
	};
	use binius_math::CompositionPoly;
	use binius_utils::felts;

	#[test]
	fn test_add() {
		type F = BinaryField8b;
		type P = PackedBinaryField8x16b;

		// 123 + x0
		let expr = Expr::Const(F::new(123)) + Expr::Var(0);
		let circuit = &ArithCircuitPoly::<F>::new(expr) as &dyn CompositionPoly<P>;
		assert_eq!(circuit.binary_tower_level(), F::TOWER_LEVEL);
		assert_eq!(circuit.degree(), 1);
		assert_eq!(circuit.n_vars(), 1);
		assert_eq!(
			circuit
				.evaluate(&[P::from_scalars(
					felts!(BinaryField16b[0, 1, 2, 3, 122, 123, 124, 125])
				)])
				.unwrap(),
			P::from_scalars(felts!(BinaryField16b[123, 122, 121, 120, 1, 0, 7, 6]))
		);
	}

	#[test]
	fn test_mul() {
		type F = BinaryField8b;
		type P = PackedBinaryField8x16b;

		// 123 * x0
		let expr = Expr::Const(F::new(123)) * Expr::Var(0);
		let circuit = &ArithCircuitPoly::<F>::new(expr) as &dyn CompositionPoly<P>;
		assert_eq!(circuit.binary_tower_level(), F::TOWER_LEVEL);
		assert_eq!(circuit.degree(), 1);
		assert_eq!(circuit.n_vars(), 1);
		assert_eq!(
			circuit
				.evaluate(&[P::from_scalars(
					felts!(BinaryField16b[0, 1, 2, 3, 122, 123, 124, 125])
				)])
				.unwrap(),
			P::from_scalars(felts!(BinaryField16b[0, 123, 157, 230, 85, 46, 154, 225]))
		);
	}

	#[test]
	fn test_pow() {
		type F = BinaryField8b;
		type P = PackedBinaryField8x16b;

		// x0^13
		let expr = Expr::Var(0).pow(13);
		let circuit = &ArithCircuitPoly::<F>::new(expr) as &dyn CompositionPoly<P>;
		assert_eq!(circuit.binary_tower_level(), F::TOWER_LEVEL);
		assert_eq!(circuit.degree(), 13);
		assert_eq!(circuit.n_vars(), 1);
		assert_eq!(
			circuit
				.evaluate(&[P::from_scalars(
					felts!(BinaryField16b[0, 1, 2, 3, 122, 123, 124, 125])
				)])
				.unwrap(),
			P::from_scalars(felts!(BinaryField16b[0, 1, 2, 3, 200, 52, 51, 115]))
		);
	}

	#[test]
	fn test_mixed() {
		type F = BinaryField8b;
		type P = PackedBinaryField8x16b;

		// x0^2 * (x1 + 123)
		let expr = Expr::Var(0).pow(2) * (Expr::Var(1) + Expr::Const(F::new(123)));
		let circuit = &ArithCircuitPoly::<F>::new(expr) as &dyn CompositionPoly<P>;

		assert_eq!(circuit.binary_tower_level(), F::TOWER_LEVEL);
		assert_eq!(circuit.degree(), 3);
		assert_eq!(circuit.n_vars(), 2);
		assert_eq!(
			circuit
				.evaluate(&[
					P::from_scalars(felts!(BinaryField16b[0, 1, 2, 3, 4, 5, 6, 7])),
					P::from_scalars(felts!(BinaryField16b[100, 101, 102, 103, 104, 105, 106, 107]))
				])
				.unwrap(),
			P::from_scalars(felts!(BinaryField16b[0, 30, 59, 36, 151, 140, 170, 176]))
		);
	}

	#[test]
	fn test_mixed_sparse() {
		type F = BinaryField8b;
		type P = PackedBinaryField8x16b;

		// x0^2 * (x1 + 123)
		let expr = Expr::Var(0).pow(2) * (Expr::Var(1) + Expr::Const(F::new(123)));
		let circuit = ArithCircuitPoly::<F>::new(expr);

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

		let mut sparse_result = vec![P::zero(); 3];
		circuit
			.batch_evaluate(
				&[
					&[query1[0], query2[0], query3[0]],
					&[query1[1], query2[1], query3[1]],
				],
				&mut sparse_result,
			)
			.unwrap();
		assert_eq!(sparse_result, vec![expected1, expected2, expected3]);
	}
}
