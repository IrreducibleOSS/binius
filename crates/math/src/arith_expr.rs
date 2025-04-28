// Copyright 2024-2025 Irreducible Inc.

use std::{
	cmp::Ordering,
	collections::{hash_map::Entry, HashMap},
	fmt::{self, Display},
	hash::{Hash, Hasher},
	iter::{Product, Sum},
	ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign},
	sync::Arc,
};

use binius_field::{Field, PackedField, TowerField};
use binius_macros::{DeserializeBytes, SerializeBytes};

use super::error::Error;

/// A builder for arithmetic expressions.
///
/// This is a simple representation of multivariate polynomials. The user can explicitly
/// specify which subexpressions are shared by using `Arc` to wrap them. The same subexpressions
/// are guaranteed to be converted to the same steps `ArithCircuit` after the conversion.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ArithExpr<F: Field> {
	Const(F),
	Var(usize),
	Add(Arc<ArithExpr<F>>, Arc<ArithExpr<F>>),
	Mul(Arc<ArithExpr<F>>, Arc<ArithExpr<F>>),
	Pow(Arc<ArithExpr<F>>, u64),
}

impl<F: Field + Display> Display for ArithExpr<F> {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		match self {
			Self::Const(v) => write!(f, "{v}"),
			Self::Var(i) => write!(f, "x{i}"),
			Self::Add(x, y) => write!(f, "({} + {})", &**x, &**y),
			Self::Mul(x, y) => write!(f, "({} * {})", &**x, &**y),
			Self::Pow(x, p) => write!(f, "({})^{p}", &**x),
		}
	}
}

impl<F: Field> ArithExpr<F> {
	pub fn pow(self, exp: u64) -> Self {
		Self::Pow(Arc::new(self), exp)
	}

	pub const fn zero() -> Self {
		Self::Const(F::ZERO)
	}

	pub const fn one() -> Self {
		Self::Const(F::ONE)
	}
}

impl<F> Default for ArithExpr<F>
where
	F: Field,
{
	fn default() -> Self {
		Self::zero()
	}
}

impl<F> Add for ArithExpr<F>
where
	F: Field,
{
	type Output = Self;

	fn add(self, rhs: Self) -> Self {
		Self::Add(Arc::new(self), Arc::new(rhs))
	}
}

impl<F> Add<Arc<Self>> for ArithExpr<F>
where
	F: Field,
{
	type Output = Self;

	fn add(self, rhs: Arc<Self>) -> Self {
		Self::Add(Arc::new(self), rhs)
	}
}

impl<F> AddAssign for ArithExpr<F>
where
	F: Field,
{
	fn add_assign(&mut self, rhs: Self) {
		*self = std::mem::take(self) + rhs;
	}
}

impl<F> AddAssign<Arc<Self>> for ArithExpr<F>
where
	F: Field,
{
	fn add_assign(&mut self, rhs: Arc<Self>) {
		*self = std::mem::take(self) + rhs;
	}
}

impl<F> Sub for ArithExpr<F>
where
	F: Field,
{
	type Output = Self;

	fn sub(self, rhs: Self) -> Self {
		Self::Add(Arc::new(self), Arc::new(rhs))
	}
}

impl<F> Sub<Arc<Self>> for ArithExpr<F>
where
	F: Field,
{
	type Output = Self;

	fn sub(self, rhs: Arc<Self>) -> Self {
		Self::Add(Arc::new(self), rhs)
	}
}

impl<F> SubAssign for ArithExpr<F>
where
	F: Field,
{
	fn sub_assign(&mut self, rhs: Self) {
		*self = std::mem::take(self) - rhs;
	}
}

impl<F> SubAssign<Arc<Self>> for ArithExpr<F>
where
	F: Field,
{
	fn sub_assign(&mut self, rhs: Arc<Self>) {
		*self = std::mem::take(self) - rhs;
	}
}

impl<F> Mul for ArithExpr<F>
where
	F: Field,
{
	type Output = Self;

	fn mul(self, rhs: Self) -> Self {
		Self::Mul(Arc::new(self), Arc::new(rhs))
	}
}

impl<F> Mul<Arc<Self>> for ArithExpr<F>
where
	F: Field,
{
	type Output = Self;

	fn mul(self, rhs: Arc<Self>) -> Self {
		Self::Mul(Arc::new(self), rhs)
	}
}

impl<F> MulAssign for ArithExpr<F>
where
	F: Field,
{
	fn mul_assign(&mut self, rhs: Self) {
		*self = std::mem::take(self) * rhs;
	}
}

impl<F> MulAssign<Arc<Self>> for ArithExpr<F>
where
	F: Field,
{
	fn mul_assign(&mut self, rhs: Arc<Self>) {
		*self = std::mem::take(self) * rhs;
	}
}

impl<F: Field> Sum for ArithExpr<F> {
	fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
		iter.reduce(|acc, item| acc + item).unwrap_or(Self::zero())
	}
}

impl<F: Field> Product for ArithExpr<F> {
	fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
		iter.reduce(|acc, item| acc * item).unwrap_or(Self::one())
	}
}

#[derive(Clone, Copy, Debug, SerializeBytes, DeserializeBytes, PartialEq, Eq)]
pub enum ArithCircuitStep<F: Field> {
	Add(usize, usize),
	Mul(usize, usize),
	Pow(usize, u64),
	Const(F),
	Var(usize),
}

impl<F: Field> Default for ArithCircuitStep<F> {
	fn default() -> Self {
		Self::Const(F::ZERO)
	}
}

/// Arithmetic expressions that can be evaluated symbolically.
///
/// Arithmetic expressions are trees, where the leaves are either constants or variables, and the
/// non-leaf nodes are arithmetic operations, such as addition, multiplication, etc. They are
/// specific representations of multivariate polynomials.
///
/// We store a tree in a form of a simple circuit.
/// This implementation isn't optimized for performance, but rather for simplicity
/// to allow easy conversion and preservation of the common subexpressions
#[derive(Clone, Debug, SerializeBytes, DeserializeBytes, Eq)]
pub struct ArithCircuit<F: Field> {
	steps: Vec<ArithCircuitStep<F>>,
}

impl<F: Field> ArithCircuit<F> {
	pub fn var(index: usize) -> Self {
		Self {
			steps: vec![ArithCircuitStep::Var(index)],
		}
	}

	pub fn constant(value: F) -> Self {
		Self {
			steps: vec![ArithCircuitStep::Const(value)],
		}
	}

	pub fn zero() -> Self {
		Self::constant(F::ZERO)
	}

	pub fn one() -> Self {
		Self::constant(F::ONE)
	}

	pub fn pow(mut self, exp: u64) -> Self {
		self.steps
			.push(ArithCircuitStep::Pow(self.steps.len() - 1, exp));

		self
	}

	/// Steps of the circuit.
	pub fn steps(&self) -> &[ArithCircuitStep<F>] {
		&self.steps
	}

	/// The total degree of the polynomial the expression represents.
	pub fn degree(&self) -> usize {
		fn step_degree<F: Field>(step: usize, steps: &[ArithCircuitStep<F>]) -> usize {
			match steps[step] {
				ArithCircuitStep::Const(_) => 0,
				ArithCircuitStep::Var(_) => 1,
				ArithCircuitStep::Add(left, right) => {
					step_degree(left, steps).max(step_degree(right, steps))
				}
				ArithCircuitStep::Mul(left, right) => {
					step_degree(left, steps) + step_degree(right, steps)
				}
				ArithCircuitStep::Pow(base, exp) => step_degree(base, steps) * (exp as usize),
			}
		}

		step_degree(self.steps.len() - 1, &self.steps)
	}

	/// The number of variables the expression contains.
	pub fn n_vars(&self) -> usize {
		self.steps
			.iter()
			.map(|step| {
				if let ArithCircuitStep::Var(index) = step {
					*index + 1
				} else {
					0
				}
			})
			.max()
			.unwrap_or(0)
	}

	/// The maximum tower level of the constant terms in the circuit.
	pub fn binary_tower_level(&self) -> usize
	where
		F: TowerField,
	{
		self.steps
			.iter()
			.map(|step| {
				if let ArithCircuitStep::Const(value) = step {
					value.min_tower_level()
				} else {
					0
				}
			})
			.max()
			.unwrap_or(0)
	}

	/// Return a new arithmetic expression that contains only the terms of highest degree
	/// (useful for interpolation at Karatsuba infinity point).
	pub fn leading_term(&self) -> Self {
		let (_, expr) = self.leading_term_with_degree(self.steps.len() - 1);
		expr
	}

	/// Same as `leading_term`, but returns the total degree as the first tuple element as well.
	fn leading_term_with_degree(&self, step: usize) -> (usize, Self) {
		match &self.steps[step] {
			ArithCircuitStep::Const(value) => (0, Self::constant(*value)),
			ArithCircuitStep::Var(index) => (1, Self::var(*index)),
			ArithCircuitStep::Add(left, right) => {
				let (lhs_degree, lhs) = self.leading_term_with_degree(*left);
				let (rhs_degree, rhs) = self.leading_term_with_degree(*right);
				match lhs_degree.cmp(&rhs_degree) {
					Ordering::Less => (rhs_degree, rhs),
					Ordering::Equal => (lhs_degree, lhs + rhs),
					Ordering::Greater => (lhs_degree, lhs),
				}
			}
			ArithCircuitStep::Mul(left, right) => {
				let (lhs_degree, lhs) = self.leading_term_with_degree(*left);
				let (rhs_degree, rhs) = self.leading_term_with_degree(*right);
				(lhs_degree + rhs_degree, lhs * rhs)
			}
			ArithCircuitStep::Pow(base, exp) => {
				let (base_degree, base) = self.leading_term_with_degree(*base);
				(base_degree * (*exp as usize), base.pow(*exp))
			}
		}
	}

	pub fn convert_field<FTgt: Field + From<F>>(&self) -> ArithCircuit<FTgt> {
		ArithCircuit {
			steps: self
				.steps
				.iter()
				.map(|step| match step {
					ArithCircuitStep::Const(value) => ArithCircuitStep::Const((*value).into()),
					ArithCircuitStep::Var(index) => ArithCircuitStep::Var(*index),
					ArithCircuitStep::Add(left, right) => ArithCircuitStep::Add(*left, *right),
					ArithCircuitStep::Mul(left, right) => ArithCircuitStep::Mul(*left, *right),
					ArithCircuitStep::Pow(base, exp) => ArithCircuitStep::Pow(*base, *exp),
				})
				.collect(),
		}
	}

	pub fn try_convert_field<FTgt: Field + TryFrom<F>>(
		&self,
	) -> Result<ArithCircuit<FTgt>, <FTgt as TryFrom<F>>::Error> {
		let steps = self
			.steps
			.iter()
			.map(|step| -> Result<ArithCircuitStep<FTgt>, <FTgt as TryFrom<F>>::Error> {
				let result = match step {
					ArithCircuitStep::Const(value) => {
						ArithCircuitStep::Const(FTgt::try_from(*value)?)
					}
					ArithCircuitStep::Var(index) => ArithCircuitStep::Var(*index),
					ArithCircuitStep::Add(left, right) => ArithCircuitStep::Add(*left, *right),
					ArithCircuitStep::Mul(left, right) => ArithCircuitStep::Mul(*left, *right),
					ArithCircuitStep::Pow(base, exp) => ArithCircuitStep::Pow(*base, *exp),
				};
				Ok(result)
			})
			.collect::<Result<Vec<_>, _>>()?;

		Ok(ArithCircuit { steps })
	}

	/// Creates a new expression with the variable indices remapped.
	///
	/// This recursively replaces the variable sub-expressions with an index `i` with the variable
	/// `indices[i]`.
	///
	/// ## Throws
	///
	/// * [`Error::IncorrectArgumentLength`] if indices has length less than the current number of
	///   variables
	pub fn remap_vars(&self, indices: &[usize]) -> Result<Self, Error> {
		let steps = self
			.steps
			.iter()
			.map(|step| -> Result<ArithCircuitStep<F>, Error> {
				if let ArithCircuitStep::Var(index) = step {
					let new_index = indices.get(*index).copied().ok_or_else(|| {
						Error::IncorrectArgumentLength {
							arg: "indices".to_string(),
							expected: *index,
						}
					})?;
					Ok(ArithCircuitStep::Var(new_index))
				} else {
					Ok(*step)
				}
			})
			.collect::<Result<Vec<_>, _>>()?;
		Ok(Self { steps })
	}

	/// Substitute variable with index `var` with a constant `value`
	pub fn const_subst(self, var: usize, value: F) -> Self {
		let steps = self
			.steps
			.iter()
			.map(|step| match step {
				ArithCircuitStep::Var(index) if *index == var => ArithCircuitStep::Const(value),
				_ => *step,
			})
			.collect();
		Self { steps }
	}

	/// Returns `Some(F)` if the expression is a constant.
	pub fn get_constant(&self) -> Option<F> {
		if let ArithCircuitStep::Const(value) =
			self.steps.last().expect("steps should not be empty")
		{
			Some(*value)
		} else {
			None
		}
	}

	/// Returns the normal form of an expression if it is linear.
	///
	/// ## Throws
	///
	/// - [`Error::NonLinearExpression`] if the expression is not linear.
	pub fn linear_normal_form(&self) -> Result<LinearNormalForm<F>, Error> {
		self.sparse_linear_normal_form().map(Into::into)
	}

	fn sparse_linear_normal_form(&self) -> Result<SparseLinearNormalForm<F>, Error> {
		fn sparse_linear_normal_form<F: Field>(
			step: usize,
			steps: &[ArithCircuitStep<F>],
		) -> Result<SparseLinearNormalForm<F>, Error> {
			match &steps[step] {
				ArithCircuitStep::Const(val) => Ok((*val).into()),
				ArithCircuitStep::Var(index) => Ok(SparseLinearNormalForm {
					constant: F::ZERO,
					dense_linear_form_len: index + 1,
					var_coeffs: [(*index, F::ONE)].into(),
				}),
				ArithCircuitStep::Add(left, right) => {
					let left = sparse_linear_normal_form(*left, steps)?;
					let right = sparse_linear_normal_form(*right, steps)?;
					Ok(left + right)
				}
				ArithCircuitStep::Mul(left, right) => {
					let left = sparse_linear_normal_form(*left, steps)?;
					let right = sparse_linear_normal_form(*right, steps)?;
					left * right
				}
				ArithCircuitStep::Pow(_, 0) => Ok(F::ONE.into()),
				ArithCircuitStep::Pow(expr, 1) => sparse_linear_normal_form(*expr, steps),
				ArithCircuitStep::Pow(expr, pow) => {
					let linear_form = sparse_linear_normal_form(*expr, steps)?;
					if linear_form.dense_linear_form_len != 0 {
						return Err(Error::NonLinearExpression);
					}
					Ok(linear_form.constant.pow(*pow).into())
				}
			}
		}

		sparse_linear_normal_form(self.steps.len() - 1, &self.steps)
	}

	/// Returns a vector of booleans indicating which variables are used in the expression.
	///
	/// The vector is indexed by variable index, and the value at index `i` is `true` if and only
	/// if the variable is used in the expression.
	pub fn vars_usage(&self) -> Vec<bool> {
		let mut usage = vec![false; self.n_vars()];

		for step in &self.steps {
			if let ArithCircuitStep::Var(index) = step {
				usage[*index] = true;
			}
		}

		usage
	}

	/// Fold constants in the circuit.
	fn optimize_constants(&mut self) {
		for step_index in 0..self.steps.len() {
			let (prev_steps, curr_steps) = self.steps.split_at_mut(step_index);
			let curr_step = &mut curr_steps[0];
			match curr_step {
				ArithCircuitStep::Const(_) | ArithCircuitStep::Var(_) => {}
				ArithCircuitStep::Add(left, right) => {
					match (&prev_steps[*left], &prev_steps[*right]) {
						(ArithCircuitStep::Const(left), ArithCircuitStep::Const(right)) => {
							*curr_step = ArithCircuitStep::Const(*left + *right);
						}
						(ArithCircuitStep::Const(left), right) if *left == F::ZERO => {
							*curr_step = *right;
						}
						(left, ArithCircuitStep::Const(right)) if *right == F::ZERO => {
							*curr_step = *left;
						}
						(left, right) if left == right && F::CHARACTERISTIC == 2 => {
							*curr_step = ArithCircuitStep::Const(F::ZERO);
						}
						_ => {}
					}
				}
				ArithCircuitStep::Mul(left, right) => {
					match (&prev_steps[*left], &prev_steps[*right]) {
						(ArithCircuitStep::Const(left), ArithCircuitStep::Const(right)) => {
							*curr_step = ArithCircuitStep::Const(*left * *right);
						}
						(ArithCircuitStep::Const(left), _) if *left == F::ZERO => {
							*curr_step = ArithCircuitStep::Const(F::ZERO);
						}
						(_, ArithCircuitStep::Const(right)) if *right == F::ZERO => {
							*curr_step = ArithCircuitStep::Const(F::ZERO);
						}
						(ArithCircuitStep::Const(left), right) if *left == F::ONE => {
							*curr_step = *right;
						}
						(left, ArithCircuitStep::Const(right)) if *right == F::ONE => {
							*curr_step = *left;
						}
						_ => {}
					}
				}
				ArithCircuitStep::Pow(base, exp) => match prev_steps[*base] {
					ArithCircuitStep::Const(value) => {
						*curr_step = ArithCircuitStep::Const(PackedField::pow(value, *exp));
					}
					ArithCircuitStep::Pow(base_inner, exp_inner) => {
						*curr_step = ArithCircuitStep::Pow(base_inner, *exp * exp_inner);
					}
					_ => {}
				},
			}
		}
	}

	/// Deduplicate steps in the circuit by using a single instance of each step.
	fn deduplicate_steps(&mut self) {
		let mut step_map = HashMap::new();
		let mut step_indices = Vec::with_capacity(self.steps.len());
		for step in 0..self.steps.len() {
			let node = StepNode {
				index: step,
				steps: &self.steps,
			};
			match step_map.entry(node) {
				Entry::Occupied(entry) => {
					step_indices.push(*entry.get());
				}
				Entry::Vacant(entry) => {
					entry.insert(step);
					step_indices.push(step);
				}
			}
		}

		for step in &mut self.steps {
			match step {
				ArithCircuitStep::Add(left, right) | ArithCircuitStep::Mul(left, right) => {
					*left = step_indices[*left];
					*right = step_indices[*right];
				}
				ArithCircuitStep::Pow(base, _) => *base = step_indices[*base],
				_ => (),
			}
		}
	}

	/// Remove the unused steps from the circuit.
	fn compress_unused_steps(&mut self) {
		fn mark_used<F: Field>(step: usize, steps: &[ArithCircuitStep<F>], used: &mut [bool]) {
			if used[step] {
				return;
			}
			used[step] = true;
			match steps[step] {
				ArithCircuitStep::Add(left, right) | ArithCircuitStep::Mul(left, right) => {
					mark_used(left, steps, used);
					mark_used(right, steps, used);
				}
				ArithCircuitStep::Pow(base, _) => mark_used(base, steps, used),
				_ => (),
			}
		}

		let mut used = vec![false; self.steps.len()];
		mark_used(self.steps.len() - 1, &self.steps, &mut used);

		let mut steps_map = (0..self.steps.len()).collect::<Vec<_>>();
		let mut target_index = 0;
		for source_index in 0..self.steps.len() {
			if used[source_index] {
				if target_index != source_index {
					match &mut self.steps[source_index] {
						ArithCircuitStep::Add(left, right) | ArithCircuitStep::Mul(left, right) => {
							*left = steps_map[*left];
							*right = steps_map[*right];
						}
						ArithCircuitStep::Pow(base, _) => *base = steps_map[*base],
						_ => (),
					}

					steps_map[source_index] = target_index;
					self.steps[target_index] = self.steps[source_index];
				}

				target_index += 1;
			}
		}

		self.steps.truncate(target_index);
	}

	/// Optimize the current instance of the circuit:
	/// - Fold constants
	/// - Extract common steps
	/// - Remove unused steps
	pub fn optimize_in_place(&mut self) {
		self.optimize_constants();
		self.deduplicate_steps();
		self.compress_unused_steps();
	}

	/// Same as `optimize_in_place`, but returns a new instance of the circuit.
	pub fn optimize(mut self) -> Self {
		self.optimize_constants();
		self.deduplicate_steps();
		self.compress_unused_steps();

		self
	}
}

impl<F: Field> From<&ArithExpr<F>> for ArithCircuit<F> {
	fn from(expr: &ArithExpr<F>) -> Self {
		fn visit_node<F: Field>(
			node: &Arc<ArithExpr<F>>,
			node_to_index: &mut HashMap<*const ArithExpr<F>, usize>,
			steps: &mut Vec<ArithCircuitStep<F>>,
		) -> usize {
			if let Some(index) = node_to_index.get(&Arc::as_ptr(node)) {
				return *index;
			}

			let step = match &**node {
				ArithExpr::Const(value) => ArithCircuitStep::Const(*value),
				ArithExpr::Var(index) => ArithCircuitStep::Var(*index),
				ArithExpr::Add(left, right) => {
					let left = visit_node(left, node_to_index, steps);
					let right = visit_node(right, node_to_index, steps);
					ArithCircuitStep::Add(left, right)
				}
				ArithExpr::Mul(left, right) => {
					let left = visit_node(left, node_to_index, steps);
					let right = visit_node(right, node_to_index, steps);
					ArithCircuitStep::Mul(left, right)
				}
				ArithExpr::Pow(base, exp) => {
					let base = visit_node(base, node_to_index, steps);
					ArithCircuitStep::Pow(base, *exp)
				}
			};

			steps.push(step);
			node_to_index.insert(Arc::as_ptr(node), steps.len() - 1);
			steps.len() - 1
		}

		let mut steps = Vec::new();
		let mut node_to_index = HashMap::new();
		match expr {
			ArithExpr::Const(c) => {
				steps.push(ArithCircuitStep::Const(*c));
			}
			ArithExpr::Var(var) => {
				steps.push(ArithCircuitStep::Var(*var));
			}
			ArithExpr::Add(left, right) => {
				let left = visit_node(left, &mut node_to_index, &mut steps);
				let right = visit_node(right, &mut node_to_index, &mut steps);
				steps.push(ArithCircuitStep::Add(left, right));
			}
			ArithExpr::Mul(left, right) => {
				let left = visit_node(left, &mut node_to_index, &mut steps);
				let right = visit_node(right, &mut node_to_index, &mut steps);
				steps.push(ArithCircuitStep::Mul(left, right));
			}
			ArithExpr::Pow(base, exp) => {
				let base = visit_node(base, &mut node_to_index, &mut steps);
				steps.push(ArithCircuitStep::Pow(base, *exp));
			}
		}

		Self { steps }
	}
}

impl<F: Field> From<ArithExpr<F>> for ArithCircuit<F> {
	fn from(expr: ArithExpr<F>) -> Self {
		Self::from(&expr)
	}
}

impl<F: Field> Display for ArithCircuit<F> {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
		fn display_step<F: Field>(
			step: usize,
			steps: &[ArithCircuitStep<F>],
			f: &mut fmt::Formatter<'_>,
		) -> Result<(), fmt::Error> {
			match &steps[step] {
				ArithCircuitStep::Const(value) => write!(f, "{}", value),
				ArithCircuitStep::Var(index) => write!(f, "x{}", index),
				ArithCircuitStep::Add(left, right) => {
					write!(f, "(")?;
					display_step(*left, steps, f)?;
					write!(f, " + ")?;
					display_step(*right, steps, f)?;
					write!(f, ")")
				}
				ArithCircuitStep::Mul(left, right) => {
					write!(f, "(")?;
					display_step(*left, steps, f)?;
					write!(f, " * ")?;
					display_step(*right, steps, f)?;
					write!(f, ")")
				}
				ArithCircuitStep::Pow(base, exp) => {
					write!(f, "(")?;
					display_step(*base, steps, f)?;
					write!(f, ")^{}", exp)
				}
			}
		}

		display_step(self.steps.len() - 1, &self.steps, f)
	}
}

impl<F: Field> PartialEq for ArithCircuit<F> {
	fn eq(&self, other: &Self) -> bool {
		StepNode {
			index: self.steps.len() - 1,
			steps: &self.steps,
		} == StepNode {
			index: other.steps.len() - 1,
			steps: &other.steps,
		}
	}
}

impl<F: Field> Add for ArithCircuit<F> {
	type Output = Self;

	fn add(mut self, rhs: Self) -> Self {
		self += rhs;
		self
	}
}

impl<F: Field> AddAssign for ArithCircuit<F> {
	fn add_assign(&mut self, mut rhs: Self) {
		let old_len = self.steps.len();
		add_offset(&mut rhs.steps, old_len);
		self.steps.extend(rhs.steps);
		self.steps
			.push(ArithCircuitStep::Add(old_len - 1, self.steps.len() - 1));
	}
}

impl<F: Field> Sub for ArithCircuit<F> {
	type Output = Self;

	fn sub(mut self, rhs: Self) -> Self {
		self -= rhs;
		self
	}
}

impl<F: Field> SubAssign for ArithCircuit<F> {
	#[allow(clippy::suspicious_op_assign_impl)]
	fn sub_assign(&mut self, rhs: Self) {
		*self += rhs;
	}
}

impl<F: Field> Mul for ArithCircuit<F> {
	type Output = Self;

	fn mul(mut self, rhs: Self) -> Self {
		self *= rhs;
		self
	}
}

impl<F: Field> MulAssign for ArithCircuit<F> {
	fn mul_assign(&mut self, mut rhs: Self) {
		let old_len = self.steps.len();
		add_offset(&mut rhs.steps, old_len);
		self.steps.extend(rhs.steps);
		self.steps
			.push(ArithCircuitStep::Mul(old_len - 1, self.steps.len() - 1));
	}
}

impl<F: Field> Sum for ArithCircuit<F> {
	fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
		iter.fold(Self::zero(), |sum, item| sum + item)
	}
}

impl<F: Field> Product for ArithCircuit<F> {
	fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
		iter.fold(Self::one(), |product, item| product * item)
	}
}

fn add_offset<F: Field>(steps: &mut [ArithCircuitStep<F>], offset: usize) {
	for step in steps.iter_mut() {
		match step {
			ArithCircuitStep::Add(left, right) | ArithCircuitStep::Mul(left, right) => {
				*left += offset;
				*right += offset;
			}
			ArithCircuitStep::Pow(base, _) => {
				*base += offset;
			}
			_ => (),
		}
	}
}
/// A normal form for a linear expression.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct LinearNormalForm<F: Field> {
	/// The constant offset of the expression.
	pub constant: F,
	/// A vector mapping variable indices to their coefficients.
	pub var_coeffs: Vec<F>,
}

struct SparseLinearNormalForm<F: Field> {
	/// The constant offset of the expression.
	pub constant: F,
	/// A map of variable indices to their coefficients.
	pub var_coeffs: HashMap<usize, F>,
	/// The `var_coeffs` vector len if converted to [`LinearNormalForm`].
	/// It is used for optimization of conversion to [`LinearNormalForm`].
	pub dense_linear_form_len: usize,
}

impl<F: Field> From<F> for SparseLinearNormalForm<F> {
	fn from(value: F) -> Self {
		Self {
			constant: value,
			dense_linear_form_len: 0,
			var_coeffs: HashMap::new(),
		}
	}
}

impl<F: Field> Add for SparseLinearNormalForm<F> {
	type Output = Self;
	fn add(self, rhs: Self) -> Self::Output {
		let (mut result, consumable) = if self.var_coeffs.len() < rhs.var_coeffs.len() {
			(rhs, self)
		} else {
			(self, rhs)
		};
		result.constant += consumable.constant;
		if consumable.dense_linear_form_len > result.dense_linear_form_len {
			result.dense_linear_form_len = consumable.dense_linear_form_len;
		}

		for (index, coeff) in consumable.var_coeffs {
			result
				.var_coeffs
				.entry(index)
				.and_modify(|res_coeff| {
					*res_coeff += coeff;
				})
				.or_insert(coeff);
		}
		result
	}
}

impl<F: Field> Mul for SparseLinearNormalForm<F> {
	type Output = Result<Self, Error>;
	fn mul(self, rhs: Self) -> Result<Self, Error> {
		if !self.var_coeffs.is_empty() && !rhs.var_coeffs.is_empty() {
			return Err(Error::NonLinearExpression);
		}
		let (mut result, consumable) = if self.var_coeffs.is_empty() {
			(rhs, self)
		} else {
			(self, rhs)
		};
		result.constant *= consumable.constant;
		for coeff in result.var_coeffs.values_mut() {
			*coeff *= consumable.constant;
		}
		Ok(result)
	}
}

impl<F: Field> From<SparseLinearNormalForm<F>> for LinearNormalForm<F> {
	fn from(value: SparseLinearNormalForm<F>) -> Self {
		let mut var_coeffs = vec![F::ZERO; value.dense_linear_form_len];
		for (i, coeff) in value.var_coeffs {
			var_coeffs[i] = coeff;
		}
		Self {
			constant: value.constant,
			var_coeffs,
		}
	}
}

/// A helper structure to be used for by-value comparison and hashing of the subexpressions
/// regardless if they are in the same ArithCircuit or not.
#[derive(Eq)]
struct StepNode<'a, F: Field> {
	index: usize,
	steps: &'a [ArithCircuitStep<F>],
}

impl<F: Field> StepNode<'_, F> {
	fn prev_step(&self, step: usize) -> Self {
		StepNode {
			index: step,
			steps: self.steps,
		}
	}
}

impl<F: Field> PartialEq for StepNode<'_, F> {
	#[allow(clippy::suspicious_operation_groupings)] // false positive
	fn eq(&self, other: &Self) -> bool {
		match (&self.steps[self.index], &other.steps[other.index]) {
			(ArithCircuitStep::Const(left), ArithCircuitStep::Const(right)) => left == right,
			(ArithCircuitStep::Var(left), ArithCircuitStep::Var(right)) => left == right,
			(
				ArithCircuitStep::Add(left, right),
				ArithCircuitStep::Add(other_left, other_right),
			)
			| (
				ArithCircuitStep::Mul(left, right),
				ArithCircuitStep::Mul(other_left, other_right),
			) => {
				self.prev_step(*left) == other.prev_step(*other_left)
					&& self.prev_step(*right) == other.prev_step(*other_right)
			}
			(ArithCircuitStep::Pow(base, exp), ArithCircuitStep::Pow(other_base, other_exp)) => {
				self.prev_step(*base) == other.prev_step(*other_base) && exp == other_exp
			}
			_ => false,
		}
	}
}

impl<F: Field> Hash for StepNode<'_, F> {
	fn hash<H: Hasher>(&self, state: &mut H) {
		match self.steps[self.index] {
			ArithCircuitStep::Const(value) => {
				0u8.hash(state);
				value.hash(state);
			}
			ArithCircuitStep::Var(index) => {
				1u8.hash(state);
				index.hash(state);
			}
			ArithCircuitStep::Add(left, right) => {
				2u8.hash(state);
				self.prev_step(left).hash(state);
				self.prev_step(right).hash(state);
			}
			ArithCircuitStep::Mul(left, right) => {
				3u8.hash(state);
				self.prev_step(left).hash(state);
				self.prev_step(right).hash(state);
			}
			ArithCircuitStep::Pow(base, exp) => {
				4u8.hash(state);
				self.prev_step(base).hash(state);
				exp.hash(state);
			}
		}
	}
}

#[cfg(test)]
mod tests {
	use std::collections::HashSet;

	use assert_matches::assert_matches;
	use binius_field::{BinaryField, BinaryField128b, BinaryField1b, BinaryField8b};
	use binius_utils::{DeserializeBytes, SerializationMode, SerializeBytes};

	use super::*;

	#[test]
	fn test_degree_with_pow() {
		let expr = ArithCircuit::constant(BinaryField8b::new(6)).pow(7);
		assert_eq!(expr.degree(), 0);

		let expr: ArithCircuit<BinaryField8b> = ArithCircuit::var(0).pow(7);
		assert_eq!(expr.degree(), 7);

		let expr: ArithCircuit<BinaryField8b> =
			(ArithCircuit::var(0) * ArithCircuit::var(1)).pow(7);
		assert_eq!(expr.degree(), 14);
	}

	#[test]
	fn test_n_vars() {
		type F = BinaryField8b;
		let expr = ArithCircuit::<F>::var(0) * ArithCircuit::constant(F::MULTIPLICATIVE_GENERATOR)
			+ ArithCircuit::var(2).pow(2);
		assert_eq!(expr.n_vars(), 3);
	}

	#[test]
	fn test_leading_term_with_degree() {
		let expr = ArithCircuit::var(0)
			* (ArithCircuit::var(1)
				* ArithCircuit::var(2)
				* ArithCircuit::constant(BinaryField8b::MULTIPLICATIVE_GENERATOR)
				+ ArithCircuit::var(4))
			+ ArithCircuit::var(5).pow(3)
			+ ArithCircuit::constant(BinaryField8b::ONE);

		let expected_expr = ArithCircuit::var(0)
			* ((ArithCircuit::var(1) * ArithCircuit::var(2))
				* ArithCircuit::constant(BinaryField8b::MULTIPLICATIVE_GENERATOR))
			+ ArithCircuit::var(5).pow(3);

		assert_eq!(expr.leading_term_with_degree(expr.steps().len() - 1), (3, expected_expr));
	}

	#[test]
	fn test_remap_vars_with_too_few_vars() {
		type F = BinaryField8b;
		let expr =
			((ArithCircuit::var(0) + ArithCircuit::constant(F::ONE)) * ArithCircuit::var(1)).pow(3);
		assert_matches!(expr.remap_vars(&[5]), Err(Error::IncorrectArgumentLength { .. }));
	}

	#[test]
	fn test_remap_vars_works() {
		type F = BinaryField8b;
		let expr =
			((ArithCircuit::var(0) + ArithCircuit::constant(F::ONE)) * ArithCircuit::var(1)).pow(3);
		let new_expr = expr.remap_vars(&[5, 3]);

		let expected =
			((ArithCircuit::var(5) + ArithCircuit::constant(F::ONE)) * ArithCircuit::var(3)).pow(3);
		assert_eq!(new_expr.unwrap(), expected);
	}

	#[test]
	fn test_optimize_identity_handling() {
		type F = BinaryField8b;
		let zero = ArithCircuit::<F>::zero();
		let one = ArithCircuit::<F>::one();

		assert_eq!((zero.clone() * ArithCircuit::<F>::var(0)).optimize(), zero);
		assert_eq!((ArithCircuit::<F>::var(0) * zero.clone()).optimize(), zero);

		assert_eq!((ArithCircuit::<F>::var(0) * one.clone()).optimize(), ArithCircuit::var(0));
		assert_eq!((one * ArithCircuit::<F>::var(0)).optimize(), ArithCircuit::var(0));

		assert_eq!((ArithCircuit::<F>::var(0) + zero.clone()).optimize(), ArithCircuit::var(0));
		assert_eq!((zero.clone() + ArithCircuit::<F>::var(0)).optimize(), ArithCircuit::var(0));

		assert_eq!((ArithCircuit::<F>::var(0) + ArithCircuit::var(0)).optimize(), zero);
	}

	#[test]
	fn test_const_subst_and_optimize() {
		// NB: this is FlushSumcheckComposition from the constraint_system
		type F = BinaryField8b;
		let expr = ArithCircuit::var(0) * ArithCircuit::var(1) + ArithCircuit::one()
			- ArithCircuit::var(1);
		assert_eq!(expr.const_subst(1, F::ZERO).optimize().get_constant(), Some(F::ONE));
	}

	#[test]
	fn test_expression_upcast() {
		type F8 = BinaryField8b;
		type F = BinaryField128b;

		let expr = ((ArithCircuit::var(0) + ArithCircuit::constant(F8::ONE))
			* ArithCircuit::constant(F8::new(222)))
		.pow(3);

		let expected = ((ArithCircuit::var(0) + ArithCircuit::constant(F::ONE))
			* ArithCircuit::constant(F::new(222)))
		.pow(3);
		assert_eq!(expr.convert_field::<F>(), expected);
	}

	#[test]
	fn test_expression_downcast() {
		type F8 = BinaryField8b;
		type F = BinaryField128b;

		let expr = ((ArithCircuit::var(0) + ArithCircuit::constant(F::ONE))
			* ArithCircuit::constant(F::new(222)))
		.pow(3);

		assert!(expr.try_convert_field::<BinaryField1b>().is_err());

		let expected = ((ArithCircuit::var(0) + ArithCircuit::constant(F8::ONE))
			* ArithCircuit::constant(F8::new(222)))
		.pow(3);
		assert_eq!(expr.try_convert_field::<BinaryField8b>().unwrap(), expected);
	}

	#[test]
	fn test_linear_normal_form() {
		type F = BinaryField128b;
		struct Case {
			expr: ArithCircuit<F>,
			expected: LinearNormalForm<F>,
		}
		let cases = vec![
			Case {
				expr: ArithCircuit::constant(F::ONE),
				expected: LinearNormalForm {
					constant: F::ONE,
					var_coeffs: vec![],
				},
			},
			Case {
				expr: (ArithCircuit::constant(F::new(2)) * ArithCircuit::constant(F::new(3)))
					.pow(2) + ArithCircuit::constant(F::new(3))
					* (ArithCircuit::constant(F::new(4)) + ArithCircuit::var(0)),
				expected: LinearNormalForm {
					constant: (F::new(2) * F::new(3)).pow(2) + F::new(3) * F::new(4),
					var_coeffs: vec![F::new(3)],
				},
			},
			Case {
				expr: ArithCircuit::constant(F::new(133))
					+ ArithCircuit::constant(F::new(42)) * ArithCircuit::var(0)
					+ ArithCircuit::var(2)
					+ ArithCircuit::constant(F::new(11))
						* ArithCircuit::constant(F::new(37))
						* ArithCircuit::var(3),
				expected: LinearNormalForm {
					constant: F::new(133),
					var_coeffs: vec![F::new(42), F::ZERO, F::ONE, F::new(11) * F::new(37)],
				},
			},
		];
		for Case { expr, expected } in cases {
			let normal_form = expr.linear_normal_form().unwrap();
			assert_eq!(normal_form.constant, expected.constant);
			assert_eq!(normal_form.var_coeffs, expected.var_coeffs);
		}
	}

	fn unique_nodes_count<F: Field>(expr: &ArithCircuit<F>) -> usize {
		let mut unique_nodes = HashSet::new();

		for step in 0..expr.steps.len() {
			unique_nodes.insert(StepNode {
				index: step,
				steps: &expr.steps,
			});
		}

		unique_nodes.len()
	}

	fn check_serialize_bytes_roundtrip<F: Field>(expr: ArithCircuit<F>) {
		let mut buf = Vec::new();

		expr.serialize(&mut buf, SerializationMode::CanonicalTower)
			.unwrap();
		let deserialized =
			ArithCircuit::<F>::deserialize(&buf[..], SerializationMode::CanonicalTower).unwrap();
		assert_eq!(expr, deserialized);
		assert_eq!(unique_nodes_count(&expr), unique_nodes_count(&deserialized));
	}

	#[test]
	fn test_serialize_bytes_roundtrip() {
		type F = BinaryField128b;
		let expr = ArithCircuit::var(0)
			* (ArithCircuit::var(1)
				* ArithCircuit::var(2)
				* ArithCircuit::constant(F::MULTIPLICATIVE_GENERATOR)
				+ ArithCircuit::var(4))
			+ ArithCircuit::var(5).pow(3)
			+ ArithCircuit::constant(F::ONE);

		check_serialize_bytes_roundtrip(expr);
	}

	#[test]
	fn test_serialize_bytes_rountrip_with_duplicates() {
		type F = BinaryField128b;
		let expr = (ArithCircuit::var(0) + ArithCircuit::constant(F::ONE))
			* (ArithCircuit::var(0) + ArithCircuit::constant(F::ONE))
			+ (ArithCircuit::var(0) + ArithCircuit::constant(F::ONE))
			+ ArithCircuit::var(1);

		check_serialize_bytes_roundtrip(expr);
	}

	#[test]
	fn test_binary_tower_level() {
		type F = BinaryField128b;
		let expr =
			ArithCircuit::constant(F::ONE) + ArithCircuit::constant(F::MULTIPLICATIVE_GENERATOR);
		assert_eq!(expr.binary_tower_level(), F::MULTIPLICATIVE_GENERATOR.min_tower_level());
	}

	#[test]
	fn test_arith_circuit_steps() {
		type F = BinaryField8b;
		let expr = (ArithCircuit::<F>::var(0) + ArithCircuit::var(1)) * ArithCircuit::var(2);
		let steps = expr.steps();
		assert_eq!(steps.len(), 5); // 3 variables, 1 addition, 1 multiplication
		assert!(matches!(steps[0], ArithCircuitStep::Var(0)));
		assert!(matches!(steps[1], ArithCircuitStep::Var(1)));
		assert!(matches!(steps[2], ArithCircuitStep::Add(_, _)));
		assert!(matches!(steps[3], ArithCircuitStep::Var(2)));
		assert!(matches!(steps[4], ArithCircuitStep::Mul(_, _)));
	}

	#[test]
	fn test_optimize_constants() {
		type F = BinaryField8b;
		let mut circuit = (ArithCircuit::<F>::var(0) + ArithCircuit::constant(F::ZERO))
			* ArithCircuit::var(1)
			+ ArithCircuit::constant(F::ONE) * ArithCircuit::var(2)
			+ ArithCircuit::constant(F::ONE).pow(4).pow(5)
			+ (ArithCircuit::var(5) + ArithCircuit::var(5));
		circuit.optimize_constants();

		let expected_ciruit = ArithCircuit::var(0) * ArithCircuit::var(1)
			+ ArithCircuit::var(2)
			+ ArithCircuit::constant(F::ONE);

		assert_eq!(circuit, expected_ciruit);
	}

	#[test]
	fn test_deduplicate_steps() {
		type F = BinaryField8b;
		let mut circuit = (ArithCircuit::<F>::var(0) + ArithCircuit::var(1))
			* (ArithCircuit::var(0) + ArithCircuit::var(1))
			+ (ArithCircuit::var(0) + ArithCircuit::var(1));
		circuit.deduplicate_steps();

		let expected_circuit = ArithCircuit::<F> {
			steps: vec![
				ArithCircuitStep::Var(0),
				ArithCircuitStep::Var(1),
				ArithCircuitStep::Add(0, 1),
				ArithCircuitStep::Mul(2, 2),
				ArithCircuitStep::Add(3, 2),
			],
		};
		assert_eq!(circuit, expected_circuit);
	}

	#[test]
	fn test_compress_unused_steps() {
		type F = BinaryField8b;
		let mut circuit = ArithCircuit::<F> {
			steps: vec![
				ArithCircuitStep::Var(0),
				ArithCircuitStep::Var(1),
				ArithCircuitStep::Var(2),
				ArithCircuitStep::Add(0, 1),
				ArithCircuitStep::Var(3),
				ArithCircuitStep::Const(F::ZERO),
				ArithCircuitStep::Var(2),
				ArithCircuitStep::Mul(3, 3),
			],
		};
		circuit.compress_unused_steps();

		let expected_circuit = ArithCircuit::<F> {
			steps: vec![
				ArithCircuitStep::Var(0),
				ArithCircuitStep::Var(1),
				ArithCircuitStep::Add(0, 1),
				ArithCircuitStep::Mul(2, 2),
			],
		};
		assert_eq!(circuit.steps, expected_circuit.steps);
	}

	#[test]
	fn test_conversion_from_expr_node_doesnt_create_duplicated_steps() {
		type F = BinaryField8b;
		let sub_expr = Arc::new(ArithExpr::<F>::Var(0) + ArithExpr::<F>::Var(1));
		let expr = ArithExpr::Mul(sub_expr.clone(), sub_expr.clone()) + sub_expr;
		let circuit = ArithCircuit::<F>::from(&expr);
		assert_eq!(circuit.steps.len(), 5);
		assert_eq!(unique_nodes_count(&circuit), 5);
	}
}
