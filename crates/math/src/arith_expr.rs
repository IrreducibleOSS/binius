// Copyright 2024-2025 Irreducible Inc.

use std::{
	cmp::Ordering,
	collections::{HashMap, HashSet},
	fmt::{self, Display},
	iter::{Product, Sum},
	ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign},
	sync::Arc,
};

use binius_field::{Field, PackedField, TowerField};
use binius_macros::{DeserializeBytes, SerializeBytes};

use super::error::Error;

/// Arithmetic expressions that can be evaluated symbolically.
///
/// Arithmetic expressions are trees, where the leaves are either constants or variables, and the
/// non-leaf nodes are arithmetic operations, such as addition, multiplication, etc. They are
/// specific representations of multivariate polynomials.
///
/// The `Arc`'s are not guaranteed to be unique, so the expression tree may contain duplicate nodes.
/// Use `deduplicate_nodes` to remove duplicate nodes from the expression tree.
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
	/// The number of variables the expression contains.
	pub fn n_vars(&self) -> usize {
		match self {
			Self::Const(_) => 0,
			Self::Var(index) => *index + 1,
			Self::Add(left, right) | Self::Mul(left, right) => left.n_vars().max(right.n_vars()),
			Self::Pow(id, _) => id.n_vars(),
		}
	}

	/// The total degree of the polynomial the expression represents.
	pub fn degree(&self) -> usize {
		match self {
			Self::Const(_) => 0,
			Self::Var(_) => 1,
			Self::Add(left, right) => left.degree().max(right.degree()),
			Self::Mul(left, right) => left.degree() + right.degree(),
			Self::Pow(base, exp) => base.degree() * *exp as usize,
		}
	}

	/// Return a new arithmetic expression that contains only the terms of highest degree
	/// (useful for interpolation at Karatsuba infinity point).
	pub fn leading_term(&self) -> Self {
		let (_, expr) = self.leading_term_with_degree();
		expr
	}

	/// Same as `leading_term`, but returns the total degree as the first tuple element as well.
	pub fn leading_term_with_degree(&self) -> (usize, Self) {
		match self {
			expr @ Self::Const(_) => (0, expr.clone()),
			expr @ Self::Var(_) => (1, expr.clone()),
			Self::Add(left, right) => {
				let (lhs_degree, lhs) = left.leading_term_with_degree();
				let (rhs_degree, rhs) = right.leading_term_with_degree();
				match lhs_degree.cmp(&rhs_degree) {
					Ordering::Less => (rhs_degree, rhs),
					Ordering::Equal => (lhs_degree, Self::Add(Arc::new(lhs), Arc::new(rhs))),
					Ordering::Greater => (lhs_degree, lhs),
				}
			}
			Self::Mul(left, right) => {
				let (lhs_degree, lhs) = left.leading_term_with_degree();
				let (rhs_degree, rhs) = right.leading_term_with_degree();
				(lhs_degree + rhs_degree, Self::Mul(Arc::new(lhs), Arc::new(rhs)))
			}
			Self::Pow(base, exp) => {
				let (base_degree, base) = base.leading_term_with_degree();
				(base_degree * *exp as usize, Self::Pow(Arc::new(base), *exp))
			}
		}
	}

	pub fn pow(self, exp: u64) -> Self {
		Self::Pow(Arc::new(self), exp)
	}

	pub const fn zero() -> Self {
		Self::Const(F::ZERO)
	}

	pub const fn one() -> Self {
		Self::Const(F::ONE)
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
	pub fn remap_vars(self, indices: &[usize]) -> Result<Self, Error> {
		fn map_var(var: usize, indices: &[usize]) -> Result<usize, Error> {
			indices
				.get(var)
				.copied()
				.ok_or_else(|| Error::IncorrectArgumentLength {
					arg: "indices".to_string(),
					expected: var,
				})
		}

		fn remap_vars_inner<F: Field>(
			expr: &Arc<ArithExpr<F>>,
			indices: &[usize],
		) -> Result<Arc<ArithExpr<F>>, Error> {
			match &**expr {
				ArithExpr::Const(_) => Ok(expr.clone()),
				ArithExpr::Var(index) => Ok(Arc::new(ArithExpr::Var(map_var(*index, indices)?))),
				ArithExpr::Add(left, right) => {
					let new_left = remap_vars_inner(left, indices)?;
					let new_right = remap_vars_inner(right, indices)?;
					Ok(Arc::new(ArithExpr::Add(new_left, new_right)))
				}
				ArithExpr::Mul(left, right) => {
					let new_left = remap_vars_inner(left, indices)?;
					let new_right = remap_vars_inner(right, indices)?;
					Ok(Arc::new(ArithExpr::Mul(new_left, new_right)))
				}
				ArithExpr::Pow(base, exp) => {
					let new_base = remap_vars_inner(base, indices)?;
					Ok(Arc::new(ArithExpr::Pow(new_base, *exp)))
				}
			}
		}

		let expr = match self {
			Self::Const(_) => self,
			Self::Var(index) => Self::Var(map_var(index, indices)?),
			Self::Add(left, right) => {
				let new_left = remap_vars_inner(&left, indices)?;
				let new_right = remap_vars_inner(&right, indices)?;
				Self::Add(new_left, new_right)
			}
			Self::Mul(left, right) => {
				let new_left = remap_vars_inner(&left, indices)?;
				let new_right = remap_vars_inner(&right, indices)?;
				Self::Mul(new_left, new_right)
			}
			Self::Pow(base, exp) => {
				let new_base = remap_vars_inner(&base, indices)?;
				Self::Pow(new_base, exp)
			}
		};
		Ok(expr)
	}

	/// Substitute variable with index `var` with a constant `value`
	pub fn const_subst(self, var: usize, value: F) -> Self {
		fn subst_var<F: Field>(index: usize, var: usize, value: F) -> ArithExpr<F> {
			if index == var {
				ArithExpr::Const(value)
			} else {
				ArithExpr::Var(index)
			}
		}

		fn const_subst_inner<F: Field>(
			expr: &Arc<ArithExpr<F>>,
			var: usize,
			value: F,
		) -> Arc<ArithExpr<F>> {
			match &**expr {
				ArithExpr::Const(_) => expr.clone(),
				ArithExpr::Var(index) => subst_var(*index, var, value).into(),
				ArithExpr::Add(left, right) => {
					let new_left = const_subst_inner(left, var, value);
					let new_right = const_subst_inner(right, var, value);
					Arc::new(ArithExpr::Add(new_left, new_right))
				}
				ArithExpr::Mul(left, right) => {
					let new_left = const_subst_inner(left, var, value);
					let new_right = const_subst_inner(right, var, value);
					Arc::new(ArithExpr::Mul(new_left, new_right))
				}
				ArithExpr::Pow(base, exp) => {
					let new_base = const_subst_inner(base, var, value);
					Arc::new(ArithExpr::Pow(new_base, *exp))
				}
			}
		}

		match self {
			Self::Const(_) => self,
			Self::Var(index) => subst_var(index, var, value),
			Self::Add(left, right) => {
				let new_left = const_subst_inner(&left, var, value);
				let new_right = const_subst_inner(&right, var, value);
				Self::Add(new_left, new_right)
			}
			Self::Mul(left, right) => {
				let new_left = const_subst_inner(&left, var, value);
				let new_right = const_subst_inner(&right, var, value);
				Self::Mul(new_left, new_right)
			}
			Self::Pow(base, exp) => {
				let new_base = const_subst_inner(&base, var, value);
				Self::Pow(new_base, exp)
			}
		}
	}

	pub fn convert_field<FTgt: Field + From<F>>(&self) -> ArithExpr<FTgt> {
		match self {
			Self::Const(val) => ArithExpr::Const((*val).into()),
			Self::Var(index) => ArithExpr::Var(*index),
			Self::Add(left, right) => {
				let new_left = left.convert_field();
				let new_right = right.convert_field();
				ArithExpr::Add(Arc::new(new_left), Arc::new(new_right))
			}
			Self::Mul(left, right) => {
				let new_left = left.convert_field();
				let new_right = right.convert_field();
				ArithExpr::Mul(Arc::new(new_left), Arc::new(new_right))
			}
			Self::Pow(base, exp) => {
				let new_base = base.convert_field();
				ArithExpr::Pow(Arc::new(new_base), *exp)
			}
		}
	}

	pub fn try_convert_field<FTgt: Field + TryFrom<F>>(
		&self,
	) -> Result<ArithExpr<FTgt>, <FTgt as TryFrom<F>>::Error> {
		Ok(match self {
			Self::Const(val) => ArithExpr::Const(FTgt::try_from(*val)?),
			Self::Var(index) => ArithExpr::Var(*index),
			Self::Add(left, right) => {
				let new_left = left.try_convert_field()?;
				let new_right = right.try_convert_field()?;
				ArithExpr::Add(Arc::new(new_left), Arc::new(new_right))
			}
			Self::Mul(left, right) => {
				let new_left = left.try_convert_field()?;
				let new_right = right.try_convert_field()?;
				ArithExpr::Mul(Arc::new(new_left), Arc::new(new_right))
			}
			Self::Pow(base, exp) => {
				let new_base = base.try_convert_field()?;
				ArithExpr::Pow(Arc::new(new_base), *exp)
			}
		})
	}

	/// Returns `Some(F)` if the expression is a constant.
	pub const fn constant(&self) -> Option<F> {
		match self {
			Self::Const(value) => Some(*value),
			_ => None,
		}
	}

	/// Creates a new optimized expression.
	///
	/// Recursively rewrites the expression for better evaluation performance. Performs constant folding,
	/// as well as leverages simple rewriting rules around additive/multiplicative identities and addition
	/// in characteristic 2.
	pub fn optimize(&self) -> Self {
		match self {
			Self::Const(_) | Self::Var(_) => self.clone(),
			Self::Add(left, right) => {
				let left = left.optimize();
				let right = right.optimize();
				match (left, right) {
					// constant folding
					(Self::Const(left), Self::Const(right)) => Self::Const(left + right),
					// 0 + a = a + 0 = a
					(Self::Const(left), right) if left == F::ZERO => right,
					(left, Self::Const(right)) if right == F::ZERO => left,
					// a + a = 0 in char 2
					// REVIEW: relies on precise structural equality, find a better way
					(left, right) if left == right && F::CHARACTERISTIC == 2 => {
						Self::Const(F::ZERO)
					}
					// fallback
					(left, right) => Self::Add(Arc::new(left), Arc::new(right)),
				}
			}
			Self::Mul(left, right) => {
				let left = left.optimize();
				let right = right.optimize();
				match (left, right) {
					// constant folding
					(Self::Const(left), Self::Const(right)) => Self::Const(left * right),
					// 0 * a = a * 0 = 0
					(left, right)
						if left == Self::Const(F::ZERO) || right == Self::Const(F::ZERO) =>
					{
						Self::Const(F::ZERO)
					}
					// 1 * a = a * 1 = a
					(Self::Const(left), right) if left == F::ONE => right,
					(left, Self::Const(right)) if right == F::ONE => left,
					// fallback
					(left, right) => Self::Mul(Arc::new(left), Arc::new(right)),
				}
			}
			Self::Pow(id, exp) => {
				let id = id.optimize();
				match id {
					Self::Const(value) => Self::Const(PackedField::pow(value, *exp)),
					Self::Pow(id_inner, exp_inner) => Self::Pow(id_inner, *exp * exp_inner),
					id => Self::Pow(Arc::new(id), *exp),
				}
			}
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
		match self {
			Self::Const(val) => Ok((*val).into()),
			Self::Var(index) => Ok(SparseLinearNormalForm {
				constant: F::ZERO,
				dense_linear_form_len: *index + 1,
				var_coeffs: [(*index, F::ONE)].into(),
			}),
			Self::Add(left, right) => {
				Ok(left.sparse_linear_normal_form()? + right.sparse_linear_normal_form()?)
			}
			Self::Mul(left, right) => {
				left.sparse_linear_normal_form()? * right.sparse_linear_normal_form()?
			}
			Self::Pow(_, 0) => Ok(F::ONE.into()),
			Self::Pow(expr, 1) => expr.sparse_linear_normal_form(),
			Self::Pow(expr, pow) => expr.sparse_linear_normal_form().and_then(|linear_form| {
				if linear_form.dense_linear_form_len != 0 {
					return Err(Error::NonLinearExpression);
				}
				Ok(linear_form.constant.pow(*pow).into())
			}),
		}
	}

	/// Returns a vector of booleans indicating which variables are used in the expression.
	///
	/// The vector is indexed by variable index, and the value at index `i` is `true` if and only
	/// if the variable is used in the expression.
	pub fn vars_usage(&self) -> Vec<bool> {
		let mut usage = vec![false; self.n_vars()];
		self.mark_vars_usage(&mut usage);
		usage
	}

	fn mark_vars_usage(&self, usage: &mut [bool]) {
		match self {
			Self::Const(_) => (),
			Self::Var(index) => usage[*index] = true,
			Self::Add(left, right) | Self::Mul(left, right) => {
				left.mark_vars_usage(usage);
				right.mark_vars_usage(usage);
			}
			Self::Pow(base, _) => base.mark_vars_usage(usage),
		}
	}

	/// Find duplicate nodes in the expression tree and replace them with a single `Arc` instance.
	pub fn deduplicate_nodes(self) -> Self {
		let mut node_set = HashSet::new();

		fn deduplicate_nodes_inner<F: Field>(
			node: Arc<ArithExpr<F>>,
			node_set: &mut HashSet<Arc<ArithExpr<F>>>,
		) -> Arc<ArithExpr<F>> {
			if let Some(node) = node_set.get(&node) {
				return Arc::clone(node);
			}

			let node = match &*node {
				ArithExpr::Const(_) | ArithExpr::Var(_) => node,
				ArithExpr::Add(left, right) => {
					let left = deduplicate_nodes_inner(Arc::clone(left), node_set);
					let right = deduplicate_nodes_inner(Arc::clone(right), node_set);
					Arc::new(ArithExpr::Add(left, right))
				}
				ArithExpr::Mul(left, right) => {
					let left = deduplicate_nodes_inner(Arc::clone(left), node_set);
					let right = deduplicate_nodes_inner(Arc::clone(right), node_set);
					Arc::new(ArithExpr::Mul(left, right))
				}
				ArithExpr::Pow(base, exp) => {
					let base = deduplicate_nodes_inner(Arc::clone(base), node_set);
					Arc::new(ArithExpr::Pow(base, *exp))
				}
			};

			node_set.insert(Arc::clone(&node));
			node
		}

		match self {
			Self::Const(_) | Self::Var(_) => self,
			Self::Add(left, right) => {
				let left = deduplicate_nodes_inner(left, &mut node_set);
				let right = deduplicate_nodes_inner(right, &mut node_set);
				Self::Add(left, right)
			}
			Self::Mul(left, right) => {
				let left = deduplicate_nodes_inner(left, &mut node_set);
				let right = deduplicate_nodes_inner(right, &mut node_set);
				Self::Mul(left, right)
			}
			Self::Pow(base, exp) => {
				let base = deduplicate_nodes_inner(base, &mut node_set);
				Self::Pow(base, exp)
			}
		}
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

/// A simple circuit representation of an arithmetic expression.
///
/// This implementation isn't optimized for performance, but rather for simplicity
/// to allow easy conversion and preservation of the
#[derive(Clone, Debug, SerializeBytes, DeserializeBytes, PartialEq, Eq)]
pub struct ArithCircuit<F: Field> {
	steps: Vec<ArithCircuitStep<F>>,
}

impl<F: Field> ArithCircuit<F> {
	/// Steps of the circuit.
	pub fn steps(&self) -> &[ArithCircuitStep<F>] {
		&self.steps
	}

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

impl<F: Field, FTgt: Field + From<F>> From<&ArithCircuit<F>> for ArithExpr<FTgt> {
	fn from(circuit: &ArithCircuit<F>) -> Self {
		fn visit_step<F: Field, FTgt: Field + From<F>>(
			step: usize,
			cached_exrs: &mut [Option<Arc<ArithExpr<FTgt>>>],
			steps: &[ArithCircuitStep<F>],
		) -> Arc<ArithExpr<FTgt>> {
			if let Some(node) = cached_exrs[step].as_ref() {
				return Arc::clone(node);
			}

			let node = match &steps[step] {
				ArithCircuitStep::Const(value) => Arc::new(ArithExpr::Const((*value).into())),
				ArithCircuitStep::Var(index) => Arc::new(ArithExpr::Var(*index)),
				ArithCircuitStep::Add(left, right) => {
					let left = visit_step(*left, cached_exrs, steps);
					let right = visit_step(*right, cached_exrs, steps);
					Arc::new(ArithExpr::Add(left, right))
				}
				ArithCircuitStep::Mul(left, right) => {
					let left = visit_step(*left, cached_exrs, steps);
					let right = visit_step(*right, cached_exrs, steps);
					Arc::new(ArithExpr::Mul(left, right))
				}
				ArithCircuitStep::Pow(base, exp) => {
					let base = visit_step(*base, cached_exrs, steps);
					Arc::new(ArithExpr::Pow(base, *exp))
				}
			};

			cached_exrs[step] = Some(Arc::clone(&node));
			node
		}

		let mut cached_exprs = vec![None; circuit.steps.len()];
		let root = visit_step(circuit.steps.len() - 1, &mut cached_exprs, &circuit.steps);

		// to remove the extra copy of the root node
		drop(cached_exprs);

		Arc::into_inner(root).expect("root must have only one reference")
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

#[cfg(test)]
mod tests {
	use assert_matches::assert_matches;
	use binius_field::{BinaryField, BinaryField128b, BinaryField1b, BinaryField8b};
	use binius_utils::{DeserializeBytes, SerializationMode, SerializeBytes};

	use super::*;

	#[test]
	fn test_degree_with_pow() {
		let expr = ArithExpr::Const(BinaryField8b::new(6)).pow(7);
		assert_eq!(expr.degree(), 0);

		let expr: ArithExpr<BinaryField8b> = ArithExpr::Var(0).pow(7);
		assert_eq!(expr.degree(), 7);

		let expr: ArithExpr<BinaryField8b> = (ArithExpr::Var(0) * ArithExpr::Var(1)).pow(7);
		assert_eq!(expr.degree(), 14);
	}

	#[test]
	fn test_leading_term_with_degree() {
		let expr = ArithExpr::Var(0)
			* (ArithExpr::Var(1)
				* ArithExpr::Var(2)
				* ArithExpr::Const(BinaryField8b::MULTIPLICATIVE_GENERATOR)
				+ ArithExpr::Var(4))
			+ ArithExpr::Var(5).pow(3)
			+ ArithExpr::Const(BinaryField8b::ONE);

		let expected_expr = ArithExpr::Var(0)
			* ((ArithExpr::Var(1) * ArithExpr::Var(2))
				* ArithExpr::Const(BinaryField8b::MULTIPLICATIVE_GENERATOR))
			+ ArithExpr::Var(5).pow(3);

		assert_eq!(expr.leading_term_with_degree(), (3, expected_expr));
	}

	#[test]
	fn test_remap_vars_with_too_few_vars() {
		type F = BinaryField8b;
		let expr = ((ArithExpr::Var(0) + ArithExpr::Const(F::ONE)) * ArithExpr::Var(1)).pow(3);
		assert_matches!(expr.remap_vars(&[5]), Err(Error::IncorrectArgumentLength { .. }));
	}

	#[test]
	fn test_remap_vars_works() {
		type F = BinaryField8b;
		let expr = ((ArithExpr::Var(0) + ArithExpr::Const(F::ONE)) * ArithExpr::Var(1)).pow(3);
		let new_expr = expr.remap_vars(&[5, 3]);

		let expected = ((ArithExpr::Var(5) + ArithExpr::Const(F::ONE)) * ArithExpr::Var(3)).pow(3);
		assert_eq!(new_expr.unwrap(), expected);
	}

	#[test]
	fn test_optimize_identity_handling() {
		type F = BinaryField8b;
		let zero = ArithExpr::<F>::zero();
		let one = ArithExpr::<F>::one();

		assert_eq!((zero.clone() * ArithExpr::<F>::Var(0)).optimize(), zero);
		assert_eq!((ArithExpr::<F>::Var(0) * zero.clone()).optimize(), zero);

		assert_eq!((ArithExpr::<F>::Var(0) * one.clone()).optimize(), ArithExpr::Var(0));
		assert_eq!((one * ArithExpr::<F>::Var(0)).optimize(), ArithExpr::Var(0));

		assert_eq!((ArithExpr::<F>::Var(0) + zero.clone()).optimize(), ArithExpr::Var(0));
		assert_eq!((zero.clone() + ArithExpr::<F>::Var(0)).optimize(), ArithExpr::Var(0));

		assert_eq!((ArithExpr::<F>::Var(0) + ArithExpr::Var(0)).optimize(), zero);
	}

	#[test]
	fn test_const_subst_and_optimize() {
		// NB: this is FlushSumcheckComposition from the constraint_system
		type F = BinaryField8b;
		let expr = ArithExpr::Var(0) * ArithExpr::Var(1) + ArithExpr::one() - ArithExpr::Var(1);
		assert_eq!(expr.const_subst(1, F::ZERO).optimize().constant(), Some(F::ONE));
	}

	#[test]
	fn test_expression_upcast() {
		type F8 = BinaryField8b;
		type F = BinaryField128b;

		let expr = ((ArithExpr::Var(0) + ArithExpr::Const(F8::ONE))
			* ArithExpr::Const(F8::new(222)))
		.pow(3);

		let expected =
			((ArithExpr::Var(0) + ArithExpr::Const(F::ONE)) * ArithExpr::Const(F::new(222))).pow(3);
		assert_eq!(expr.convert_field::<F>(), expected);
	}

	#[test]
	fn test_expression_downcast() {
		type F8 = BinaryField8b;
		type F = BinaryField128b;

		let expr =
			((ArithExpr::Var(0) + ArithExpr::Const(F::ONE)) * ArithExpr::Const(F::new(222))).pow(3);

		assert!(expr.try_convert_field::<BinaryField1b>().is_err());

		let expected = ((ArithExpr::Var(0) + ArithExpr::Const(F8::ONE))
			* ArithExpr::Const(F8::new(222)))
		.pow(3);
		assert_eq!(expr.try_convert_field::<BinaryField8b>().unwrap(), expected);
	}

	#[test]
	fn test_linear_normal_form() {
		type F = BinaryField128b;
		use ArithExpr::{Const, Var};
		struct Case {
			expr: ArithExpr<F>,
			expected: LinearNormalForm<F>,
		}
		let cases = vec![
			Case {
				expr: Const(F::ONE),
				expected: LinearNormalForm {
					constant: F::ONE,
					var_coeffs: vec![],
				},
			},
			Case {
				expr: (Const(F::new(2)) * Const(F::new(3))).pow(2)
					+ Const(F::new(3)) * (Const(F::new(4)) + Var(0)),
				expected: LinearNormalForm {
					constant: (F::new(2) * F::new(3)).pow(2) + F::new(3) * F::new(4),
					var_coeffs: vec![F::new(3)],
				},
			},
			Case {
				expr: Const(F::new(133))
					+ Const(F::new(42)) * Var(0)
					+ Var(2) + Const(F::new(11)) * Const(F::new(37)) * Var(3),
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

	fn unique_nodes_count<F: Field>(expr: &ArithExpr<F>) -> usize {
		let mut unique_nodes = HashSet::new();

		fn visit_node<F: Field>(
			node: &Arc<ArithExpr<F>>,
			unique_nodes: &mut HashSet<*const ArithExpr<F>>,
		) {
			if unique_nodes.insert(Arc::as_ptr(node)) {
				match &**node {
					ArithExpr::Const(_) | ArithExpr::Var(_) => (),
					ArithExpr::Add(left, right) | ArithExpr::Mul(left, right) => {
						visit_node(left, unique_nodes);
						visit_node(right, unique_nodes);
					}
					ArithExpr::Pow(base, _) => visit_node(base, unique_nodes),
				}
			}
		}

		match expr {
			ArithExpr::Const(_) | ArithExpr::Var(_) => 1,
			ArithExpr::Add(left, right) | ArithExpr::Mul(left, right) => {
				visit_node(left, &mut unique_nodes);
				visit_node(right, &mut unique_nodes);
				unique_nodes.len() + 1
			}
			ArithExpr::Pow(base, _) => {
				visit_node(base, &mut unique_nodes);
				unique_nodes.len() + 1
			}
		}
	}

	fn check_serialize_bytes_roundtrip<F: Field>(expr: ArithExpr<F>) {
		let mut buf = Vec::new();
		let circut = ArithCircuit::from(&expr);
		circut
			.serialize(&mut buf, SerializationMode::CanonicalTower)
			.unwrap();
		let deserialized =
			ArithCircuit::<F>::deserialize(&buf[..], SerializationMode::CanonicalTower).unwrap();
		let deserialized = ArithExpr::from(&deserialized);
		assert_eq!(expr, deserialized);
		assert_eq!(unique_nodes_count(&expr), unique_nodes_count(&deserialized));
	}

	#[test]
	fn test_serialize_bytes_roundtrip() {
		type F = BinaryField128b;
		let expr = ArithExpr::Var(0)
			* (ArithExpr::Var(1)
				* ArithExpr::Var(2)
				* ArithExpr::Const(F::MULTIPLICATIVE_GENERATOR)
				+ ArithExpr::Var(4))
			+ ArithExpr::Var(5).pow(3)
			+ ArithExpr::Const(F::ONE);

		check_serialize_bytes_roundtrip(expr);
	}

	#[test]
	fn test_serialize_bytes_rountrip_with_duplicates() {
		type F = BinaryField128b;
		let expr = (ArithExpr::Var(0) + ArithExpr::Const(F::ONE))
			* (ArithExpr::Var(0) + ArithExpr::Const(F::ONE))
			+ (ArithExpr::Var(0) + ArithExpr::Const(F::ONE))
			+ ArithExpr::Var(1);

		check_serialize_bytes_roundtrip(expr);
	}

	#[test]
	fn test_deduplication() {
		type F = BinaryField128b;
		let expr = (ArithExpr::Var(0) + ArithExpr::Const(F::ONE))
			* (ArithExpr::Var(1) + ArithExpr::Const(F::ONE))
			+ (ArithExpr::Var(0) + ArithExpr::Const(F::ONE))
				* (ArithExpr::Var(1) + ArithExpr::Const(F::ONE))
			+ ArithExpr::Var(1);

		let expr = expr.deduplicate_nodes();
		assert_eq!(unique_nodes_count(&expr), 8);
	}

	#[test]
	fn test_arith_circuit_degree() {
		type F = BinaryField8b;
		let expr = ((ArithExpr::<F>::Var(0) + ArithExpr::Var(1)) * ArithExpr::Var(2)).pow(3);
		let circuit = ArithCircuit::from(&expr);
		assert_eq!(circuit.degree(), expr.degree());
	}

	#[test]
	fn test_arith_circuit_n_vars() {
		type F = BinaryField8b;
		let expr = ArithExpr::<F>::Var(0) + ArithExpr::Var(1) + ArithExpr::Var(2);
		let circuit = ArithCircuit::from(&expr);
		assert_eq!(circuit.n_vars(), expr.n_vars());
	}

	#[test]
	fn test_arith_circuit_remap_vars() {
		type F = BinaryField8b;
		let expr = ArithExpr::<F>::Var(0) + ArithExpr::Var(1) * ArithExpr::Var(2);
		let circuit = ArithCircuit::from(&expr);
		let remapped_circuit = circuit.remap_vars(&[2, 0, 1]).unwrap();
		let expected_expr = ArithExpr::Var(2) + ArithExpr::Var(0) * ArithExpr::Var(1);
		let expected_circuit = ArithCircuit::from(&expected_expr);
		assert_eq!(remapped_circuit, expected_circuit);
	}

	#[test]
	fn test_arith_circuit_convert_field() {
		type F8 = BinaryField8b;
		type F128 = BinaryField128b;

		let expr = (ArithExpr::Var(0) + ArithExpr::Const(F8::new(5))) * ArithExpr::Var(1);
		let circuit = ArithCircuit::from(&expr);
		let converted_circuit = circuit.convert_field::<F128>();

		let expected_expr =
			(ArithExpr::Var(0) + ArithExpr::Const(F128::new(5))) * ArithExpr::Var(1);
		let expected_circuit = ArithCircuit::from(&expected_expr);

		assert_eq!(converted_circuit, expected_circuit);
	}

	#[test]
	fn test_arith_circuit_try_convert_field() {
		type F8 = BinaryField8b;
		type F128 = BinaryField128b;

		let expr = (ArithExpr::Var(0) + ArithExpr::Const(F128::new(5))) * ArithExpr::Var(1);
		let circuit = ArithCircuit::from(&expr);

		// Successful conversion
		let converted_circuit = circuit.try_convert_field::<F8>().unwrap();
		let expected_expr = (ArithExpr::Var(0) + ArithExpr::Const(F8::new(5))) * ArithExpr::Var(1);
		let expected_circuit = ArithCircuit::from(&expected_expr);
		assert_eq!(converted_circuit, expected_circuit);

		// Failing conversion
		let invalid_expr =
			(ArithExpr::Var(0) + ArithExpr::Const(F128::new(256))) * ArithExpr::Var(1);
		let invalid_circuit = ArithCircuit::from(&invalid_expr);
		assert!(invalid_circuit.try_convert_field::<F8>().is_err());
	}

	#[test]
	fn test_arith_circuit_binary_tower_level() {
		type F = BinaryField128b;
		let expr = ArithExpr::Const(F::ONE) + ArithExpr::Const(F::MULTIPLICATIVE_GENERATOR);
		let circuit = ArithCircuit::from(&expr);
		assert_eq!(circuit.binary_tower_level(), F::MULTIPLICATIVE_GENERATOR.min_tower_level());
	}

	#[test]
	fn test_arith_circuit_steps() {
		type F = BinaryField8b;
		let expr = (ArithExpr::<F>::Var(0) + ArithExpr::Var(1)) * ArithExpr::Var(2);
		let circuit = ArithCircuit::from(&expr);
		let steps = circuit.steps();
		assert_eq!(steps.len(), 5); // 3 variables, 1 addition, 1 multiplication
		assert!(matches!(steps[0], ArithCircuitStep::Var(0)));
		assert!(matches!(steps[1], ArithCircuitStep::Var(1)));
		assert!(matches!(steps[2], ArithCircuitStep::Add(_, _)));
		assert!(matches!(steps[3], ArithCircuitStep::Var(2)));
		assert!(matches!(steps[4], ArithCircuitStep::Mul(_, _)));
	}
}
