// Copyright 2024-2025 Irreducible Inc.

use std::{
	cmp::Ordering,
	collections::{HashMap, HashSet},
	fmt::{self, Display},
	iter::{Product, Sum},
	marker::PhantomData,
	ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign},
	sync::Arc,
};

use binius_field::{Field, PackedField, TowerField};
use binius_macros::{DeserializeBytes, SerializeBytes};
use binius_utils::{DeserializeBytes, SerializationError, SerializationMode, SerializeBytes};
use bytes::{Buf, BufMut};
use derive_more::Display;
use stackalloc::stackalloc_with_default;

use super::error::Error;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ArithExprNode<F: Field> {
	Const(F),
	Var(usize),
	Add(Arc<ArithExprNode<F>>, Arc<ArithExprNode<F>>),
	Mul(Arc<ArithExprNode<F>>, Arc<ArithExprNode<F>>),
	Pow(Arc<ArithExprNode<F>>, u64),
}

impl<F> ArithExprNode<F>
where
	F: Field,
{
	pub fn one() -> Self {
		Self::Const(F::ONE)
	}

	pub fn zero() -> Self {
		Self::Const(F::ZERO)
	}

	pub fn pow(self, exp: u64) -> Self {
		Self::Pow(Arc::new(self), exp)
	}

	pub fn n_vars(&self) -> usize {
		match self {
			Self::Const(_) => 0,
			Self::Var(index) => *index + 1,
			Self::Add(left, right) | Self::Mul(left, right) => left.n_vars().max(right.n_vars()),
			Self::Pow(id, _) => id.n_vars(),
		}
	}

	pub fn degree(&self) -> usize {
		match self {
			Self::Const(_) => 0,
			Self::Var(_) => 1,
			Self::Add(left, right) => left.degree().max(right.degree()),
			Self::Mul(left, right) => left.degree() + right.degree(),
			Self::Pow(base, exp) => base.degree() * *exp as usize,
		}
	}

	fn leading_term_with_degree(self: &Arc<Self>) -> (usize, Arc<Self>) {
		match &**self {
			Self::Const(_) => (0, self.clone()),
			Self::Var(_) => (1, self.clone()),
			Self::Add(left, right) => {
				let (lhs_degree, lhs) = left.leading_term_with_degree();
				let (rhs_degree, rhs) = right.leading_term_with_degree();
				match lhs_degree.cmp(&rhs_degree) {
					Ordering::Less => (rhs_degree, rhs),
					Ordering::Equal => (lhs_degree, Arc::new(Self::Add(lhs, rhs))),
					Ordering::Greater => (lhs_degree, lhs),
				}
			}
			Self::Mul(left, right) => {
				let (lhs_degree, lhs) = left.leading_term_with_degree();
				let (rhs_degree, rhs) = right.leading_term_with_degree();
				(lhs_degree + rhs_degree, Arc::new(Self::Mul(lhs, rhs)))
			}
			Self::Pow(base, exp) => {
				let (base_degree, base) = base.leading_term_with_degree();
				(base_degree * *exp as usize, Arc::new(Self::Pow(base, *exp)))
			}
		}
	}

	fn remap_vars(self: &Arc<Self>, indices: &[usize]) -> Result<Arc<Self>, Error> {
		let expr = match &**self {
			Self::Const(_) => Arc::clone(self),
			Self::Var(index) => {
				let new_index =
					indices
						.get(*index)
						.ok_or_else(|| Error::IncorrectArgumentLength {
							arg: "subset".to_string(),
							expected: *index,
						})?;

				if *new_index == *index {
					Arc::clone(self)
				} else {
					Arc::new(Self::Var(*new_index))
				}
			}
			Self::Add(left, right) => {
				let new_left = left.remap_vars(indices)?;
				let new_right = right.remap_vars(indices)?;

				if Arc::ptr_eq(left, &new_left) && Arc::ptr_eq(right, &new_right) {
					Arc::clone(self)
				} else {
					Arc::new(Self::Add(new_left, new_right))
				}
			}
			Self::Mul(left, right) => {
				let new_left = left.remap_vars(indices)?;
				let new_right = right.remap_vars(indices)?;

				if Arc::ptr_eq(left, &new_left) && Arc::ptr_eq(right, &new_right) {
					Arc::clone(self)
				} else {
					Arc::new(Self::Mul(new_left, new_right))
				}
			}
			Self::Pow(base, exp) => {
				let new_base = base.remap_vars(indices)?;

				if Arc::ptr_eq(base, &new_base) {
					Arc::clone(self)
				} else {
					Arc::new(Self::Pow(new_base, *exp))
				}
			}
		};
		Ok(expr)
	}

	/// Substitutes the variable with index `var` with the node `value`.
	fn subst_var(self: &Arc<Self>, var: usize, value: &Arc<Self>) -> Arc<Self> {
		match &**self {
			Self::Const(_) => Arc::clone(self),
			Self::Var(index) => {
				if *index == var {
					Arc::clone(value)
				} else {
					Arc::clone(self)
				}
			}
			Self::Add(left, right) => {
				let new_left = left.subst_var(var, value);
				let new_right = right.subst_var(var, value);
				Arc::new(Self::Add(new_left, new_right))
			}
			Self::Mul(left, right) => {
				let new_left = left.subst_var(var, value);
				let new_right = right.subst_var(var, value);
				Arc::new(Self::Mul(new_left, new_right))
			}
			Self::Pow(base, exp) => {
				let new_base = base.subst_var(var, value);
				Arc::new(Self::Pow(new_base, *exp))
			}
		}
	}

	fn convert_field<FTgt: Field + From<F>>(&self) -> ArithExprNode<FTgt> {
		match self {
			Self::Const(value) => ArithExprNode::Const((*value).into()),
			Self::Var(index) => ArithExprNode::Var(*index),
			Self::Add(left, right) => {
				let new_left = left.convert_field();
				let new_right = right.convert_field();
				ArithExprNode::Add(Arc::new(new_left), Arc::new(new_right))
			}
			Self::Mul(left, right) => {
				let new_left = left.convert_field();
				let new_right = right.convert_field();
				ArithExprNode::Mul(Arc::new(new_left), Arc::new(new_right))
			}
			Self::Pow(base, exp) => {
				let new_base = base.convert_field();
				ArithExprNode::Pow(Arc::new(new_base), *exp)
			}
		}
	}

	fn try_convert_field<FTgt: Field + TryFrom<F>>(
		&self,
	) -> Result<ArithExprNode<FTgt>, <FTgt as TryFrom<F>>::Error> {
		match self {
			Self::Const(value) => Ok(ArithExprNode::Const(FTgt::try_from(*value)?)),
			Self::Var(index) => Ok(ArithExprNode::Var(*index)),
			Self::Add(left, right) => {
				let new_left = left.try_convert_field()?;
				let new_right = right.try_convert_field()?;
				Ok(ArithExprNode::Add(Arc::new(new_left), Arc::new(new_right)))
			}
			Self::Mul(left, right) => {
				let new_left = left.try_convert_field()?;
				let new_right = right.try_convert_field()?;
				Ok(ArithExprNode::Mul(Arc::new(new_left), Arc::new(new_right)))
			}
			Self::Pow(base, exp) => {
				let new_base = base.try_convert_field()?;
				Ok(ArithExprNode::Pow(Arc::new(new_base), *exp))
			}
		}
	}

	pub const fn is_composite(&self) -> bool {
		match self {
			Self::Const(_) | Self::Var(_) => false,
			Self::Add(_, _) | Self::Mul(_, _) | Self::Pow(_, _) => true,
		}
	}

	pub const fn constant(&self) -> Option<F> {
		if let Self::Const(val) = self {
			Some(*val)
		} else {
			None
		}
	}

	/// Creates a new optimized expression.
	///
	/// Recursively rewrites the expression for better evaluation performance. Performs constant folding,
	/// as well as leverages simple rewriting rules around additive/multiplicative identities and addition
	/// in characteristic 2.
	fn optimize(self: &Arc<Self>) -> Arc<Self> {
		match &**self {
			Self::Const(_) | Self::Var(_) => Arc::clone(self),
			Self::Add(left, right) => {
				let left = left.optimize();
				let right = right.optimize();
				match (&*left, &*right) {
					// constant folding
					(Self::Const(lhs), Self::Const(rhs)) => Arc::new(Self::Const(*lhs + *rhs)),
					// 0 + a = a + 0 = a
					(Self::Const(lhs), _) if *lhs == F::ZERO => right,
					(_, Self::Const(rhs)) if *rhs == F::ZERO => left,
					// a + a = 0 in char 2
					// REVIEW: relies on precise structural equality, find a better way
					(lhs, rhs) if lhs == rhs && F::CHARACTERISTIC == 2 => {
						Arc::new(Self::Const(F::ZERO))
					}
					// fallback
					(_, _) => Arc::new(Self::Add(left, right)),
				}
			}
			Self::Mul(left, right) => {
				let left = left.optimize();
				let right = right.optimize();
				match (&*left, &*right) {
					// constant folding
					(Self::Const(lhs), Self::Const(rhs)) => Arc::new(Self::Const(*lhs * *rhs)),
					// 0 * a = a * 0 = 0
					(lhs, rhs) if lhs == &Self::Const(F::ZERO) || rhs == &Self::Const(F::ZERO) => {
						Arc::new(Self::Const(F::ZERO))
					}
					// 1 * a = a * 1 = a
					(Self::Const(lhs), _) if *lhs == F::ONE => right,
					(_, Self::Const(rhs)) if *rhs == F::ONE => left,
					// fallback
					(_, _) => Arc::new(Self::Mul(left, right)),
				}
			}
			Self::Pow(id, exp) => {
				let id = id.optimize();
				match &*id {
					Self::Const(value) => Arc::new(Self::Const(PackedField::pow(*value, *exp))),
					Self::Pow(id_inner, exp_inner) => {
						Arc::new(Self::Pow(Arc::clone(id_inner), *exp * exp_inner))
					}
					_ => Arc::new(Self::Pow(id, *exp)),
				}
			}
		}
	}

	fn evaluate(&self, vars: &[F]) -> F {
		match self {
			Self::Const(val) => *val,
			Self::Var(index) => vars[*index],
			Self::Add(left, right) => left.evaluate(vars) + right.evaluate(vars),
			Self::Mul(left, right) => left.evaluate(vars) * right.evaluate(vars),
			Self::Pow(base, exp) => base.evaluate(vars).pow(*exp),
		}
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

	fn binary_tower_level(&self) -> usize
	where
		F: TowerField,
	{
		match self {
			Self::Const(value) => (*value).min_tower_level(),
			Self::Var(_) => 0,
			Self::Add(left, right) | Self::Mul(left, right) => {
				left.binary_tower_level().max(right.binary_tower_level())
			}
			Self::Pow(base, _) => base.binary_tower_level(),
		}
	}

	fn nodes(&self) -> usize {
		1 + match self {
			Self::Const(_) | Self::Var(_) => 0,
			Self::Add(left, right) | Self::Mul(left, right) => left.nodes() + right.nodes(),
			Self::Pow(base, _) => base.nodes(),
		}
	}

	fn unique_nodes(self: &Arc<Self>) -> usize {
		let mut unique_nodes = HashSet::new();

		fn collect_unique_nodes<F: Field>(
			node: &Arc<ArithExprNode<F>>,
			unique_nodes: &mut HashSet<*const ArithExprNode<F>>,
		) {
			if unique_nodes.insert(Arc::as_ptr(node)) {
				match &**node {
					ArithExprNode::Const(_) | ArithExprNode::Var(_) => (),
					ArithExprNode::Add(left, right) | ArithExprNode::Mul(left, right) => {
						collect_unique_nodes(left, unique_nodes);
						collect_unique_nodes(right, unique_nodes);
					}
					ArithExprNode::Pow(base, _) => collect_unique_nodes(base, unique_nodes),
				}
			}
		}

		collect_unique_nodes(self, &mut unique_nodes);
		unique_nodes.len()
	}
}

impl<F> Display for ArithExprNode<F>
where
	F: Field + Display,
{
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

impl<F> Default for ArithExprNode<F>
where
	F: Field,
{
	fn default() -> Self {
		Self::zero()
	}
}

impl<F> Add for ArithExprNode<F>
where
	F: Field,
{
	type Output = Self;

	fn add(self, rhs: Self) -> Self {
		Self::Add(Arc::new(self), Arc::new(rhs))
	}
}

impl<F> AddAssign for ArithExprNode<F>
where
	F: Field,
{
	fn add_assign(&mut self, rhs: Self) {
		*self = std::mem::take(self) + rhs;
	}
}

impl<F> Sub for ArithExprNode<F>
where
	F: Field,
{
	type Output = Self;

	fn sub(self, rhs: Self) -> Self {
		Self::Add(Arc::new(self), Arc::new(rhs))
	}
}

impl<F> SubAssign for ArithExprNode<F>
where
	F: Field,
{
	fn sub_assign(&mut self, rhs: Self) {
		*self = std::mem::take(self) - rhs;
	}
}

impl<F> Mul for ArithExprNode<F>
where
	F: Field,
{
	type Output = Self;

	fn mul(self, rhs: Self) -> Self {
		Self::Mul(Arc::new(self), Arc::new(rhs))
	}
}

impl<F> MulAssign for ArithExprNode<F>
where
	F: Field,
{
	fn mul_assign(&mut self, rhs: Self) {
		*self = std::mem::take(self) * rhs;
	}
}

impl<F: Field> Sum for ArithExprNode<F> {
	fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
		iter.reduce(|acc, item| acc + item)
			.unwrap_or_else(Self::zero)
	}
}

impl<F: Field> Product for ArithExprNode<F> {
	fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
		iter.reduce(|acc, item| acc * item)
			.unwrap_or_else(Self::one)
	}
}

impl<F: Field> From<Arc<ArithExprNode<F>>> for ArithExpr<F> {
	fn from(node: Arc<ArithExprNode<F>>) -> Self {
		let mut node_set = HashSet::new();

		fn convert_node<F: Field>(
			node: Arc<ArithExprNode<F>>,
			node_set: &mut HashSet<Arc<ArithExprNode<F>>>,
		) -> Arc<ArithExprNode<F>> {
			if let Some(node) = node_set.get(&node) {
				return Arc::clone(node);
			}

			let node = match &*node {
				ArithExprNode::Const(_) | ArithExprNode::Var(_) => Arc::clone(&node),
				ArithExprNode::Add(left, right) => {
					let left = convert_node(Arc::clone(left), node_set);
					let right = convert_node(Arc::clone(right), node_set);
					Arc::new(ArithExprNode::Add(left, right))
				}
				ArithExprNode::Mul(left, right) => {
					let left = convert_node(Arc::clone(left), node_set);
					let right = convert_node(Arc::clone(right), node_set);
					Arc::new(ArithExprNode::Mul(left, right))
				}
				ArithExprNode::Pow(base, exp) => {
					let base = convert_node(Arc::clone(base), node_set);
					Arc::new(ArithExprNode::Pow(base, *exp))
				}
			};

			node_set.insert(Arc::clone(&node));
			node
		}

		let root = convert_node(node, &mut node_set);
		Self { root }
	}
}

impl<F: Field> From<ArithExprNode<F>> for ArithExpr<F> {
	fn from(node: ArithExprNode<F>) -> Self {
		ArithExpr::from(Arc::new(node))
	}
}
/// Arithmetic expressions that can be evaluated symbolically.
///
/// Arithmetic expressions are trees, where the leaves are either constants or variables, and the
/// non-leaf nodes are arithmetic operations, such as addition, multiplication, etc. They are
/// specific representations of multivariate polynomials.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Display)]
pub struct ArithExpr<F: Field> {
	root: Arc<ArithExprNode<F>>,
}

impl<F: Field> ArithExpr<F> {
	/// The number of variables the expression contains.
	pub fn n_vars(&self) -> usize {
		self.root.n_vars()
	}

	/// The total degree of the polynomial the expression represents.
	pub fn degree(&self) -> usize {
		self.root.degree()
	}

	pub fn zero() -> Self {
		Self {
			root: Arc::new(ArithExprNode::zero()),
		}
	}

	pub fn one() -> Self {
		Self {
			root: Arc::new(ArithExprNode::one()),
		}
	}

	pub fn pow(self, exp: u64) -> Self {
		// No duplicated nodes can be created, so we can just call the method on the root
		Self {
			root: Arc::new(ArithExprNode::Pow(self.root, exp)),
		}
	}

	/// Return a new arithmetic expression that contains only the terms of highest degree
	/// (useful for interpolation at Karatsuba infinity point).
	pub fn leading_term(&self) -> Self {
		let (degree, leading_term) = self.root.leading_term_with_degree();
		if degree == 0 {
			return Self::zero();
		}
		Self { root: leading_term }
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
		let remapped_root = self.root.remap_vars(indices)?;

		// Convert to deduplicate nodes
		Ok(remapped_root.into())
	}

	/// Substitute variable with index `var` with a constant `value`
	pub fn const_subst(self, var: usize, value: F) -> Self {
		let value = Arc::new(ArithExprNode::Const(value));
		let subst_root = self.root.subst_var(var, &value);

		// Convert to deduplicate nodes
		subst_root.into()
	}

	pub fn convert_field<FTgt: Field + From<F>>(&self) -> ArithExpr<FTgt> {
		self.root.convert_field().into()
	}

	pub fn try_convert_field<FTgt: Field + TryFrom<F>>(
		&self,
	) -> Result<ArithExpr<FTgt>, <FTgt as TryFrom<F>>::Error> {
		self.root.try_convert_field().map(ArithExpr::from)
	}

	/// Whether expression is a composite node, and not a leaf.
	pub fn is_composite(&self) -> bool {
		self.root.is_composite()
	}

	/// Returns `Some(F)` if the expression is a constant.
	pub fn constant(&self) -> Option<F> {
		self.root.constant()
	}

	/// Creates a new optimized expression.
	///
	/// Recursively rewrites the expression for better evaluation performance. Performs constant folding,
	/// as well as leverages simple rewriting rules around additive/multiplicative identities and addition
	/// in characteristic 2.
	pub fn optimize(&self) -> Self {
		// Convert to deduplicate nodes
		self.root.optimize().into()
	}

	/// Returns the normal form of an expression if it is linear.
	///
	/// ## Throws
	///
	/// - [`Error::NonLinearExpression`] if the expression is not linear.
	pub fn linear_normal_form(&self) -> Result<LinearNormalForm<F>, Error> {
		if self.degree() > 1 {
			return Err(Error::NonLinearExpression);
		}

		let n_vars = self.n_vars();

		// Linear normal form: f(x0, x1, ... x{n-1}) = c + a0*x0 + a1*x1 + ... + a{n-1}*x{n-1}
		// Evaluating with all variables set to 0, should give the constant term
		let constant = self.root.evaluate(&vec![F::ZERO; n_vars]);

		// Evaluating with x{k} set to 1 and all other x{i} set to 0, gives us `constant + a{k}`
		// That means we can subtract the constant from the evaluated expression to get the coefficient a{k}
		let var_coeffs = (0..n_vars)
			.map(|i| {
				let mut vars = vec![F::ZERO; n_vars];
				vars[i] = F::ONE;
				self.root.evaluate(&vars) - constant
			})
			.collect();
		Ok(LinearNormalForm {
			constant,
			var_coeffs,
		})
	}

	/// Returns a vector of booleans indicating which variables are used in the expression.
	///
	/// The vector is indexed by variable index, and the value at index `i` is `true` if and only
	/// if the variable is used in the expression.
	pub fn vars_usage(&self) -> Vec<bool> {
		let mut usage = vec![false; self.n_vars()];
		self.root.mark_vars_usage(&mut usage);
		usage
	}

	pub fn binary_tower_level(&self) -> usize
	where
		F: TowerField,
	{
		self.root.binary_tower_level()
	}

	pub fn root(&self) -> &Arc<ArithExprNode<F>> {
		&self.root
	}

	pub fn nodes(&self) -> usize {
		self.root.nodes()
	}

	pub fn unique_nodes(&self) -> usize {
		self.root.unique_nodes()
	}
}

impl<F: Field> Add<ArithExprNode<F>> for ArithExpr<F> {
	type Output = Self;

	fn add(self, rhs: ArithExprNode<F>) -> Self {
		let new_expr = (*self.root).clone() + rhs;
		new_expr.into()
	}
}

impl<F: Field> Add for ArithExpr<F> {
	type Output = Self;

	fn add(self, rhs: ArithExpr<F>) -> Self {
		let new_expr = (*self.root).clone() + (*rhs.root).clone();
		new_expr.into()
	}
}

impl<F: Field> Sub<ArithExprNode<F>> for ArithExpr<F> {
	type Output = Self;

	fn sub(self, rhs: ArithExprNode<F>) -> Self {
		let new_expr = (*self.root).clone() - rhs;
		new_expr.into()
	}
}

impl<F: Field> Sub for ArithExpr<F> {
	type Output = Self;

	fn sub(self, rhs: ArithExpr<F>) -> Self {
		let new_expr = (*self.root).clone() - (*rhs.root).clone();
		new_expr.into()
	}
}

impl<F: Field> Mul<ArithExprNode<F>> for ArithExpr<F> {
	type Output = Self;

	fn mul(self, rhs: ArithExprNode<F>) -> Self {
		let new_expr = (*self.root).clone() * rhs;
		new_expr.into()
	}
}

impl<F: Field> Mul for ArithExpr<F> {
	type Output = Self;

	fn mul(self, rhs: ArithExpr<F>) -> Self {
		let new_expr = (*self.root).clone() * (*rhs.root).clone();
		new_expr.into()
	}
}

impl<F: Field> SerializeBytes for ArithExpr<F> {
	fn serialize(
		&self,
		write_buf: impl BufMut,
		mode: SerializationMode,
	) -> Result<(), SerializationError> {
		stackalloc_with_default::<ArithCircuitStep<F>, _, _>(self.nodes(), |mut steps_buf| {
			let circuit = ArithCircuit::from_expr(self, &mut steps_buf);
			circuit.serialize(write_buf, mode)
		})
	}
}

impl<F: Field> DeserializeBytes for ArithExpr<F> {
	fn deserialize(
		read_buf: impl Buf,
		mode: SerializationMode,
	) -> Result<Self, SerializationError> {
		let circuit = ArithCircuit::<F, Vec<ArithCircuitStep<F>>>::deserialize(read_buf, mode)?;
		let expr = circuit.to_expr();
		Ok(expr)
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

#[derive(Clone, Debug, SerializeBytes, DeserializeBytes)]
enum CircuitArg {
	Var(usize),
	Step(usize),
}

#[derive(Clone, Debug, SerializeBytes, DeserializeBytes)]
enum ArithCircuitStep<F: Field> {
	Add(CircuitArg, CircuitArg),
	Mul(CircuitArg, CircuitArg),
	Pow(CircuitArg, u64),
	Const(F),
}

impl<F: Field> Default for ArithCircuitStep<F> {
	fn default() -> Self {
		Self::Const(F::ZERO)
	}
}

#[derive(Clone, Debug, SerializeBytes, DeserializeBytes)]
struct ArithCircuit<F: Field, Data: AsRef<[ArithCircuitStep<F>]>> {
	steps: Data,
	_pd: PhantomData<F>,
}

impl<F: Field, Data: AsRef<[ArithCircuitStep<F>]>> ArithCircuit<F, Data> {
	fn from_expr<'a>(
		expr: &ArithExpr<F>,
		data: &'a mut Data,
	) -> ArithCircuit<F, &'a [ArithCircuitStep<F>]>
	where
		Data: AsMut<[ArithCircuitStep<F>]>,
	{
		let mut node_to_index = HashMap::new();

		fn visit_node<F: Field>(
			node: &Arc<ArithExprNode<F>>,
			node_to_index: &mut HashMap<*const ArithExprNode<F>, usize>,
			steps: &mut [ArithCircuitStep<F>],
			current_len: &mut usize,
		) -> CircuitArg {
			if let Some(index) = node_to_index.get(&Arc::as_ptr(node)) {
				return CircuitArg::Step(*index);
			}

			let result = match &**node {
				ArithExprNode::Const(value) => {
					let step = ArithCircuitStep::Const(*value);
					steps[*current_len] = step;

					CircuitArg::Step(*current_len)
				}
				ArithExprNode::Var(index) => CircuitArg::Var(*index),
				ArithExprNode::Add(left, right) => {
					let left = visit_node(left, node_to_index, steps, current_len);
					let right = visit_node(right, node_to_index, steps, current_len);
					let step = ArithCircuitStep::Add(left, right);
					steps[*current_len] = step;

					CircuitArg::Step(*current_len)
				}
				ArithExprNode::Mul(left, right) => {
					let left = visit_node(left, node_to_index, steps, current_len);
					let right = visit_node(right, node_to_index, steps, current_len);
					let step = ArithCircuitStep::Mul(left, right);
					steps[*current_len] = step;
					CircuitArg::Step(*current_len)
				}
				ArithExprNode::Pow(base, exp) => {
					let base = visit_node(base, node_to_index, steps, current_len);
					let step = ArithCircuitStep::Pow(base, *exp);
					steps[*current_len] = step;
					CircuitArg::Step(*current_len)
				}
			};

			// Node can't be a child of itself and so one, so we can add it to the map after
			// visiting the child nodes.
			node_to_index.insert(Arc::as_ptr(node), *current_len);
			*current_len += 1;

			result
		}

		let mut current_len = 0;
		visit_node(&expr.root(), &mut node_to_index, data.as_mut(), &mut current_len);

		ArithCircuit {
			steps: &data.as_mut()[..current_len],
			_pd: PhantomData,
		}
	}

	fn to_expr(&self) -> ArithExpr<F> {
		let root = stackalloc_with_default::<Option<Arc<ArithExprNode<F>>>, _, _>(
			self.steps.as_ref().len(),
			|cached_exrs| {
				fn visit_arg<F: Field>(
					arg: &CircuitArg,
					cached_exrs: &mut [Option<Arc<ArithExprNode<F>>>],
					steps: &[ArithCircuitStep<F>],
				) -> Arc<ArithExprNode<F>> {
					match arg {
						CircuitArg::Var(index) => Arc::new(ArithExprNode::Var(*index)),
						CircuitArg::Step(index) => visit_step(*index, cached_exrs, steps),
					}
				}

				fn visit_step<F: Field>(
					step: usize,
					cached_exrs: &mut [Option<Arc<ArithExprNode<F>>>],
					steps: &[ArithCircuitStep<F>],
				) -> Arc<ArithExprNode<F>> {
					if let Some(node) = cached_exrs[step].as_ref() {
						return Arc::clone(node);
					}

					let node = match &steps[step] {
						ArithCircuitStep::Const(value) => Arc::new(ArithExprNode::Const(*value)),
						ArithCircuitStep::Add(left, right) => {
							let left = visit_arg(&left, cached_exrs, steps);
							let right = visit_arg(&right, cached_exrs, steps);
							Arc::new(ArithExprNode::Add(left, right))
						}
						ArithCircuitStep::Mul(left, right) => {
							let left = visit_arg(&left, cached_exrs, steps);
							let right = visit_arg(&right, cached_exrs, steps);
							Arc::new(ArithExprNode::Mul(left, right))
						}
						ArithCircuitStep::Pow(base, exp) => {
							let base = visit_arg(&base, cached_exrs, steps);
							Arc::new(ArithExprNode::Pow(base, *exp))
						}
					};

					cached_exrs[step] = Some(Arc::clone(&node));
					node
				}

				visit_step(self.steps.as_ref().len() - 1, cached_exrs, self.steps.as_ref())
			},
		);

		ArithExpr { root }
	}
}

#[cfg(test)]
mod tests {
	use assert_matches::assert_matches;
	use binius_field::{BinaryField, BinaryField128b, BinaryField1b, BinaryField8b};

	use super::*;

	#[test]
	fn test_degree_with_pow() {
		let expr = ArithExprNode::Const(BinaryField8b::new(6)).pow(7);
		assert_eq!(expr.degree(), 0);
		assert_eq!(ArithExpr::from(expr).degree(), 0);

		let expr: ArithExprNode<BinaryField8b> = ArithExprNode::Var(0).pow(7);
		assert_eq!(expr.degree(), 7);
		assert_eq!(ArithExpr::from(expr).degree(), 7);

		let expr: ArithExprNode<BinaryField8b> =
			(ArithExprNode::Var(0) * ArithExprNode::Var(1)).pow(7);
		assert_eq!(expr.degree(), 14);
		assert_eq!(ArithExpr::from(expr).degree(), 14);
	}

	#[test]
	fn test_leading_term() {
		let expr = ArithExprNode::Var(0)
			* (ArithExprNode::Var(1)
				* ArithExprNode::Var(2)
				* ArithExprNode::Const(BinaryField8b::MULTIPLICATIVE_GENERATOR)
				+ ArithExprNode::Var(4))
			+ ArithExprNode::Var(5).pow(3)
			+ ArithExprNode::Const(BinaryField8b::ONE);

		let expected_expr = ArithExprNode::Var(0)
			* ((ArithExprNode::Var(1) * ArithExprNode::Var(2))
				* ArithExprNode::Const(BinaryField8b::MULTIPLICATIVE_GENERATOR))
			+ ArithExprNode::Var(5).pow(3);

		let expr = ArithExpr::from(expr);
		assert_eq!(&**expr.leading_term().root(), &expected_expr);
	}

	#[test]
	fn test_remap_vars_with_too_few_vars() {
		type F = BinaryField8b;
		let expr =
			((ArithExprNode::Var(0) + ArithExprNode::Const(F::ONE)) * ArithExprNode::Var(1)).pow(3);
		let expr = ArithExpr::from(expr);
		assert_matches!(expr.remap_vars(&[5]), Err(Error::IncorrectArgumentLength { .. }));
	}

	#[test]
	fn test_remap_vars_works() {
		type F = BinaryField8b;
		let expr =
			((ArithExprNode::Var(0) + ArithExprNode::Const(F::ONE)) * ArithExprNode::Var(1)).pow(3);
		let expr = ArithExpr::from(expr);
		let new_expr = expr.remap_vars(&[5, 3]);

		let expected =
			((ArithExprNode::Var(5) + ArithExprNode::Const(F::ONE)) * ArithExprNode::Var(3)).pow(3);

		assert_eq!(&**new_expr.unwrap().root(), &expected);
	}

	fn check_optimize<F: Field>(expr: ArithExprNode<F>, expected: &ArithExprNode<F>) {
		let optimized = ArithExpr::from(expr).optimize();
		assert_eq!(&**optimized.root(), expected);
	}

	#[test]
	fn test_optimize_identity_handling() {
		type F = BinaryField8b;
		let zero = ArithExprNode::<F>::zero();
		let one = ArithExprNode::<F>::one();

		check_optimize(zero.clone() * ArithExprNode::<F>::Var(0), &zero);
		check_optimize(ArithExprNode::<F>::Var(0) * zero.clone(), &zero);

		check_optimize(one.clone() * ArithExprNode::<F>::Var(0), &ArithExprNode::Var(0));
		check_optimize(ArithExprNode::<F>::Var(0) * one.clone(), &ArithExprNode::Var(0));

		check_optimize(zero.clone() + ArithExprNode::<F>::Var(0), &ArithExprNode::Var(0));
		check_optimize(ArithExprNode::<F>::Var(0) + zero.clone(), &ArithExprNode::Var(0));

		check_optimize(ArithExprNode::<F>::Var(0) + ArithExprNode::<F>::Var(0), &zero);
	}

	#[test]
	fn test_const_subst_and_optimize() {
		// NB: this is FlushSumcheckComposition from the constraint_system
		type F = BinaryField8b;
		let expr = ArithExprNode::Var(0) * ArithExprNode::Var(1) + ArithExprNode::one()
			- ArithExprNode::Var(1);
		let expr = ArithExpr::from(expr);
		assert_eq!(expr.const_subst(1, F::ZERO).optimize().constant(), Some(F::ONE));
	}

	#[test]
	fn test_expression_upcast() {
		type F8 = BinaryField8b;
		type F = BinaryField128b;

		let expr = ((ArithExprNode::Var(0) + ArithExprNode::Const(F8::ONE))
			* ArithExprNode::Const(F8::new(222)))
		.pow(3);

		let expected = ((ArithExprNode::Var(0) + ArithExprNode::Const(F::ONE))
			* ArithExprNode::Const(F::new(222)))
		.pow(3);
		assert_eq!(expr.convert_field::<F>(), expected);
	}

	#[test]
	fn test_expression_downcast() {
		type F8 = BinaryField8b;
		type F = BinaryField128b;

		let expr = ((ArithExprNode::Var(0) + ArithExprNode::Const(F::ONE))
			* ArithExprNode::Const(F::new(222)))
		.pow(3);

		assert!(expr.try_convert_field::<BinaryField1b>().is_err());

		let expected = ((ArithExprNode::Var(0) + ArithExprNode::Const(F8::ONE))
			* ArithExprNode::Const(F8::new(222)))
		.pow(3);
		assert_eq!(expr.try_convert_field::<BinaryField8b>().unwrap(), expected);
	}

	#[test]
	fn test_linear_normal_form() {
		type F = BinaryField128b;
		use ArithExprNode::{Const, Var};
		let expr = Const(F::new(133))
			+ Const(F::new(42)) * Var(0)
			+ Var(2) + Const(F::new(11)) * Const(F::new(37)) * Var(3);
		let expr = ArithExpr::from(expr);
		let normal_form = expr.linear_normal_form().unwrap();
		assert_eq!(normal_form.constant, F::new(133));
		assert_eq!(
			normal_form.var_coeffs,
			vec![F::new(42), F::ZERO, F::ONE, F::new(11) * F::new(37)]
		);
	}

	#[test]
	fn test_deduplication() {
		type F = BinaryField128b;
		let expr = (ArithExprNode::Var(0) + ArithExprNode::Const(F::ONE))
			* (ArithExprNode::Var(1) + ArithExprNode::Const(F::ONE))
			+ (ArithExprNode::Var(0) + ArithExprNode::Const(F::ONE))
				* (ArithExprNode::Var(1) + ArithExprNode::Const(F::ONE))
			+ ArithExprNode::Var(1);
		assert_eq!(expr.nodes(), 17);

		let expr = ArithExpr::from(expr);
		assert_eq!(expr.unique_nodes(), 8);
	}

	fn check_serialize_bytes_roundtrip<F: Field>(expr: ArithExprNode<F>) {
		let expr = ArithExpr::from(expr);
		let mut buf = Vec::new();
		expr.serialize(&mut buf, SerializationMode::CanonicalTower)
			.unwrap();
		let deserialized =
			ArithExpr::<F>::deserialize(&buf[..], SerializationMode::CanonicalTower).unwrap();
		assert_eq!(expr, deserialized);
		assert_eq!(expr.unique_nodes(), deserialized.unique_nodes());
	}

	#[test]
	fn test_serialize_bytes_roundtrip() {
		type F = BinaryField128b;
		let expr = ArithExprNode::Var(0)
			* (ArithExprNode::Var(1)
				* ArithExprNode::Var(2)
				* ArithExprNode::Const(F::MULTIPLICATIVE_GENERATOR)
				+ ArithExprNode::Var(4))
			+ ArithExprNode::Var(5).pow(3)
			+ ArithExprNode::Const(F::ONE);

		check_serialize_bytes_roundtrip(expr);
	}

	#[test]
	fn test_serialize_bytes_rountrip_with_duplicates() {
		type F = BinaryField128b;
		let expr = (ArithExprNode::Var(0) + ArithExprNode::Const(F::ONE))
			* (ArithExprNode::Var(0) + ArithExprNode::Const(F::ONE))
			+ (ArithExprNode::Var(0) + ArithExprNode::Const(F::ONE))
			+ ArithExprNode::Var(1);

		check_serialize_bytes_roundtrip(expr);
	}
}
