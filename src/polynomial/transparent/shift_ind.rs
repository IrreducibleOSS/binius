// Copyright 2024 Ulvetanna Inc.

use crate::field::{util::eq, Field};

use crate::polynomial::{Error, MultilinearExtension, MultivariatePoly};

/// Represents MLE of circular right shift indicator $f_{b, o}(X, Y)$ on $2*b$ variables
/// partially evaluated at $Y = r$
///
/// # Formal Definition
/// Let $x, y \in \{0, 1\}^b$
///     * $f(x, y) = 1$ if $\{y\} \equiv \{x\} + \{o\} (\text{mod } 2^b)$
///     * $f(x, y) = 0$ otw
/// where:
///     * $\{x\}$ is the integer representation of the hypercube point $x \in \{0, 1\}^b$,
///     * $b$ is the block size parameter'
///     * $o$ is the shift offset parameter.
/// Observe $\forall x \in \{0, 1\}^b$, there is exactly one $y \in \{0, 1\}^b$ s.t. $f(x, y) = 1$
///
/// # Intuition
/// Consider the lexicographic ordering of each point on the $b$-variate hypercube into a $2^b$ length array.
/// Thus, we can give each element on the hypercube a unique index $\in \{0, \ldots, 2^b - 1\}$
/// Let $x, y \in \{0, 1\}^{b}$ be s.t. $\{x\} = i$ and $\{y\} = j$
/// $f(x, y) = 1$ iff:
///     * taking $o$ steps to the right from $i$ gets you to $j$ (allowing for wrap around).
///
/// # Note
/// This corresponds to the regular shift indicator in Section 4.3 of the Succinct Arguments over Towers of
/// Binary Fields paper.
///
/// # Example
/// Let $b$ = 2, $o$ = 1. Then hypercube points (0, 0), (1, 0), (0, 1), (1, 1) can be lexicographically
/// ordered into an array [(0, 0), (1, 0), (0, 1), (1, 1)]
/// Then, by considering the index of each hypercube point in the above array, we observe:
///     * $f((0, 0), (1, 0)) = 1$ because $1 = 0 + o$ mod $(2^b)$
///     * $f((1, 0), (0, 1)) = 1$ because $2 = 1 + o$ mod $(2^b)$
///     * $f((0, 1), (1, 1)) = 1$ because $3 = 2 + o$ mod $(2^b)$
///     * $f((1, 1), (0, 0)) = 1$ because $0 = 3 + o$ mod $(2^b)$
/// and every other pair of $b$-variate hypercube points $x, y \in \{0, 1\}^{b}$ is s.t. f(x, y) = 0.
#[derive(Debug)]
pub struct CircularRightShiftIndPartialEval<F: Field> {
	/// Block size $b$, also the number of variables
	block_size: usize,
	/// Right shift offset $o \in \{1, \ldots, 2^b - 1\}$
	right_shift_offset: usize,
	/// partial evaluation point $r$, typically lowest $b$ coords
	/// from a larger challenge point.
	r: Vec<F>,
}

impl<F: Field> CircularRightShiftIndPartialEval<F> {
	pub fn new(block_size: usize, shift_offset: usize, r: Vec<F>) -> Result<Self, Error> {
		assert_valid_shift_ind_args(block_size, shift_offset, &r)?;
		Ok(Self {
			block_size,
			right_shift_offset: shift_offset,
			r,
		})
	}

	/// Evaluates this partially evaluated circular shift indicator MLE $f(X, r)$
	/// over the entire $b$-variate hypercube
	pub fn multilinear_extension(&self) -> Result<MultilinearExtension<'static, F>, Error> {
		let (ps, pps) =
			partial_evaluate_hypercube_help(self.block_size, self.right_shift_offset, &self.r)?;
		let values = ps
			.iter()
			.zip(pps)
			.map(|(p, pp)| *p + pp)
			.collect::<Vec<_>>();
		MultilinearExtension::from_values(values)
	}

	/// Evaluates this partial circular shift indicator MLE $f(X, r)$ with $X=x$
	fn evaluate_at_point(&self, x: &[F]) -> Result<F, Error> {
		if x.len() != self.block_size {
			return Err(Error::IncorrectQuerySize {
				expected: self.block_size,
			});
		}
		let (p_res, pp_res) =
			evaluate_shift_ind_help(self.block_size, self.right_shift_offset, x, &self.r)?;
		Ok(p_res + pp_res)
	}
}

/// Represents MLE of logical right shift indicator $f_{b, o}(X, Y)$ on $2*b$ variables
/// partially evaluated at $Y = r$
///
/// # Formal Definition
/// Let $x, y \in \{0, 1\}^b$
///     * $f(x, y) = 1$ if $\{y\} \equiv \{x\} + \{o\}$
///     * $f(x, y) = 0$ otw
/// where:
///     * $\{x\}$ is the integer representation of the hypercube point $x \in \{0, 1\}^b$,
///     * $b$ is the block size parameter'
///     * $o$ is the shift offset parameter.
/// Observe $\forall x \in \{0, 1\}^b$, there is at most one $y \in \{0, 1\}^b$ s.t. $f(x, y) = 1$
///
/// # Intuition
/// Consider the lexicographic ordering of each point on the $b$-variate hypercube into a $2^b$ length array.
/// Thus, we can give each element on the hypercube a unique index $\in \{0, \ldots, 2^b - 1\}$
/// Let $x, y \in \{0, 1\}^{b}$ be s.t. $\{x\} = i$ and $\{y\} = j$
/// $f(x, y) = 1$ iff:
///     * taking $o$ steps to the right from $i$ gets you to $j$ (not allowing for wrap around).
///
/// # Note
/// This corresponds to the shift prime indicator in Section 4.3 of the Succinct Arguments over Towers of
/// Binary Fields paper.
///
/// # Example
/// Let $b$ = 2, $o$ = 1. Then hypercube points (0, 0), (1, 0), (0, 1), (1, 1) can be lexicographically
/// ordered into an array [(0, 0), (1, 0), (0, 1), (1, 1)]
/// Then, by considering the index of each hypercube point in the above array, we observe:
///     * $f((0, 0), (1, 0)) = 1$ because $1 = 0 + o$
///     * $f((1, 0), (0, 1)) = 1$ because $2 = 1 + o$
///     * $f((0, 1), (1, 1)) = 1$ because $3 = 2 + o$
/// and every other pair of $b$-variate hypercube points $x, y \in \{0, 1\}^{b}$ is s.t. f(x, y) = 0.
#[derive(Debug)]
pub struct LogicalRightShiftIndPartialEval<F: Field> {
	/// Block size $b$, also the number of variables
	block_size: usize,
	/// Right shift offset $o \in \{1, \ldots, 2^b - 1\}$
	right_shift_offset: usize,
	/// partial evaluation point $r$, typically lowest $b$ coords
	/// from a larger challenge point.
	r: Vec<F>,
}

impl<F: Field> LogicalRightShiftIndPartialEval<F> {
	pub fn new(block_size: usize, shift_offset: usize, r: Vec<F>) -> Result<Self, Error> {
		assert_valid_shift_ind_args(block_size, shift_offset, &r)?;
		Ok(Self {
			block_size,
			right_shift_offset: shift_offset,
			r,
		})
	}

	/// Evaluates this partially evaluated logical right shift indicator MLE $f(X, r)$
	/// over the entire $b$-variate hypercube
	pub fn multilinear_extension(&self) -> Result<MultilinearExtension<'static, F>, Error> {
		let (ps, _) =
			partial_evaluate_hypercube_help(self.block_size, self.right_shift_offset, &self.r)?;
		MultilinearExtension::from_values(ps)
	}

	/// Evaluates this partial logical right shift indicator MLE $f(X, r)$ with $X=x$
	fn evaluate_at_point(&self, x: &[F]) -> Result<F, Error> {
		if x.len() != self.block_size {
			return Err(Error::IncorrectQuerySize {
				expected: self.block_size,
			});
		}
		let (p_res, _) =
			evaluate_shift_ind_help(self.block_size, self.right_shift_offset, x, &self.r)?;
		Ok(p_res)
	}
}

/// Represents MLE of logical left shift indicator $f_{b, o}(X, Y)$ on $2*b$ variables
/// partially evaluated at $Y = r$
///
/// # Formal Definition
/// Let $x, y \in \{0, 1\}^b$
///     * $f(x, y) = 1$ if $\{y\} \equiv \{x\} - \{o\}$
///     * $f(x, y) = 0$ otw
/// where:
///     * $\{x\}$ is the integer representation of the hypercube point $x \in \{0, 1\}^b$,
///     * $b$ is the block size parameter'
///     * $o$ is the shift offset parameter.
/// Observe $\forall x \in \{0, 1\}^b$, there is at most one $y \in \{0, 1\}^b$ s.t. $f(x, y) = 1$
///
/// # Intuition
/// Consider the lexicographic ordering of each point on the $b$-variate hypercube into a $2^b$ length array.
/// Thus, we can give each element on the hypercube a unique index $\in \{0, \ldots, 2^b - 1\}$
/// Let $x, y \in \{0, 1\}^{b}$ be s.t. $\{x\} = i$ and $\{y\} = j$
/// $f(x, y) = 1$ iff:
///     * taking $o$ steps to the left from $i$ gets you to $j$ (not allowing for wrap around).
///
/// # Note
/// This corresponds to the shift double prime indicator in Section 4.3 of the Succinct Arguments
/// over Towers of Binary Fields paper. In the paper, the shift offset is always a right shift,
/// so the shift_offset for the corresponding shift double prime indicator would be $2^b - o$.
///
/// # Example
/// Let $b$ = 2, $o$ = 3. Then hypercube points (0, 0), (1, 0), (0, 1), (1, 1) can be lexicographically
/// ordered into an array [(0, 0), (1, 0), (0, 1), (1, 1)]
/// Then, by considering the index of each hypercube point in the above array, we observe:
///     * $f((1, 1), (0, 0)) = 1$ because $0 = 3 - o$
/// and every other pair of $b$-variate hypercube points $x, y \in \{0, 1\}^{b}$ is s.t. f(x, y) = 0.
#[derive(Debug)]
pub struct LogicalLeftShiftIndPartialEval<F: Field> {
	/// Block size $b$, also the number of variables
	block_size: usize,
	/// Left shift offset $o \in \{1, \ldots, 2^b - 1\}$
	left_shift_offset: usize,
	/// partial evaluation point $r$, typically lowest $b$ coords
	/// from a larger challenge point.
	r: Vec<F>,
}

impl<F: Field> LogicalLeftShiftIndPartialEval<F> {
	pub fn new(block_size: usize, left_shift_offset: usize, r: Vec<F>) -> Result<Self, Error> {
		assert_valid_shift_ind_args(block_size, left_shift_offset, &r)?;
		Ok(Self {
			block_size,
			left_shift_offset,
			r,
		})
	}

	/// Evaluates this partially evaluated logical left shift indicator MLE $f(X, r)$
	/// over the entire $b$-variate hypercube
	pub fn multilinear_extension(&self) -> Result<MultilinearExtension<'static, F>, Error> {
		let right_shift_offset = get_right_shift_offset(self.block_size, self.left_shift_offset);
		let (_, pps) =
			partial_evaluate_hypercube_help(self.block_size, right_shift_offset, &self.r)?;
		MultilinearExtension::from_values(pps)
	}

	/// Evaluates this partial logical left shift indicator MLE $f(X, r)$ with $X=x$
	fn evaluate_at_point(&self, x: &[F]) -> Result<F, Error> {
		if x.len() != self.block_size {
			return Err(Error::IncorrectQuerySize {
				expected: self.block_size,
			});
		}
		let right_shift_offset = get_right_shift_offset(self.block_size, self.left_shift_offset);
		let (_, pp_res) = evaluate_shift_ind_help(self.block_size, right_shift_offset, x, &self.r)?;
		Ok(pp_res)
	}
}

impl<F: Field> MultivariatePoly<F> for CircularRightShiftIndPartialEval<F> {
	fn n_vars(&self) -> usize {
		self.block_size
	}

	fn degree(&self) -> usize {
		self.block_size
	}

	fn evaluate(&self, query: &[F]) -> Result<F, Error> {
		self.evaluate_at_point(query)
	}
}

impl<F: Field> MultivariatePoly<F> for LogicalRightShiftIndPartialEval<F> {
	fn n_vars(&self) -> usize {
		self.block_size
	}

	fn degree(&self) -> usize {
		self.block_size
	}

	fn evaluate(&self, query: &[F]) -> Result<F, Error> {
		self.evaluate_at_point(query)
	}
}

impl<F: Field> MultivariatePoly<F> for LogicalLeftShiftIndPartialEval<F> {
	fn n_vars(&self) -> usize {
		self.block_size
	}

	fn degree(&self) -> usize {
		self.block_size
	}

	fn evaluate(&self, query: &[F]) -> Result<F, Error> {
		self.evaluate_at_point(query)
	}
}

/// Gets right shift offset from left shift offset
fn get_right_shift_offset(block_size: usize, left_shift_offset: usize) -> usize {
	(1 << block_size) - left_shift_offset
}

/// Checks validity of shift indicator arguments
fn assert_valid_shift_ind_args<F: Field>(
	block_size: usize,
	shift_offset: usize,
	partial_query_point: &[F],
) -> Result<(), Error> {
	if partial_query_point.len() != block_size {
		return Err(Error::IncorrectQuerySize {
			expected: block_size,
		});
	}
	if shift_offset == 0 || shift_offset >= 1 << block_size {
		return Err(Error::InvalidShiftOffset {
			max_shift_offset: (1 << block_size) - 1,
			shift_offset,
		});
	}

	Ok(())
}

/// Evaluates the LogicalRightShift and LogicalLeftShift indicators at the point $(x, y)$
///
/// Requires length of x (and y) is block_size
/// Requires shift offset is at most $2^b$ where $b$ is block_size
fn evaluate_shift_ind_help<F: Field>(
	block_size: usize,
	shift_offset: usize,
	x: &[F],
	y: &[F],
) -> Result<(F, F), Error> {
	if x.len() != block_size {
		return Err(Error::IncorrectQuerySize {
			expected: block_size,
		});
	}
	assert_valid_shift_ind_args(block_size, shift_offset, y)?;

	let (mut s_ind_p, mut s_ind_pp) = (F::ONE, F::ZERO);
	let (mut temp_p, mut temp_pp) = (F::default(), F::default());
	(0..block_size).for_each(|k| {
		let o_k = shift_offset >> k;
		if o_k % 2 == 1 {
			temp_p = (F::ONE - x[k]) * y[k] * s_ind_p;
			temp_pp = x[k] * (F::ONE - y[k]) * s_ind_p + eq(x[k], y[k]) * s_ind_pp;
		} else {
			temp_p = eq(x[k], y[k]) * s_ind_p + (F::ONE - x[k]) * y[k] * s_ind_pp;
			temp_pp = x[k] * (F::ONE - y[k]) * s_ind_pp;
		}
		// roll over results
		s_ind_p = temp_p;
		s_ind_pp = temp_pp;
	});

	Ok((s_ind_p, s_ind_pp))
}

/// Evaluates the LogicalRightShift and LogicalLeftShift indicators over the entire hypercube
///
/// Total time is O(2^b) field operations (optimal in light of output size)
/// Requires length of $r$ is exactly block_size
/// Requires shift offset is at most $2^b$ where $b$ is block_size
fn partial_evaluate_hypercube_help<F: Field>(
	block_size: usize,
	shift_offset: usize,
	r: &[F],
) -> Result<(Vec<F>, Vec<F>), Error> {
	assert_valid_shift_ind_args(block_size, shift_offset, r)?;
	let mut s_ind_p = vec![F::ONE; 1 << block_size];
	let mut s_ind_pp = vec![F::ZERO; 1 << block_size];

	(0..block_size).for_each(|k| {
		let o_k = shift_offset >> k;
		// complexity: just two multiplications per iteration!
		(0..(1 << k)).for_each(|i| {
			if o_k % 2 == 1 {
				s_ind_pp[1 << k | i] = s_ind_pp[i] * r[k];
				let tmp = s_ind_pp[1 << k | i];
				s_ind_pp[i] -= tmp;
				s_ind_p[1 << k | i] = s_ind_p[i] * r[k]; // gimmick: use this as a stash slot
				s_ind_pp[1 << k | i] += s_ind_p[i] - s_ind_p[1 << k | i]; // * 1 - r
				s_ind_p[i] = s_ind_p[1 << k | i]; // now move to lower half
				s_ind_p[1 << k | i] = F::ZERO; // clear upper half
			} else {
				s_ind_p[1 << k | i] = s_ind_p[i] * r[k];
				let tmp = s_ind_p[1 << k | i];
				s_ind_p[i] -= tmp;
				s_ind_pp[1 << k | i] = s_ind_pp[i] * (F::ONE - r[k]);
				s_ind_p[i] += s_ind_pp[i] - s_ind_pp[1 << k | i];
				s_ind_pp[i] = F::ZERO; // clear lower half
			}
		})
	});

	Ok((s_ind_p, s_ind_pp))
}

#[cfg(test)]
mod tests {
	use rand::{rngs::StdRng, SeedableRng};

	use super::{
		CircularRightShiftIndPartialEval, LogicalLeftShiftIndPartialEval,
		LogicalRightShiftIndPartialEval,
	};
	use crate::{
		field::{BinaryField32b, Field},
		polynomial::{multilinear_query::MultilinearQuery, MultivariatePoly},
	};
	use std::iter::repeat_with;

	// Consistency Tests
	fn test_circular_right_shift_consistency_help<F: Field>(
		block_size: usize,
		right_shift_offset: usize,
	) {
		let mut rng = StdRng::seed_from_u64(0);
		let r = repeat_with(|| F::random(&mut rng))
			.take(block_size)
			.collect::<Vec<_>>();
		let eval_point = &repeat_with(|| F::random(&mut rng))
			.take(block_size)
			.collect::<Vec<_>>();

		// Get Multivariate Poly version
		let shift_r_mvp =
			CircularRightShiftIndPartialEval::new(block_size, right_shift_offset, r).unwrap();
		let eval_mvp = shift_r_mvp.evaluate(eval_point).unwrap();

		// Get MultilinearExtension version
		let shift_r_mle = shift_r_mvp.multilinear_extension().unwrap();
		let multilin_query = MultilinearQuery::<F>::with_full_query(eval_point).unwrap();
		let eval_mle = shift_r_mle.evaluate(&multilin_query).unwrap();

		// Assert equality
		assert_eq!(eval_mle, eval_mvp);
	}

	fn test_logical_right_shift_consistency_help<F: Field>(
		block_size: usize,
		right_shift_offset: usize,
	) {
		let mut rng = StdRng::seed_from_u64(0);
		let r = repeat_with(|| F::random(&mut rng))
			.take(block_size)
			.collect::<Vec<_>>();
		let eval_point = &repeat_with(|| F::random(&mut rng))
			.take(block_size)
			.collect::<Vec<_>>();

		// Get Multivariate Poly version
		let shift_r_mvp =
			LogicalRightShiftIndPartialEval::new(block_size, right_shift_offset, r).unwrap();
		let eval_mvp = shift_r_mvp.evaluate(eval_point).unwrap();

		// Get MultilinearExtension version
		let shift_r_mle = shift_r_mvp.multilinear_extension().unwrap();
		let multilin_query = MultilinearQuery::<F>::with_full_query(eval_point).unwrap();
		let eval_mle = shift_r_mle.evaluate(&multilin_query).unwrap();

		// Assert equality
		assert_eq!(eval_mle, eval_mvp);
	}

	fn test_logical_left_shift_consistency_help<F: Field>(
		block_size: usize,
		left_shift_offset: usize,
	) {
		let mut rng = StdRng::seed_from_u64(0);
		let r = repeat_with(|| F::random(&mut rng))
			.take(block_size)
			.collect::<Vec<_>>();
		let eval_point = &repeat_with(|| F::random(&mut rng))
			.take(block_size)
			.collect::<Vec<_>>();

		// Get Multivariate Poly version
		let shift_r_mvp =
			LogicalLeftShiftIndPartialEval::new(block_size, left_shift_offset, r).unwrap();
		let eval_mvp = shift_r_mvp.evaluate(eval_point).unwrap();

		// Get MultilinearExtension version
		let shift_r_mle = shift_r_mvp.multilinear_extension().unwrap();
		let multilin_query = MultilinearQuery::<F>::with_full_query(eval_point).unwrap();
		let eval_mle = shift_r_mle.evaluate(&multilin_query).unwrap();

		// Assert equality
		assert_eq!(eval_mle, eval_mvp);
	}

	#[test]
	fn test_logical_right_shift_consistency_schwartz_zippel() {
		for block_size in 2..=10 {
			for right_shift_offset in [1, 2, 3, (1 << block_size) - 1, (1 << block_size) / 2] {
				test_logical_right_shift_consistency_help::<BinaryField32b>(
					block_size,
					right_shift_offset,
				);
			}
		}
	}

	#[test]
	fn test_circular_right_shift_consistency_schwartz_zippel() {
		for block_size in 2..=10 {
			for right_shift_offset in [1, 2, 3, (1 << block_size) - 1, (1 << block_size) / 2] {
				test_circular_right_shift_consistency_help::<BinaryField32b>(
					block_size,
					right_shift_offset,
				);
			}
		}
	}

	#[test]
	fn test_logical_left_shift_consistency_schwartz_zippel() {
		for block_size in 2..=10 {
			for left_shift_offset in [1, 2, 3, (1 << block_size) - 1, (1 << block_size) / 2] {
				test_logical_left_shift_consistency_help::<BinaryField32b>(
					block_size,
					left_shift_offset,
				);
			}
		}
	}

	// Functionality Tests
	fn decompose_index_to_hypercube_point<F: Field>(block_size: usize, index: usize) -> Vec<F> {
		(0..block_size)
			.map(|k| {
				if (index >> k) % 2 == 1 {
					F::ONE
				} else {
					F::ZERO
				}
			})
			.collect::<Vec<_>>()
	}

	fn test_circular_right_shift_functionality_help<F: Field>(
		block_size: usize,
		right_shift_offset: usize,
	) {
		(0..(1 << block_size)).for_each(|i| {
			let r = decompose_index_to_hypercube_point::<F>(block_size, i);
			let shift_r_mvp =
				CircularRightShiftIndPartialEval::new(block_size, right_shift_offset, r).unwrap();
			(0..(1 << block_size)).for_each(|j| {
				let x = decompose_index_to_hypercube_point::<F>(block_size, j);
				let eval_mvp = shift_r_mvp.evaluate(&x).unwrap();
				if (j + right_shift_offset) % (1 << block_size) == i {
					assert_eq!(eval_mvp, F::ONE);
				} else {
					assert_eq!(eval_mvp, F::ZERO);
				}
			});
		});
	}
	fn test_logical_right_shift_functionality_help<F: Field>(
		block_size: usize,
		right_shift_offset: usize,
	) {
		(0..(1 << block_size)).for_each(|i| {
			let r = decompose_index_to_hypercube_point::<F>(block_size, i);
			let shift_r_mvp =
				LogicalRightShiftIndPartialEval::new(block_size, right_shift_offset, r).unwrap();
			(0..(1 << block_size)).for_each(|j| {
				let x = decompose_index_to_hypercube_point::<F>(block_size, j);
				let eval_mvp = shift_r_mvp.evaluate(&x).unwrap();
				if j + right_shift_offset == i {
					assert_eq!(eval_mvp, F::ONE);
				} else {
					assert_eq!(eval_mvp, F::ZERO);
				}
			});
		});
	}
	fn test_logical_left_shift_functionality_help<F: Field>(
		block_size: usize,
		left_shift_offset: usize,
	) {
		(0..(1 << block_size)).for_each(|i| {
			let r = decompose_index_to_hypercube_point::<F>(block_size, i);
			let shift_r_mvp =
				LogicalLeftShiftIndPartialEval::new(block_size, left_shift_offset, r).unwrap();
			(0..(1 << block_size)).for_each(|j| {
				let x = decompose_index_to_hypercube_point::<F>(block_size, j);
				let eval_mvp = shift_r_mvp.evaluate(&x).unwrap();
				if j >= left_shift_offset && j - left_shift_offset == i {
					assert_eq!(eval_mvp, F::ONE);
				} else {
					assert_eq!(eval_mvp, F::ZERO);
				}
			});
		});
	}

	#[test]
	fn test_circular_right_shift_functionality() {
		for block_size in 3..5 {
			for right_shift_offset in [
				1,
				3,
				(1 << block_size) - 1,
				(1 << block_size) - 2,
				(1 << (block_size - 1)),
			] {
				test_circular_right_shift_functionality_help::<BinaryField32b>(
					block_size,
					right_shift_offset,
				);
			}
		}
	}
	#[test]
	fn test_logical_right_shift_functionality() {
		for block_size in 3..5 {
			for right_shift_offset in [
				1,
				3,
				(1 << block_size) - 1,
				(1 << block_size) - 2,
				(1 << (block_size - 1)),
			] {
				test_logical_right_shift_functionality_help::<BinaryField32b>(
					block_size,
					right_shift_offset,
				);
			}
		}
	}
	#[test]
	fn test_logical_left_shift_functionality() {
		for block_size in 3..5 {
			for left_shift_offset in [
				1,
				3,
				(1 << block_size) - 1,
				(1 << block_size) - 2,
				(1 << (block_size - 1)),
			] {
				test_logical_left_shift_functionality_help::<BinaryField32b>(
					block_size,
					left_shift_offset,
				);
			}
		}
	}
}
