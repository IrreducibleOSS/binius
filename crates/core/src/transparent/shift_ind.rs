// Copyright 2024 Irreducible Inc.

use crate::{
	oracle::ShiftVariant,
	polynomial::{Error, MultivariatePoly},
};
use binius_field::{util::eq, Field, PackedFieldIndexable, TowerField};
use binius_math::MultilinearExtension;
use binius_utils::bail;

/// Represents MLE of shift indicator $f_{b, o}(X, Y)$ on $2*b$ variables
/// partially evaluated at $Y = r$
///
/// # Formal Definition
/// Let $x, y \in \{0, 1\}^b$
/// If ShiftVariant is CircularLeft:
///     * $f(x, y) = 1$ if $\{y\} - \{o\} \equiv \{x\} (\text{mod } 2^b)$
///     * $f(x, y) = 0$ otw
///
/// Else if ShiftVariant is LogicalLeft:
///    * $f(x, y) = 1$ if $\{y\} - \{o\} \equiv \{x\}$
///    * $f(x, y) = 0$ otw
///
/// Else, ShiftVariant is LogicalRight:
///    * $f(x, y) = 1$ if $\{y\} + \{o\} \equiv \{x\}$
///    * $f(x, y) = 0$ otw
///
/// where:
///    * $\{x\}$ is the integer representation of the hypercube point $x \in \{0, 1\}^b$,
///    * $b$ is the block size parameter'
///    * $o$ is the shift offset parameter.
///
/// Observe $\forall x \in \{0, 1\}^b$, there is at most one $y \in \{0, 1\}^b$ s.t. $f(x, y) = 1$
///
/// # Intuition
/// Consider the lexicographic ordering of each point on the $b$-variate hypercube into a $2^b$ length array.
/// Thus, we can give each element on the hypercube a unique index $\in \{0, \ldots, 2^b - 1\}$
/// Let $x, y \in \{0, 1\}^{b}$ be s.t. $\{x\} = i$ and $\{y\} = j$
/// $f(x, y) = 1$ iff:
///     * taking $o$ steps from $j$ gets you to $i$
/// (wrap around if ShiftVariant is Circular + direction of steps depending on ShiftVariant's direction)
///
/// # Note
/// CircularLeft corresponds to the shift indicator in Section 4.3.
/// LogicalLeft corresponds to the shift prime indicator in Section 4.3.
/// LogicalRight corresponds to the shift double prime indicator in Section 4.3.
///
/// [DP23]: https://eprint.iacr.org/2023/1784
///
/// # Example
/// Let $b$ = 2, $o$ = 1, variant = CircularLeft.
/// The hypercube points (0, 0), (1, 0), (0, 1), (1, 1) can be lexicographically
/// ordered into an array [(0, 0), (1, 0), (0, 1), (1, 1)]
/// Then, by considering the index of each hypercube point in the above array, we observe:
///     * $f((0, 0), (1, 0)) = 1$ because $1 - 1 = 0$ mod $4$
///     * $f((1, 0), (0, 1)) = 1$ because $2 - 1 = 1$ mod $4$
///     * $f((0, 1), (1, 1)) = 1$ because $3 - 1 = 2$ mod $4$
///     * $f((1, 1), (0, 0)) = 1$ because $0 - 1 = 3$ mod $4$
/// and every other pair of $b$-variate hypercube points $x, y \in \{0, 1\}^{b}$ is s.t. f(x, y) = 0.
/// Using these shift params, if f = [[a_i, b_i, c_i, d_i]_i], then shifted_f = [[b_i, c_i, d_i, a_i]_i]
///
/// # Example
/// Let $b$ = 2, $o$ = 1, variant = LogicalLeft.
/// The hypercube points (0, 0), (1, 0), (0, 1), (1, 1) can be lexicographically
/// ordered into an array [(0, 0), (1, 0), (0, 1), (1, 1)]
/// Then, by considering the index of each hypercube point in the above array, we observe:
///     * $f((0, 0), (1, 0)) = 1$ because $1 - 1 = 0$
///     * $f((1, 0), (0, 1)) = 1$ because $2 - 1 = 1$
///     * $f((0, 1), (1, 1)) = 1$ because $3 - 1 = 2$
/// and every other pair of $b$-variate hypercube points $x, y \in \{0, 1\}^{b}$ is s.t. f(x, y) = 0.
/// Using these shift params, if f = [[a_i, b_i, c_i, d_i]_i], then shifted_f = [[b_i, c_i, d_i, 0]_i]
///
/// # Example
/// Let $b$ = 2, $o$ = 1, variant = LogicalRight.
/// The hypercube points (0, 0), (1, 0), (0, 1), (1, 1) can be lexicographically
/// ordered into an array [(0, 0), (1, 0), (0, 1), (1, 1)]
/// Then, by considering the index of each hypercube point in the above array, we observe:
///     * $f((1, 0), (0, 0)) = 1$ because $0 + 1 = 1$
///     * $f((0, 1), (1, 0)) = 1$ because $1 + 1 = 2$
///     * $f((1, 1), (0, 1)) = 1$ because $2 + 1 = 3$
/// and every other pair of $b$-variate hypercube points $x, y \in \{0, 1\}^{b}$ is s.t. f(x, y) = 0.
/// Using these shift params, if f = [[a_i, b_i, c_i, d_i]_i], then shifted_f = [[0, a_i, b_i, c_i]_i]
#[derive(Debug, Clone)]
pub struct ShiftIndPartialEval<F: Field> {
	/// Block size $b$, also the number of variables
	block_size: usize,
	/// shift offset $o \in \{1, \ldots, 2^b - 1\}$
	shift_offset: usize,
	/// Shift variant
	shift_variant: ShiftVariant,
	/// partial evaluation point $r$, typically lowest $b$ coords
	/// from a larger challenge point.
	r: Vec<F>,
}

impl<F: Field> ShiftIndPartialEval<F> {
	pub fn new(
		block_size: usize,
		shift_offset: usize,
		shift_variant: ShiftVariant,
		r: Vec<F>,
	) -> Result<Self, Error> {
		assert_valid_shift_ind_args(block_size, shift_offset, &r)?;
		Ok(Self {
			block_size,
			shift_offset,
			r,
			shift_variant,
		})
	}

	fn multilinear_extension_circular<P>(&self) -> Result<MultilinearExtension<P>, Error>
	where
		P: PackedFieldIndexable<Scalar = F>,
	{
		let (ps, pps) =
			partial_evaluate_hypercube_impl::<P>(self.block_size, self.shift_offset, &self.r)?;
		let values = ps
			.iter()
			.zip(pps)
			.map(|(p, pp)| *p + pp)
			.collect::<Vec<_>>();
		Ok(MultilinearExtension::from_values(values)?)
	}

	fn multilinear_extension_logical_left<P>(&self) -> Result<MultilinearExtension<P>, Error>
	where
		P: PackedFieldIndexable<Scalar = F>,
	{
		let (ps, _) =
			partial_evaluate_hypercube_impl::<P>(self.block_size, self.shift_offset, &self.r)?;
		Ok(MultilinearExtension::from_values(ps)?)
	}

	fn multilinear_extension_logical_right<P>(&self) -> Result<MultilinearExtension<P>, Error>
	where
		P: PackedFieldIndexable<Scalar = F>,
	{
		let right_shift_offset = get_left_shift_offset(self.block_size, self.shift_offset);
		let (_, pps) =
			partial_evaluate_hypercube_impl::<P>(self.block_size, right_shift_offset, &self.r)?;
		Ok(MultilinearExtension::from_values(pps)?)
	}

	/// Evaluates this partially evaluated circular shift indicator MLE $f(X, r)$
	/// over the entire $b$-variate hypercube
	pub fn multilinear_extension<P>(&self) -> Result<MultilinearExtension<P>, Error>
	where
		P: PackedFieldIndexable<Scalar = F>,
	{
		match self.shift_variant {
			ShiftVariant::CircularLeft => self.multilinear_extension_circular(),
			ShiftVariant::LogicalLeft => self.multilinear_extension_logical_left(),
			ShiftVariant::LogicalRight => self.multilinear_extension_logical_right(),
		}
	}

	/// Evaluates this partial circular shift indicator MLE $f(X, r)$ with $X=x$
	fn evaluate_at_point(&self, x: &[F]) -> Result<F, Error> {
		if x.len() != self.block_size {
			bail!(Error::IncorrectQuerySize {
				expected: self.block_size,
			});
		}

		let left_shift_offset = match self.shift_variant {
			ShiftVariant::CircularLeft => self.shift_offset,
			ShiftVariant::LogicalLeft => self.shift_offset,
			ShiftVariant::LogicalRight => get_left_shift_offset(self.block_size, self.shift_offset),
		};

		let (p_res, pp_res) =
			evaluate_shift_ind_help(self.block_size, left_shift_offset, x, &self.r)?;

		match self.shift_variant {
			ShiftVariant::CircularLeft => Ok(p_res + pp_res),
			ShiftVariant::LogicalLeft => Ok(p_res),
			ShiftVariant::LogicalRight => Ok(pp_res),
		}
	}
}

impl<F: TowerField> MultivariatePoly<F> for ShiftIndPartialEval<F> {
	fn n_vars(&self) -> usize {
		self.block_size
	}

	fn degree(&self) -> usize {
		self.block_size
	}

	fn evaluate(&self, query: &[F]) -> Result<F, Error> {
		self.evaluate_at_point(query)
	}

	fn binary_tower_level(&self) -> usize {
		F::TOWER_LEVEL
	}
}

/// Gets right shift offset from left shift offset
fn get_left_shift_offset(block_size: usize, right_shift_offset: usize) -> usize {
	(1 << block_size) - right_shift_offset
}

/// Checks validity of shift indicator arguments
fn assert_valid_shift_ind_args<F: Field>(
	block_size: usize,
	shift_offset: usize,
	partial_query_point: &[F],
) -> Result<(), Error> {
	if partial_query_point.len() != block_size {
		bail!(Error::IncorrectQuerySize {
			expected: block_size,
		});
	}
	if shift_offset == 0 || shift_offset >= 1 << block_size {
		bail!(Error::InvalidShiftOffset {
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
		bail!(Error::IncorrectQuerySize {
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
fn partial_evaluate_hypercube_impl<P: PackedFieldIndexable>(
	block_size: usize,
	shift_offset: usize,
	r: &[P::Scalar],
) -> Result<(Vec<P>, Vec<P>), Error> {
	assert_valid_shift_ind_args(block_size, shift_offset, r)?;
	let mut s_ind_p = vec![P::one(); 1 << (block_size - P::LOG_WIDTH)];
	let mut s_ind_pp = vec![P::zero(); 1 << (block_size - P::LOG_WIDTH)];

	partial_evaluate_hypercube_with_buffers(
		block_size.min(P::LOG_WIDTH),
		shift_offset,
		r,
		P::unpack_scalars_mut(&mut s_ind_p),
		P::unpack_scalars_mut(&mut s_ind_pp),
	);
	if block_size > P::LOG_WIDTH {
		partial_evaluate_hypercube_with_buffers(
			block_size - P::LOG_WIDTH,
			shift_offset >> P::LOG_WIDTH,
			&r[P::LOG_WIDTH..],
			&mut s_ind_p,
			&mut s_ind_pp,
		);
	}

	Ok((s_ind_p, s_ind_pp))
}

fn partial_evaluate_hypercube_with_buffers<P: PackedFieldIndexable>(
	block_size: usize,
	shift_offset: usize,
	r: &[P::Scalar],
	s_ind_p: &mut [P],
	s_ind_pp: &mut [P],
) {
	for k in 0..block_size {
		// complexity: just two multiplications per iteration!
		if (shift_offset >> k) % 2 == 1 {
			for i in 0..(1 << k) {
				let mut pp_lo = s_ind_pp[i];
				let mut pp_hi = pp_lo * r[k];

				pp_lo -= pp_hi;

				let p_lo = s_ind_p[i];
				let p_hi = p_lo * r[k];
				pp_hi += p_lo - p_hi; // * 1 - r

				s_ind_pp[i] = pp_lo;
				s_ind_pp[1 << k | i] = pp_hi;

				s_ind_p[i] = p_hi;
				s_ind_p[1 << k | i] = P::zero(); // clear upper half
			}
		} else {
			for i in 0..(1 << k) {
				let mut p_lo = s_ind_p[i];
				let p_hi = p_lo * r[k];
				p_lo -= p_hi;

				let pp_lo = s_ind_pp[i];
				let pp_hi = pp_lo * (P::one() - r[k]);
				p_lo += pp_lo - pp_hi;

				s_ind_p[i] = p_lo;
				s_ind_p[1 << k | i] = p_hi;

				s_ind_pp[i] = P::zero(); // clear lower half
				s_ind_pp[1 << k | i] = pp_hi;
			}
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::polynomial::test_utils::decompose_index_to_hypercube_point;
	use binius_field::{BinaryField32b, PackedBinaryField4x32b};
	use binius_hal::{make_portable_backend, ComputationBackendExt};
	use rand::{rngs::StdRng, SeedableRng};
	use std::iter::repeat_with;

	// Consistency Tests for each shift variant
	fn test_circular_left_shift_consistency_help<
		F: TowerField,
		P: PackedFieldIndexable<Scalar = F>,
	>(
		block_size: usize,
		right_shift_offset: usize,
	) {
		let mut rng = StdRng::seed_from_u64(0);
		let backend = make_portable_backend();
		let r = repeat_with(|| F::random(&mut rng))
			.take(block_size)
			.collect::<Vec<_>>();
		let eval_point = &repeat_with(|| F::random(&mut rng))
			.take(block_size)
			.collect::<Vec<_>>();

		// Get Multivariate Poly version
		let shift_variant = ShiftVariant::CircularLeft;
		let shift_r_mvp =
			ShiftIndPartialEval::new(block_size, right_shift_offset, shift_variant, r).unwrap();
		let eval_mvp = shift_r_mvp.evaluate(eval_point).unwrap();

		// Get MultilinearExtension version
		let shift_r_mle = shift_r_mvp.multilinear_extension::<P>().unwrap();
		let multilin_query = backend.multilinear_query::<P>(eval_point).unwrap();
		let eval_mle = shift_r_mle.evaluate(&multilin_query).unwrap();

		// Assert equality
		assert_eq!(eval_mle, eval_mvp);
	}

	fn test_logical_left_shift_consistency_help<
		F: TowerField,
		P: PackedFieldIndexable<Scalar = F>,
	>(
		block_size: usize,
		right_shift_offset: usize,
	) {
		let mut rng = StdRng::seed_from_u64(0);
		let backend = make_portable_backend();
		let r = repeat_with(|| F::random(&mut rng))
			.take(block_size)
			.collect::<Vec<_>>();
		let eval_point = &repeat_with(|| F::random(&mut rng))
			.take(block_size)
			.collect::<Vec<_>>();

		// Get Multivariate Poly version
		let shift_variant = ShiftVariant::LogicalLeft;
		let shift_r_mvp =
			ShiftIndPartialEval::new(block_size, right_shift_offset, shift_variant, r).unwrap();
		let eval_mvp = shift_r_mvp.evaluate(eval_point).unwrap();

		// Get MultilinearExtension version
		let shift_r_mle = shift_r_mvp.multilinear_extension::<P>().unwrap();
		let multilin_query = backend.multilinear_query::<P>(eval_point).unwrap();
		let eval_mle = shift_r_mle.evaluate(&multilin_query).unwrap();

		// Assert equality
		assert_eq!(eval_mle, eval_mvp);
	}

	fn test_logical_right_shift_consistency_help<
		F: TowerField,
		P: PackedFieldIndexable<Scalar = F>,
	>(
		block_size: usize,
		left_shift_offset: usize,
	) {
		let mut rng = StdRng::seed_from_u64(0);
		let backend = make_portable_backend();
		let r = repeat_with(|| F::random(&mut rng))
			.take(block_size)
			.collect::<Vec<_>>();
		let eval_point = &repeat_with(|| F::random(&mut rng))
			.take(block_size)
			.collect::<Vec<_>>();

		// Get Multivariate Poly version
		let shift_variant = ShiftVariant::LogicalRight;
		let shift_r_mvp =
			ShiftIndPartialEval::new(block_size, left_shift_offset, shift_variant, r).unwrap();
		let eval_mvp = shift_r_mvp.evaluate(eval_point).unwrap();

		// Get MultilinearExtension version
		let shift_r_mle = shift_r_mvp.multilinear_extension::<P>().unwrap();
		let multilin_query = backend.multilinear_query::<P>(eval_point).unwrap();
		let eval_mle = shift_r_mle.evaluate(&multilin_query).unwrap();

		// Assert equality
		assert_eq!(eval_mle, eval_mvp);
	}

	#[test]
	fn test_circular_left_shift_consistency_schwartz_zippel() {
		for block_size in 2..=10 {
			for right_shift_offset in [1, 2, 3, (1 << block_size) - 1, (1 << block_size) / 2] {
				test_circular_left_shift_consistency_help::<_, PackedBinaryField4x32b>(
					block_size,
					right_shift_offset,
				);
			}
		}
	}

	#[test]
	fn test_logical_left_shift_consistency_schwartz_zippel() {
		for block_size in 2..=10 {
			for right_shift_offset in [1, 2, 3, (1 << block_size) - 1, (1 << block_size) / 2] {
				test_logical_left_shift_consistency_help::<_, PackedBinaryField4x32b>(
					block_size,
					right_shift_offset,
				);
			}
		}
	}

	#[test]
	fn test_logical_right_shift_consistency_schwartz_zippel() {
		for block_size in 2..=10 {
			for left_shift_offset in [1, 2, 3, (1 << block_size) - 1, (1 << block_size) / 2] {
				test_logical_right_shift_consistency_help::<_, PackedBinaryField4x32b>(
					block_size,
					left_shift_offset,
				);
			}
		}
	}

	// Functionality Tests for each shift variant
	fn test_circular_left_shift_functionality_help<F: TowerField>(
		block_size: usize,
		right_shift_offset: usize,
	) {
		let shift_variant = ShiftVariant::CircularLeft;
		(0..(1 << block_size)).for_each(|i| {
			let r = decompose_index_to_hypercube_point::<F>(block_size, i);
			let shift_r_mvp =
				ShiftIndPartialEval::new(block_size, right_shift_offset, shift_variant, r).unwrap();
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
	fn test_logical_left_shift_functionality_help<F: TowerField>(
		block_size: usize,
		right_shift_offset: usize,
	) {
		let shift_variant = ShiftVariant::LogicalLeft;
		(0..(1 << block_size)).for_each(|i| {
			let r = decompose_index_to_hypercube_point::<F>(block_size, i);
			let shift_r_mvp =
				ShiftIndPartialEval::new(block_size, right_shift_offset, shift_variant, r).unwrap();
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

	fn test_logical_right_shift_functionality_help<F: TowerField>(
		block_size: usize,
		left_shift_offset: usize,
	) {
		let shift_variant = ShiftVariant::LogicalRight;
		(0..(1 << block_size)).for_each(|i| {
			let r = decompose_index_to_hypercube_point::<F>(block_size, i);
			let shift_r_mvp =
				ShiftIndPartialEval::new(block_size, left_shift_offset, shift_variant, r).unwrap();
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
	fn test_circular_left_shift_functionality() {
		for block_size in 3..5 {
			for right_shift_offset in [
				1,
				3,
				(1 << block_size) - 1,
				(1 << block_size) - 2,
				(1 << (block_size - 1)),
			] {
				test_circular_left_shift_functionality_help::<BinaryField32b>(
					block_size,
					right_shift_offset,
				);
			}
		}
	}
	#[test]
	fn test_logical_left_shift_functionality() {
		for block_size in 3..5 {
			for right_shift_offset in [
				1,
				3,
				(1 << block_size) - 1,
				(1 << block_size) - 2,
				(1 << (block_size - 1)),
			] {
				test_logical_left_shift_functionality_help::<BinaryField32b>(
					block_size,
					right_shift_offset,
				);
			}
		}
	}
	#[test]
	fn test_logical_right_shift_functionality() {
		for block_size in 3..5 {
			for left_shift_offset in [
				1,
				3,
				(1 << block_size) - 1,
				(1 << block_size) - 2,
				(1 << (block_size - 1)),
			] {
				test_logical_right_shift_functionality_help::<BinaryField32b>(
					block_size,
					left_shift_offset,
				);
			}
		}
	}
}
