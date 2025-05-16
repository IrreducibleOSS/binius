// Copyright 2024-2025 Irreducible Inc.

use binius_field::BinaryField;
use binius_math::Matrix;
use binius_utils::{bail, checked_arithmetics::log2_ceil_usize};

use crate::{
	additive_ntt::{AdditiveNTT, NTTShape},
	error::Error,
	twiddle::TwiddleAccess,
};

pub struct OddInterpolate<F: BinaryField> {
	vandermonde_inverse: Matrix<F>,
	ell: usize,
}

impl<F: BinaryField> OddInterpolate<F> {
	/// Create a new odd interpolator into novel polynomial basis for domains of size $d \times
	/// 2^{\ell}$. Takes a reference to NTT twiddle factors to seed the "Vandermonde" matrix and
	/// compute its inverse. Time complexity is $\mathcal{O}(d^3).$
	pub fn new<TA>(d: usize, ell: usize, twiddle_access: &[TA]) -> Result<Self, Error>
	where
		TA: TwiddleAccess<F>,
	{
		// TODO: This constructor should accept an `impl AdditiveNTT` instead of an
		// `impl TwiddleAccess`. It can use `AdditiveNTT::get_subspace_eval` instead of the twiddle
		// accessors directly. `AdditiveNTT` is a more public interface.
		let vandermonde = novel_vandermonde(d, ell, twiddle_access)?;

		let mut vandermonde_inverse = Matrix::zeros(d, d);
		vandermonde.inverse_into(&mut vandermonde_inverse)?;

		Ok(Self {
			vandermonde_inverse,
			ell,
		})
	}

	/// Let $L/\mathbb F_2$ be a binary field, and fix an $\mathbb F_2$-basis $1=:\beta_0,\ldots,
	/// \beta_{r-1}$ as usual. Let $d\geq 1$ be an odd integer and let $\ell\geq 0$ be an integer.
	/// Let $[a_0,\ldots, a_{d\times 2^{\ell} - 1}]$ be a list of elements of $L$. There is a
	/// unique univariate polynomial $P(X)\in L\[X\]$ of degree less than $d\times 2^{\ell}$ such
	/// that the *evaluations* of $P$ on the "first" $d\times 2^{\ell}$ elements of $L$ (in
	/// little-Endian binary counting order with respect to the basis $\beta_0,\ldots, \beta_{r}$)
	/// are precisely $a_0,\ldots, a_{d\times 2^{\ell} - 1}$.
	///
	/// We efficiently compute the coefficients of $P(X)$ with respect to the Novel Polynomial Basis
	/// (itself taken with respect to the given ordered list $\beta_0,\ldots, \beta_{r-1}$).
	///
	/// Time complexity is $\mathcal{O}(d^2\times 2^{\ell} + \ell 2^{\ell})$, thus this routine is
	/// intended to be used for small values of $d$.
	pub fn inverse_transform<NTT>(&self, ntt: &NTT, data: &mut [F]) -> Result<(), Error>
	where
		// REVIEW: generalize this to any P: PackedField<Scalar=F>
		NTT: AdditiveNTT<F>,
	{
		let d = self.vandermonde_inverse.m();
		let ell = self.ell;

		if data.len() != d << ell {
			bail!(Error::OddInterpolateIncorrectLength {
				expected_len: d << ell
			});
		}

		let coset_bits = log2_ceil_usize(d);
		let log_required_domain_size = coset_bits + ell;
		if ntt.log_domain_size() < log_required_domain_size {
			bail!(Error::DomainTooSmall {
				log_required_domain_size
			});
		}

		let shape = NTTShape {
			log_y: ell,
			..Default::default()
		};
		for (i, chunk) in data.chunks_exact_mut(1 << ell).enumerate() {
			ntt.inverse_transform(chunk, shape, i, coset_bits, 0)?;
		}

		// Given M and a vector v, do the "strided product" M v. In more detail: we assume matrix is
		// $d\times d$, and vector is $d\times 2^{\ell}$. For each $i$ in $0,\ldots, 2^{\ell-1}$,
		// let $v_i$ be the subvect given by those entries whose index is congruent to $i$ mod
		// $2^{\ell}$. Then this computes $M v_i$, and finally "interleaves" the result (which
		// means that we treat $M v_i = w_i$ for each $i$ and then conjure up the associated
		// vector $w$.)
		let mut bases = vec![F::ZERO; d];
		let mut novel = vec![F::ZERO; d];
		// TODO: use `Matrix::mul_into`, implement when data is a slice of type `P:
		// PackedField<Scalar=F>`.
		for stride in 0..1 << ell {
			(0..d).for_each(|i| bases[i] = data[i << ell | stride]);
			self.vandermonde_inverse.mul_vec_into(&bases, &mut novel);
			(0..d).for_each(|i| data[i << ell | stride] = novel[i]);
		}

		Ok(())
	}
}

/// Compute the Vandermonde matrix: $X^{(\ell)}_i(w^{\ell}_j)$, where $w^{\ell}_j$ is the
/// $j^{\text{th}}$ element of the field with respect to the $\beta^{(\ell)}_i$ in little Endian
/// order. The matrix has dimensions $d\times d$. The key trick is that
/// $\widehat{W}^{(\ell)}_i(\beta^{\ell}_j) = $\widehat{W}_{i+\ell}(\beta_{j+\ell})$.
fn novel_vandermonde<F, TA>(d: usize, ell: usize, twiddle_access: &[TA]) -> Result<Matrix<F>, Error>
where
	F: BinaryField,
	TA: TwiddleAccess<F>,
{
	if d == 0 {
		return Ok(Matrix::zeros(0, 0));
	}

	let log_d = log2_ceil_usize(d);

	// This will contain the evaluations of $X^{(\ell)}_{j}(w^{(\ell)}_i)$. As usual, indexing goes
	// from 0..d-1.
	let mut x_ell = Matrix::zeros(d, d);

	// $X_0$ is the function "1".
	(0..d).for_each(|j| x_ell[(j, 0)] = F::ONE);

	let log_required_domain_size = log_d + ell;
	if twiddle_access.len() < log_required_domain_size {
		bail!(Error::DomainTooSmall {
			log_required_domain_size
		});
	}

	let twiddle_access = &twiddle_access[twiddle_access.len() - log_d..];
	for (j, twiddle_access_j_plus_ell) in twiddle_access.iter().enumerate() {
		assert!(twiddle_access_j_plus_ell.log_n() >= log_d - 1 - j);

		for i in 0..d {
			x_ell[(i, 1 << j)] = twiddle_access_j_plus_ell.get(i >> (j + 1))
				+ if (i >> j) & 1 == 1 { F::ONE } else { F::ZERO };
		}

		// Note that the jth column of x_ell is the ordered list of values $X_j(w_i)$ for i = 0,
		// ..., d-1.
		for k in 1..(1 << j).min(d - (1 << j)) {
			for t in 0..d {
				x_ell[(t, k + (1 << j))] = x_ell[(t, k)] * x_ell[(t, 1 << j)];
			}
		}
	}

	Ok(x_ell)
}

#[cfg(test)]
mod tests {
	use std::iter::repeat_with;

	use binius_field::{BinaryField32b, Field};
	use rand::{rngs::StdRng, SeedableRng};

	use super::*;
	use crate::single_threaded::SingleThreadedNTT;

	#[test]
	fn test_interpolate_odd() {
		type F = BinaryField32b;
		let max_ell = 8;
		let max_d = 10;

		let mut rng = StdRng::seed_from_u64(0);
		let ntt = SingleThreadedNTT::<F>::new(max_ell + log2_ceil_usize(max_d)).unwrap();

		for ell in 0..max_ell {
			for d in 0..max_d {
				let expected_novel = repeat_with(|| F::random(&mut rng))
					.take(d << ell)
					.collect::<Vec<_>>();

				let mut ntt_evals = expected_novel.clone();
				// zero-pad to the next power of two to apply the forward transform.
				let next_log_n = log2_ceil_usize(expected_novel.len());
				ntt_evals.resize(1 << next_log_n, F::ZERO);
				// apply forward transform and then run our odd interpolation routine.
				let shape = NTTShape {
					log_y: next_log_n,
					..Default::default()
				};
				ntt.forward_transform(&mut ntt_evals, shape, 0, 0, 0)
					.unwrap();

				let odd_interpolate = OddInterpolate::new(d, ell, &ntt.s_evals).unwrap();
				odd_interpolate
					.inverse_transform(&ntt, &mut ntt_evals[..d << ell])
					.unwrap();

				assert_eq!(expected_novel, &ntt_evals[..d << ell]);
			}
		}
	}
}
