// Copyright 2024-2025 Irreducible Inc.

use binius_field::Field;
use tracing::instrument;

use crate::{Error, extrapolate_line_scalar};

/// Evaluate a piecewise multilinear polynomial at a point, given the evaluations of the pieces.
///
/// A piecewise multilinear is defined by a sequence of multilinear polynomials, ordered from
/// most number of variables to least. We define it over a boolean hypercube by identifying
/// the smallest hypercube larger than the total number of hypercube evaluations of all the pieces,
/// then concatenating the evaluations, flattened into vectors in little-endian order. The
/// evaluation vector is right-padded with zeros up to the hypercube size of the containing cube.
/// Equivalently, we can view the piecewise multilinear as defined by an inductive linear
/// interpolation of pairs of polynomials.
///
/// This function takes a description of a piecewise multilinear polynomial and the evaluations
/// of the pieces at prefixes of the given point, then evaluates the concatenated multilinear at
/// the point.
///
/// ## Arguments
///
/// * `point` - the evaluation point. The length specifies the number of variables in the
///   concatenated multilinear.
/// * `n_pieces_by_vars` - the number of multilinear pieces, indexed by the number of variables.
///   Entry at index `i` is the number of multilinears with `i` variables. The sum of
///   `n_pieces_by_vars[i] * 2^i` for all indices `i` must be at most `2^n`, where `n` is the length
///   of `point.`.
/// * `piece_evals` - the evaluations of the multilinear pieces at the corresponding prefixes of
///   `point`. The length must be equal to the sum of the values in `n_pieces_by_vars`. This must be
///   in the *same order* as the corresponding multilinears are implicitly concatenated.
///
/// ## Example
///
/// Suppose we have three multilinear functions $f_0$, $f_1$, and $f_2$, which have $2$, $2$, and
/// $1$ variables respectively. There exists a unique $4$-variate multilinear function $\tilde{f}$,
/// which is defined by concatenating the evaluations of $f_0$, $f_1$, and $f_2$ along their
/// respective defining Boolean hypercubes and then zero-padding. (I.e., we simply decree that the
/// resulting list of length 16 is the ordered list of evaluations of a multilinear $\tilde{f}$
/// on $\mathcal B_4$.) We wish to evaluate $\tilde{f}$ at the point $(r_0, r_1, r_2, r_3)$,
/// and we are given the evaluations of $v_0:=f_0(r_0,r_1)$, $v_1:=f_1(r_0,r_1)$, and
/// $v_2:=f_2(r_0)$. In this situation, we have the following:
/// `n_pieces_by_vars` is `[0, 1, 2]`, and `piece_evals` is `[v_0, v_1, v_2]`.
#[instrument(skip_all)]
pub fn evaluate_piecewise_multilinear<F: Field>(
	point: &[F],
	n_pieces_by_vars: &[usize],
	piece_evals: &mut [F],
) -> Result<F, Error> {
	// dimension of the big hypercube
	let total_n_vars = point.len();
	// number of multilinears
	let num_polys = n_pieces_by_vars.iter().sum();
	// total number of coefficients in all of the multilinears.
	let total_length: usize = n_pieces_by_vars
		.iter()
		.enumerate()
		.map(|(i, &n)| n << i)
		.sum();
	if total_length > 1 << total_n_vars {
		return Err(Error::PiecewiseMultilinearTooLong {
			total_length,
			total_n_vars,
		});
	}
	if piece_evals.len() != num_polys {
		return Err(Error::PiecewiseMultilinearIncompatibleEvals {
			expected: num_polys,
			actual: piece_evals.len(),
		});
	}

	// The logic is that we can iteratively compute the "root claim" via a "folding" procedure,
	// reading from right to left.
	// To demonstrate what is happening, suppose we have 3 multilinears f_0, f_1, f_2, of lengths
	// 2, 2, 1 respectively.
	// We want to evaluate the concatenation at point [r_0, r_1, r_2, r_3]. We are given the
	// evaluations:
	// v_0 := f_0(r_0, r_1), v_1 := f_1(r_0, r_1), and v_2 := f_2(r_0).
	// we first replace v_2 with (1-r_1)*v_2 + r_1 * 0, which is implicitly the evaluation of the
	// 2-variate multilinear corresponding to f_2(evals)||0, 0 at the point (r_0, r_1).
	// we then replace v_0 with (1-r_2)*v_0 + r_2 * v_1 and v_1 with (1-r_2)*v_1 + r_2 * 0.
	// this list of length 2 represents the evaluation claims of the trivariate multilinears
	// corresponding to:
	// 	1. f_0(evals)||f_1(evals); and
	// 	2. f_2(evals)||0, 0, 0, 0
	// at the point (r_0, r_1, r_2).
	// finally, we fold our final list of length 2 via r_3 to obtain the root claim.
	let mut index = piece_evals.len();
	let mut n_to_fold = 0;
	for (i, &point_i) in point.iter().enumerate() {
		n_to_fold += n_pieces_by_vars.get(i).copied().unwrap_or(0);
		fold_segment(&mut piece_evals[index - n_to_fold..index], point_i);
		let n_folded_out = n_to_fold / 2;
		index -= n_folded_out;
		n_to_fold -= n_folded_out;
	}

	Ok(piece_evals[0])
}

/// Folds a sequence of pairs of field elements by linear extrapolation at a given point.
///
/// Given a list $[a_0,\ldots, a_{n-1}]$ and a parameter $r$, mutate to the list
/// $[a_0(1-r) + a_1r, a_2(1-r) + a_3r, \ldots, 0, \ldots]$. If $n$ is odd, then the last
/// potentially non-zero element is $a_{n-1}(1-r)$, at index $(n-1)/2$.
fn fold_segment<F: Field>(segment: &mut [F], z: F) {
	let n_full_pairs = segment.len() / 2;
	for i in 0..n_full_pairs {
		segment[i] = extrapolate_line_scalar(segment[2 * i], segment[2 * i + 1], z);
	}
	if segment.len() % 2 == 1 {
		let i = segment.len() / 2;
		segment[i] = extrapolate_line_scalar(segment[2 * i], F::ZERO, z);
	}
}

#[cfg(test)]
mod tests {
	use std::iter::repeat_with;

	use binius_field::BinaryField32b;
	use binius_utils::checked_arithmetics::{log2_ceil_usize, log2_strict_usize};
	use rand::{SeedableRng, prelude::StdRng};

	use super::*;
	use crate::{MultilinearExtension, MultilinearQuery};

	fn test_piecewise_multilinear(number_of_variables: &[usize]) {
		type F = BinaryField32b;
		assert!(
			number_of_variables.windows(2).all(|w| w[0] >= w[1]),
			"Number of variables must be sorted in non-increasing order"
		);

		let mut n_pieces_by_vars = vec![0; number_of_variables[0] + 1];
		for &n_vars in number_of_variables {
			n_pieces_by_vars[n_vars] += 1;
		}

		let mut rng = StdRng::seed_from_u64(0);

		// build random multilinears with the given number of variables.
		let mut multilinear_coefficients: Vec<Vec<F>> = vec![];
		for &n_vars in number_of_variables {
			multilinear_coefficients.push(gen_random_multilinear(n_vars, &mut rng));
		}

		let total_num_coeffs: usize = number_of_variables.iter().map(|&n| 1 << n).sum();
		let total_number_of_variables = log2_ceil_usize(total_num_coeffs);
		let zero_padding_length = (1 << total_number_of_variables) - total_num_coeffs;

		// zero-padded concatenated multilinear
		let concatenated_multilinear_coeffs = multilinear_coefficients
			.iter()
			.flat_map(|coeffs| coeffs.iter())
			.copied()
			.chain(repeat_with(|| F::ZERO).take(zero_padding_length))
			.collect::<Vec<_>>();

		let concatenated_multilinear = MultilinearExtension::<F, Vec<F>>::new(
			total_number_of_variables,
			concatenated_multilinear_coeffs,
		)
		.unwrap();

		let eval_point = repeat_with(|| <F as Field>::random(&mut rng))
			.take(total_number_of_variables)
			.collect::<Vec<_>>();

		let mlq_eval_point = MultilinearQuery::<F>::expand(&eval_point);
		// compute individual claims.
		let mut piece_evals =
			eval_multilinears_at_common_prefix(&multilinear_coefficients, &eval_point);

		// compute claim of concatenated polynomial directly.
		let concatenate_and_evaluate = concatenated_multilinear.evaluate(&mlq_eval_point).unwrap();

		// compute claim of concatenated polynomial via `evaluation_piecewise_multilinear`
		let compute_via_piecewise = evaluate_piecewise_multilinear(
			eval_point.as_slice(),
			&n_pieces_by_vars,
			&mut piece_evals,
		)
		.unwrap();
		assert_eq!(concatenate_and_evaluate, compute_via_piecewise);
	}

	#[test]
	fn test_piecewise_multilinear_4_4() {
		test_piecewise_multilinear(&[4, 4]);
	}

	#[test]
	fn test_piecewise_multilinear_5_3_2() {
		test_piecewise_multilinear(&[5, 3, 2]);
	}

	#[test]
	fn test_piecewise_multilinear_5_5_3_3_2_2_2() {
		test_piecewise_multilinear(&[5, 5, 3, 3, 2, 2, 2]);
	}

	#[test]
	fn test_piecewise_multilinear_2_2_1() {
		test_piecewise_multilinear(&[2, 2, 1]);
	}

	#[test]
	fn test_piecewise_multilinear_3_1_0_0_0() {
		test_piecewise_multilinear(&[3, 1, 0, 0, 0]);
	}

	fn gen_random_multilinear<F: Field>(n_vars: usize, mut rng: &mut StdRng) -> Vec<F> {
		repeat_with(|| F::random(&mut rng))
			.take(1 << n_vars)
			.collect::<Vec<_>>()
	}

	fn eval_multilinears_at_common_prefix<F: Field>(
		multilinears: &[Vec<F>],
		prefix: &[F],
	) -> Vec<F> {
		let mut result = Vec::new();
		for multilinear_coeffs in multilinears {
			let log_len = log2_strict_usize(multilinear_coeffs.len());
			let multilinear =
				MultilinearExtension::<F, Vec<F>>::new(log_len, multilinear_coeffs.clone())
					.unwrap();
			let mlq_eval_point = MultilinearQuery::<F>::expand(&prefix[..log_len]);
			let eval = multilinear.evaluate(&mlq_eval_point).unwrap();
			result.push(eval);
		}
		result
	}

	#[test]
	fn test_fold_segment_basic() {
		let s0 = BinaryField32b::from(2);
		let s1 = BinaryField32b::from(4);
		let s2 = BinaryField32b::from(6);
		let s3 = BinaryField32b::from(8);

		let mut segment = vec![s0, s1, s2, s3];
		let z = BinaryField32b::from(3);

		fold_segment(&mut segment, z);

		assert_eq!(
			segment,
			vec![
				extrapolate_line_scalar(s0, s1, z),
				extrapolate_line_scalar(s2, s3, z),
				s2,
				s3
			]
		);
	}

	#[test]
	fn test_fold_segment_single_element() {
		let s0 = BinaryField32b::from(5);
		let mut segment = vec![s0];
		let z = BinaryField32b::from(3);

		fold_segment(&mut segment, z);

		assert_eq!(segment, vec![extrapolate_line_scalar(s0, BinaryField32b::ZERO, z),]);
	}

	#[test]
	fn test_fold_segment_odd_length() {
		let s0 = BinaryField32b::from(1);
		let s1 = BinaryField32b::from(3);
		let s2 = BinaryField32b::from(5);

		let mut segment = vec![s0, s1, s2];
		let z = BinaryField32b::from(2);

		fold_segment(&mut segment, z);

		assert_eq!(
			segment,
			vec![
				extrapolate_line_scalar(s0, s1, z),
				extrapolate_line_scalar(s2, BinaryField32b::ZERO, z),
				s2
			]
		);
	}
}
