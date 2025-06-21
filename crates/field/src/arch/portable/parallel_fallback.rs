// Copyright 2024-2025 Irreducible Inc.

//! Portable fallback implementations for parallel field operations.
//!
//! This module provides the same function signatures as the ARM NEON parallel
//! implementations but falls back to sequential processing for other architectures.

use crate::{
	PackedBinaryField16x8b, PackedAESBinaryField16x8b,
	arithmetic_traits::{MulAlpha, Square, InvertOrZero},
};

/// Fallback parallel multiplication - processes sequentially
pub fn packed_tower_16x8b_multiply_batch_parallel(
	lhs_batch: &[PackedBinaryField16x8b],
	rhs_batch: &[PackedBinaryField16x8b],
	output_batch: &mut [PackedBinaryField16x8b],
) {
	for ((lhs, rhs), output) in lhs_batch.iter().zip(rhs_batch.iter()).zip(output_batch.iter_mut()) {
		*output = *lhs * *rhs;
	}
}

/// Fallback parallel squaring - processes sequentially
pub fn packed_tower_16x8b_square_batch_parallel(
	input_batch: &[PackedBinaryField16x8b],
	output_batch: &mut [PackedBinaryField16x8b],
) {
	for (input, output) in input_batch.iter().zip(output_batch.iter_mut()) {
		*output = Square::square(*input);
	}
}

/// Fallback parallel inversion - processes sequentially
pub fn packed_tower_16x8b_invert_batch_parallel(
	input_batch: &[PackedBinaryField16x8b],
	output_batch: &mut [PackedBinaryField16x8b],
) {
	for (input, output) in input_batch.iter().zip(output_batch.iter_mut()) {
		*output = InvertOrZero::invert_or_zero(*input);
	}
}

/// Fallback parallel multiply alpha - processes sequentially
pub fn packed_tower_16x8b_multiply_alpha_batch_parallel(
	input_batch: &[PackedBinaryField16x8b],
	output_batch: &mut [PackedBinaryField16x8b],
) {
	for (input, output) in input_batch.iter().zip(output_batch.iter_mut()) {
		*output = input.mul_alpha();
	}
}

/// Fallback parallel AES multiplication - processes sequentially
pub fn packed_aes_16x8b_multiply_batch_parallel(
	lhs_batch: &[PackedAESBinaryField16x8b],
	rhs_batch: &[PackedAESBinaryField16x8b],
	output_batch: &mut [PackedAESBinaryField16x8b],
) {
	for ((lhs, rhs), output) in lhs_batch.iter().zip(rhs_batch.iter()).zip(output_batch.iter_mut()) {
		*output = *lhs * *rhs;
	}
}

/// Fallback parallel AES to tower conversion - processes sequentially
pub fn packed_aes_to_tower_batch_parallel(
	input_batch: &[PackedAESBinaryField16x8b],
	output_batch: &mut [PackedBinaryField16x8b],
) {
	for (input, output) in input_batch.iter().zip(output_batch.iter_mut()) {
		*output = PackedBinaryField16x8b::from(*input);
	}
}

/// Fallback parallel tower to AES conversion - processes sequentially
pub fn packed_tower_to_aes_batch_parallel(
	input_batch: &[PackedBinaryField16x8b],
	output_batch: &mut [PackedAESBinaryField16x8b],
) {
	for (input, output) in input_batch.iter().zip(output_batch.iter_mut()) {
		*output = PackedAESBinaryField16x8b::from(*input);
	}
}

/// Fallback parallel linear combination - processes sequentially
pub fn packed_tower_16x8b_linear_combination_parallel(
	coeffs: &[PackedBinaryField16x8b],
	values: &[PackedBinaryField16x8b],
	output: &mut PackedBinaryField16x8b,
) {
	*output = PackedBinaryField16x8b::default();
	for (coeff, value) in coeffs.iter().zip(values.iter()) {
		*output += *coeff * *value;
	}
}

/// Fallback parallel multilinear evaluation - processes sequentially
pub fn packed_tower_16x8b_multilinear_eval_parallel(
	coeffs: &[PackedBinaryField16x8b],
	eval_point: &[PackedBinaryField16x8b],
	output: &mut PackedBinaryField16x8b,
) {
	// Simple multilinear evaluation using horner's method
	*output = PackedBinaryField16x8b::default();
	for (coeff, point) in coeffs.iter().zip(eval_point.iter()) {
		*output = *output * *point + *coeff;
	}
}

/// Fallback parallel interpolation - processes sequentially
pub fn packed_tower_16x8b_interpolate_parallel(
	points: &[PackedBinaryField16x8b],
	values: &[PackedBinaryField16x8b],
	eval_point: PackedBinaryField16x8b,
	output: &mut PackedBinaryField16x8b,
) {
	// Simple Lagrange interpolation
	*output = PackedBinaryField16x8b::default();
	let n = points.len();
	
	for i in 0..n {
		let mut term = values[i];
		for j in 0..n {
			if i != j {
				term = term * (eval_point - points[j]) * InvertOrZero::invert_or_zero(points[i] - points[j]);
			}
		}
		*output += term;
	}
} 