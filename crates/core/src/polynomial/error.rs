// Copyright 2023-2025 Irreducible Inc.

use std::ops::Range;

use binius_field::Error as FieldError;

#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error("the query must have size {expected}, and istead has {actual}")]
	IncorrectQuerySize { expected: usize, actual: usize },
	#[error("all polynomials in mixed composition should have {expected} vars")]
	IncorrectArityInMixedComposition { expected: usize },
	#[error("array of inner composition evaluations is of incorrect length")]
	IncorrectInnerEvalsLength,
	#[error("the polynomial is expected to have {expected} variables, and instead has {actual}")]
	IncorrectNumberOfVariables { expected: usize, actual: usize },
	#[error("block size must between 1 and {n_vars} (inclusive)")]
	InvalidBlockSize { n_vars: usize },
	#[error("shift offset must be between 1 and {max_shift_offset} inclusive, got {shift_offset}")]
	InvalidShiftOffset {
		max_shift_offset: usize,
		shift_offset: usize,
	},
	#[error("invalid tower height: {actual}. tower height must be 0, 3, 4, 5, 6, or 7")]
	InvalidTowerHeight { actual: usize },
	#[error("indices provided to IndexComposition constructor do not match number of variables")]
	IndexCompositionIndicesOutOfBounds,
	#[error("mixed polynomial was not provided")]
	MixedMultilinearNotFound,
	#[error("MultilinearComposite constructed with incorrect arguments: {0}")]
	MultilinearCompositeValidation(String),
	#[error("argument {arg} must be in the range {range:?}")]
	ArgumentRangeError { arg: String, range: Range<usize> },
	#[error("{0}")]
	FieldError(#[from] FieldError),
	#[error(
		"not enough field elements to fill a single packed field element ({length} / {packed_width})"
	)]
	PackedFieldNotFilled { length: usize, packed_width: usize },
	#[error(
		"one of the defining inputs to the ring switching eq-indicator function has length {actual}, rather than {expected}"
	)]
	RingSwitchWrongLength { expected: usize, actual: usize },
	#[error("{0}")]
	MathError(#[from] binius_math::Error),
	#[error("{0}")]
	HalError(#[from] binius_hal::Error),
}
