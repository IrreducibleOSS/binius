// Copyright 2023 Ulvetanna Inc.

use binius_field::Error as FieldError;
use std::ops::Range;

#[derive(Debug, Clone, thiserror::Error)]
pub enum Error {
	#[error("the query must have size {expected}")]
	IncorrectQuerySize { expected: usize },
	#[error("all polynomials in mixed composition should have {expected} vars")]
	IncorrectArityInMixedComposition { expected: usize },
	#[error("array of inner composition evaluations is of incorrect length")]
	IncorrectInnerEvalsLength,
	#[error("the output polynomial must have size {expected}")]
	IncorrectOutputPolynomialSize { expected: usize },
	#[error("the polynomial is expected to have {expected} variables, and instead has {actual}")]
	IncorrectNumberOfVariables { expected: usize, actual: usize },
	#[error("block size must between 1 and {n_vars} (inclusive)")]
	InvalidBlockSize { n_vars: usize },
	#[error("shift offset must be between 1 and {max_shift_offset} inclusive, got {shift_offset}")]
	InvalidShiftOffset {
		max_shift_offset: usize,
		shift_offset: usize,
	},
	#[error("argument length must be a power of two")]
	PowerOfTwoLengthRequired,
	#[error("cannot operate on polynomials with more than 31 variables")]
	TooManyVariables,
	#[error("indexed point on hypercube is out of range: index={index}")]
	HypercubeIndexOutOfRange { index: usize },
	#[error("indices provided to IndexComposition constructor do not match number of variables")]
	IndexCompositionIndicesOutOfBounds,
	#[error("MultilinearQuery is full, cannot update further. Has {max_query_vars} variables")]
	MultilinearQueryFull { max_query_vars: usize },
	#[error("mixed polynomial was not provided")]
	MixedMultilinearNotFound,
	#[error("MultilinearComposite constructed with incorrect arguments: {0}")]
	MultilinearCompositeValidation(String),
	// TODO: Change range to bounds: Box<dyn RangeBounds + Send + Sync + 'static>
	#[error("argument {arg} must be in the range {range:?}")]
	ArgumentRangeError { arg: String, range: Range<usize> },
	#[error("{0}")]
	FieldError(#[from] FieldError),
	#[error("not enough field elements to fill a single packed field element ({length} / {packed_width})")]
	PackedFieldNotFilled { length: usize, packed_width: usize },
	#[error("{0}")]
	MathError(#[from] binius_math::Error),
}
