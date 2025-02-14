// Copyright 2024-2025 Irreducible Inc.

use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
	#[error("table by name `{0}` not found")]
	MissingTable(String),
	#[error("all tables_stuffs must be dropped before calling `build`")]
	RemainingRcRefs,
	#[error("the number of slices you gave us for table {0} isn't as expected, i expected {1}")]
	IncorrectSliceCount(usize, usize),
	#[error("error making multilinear extension: {0}")]
	MultilinearExtensionError(#[from] binius_math::Error),
	#[error("error updating extension index: {0}")]
	UpdateExtensionIndexError(#[from] binius_core::witness::Error),
	#[error(
		"underliers for table {0} column {1} wasn't given a power of 2 length, had length {2}"
	)]
	NonPowerOfTwoLengthUnderliers(String, String, usize),
}
