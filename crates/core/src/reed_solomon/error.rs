// Copyright 2025 Irreducible Inc.

#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error("the evaluation domain of the code does not match the subspace of the NTT encoder")]
	EncoderSubspaceMismatch,
	#[error("the dimension of the evaluation domain of the code does not match the parameters")]
	SubspaceDimensionMismatch,
	#[error("math error: {0}")]
	Math(#[from] binius_math::Error),
	#[error("NTT error: {0}")]
	NTT(#[from] binius_ntt::Error),
}
