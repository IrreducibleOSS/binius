// Copyright 2023 Ulvetanna Inc.

//! [Reed–Solomon] codes over binary fields.
//!
//! The Reed–Solomon code admits an efficient encoding algorithm over binary fields due to [LCH14].
//! The additive NTT encoding algorithm encodes messages interpreted as the coefficients of a
//! polynomial in a non-standard, novel polynomial basis and the codewords are the polynomial
//! evaluations over a linear subspace of the field. See the [binius_ntt] crate for more details.
//!
//! [Reed–Solomon]: <https://en.wikipedia.org/wiki/Reed%E2%80%93Solomon_error_correction>
//! [LCH14]: <https://arxiv.org/abs/1404.3458>

use crate::linear_code::{LinearCode, LinearCodeWithExtensionEncoding};
use binius_field::{
	BinaryField, ExtensionField, PackedExtension, PackedField, PackedFieldIndexable,
	RepackedExtension,
};
use binius_ntt::{AdditiveNTT, AdditiveNTTWithOTFCompute, Error};
use rayon::prelude::*;
use std::marker::PhantomData;

#[derive(Debug)]
pub struct ReedSolomonCode<P>
where
	P: PackedField,
	P::Scalar: BinaryField,
{
	// TODO: Genericize whether to use AdditiveNTT or AdditiveNTTWithPrecompute
	ntt: AdditiveNTTWithOTFCompute<P::Scalar>,
	log_dimension: usize,
	log_inv_rate: usize,
	_p_marker: PhantomData<P>,
}

impl<P> ReedSolomonCode<P>
where
	P: PackedField,
	P::Scalar: BinaryField,
{
	pub fn new(log_dimension: usize, log_inv_rate: usize) -> Result<Self, Error> {
		let ntt = AdditiveNTTWithOTFCompute::new(log_dimension + log_inv_rate)?;
		Ok(Self {
			ntt,
			log_dimension,
			log_inv_rate,
			_p_marker: PhantomData,
		})
	}
}

impl<P, F> LinearCode for ReedSolomonCode<P>
where
	P: PackedField<Scalar = F> + PackedExtension<F> + PackedFieldIndexable,
	F: BinaryField,
{
	type P = P;
	type EncodeError = Error;

	fn len(&self) -> usize {
		1 << (self.log_dimension + self.log_inv_rate)
	}

	fn dim_bits(&self) -> usize {
		self.log_dimension
	}

	fn min_dist(&self) -> usize {
		self.len() - self.dim() + 1
	}

	fn inv_rate(&self) -> usize {
		1 << self.log_inv_rate
	}

	fn encode_batch_inplace(
		&self,
		code: &mut [Self::P],
		log_batch_size: usize,
	) -> Result<(), Self::EncodeError> {
		if (code.len() << log_batch_size) < self.len() {
			return Err(Error::BufferTooSmall {
				log_code_len: self.len(),
			});
		}
		if self.dim() % P::WIDTH != 0 {
			return Err(Error::PackingWidthMustDivideDimension);
		}

		let msgs_len = (self.dim() / P::WIDTH) << log_batch_size;
		for i in 1..(1 << self.log_inv_rate) {
			code.copy_within(0..msgs_len, i * msgs_len);
		}

		(0..(1 << self.log_inv_rate))
			.into_par_iter()
			.zip(code.par_chunks_exact_mut(msgs_len))
			.try_for_each(|(i, data)| self.ntt.forward_transform(data, i, log_batch_size))
	}
}

impl<P, F> LinearCodeWithExtensionEncoding for ReedSolomonCode<P>
where
	P: PackedField<Scalar = F> + PackedExtension<F> + PackedFieldIndexable,
	F: BinaryField,
{
	fn encode_extension_inplace<PE>(&self, code: &mut [PE]) -> Result<(), Self::EncodeError>
	where
		PE: RepackedExtension<P>,
		PE::Scalar: ExtensionField<<Self::P as PackedField>::Scalar>,
	{
		if code.len() * PE::WIDTH < self.len() {
			return Err(Error::BufferTooSmall {
				log_code_len: self.len(),
			});
		}
		if self.dim() % PE::WIDTH != 0 {
			return Err(Error::PackingWidthMustDivideDimension);
		}

		let dim = self.dim() / PE::WIDTH;
		for i in 1..(1 << self.log_inv_rate) {
			code.copy_within(0..dim, i * dim);
		}
		(0..(1 << self.log_inv_rate))
			.into_par_iter()
			.zip(code.par_chunks_exact_mut(dim))
			.try_for_each(|(i, data)| self.ntt.forward_transform_ext(data, i))
	}
}
