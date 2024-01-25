// Copyright 2023 Ulvetanna Inc.

use crate::{
	field::{BinaryField, ExtensionField, PackedExtensionField, PackedField},
	linear_code::{LinearCode, LinearCodeWithExtensionEncoding},
	reed_solomon::additive_ntt::AdditiveNTT,
};
use std::marker::PhantomData;

use super::{additive_ntt::AdditiveNTTWithOTFCompute, error::Error};

pub struct ReedSolomonCode<P>
where
	P: PackedField,
	P::Scalar: BinaryField,
{
	// TODO: Genericize whether to use AdditiveNTT or AdditiveNTTWithPrecompute
	ntt: AdditiveNTTWithOTFCompute<P::Scalar>,
	log_dimension: usize,
	log_inv_rate: usize,
	n_test_queries: usize,
	_p_marker: PhantomData<P>,
}

impl<P> ReedSolomonCode<P>
where
	P: PackedField,
	P::Scalar: BinaryField,
{
	pub fn new(
		log_dimension: usize,
		log_inv_rate: usize,
		n_test_queries: usize,
	) -> Result<Self, Error> {
		let ntt = AdditiveNTTWithOTFCompute::new(log_dimension + log_inv_rate)?;
		Ok(Self {
			ntt,
			log_dimension,
			log_inv_rate,
			n_test_queries,
			_p_marker: PhantomData,
		})
	}
}

impl<P, F> LinearCode for ReedSolomonCode<P>
where
	P: PackedField<Scalar = F> + PackedExtensionField<F>,
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

	fn n_test_queries(&self) -> usize {
		self.n_test_queries
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
		for i in 0..(1 << self.log_inv_rate) {
			self.ntt.forward_transform(
				&mut code[i * msgs_len..(i + 1) * msgs_len],
				i as u32,
				log_batch_size,
			)?;
		}
		Ok(())
	}
}

impl<P, F> LinearCodeWithExtensionEncoding for ReedSolomonCode<P>
where
	P: PackedField<Scalar = F> + PackedExtensionField<F>,
	F: BinaryField,
{
	fn encode_extension_inplace<PE>(&self, code: &mut [PE]) -> Result<(), Self::EncodeError>
	where
		PE: PackedExtensionField<Self::P>,
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
		for i in 0..(1 << self.log_inv_rate) {
			self.ntt
				.forward_transform_ext(&mut code[i * dim..(i + 1) * dim], i as u32)?;
		}
		Ok(())
	}
}
