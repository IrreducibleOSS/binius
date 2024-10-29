// Copyright 2023-2024 Irreducible Inc.

//! [Reed–Solomon] codes over binary fields.
//!
//! The Reed–Solomon code admits an efficient encoding algorithm over binary fields due to [LCH14].
//! The additive NTT encoding algorithm encodes messages interpreted as the coefficients of a
//! polynomial in a non-standard, novel polynomial basis and the codewords are the polynomial
//! evaluations over a linear subspace of the field. See the [binius_ntt] crate for more details.
//!
//! [Reed–Solomon]: <https://en.wikipedia.org/wiki/Reed%E2%80%93Solomon_error_correction>
//! [LCH14]: <https://arxiv.org/abs/1404.3458>

use crate::linear_code::LinearCode;
use binius_field::{BinaryField, PackedField};
use binius_ntt::{AdditiveNTT, DynamicDispatchNTT, Error, NTTOptions, ThreadingSettings};
use binius_utils::bail;
use getset::CopyGetters;
use rayon::prelude::*;
use std::marker::PhantomData;

#[derive(Debug, CopyGetters)]
pub struct ReedSolomonCode<P>
where
	P: PackedField,
	P::Scalar: BinaryField,
{
	ntt: DynamicDispatchNTT<P::Scalar>,
	log_dimension: usize,
	#[getset(get_copy = "pub")]
	log_inv_rate: usize,
	multithreaded: bool,
	_p_marker: PhantomData<P>,
}

impl<P> ReedSolomonCode<P>
where
	P: PackedField<Scalar: BinaryField>,
{
	pub fn new(
		log_dimension: usize,
		log_inv_rate: usize,
		ntt_options: NTTOptions,
	) -> Result<Self, Error> {
		// Since we split work between log_inv_rate threads, we need to decrease the number of threads per each NTT transformation.
		let ntt_log_threads = ntt_options
			.thread_settings
			.log_threads_count()
			.saturating_sub(log_inv_rate);
		let ntt = DynamicDispatchNTT::new(
			log_dimension + log_inv_rate,
			NTTOptions {
				thread_settings: ThreadingSettings::ExplicitThreadsCount {
					log_threads: ntt_log_threads,
				},
				..ntt_options
			},
		)?;

		let multithreaded =
			!matches!(ntt_options.thread_settings, ThreadingSettings::SingleThreaded);

		Ok(Self {
			ntt,
			log_dimension,
			log_inv_rate,
			multithreaded,
			_p_marker: PhantomData,
		})
	}

	pub fn get_ntt(&self) -> &impl AdditiveNTT<P> {
		&self.ntt
	}

	pub fn log_dim(&self) -> usize {
		self.log_dimension
	}

	pub fn log_len(&self) -> usize {
		self.log_dimension + self.log_inv_rate
	}
}

impl<P, F> LinearCode for ReedSolomonCode<P>
where
	P: PackedField<Scalar = F>,
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
		let _scope = tracing::debug_span!(
			"Reed–Solomon encode",
			log_len = self.log_len(),
			log_batch_size = log_batch_size,
			symbol_bits = F::N_BITS,
		)
		.entered();
		if (code.len() << log_batch_size) < self.len() {
			bail!(Error::BufferTooSmall {
				log_code_len: self.len(),
			});
		}
		if self.dim() % P::WIDTH != 0 {
			bail!(Error::PackingWidthMustDivideDimension);
		}

		let msgs_len = (self.dim() / P::WIDTH) << log_batch_size;
		for i in 1..(1 << self.log_inv_rate) {
			code.copy_within(0..msgs_len, i * msgs_len);
		}

		if self.multithreaded {
			(0..(1 << self.log_inv_rate))
				.into_par_iter()
				.zip(code.par_chunks_exact_mut(msgs_len))
				.try_for_each(|(i, data)| self.ntt.forward_transform(data, i, log_batch_size))
		} else {
			(0..(1 << self.log_inv_rate))
				.zip(code.chunks_exact_mut(msgs_len))
				.try_for_each(|(i, data)| self.ntt.forward_transform(data, i, log_batch_size))
		}
	}
}
