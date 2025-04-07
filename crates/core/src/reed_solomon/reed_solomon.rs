// Copyright 2023-2025 Irreducible Inc.

//! [Reed–Solomon] codes over binary fields.
//!
//! The Reed–Solomon code admits an efficient encoding algorithm over binary fields due to [LCH14].
//! The additive NTT encoding algorithm encodes messages interpreted as the coefficients of a
//! polynomial in a non-standard, novel polynomial basis and the codewords are the polynomial
//! evaluations over a linear subspace of the field. See the [binius_ntt] crate for more details.
//!
//! [Reed–Solomon]: <https://en.wikipedia.org/wiki/Reed%E2%80%93Solomon_error_correction>
//! [LCH14]: <https://arxiv.org/abs/1404.3458>

use std::marker::PhantomData;

use binius_field::{BinaryField, ExtensionField, PackedField, RepackedExtension};
use binius_maybe_rayon::prelude::*;
use binius_ntt::{AdditiveNTT, DynamicDispatchNTT, Error, NTTOptions, ThreadingSettings};
use binius_utils::bail;
use getset::CopyGetters;
use tracing::instrument;

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
		ntt_options: &NTTOptions,
	) -> Result<Self, Error> {
		// Since we split work between log_inv_rate threads, we need to decrease the number of threads per each NTT transformation.
		let ntt_log_threads = ntt_options
			.thread_settings
			.log_threads_count()
			.saturating_sub(log_inv_rate);
		let ntt = DynamicDispatchNTT::new(
			log_dimension + log_inv_rate,
			&NTTOptions {
				thread_settings: ThreadingSettings::ExplicitThreadsCount {
					log_threads: ntt_log_threads,
				},
				precompute_twiddles: ntt_options.precompute_twiddles,
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

	pub const fn get_ntt(&self) -> &impl AdditiveNTT<P::Scalar> {
		&self.ntt
	}

	/// The dimension.
	pub const fn dim(&self) -> usize {
		1 << self.dim_bits()
	}

	pub const fn log_dim(&self) -> usize {
		self.log_dimension
	}

	pub const fn log_len(&self) -> usize {
		self.log_dimension + self.log_inv_rate
	}

	/// The block length.
	#[allow(clippy::len_without_is_empty)]
	pub const fn len(&self) -> usize {
		1 << (self.log_dimension + self.log_inv_rate)
	}

	/// The base-2 log of the dimension.
	const fn dim_bits(&self) -> usize {
		self.log_dimension
	}

	/// The reciprocal of the rate, ie. `self.len() / self.dim()`.
	pub const fn inv_rate(&self) -> usize {
		1 << self.log_inv_rate
	}

	/// Encode a batch of interleaved messages in-place in a provided buffer.
	///
	/// The message symbols are interleaved in the buffer, which improves the cache-efficiency of
	/// the encoding procedure. The interleaved codeword is stored in the buffer when the method
	/// completes.
	///
	/// ## Throws
	///
	/// * If the `code` buffer does not have capacity for `len() << log_batch_size` field
	///   elements.
	fn encode_batch_inplace(&self, code: &mut [P], log_batch_size: usize) -> Result<(), Error> {
		let _scope = tracing::trace_span!(
			"Reed–Solomon encode",
			log_len = self.log_len(),
			log_batch_size = log_batch_size,
			symbol_bits = P::Scalar::N_BITS,
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
				.try_for_each(|(i, data)| {
					self.ntt
						.forward_transform(data, i, log_batch_size, 0, self.log_dim())
				})
		} else {
			(0..(1 << self.log_inv_rate))
				.zip(code.chunks_exact_mut(msgs_len))
				.try_for_each(|(i, data)| {
					self.ntt
						.forward_transform(data, i, log_batch_size, 0, self.log_dim())
				})
		}
	}

	/// Encode a batch of interleaved messages of extension field elements in-place in a provided
	/// buffer.
	///
	/// A linear code can be naturally extended to a code over extension fields by encoding each
	/// dimension of the extension as a vector-space separately.
	///
	/// ## Preconditions
	///
	/// * `PE::Scalar::DEGREE` must be a power of two.
	///
	/// ## Throws
	///
	/// * If the `code` buffer does not have capacity for `len() << log_batch_size` field elements.
	#[instrument(skip_all, level = "debug")]
	pub fn encode_ext_batch_inplace<PE: RepackedExtension<P>>(
		&self,
		code: &mut [PE],
		log_batch_size: usize,
	) -> Result<(), Error> {
		self.encode_batch_inplace(PE::cast_bases_mut(code), log_batch_size + PE::Scalar::LOG_DEGREE)
	}
}
