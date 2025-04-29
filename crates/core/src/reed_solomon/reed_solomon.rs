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

use binius_field::{BinaryField, ExtensionField, PackedExtension, PackedField};
use binius_math::BinarySubspace;
use binius_maybe_rayon::prelude::*;
use binius_ntt::{AdditiveNTT, Error, NTTShape};
use binius_utils::bail;
use getset::{CopyGetters, Getters};

#[derive(Debug, Getters, CopyGetters)]
pub struct ReedSolomonCode<F: BinaryField> {
	#[get = "pub"]
	subspace: BinarySubspace<F>,
	log_dimension: usize,
	#[get_copy = "pub"]
	log_inv_rate: usize,
}

impl<F: BinaryField> ReedSolomonCode<F> {
	pub fn new(log_dimension: usize, log_inv_rate: usize) -> Result<Self, Error> {
		Self::with_subspace(
			BinarySubspace::with_dim(log_dimension + log_inv_rate)?,
			log_dimension,
			log_inv_rate,
		)
	}

	pub fn with_subspace(
		subspace: BinarySubspace<F>,
		log_dimension: usize,
		log_inv_rate: usize,
	) -> Result<Self, Error> {
		if subspace.dim() != log_dimension + log_inv_rate {
			return Err(Error::DomainTooSmall {
				log_required_domain_size: log_dimension + log_inv_rate,
			});
		}
		Ok(Self {
			subspace,
			log_dimension,
			log_inv_rate,
		})
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
	fn encode_batch_inplace<P: PackedField<Scalar = F>, NTT: AdditiveNTT<F> + Sync>(
		&self,
		ntt: &NTT,
		code: &mut [P],
		log_batch_size: usize,
	) -> Result<(), Error> {
		let _scope = tracing::trace_span!(
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

		let shape = NTTShape {
			log_x: log_batch_size,
			log_y: self.log_dim(),
			..Default::default()
		};
		(0..(1 << self.log_inv_rate))
			.into_par_iter()
			.zip(code.par_chunks_exact_mut(msgs_len))
			.try_for_each(|(i, data)| ntt.forward_transform(data, shape, i))
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
	pub fn encode_ext_batch_inplace<PE: PackedExtension<F>, NTT: AdditiveNTT<F> + Sync>(
		&self,
		ntt: &NTT,
		code: &mut [PE],
		log_batch_size: usize,
	) -> Result<(), Error> {
		self.encode_batch_inplace(
			ntt,
			PE::cast_bases_mut(code),
			log_batch_size + PE::Scalar::LOG_DEGREE,
		)
	}
}
