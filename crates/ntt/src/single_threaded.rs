// Copyright 2024-2025 Irreducible Inc.

use std::{cmp, marker::PhantomData};

use binius_field::{BinaryField, PackedField, TowerField};
use binius_math::BinarySubspace;

use super::{additive_ntt::AdditiveNTT, error::Error, twiddle::TwiddleAccess};
use crate::twiddle::{expand_subspace_evals, OnTheFlyTwiddleAccess, PrecomputedTwiddleAccess};

/// Implementation of `AdditiveNTT` that performs the computation single-threaded.
#[derive(Debug)]
pub struct SingleThreadedNTT<F: BinaryField, TA: TwiddleAccess<F> = OnTheFlyTwiddleAccess<F>> {
	// TODO: Figure out how to make this private, it should not be `pub(super)`.
	pub(super) s_evals: Vec<TA>,
	_marker: PhantomData<F>,
}

impl<F: BinaryField> SingleThreadedNTT<F> {
	/// Default constructor constructs an NTT over the canonical subspace for the field using
	/// on-the-fly computed twiddle factors.
	pub fn new(log_domain_size: usize) -> Result<Self, Error> {
		let subspace = BinarySubspace::with_dim(log_domain_size)?;
		let twiddle_access = OnTheFlyTwiddleAccess::generate(&subspace)?;
		Ok(Self::with_twiddle_access(twiddle_access))
	}

	/// Constructs an NTT over an isomorphic subspace for the given domain field using on-the-fly
	/// computed twiddle factors.
	pub fn with_domain_field<FDomain>(log_domain_size: usize) -> Result<Self, Error>
	where
		FDomain: BinaryField,
		F: From<FDomain>,
	{
		let subspace = BinarySubspace::<FDomain>::with_dim(log_domain_size)?.isomorphic();
		let twiddle_access = OnTheFlyTwiddleAccess::generate(&subspace)?;
		Ok(Self::with_twiddle_access(twiddle_access))
	}

	pub fn precompute_twiddles(&self) -> SingleThreadedNTT<F, PrecomputedTwiddleAccess<F>> {
		SingleThreadedNTT::with_twiddle_access(expand_subspace_evals(&self.s_evals))
	}
}

impl<F: TowerField> SingleThreadedNTT<F> {
	/// A specialization of [`with_domain_field`](Self::with_domain_field) to the canonical tower field.
	pub fn with_canonical_field(log_domain_size: usize) -> Result<Self, Error> {
		Self::with_domain_field::<F::Canonical>(log_domain_size)
	}
}

impl<F: BinaryField, TA: TwiddleAccess<F>> SingleThreadedNTT<F, TA> {
	const fn with_twiddle_access(twiddle_access: Vec<TA>) -> Self {
		Self {
			s_evals: twiddle_access,
			_marker: PhantomData,
		}
	}
}

impl<F: BinaryField, TA: TwiddleAccess<F>> SingleThreadedNTT<F, TA> {
	pub fn twiddles(&self) -> &[TA] {
		&self.s_evals
	}
}

impl<F, TA> AdditiveNTT<F> for SingleThreadedNTT<F, TA>
where
	F: BinaryField,
	TA: TwiddleAccess<F>,
{
	fn log_domain_size(&self) -> usize {
		self.s_evals.len()
	}

	fn subspace(&self, i: usize) -> BinarySubspace<F> {
		let (subspace, shift) = self.s_evals[i].affine_subspace();
		debug_assert_eq!(shift, F::ZERO, "s_evals subspaces must be linear by construction");
		subspace
	}

	fn get_subspace_eval(&self, i: usize, j: usize) -> F {
		self.s_evals[i].get(j)
	}

	fn forward_transform<P: PackedField<Scalar = F>>(
		&self,
		data: &mut [P],
		coset: u32,
		log_batch_size: usize,
		log_n: usize,
	) -> Result<(), Error> {
		forward_transform(self.log_domain_size(), &self.s_evals, data, coset, log_batch_size, log_n)
	}

	fn inverse_transform<P: PackedField<Scalar = F>>(
		&self,
		data: &mut [P],
		coset: u32,
		log_batch_size: usize,
		log_n: usize,
	) -> Result<(), Error> {
		inverse_transform(self.log_domain_size(), &self.s_evals, data, coset, log_batch_size, log_n)
	}
}

pub fn forward_transform<F: BinaryField, P: PackedField<Scalar = F>>(
	log_domain_size: usize,
	s_evals: &[impl TwiddleAccess<F>],
	data: &mut [P],
	coset: u32,
	log_batch_size: usize,
	log_n: usize,
) -> Result<(), Error> {
	match data.len() {
		0 => return Ok(()),
		1 => {
			return match P::LOG_WIDTH {
				0 => Ok(()),
				_ => {
					// Special case when there is only one packed element: since we cannot
					// interleave with another packed element, the code below will panic when there
					// is only one.
					//
					// Handle the case of one packed element by batch transforming the original
					// data with dummy data and extracting the transformed result.
					let mut buffer = [data[0], P::zero()];

					forward_transform(
						log_domain_size,
						s_evals,
						&mut buffer,
						coset,
						log_batch_size,
						log_n,
					)?;

					data[0] = buffer[0];

					Ok(())
				}
			};
		}
		_ => {}
	};

	let log_b = log_batch_size;

	let log_w = P::LOG_WIDTH;

	check_batch_transform_inputs_and_params(log_domain_size, data, coset, log_batch_size, log_n)?;

	// Cutoff is the stage of the NTT where each the butterfly units are contained within
	// packed base field elements.
	let cutoff = log_w.saturating_sub(log_b);

	for i in (cutoff..log_n).rev() {
		let coset_twiddle = s_evals[i].coset(log_domain_size - log_n, coset as usize);

		for j in 0..1 << (log_n - 1 - i) {
			let twiddle = P::broadcast(coset_twiddle.get(j));
			for k in 0..1 << (i + log_b - log_w) {
				let idx0 = j << (i + log_b - log_w + 1) | k;
				let idx1 = idx0 | 1 << (i + log_b - log_w);
				data[idx0] += data[idx1] * twiddle;
				data[idx1] += data[idx0];
			}
		}
	}

	for i in (0..cmp::min(cutoff, log_n)).rev() {
		let coset_twiddle = s_evals[i].coset(log_domain_size - log_n, coset as usize);

		// A block is a block of butterfly units that all have the same twiddle factor. Since we
		// are below the cutoff round, the block length is less than the packing width, and
		// therefore each packed multiplication is with a non-uniform twiddle. Since the subspace
		// polynomials are linear, we can calculate an additive factor that can be added to the
		// packed twiddles for all packed butterfly units.
		let log_block_len = i + log_b;
		let block_twiddle = calculate_twiddle::<P>(
			&s_evals[i].coset(log_domain_size - 1 - cutoff, 0),
			log_block_len,
		);

		for j in 0..data.len() / 2 {
			let twiddle = P::broadcast(coset_twiddle.get(j << (cutoff - i))) + block_twiddle;
			let (mut u, mut v) = data[j << 1].interleave(data[j << 1 | 1], log_block_len);
			u += v * twiddle;
			v += u;
			(data[j << 1], data[j << 1 | 1]) = u.interleave(v, log_block_len);
		}
	}

	Ok(())
}

pub fn inverse_transform<F: BinaryField, P: PackedField<Scalar = F>>(
	log_domain_size: usize,
	s_evals: &[impl TwiddleAccess<F>],
	data: &mut [P],
	coset: u32,
	log_batch_size: usize,
	log_n: usize,
) -> Result<(), Error> {
	match data.len() {
		0 => return Ok(()),
		1 => {
			return match P::LOG_WIDTH {
				0 => Ok(()),
				_ => {
					// Special case when there is only one packed element: since we cannot
					// interleave with another packed element, the code below will panic when there
					// is only one.
					//
					// Handle the case of one packed element by batch transforming the original
					// data with dummy data and extracting the transformed result.
					let mut buffer = [data[0], P::zero()];

					inverse_transform(
						log_domain_size,
						s_evals,
						&mut buffer,
						coset,
						log_batch_size,
						log_n,
					)?;

					data[0] = buffer[0];
					Ok(())
				}
			};
		}
		_ => {}
	};

	let log_w = P::LOG_WIDTH;

	let log_b = log_batch_size;

	check_batch_transform_inputs_and_params(log_domain_size, data, coset, log_batch_size, log_n)?;

	// Cutoff is the stage of the NTT where each the butterfly units are contained within
	// packed base field elements.
	let cutoff = log_w.saturating_sub(log_b);

	for (i, s_eval) in s_evals.iter().enumerate().take(cmp::min(cutoff, log_n)) {
		let coset_twiddle = s_eval.coset(log_domain_size - log_n, coset as usize);

		// A block is a block of butterfly units that all have the same twiddle factor. Since we
		// are below the cutoff round, the block length is less than the packing width, and
		// therefore each packed multiplication is with a non-uniform twiddle. Since the subspace
		// polynomials are linear, we can calculate an additive factor that can be added to the
		// packed twiddles for all packed butterfly units.
		let log_block_len = i + log_b;
		let block_twiddle =
			calculate_twiddle::<P>(&s_eval.coset(log_domain_size - 1 - cutoff, 0), log_block_len);

		for j in 0..data.len() / 2 {
			let twiddle = P::broadcast(coset_twiddle.get(j << (cutoff - i))) + block_twiddle;
			let (mut u, mut v) = data[j << 1].interleave(data[j << 1 | 1], log_block_len);
			v += u;
			u += v * twiddle;
			(data[j << 1], data[j << 1 | 1]) = u.interleave(v, log_block_len);
		}
	}

	for (i, s_eval) in s_evals
		.iter()
		.enumerate()
		.skip(cutoff)
		.take(log_n)
		.skip(cutoff)
	{
		let coset_twiddle = s_eval.coset(log_domain_size - log_n, coset as usize);

		for j in 0..1 << (log_n - 1 - i) {
			let twiddle = P::broadcast(coset_twiddle.get(j));
			for k in 0..1 << (i + log_b - log_w) {
				let idx0 = j << (i + log_b - log_w + 1) | k;
				let idx1 = idx0 | 1 << (i + log_b - log_w);
				data[idx1] += data[idx0];
				data[idx0] += data[idx1] * twiddle;
			}
		}
	}

	Ok(())
}

pub fn check_batch_transform_inputs_and_params<PB: PackedField>(
	log_domain_size: usize,
	data: &[PB],
	coset: u32,
	log_batch_size: usize,
	log_n: usize,
) -> Result<(), Error> {
	if !data.len().is_power_of_two() {
		return Err(Error::PowerOfTwoLengthRequired);
	}
	if !PB::WIDTH.is_power_of_two() {
		return Err(Error::PackingWidthMustDivideDimension);
	}

	let full_sized_n = (data.len() * PB::WIDTH) >> log_batch_size;

	// Verify that our log_n exactly matches the data length, except when we are NTT-ing one packed field
	if (1 << log_n != full_sized_n && data.len() > 2) || (1 << log_n > full_sized_n) {
		return Err(Error::BatchTooLarge);
	}

	let coset_bits = 32 - coset.leading_zeros() as usize;

	// The domain size should be at least large enough to represent the given coset;
	// on the lower end, there is a fallback for data.len() == 1 which reduces to
	// a forward/inverse NTT on the [PB; 2], which demands log_domain_size of
	// at least min(PB::LOG_WIDTH + 1 - log_batch_size, 0).
	// Not enforcing this bound makes some twiddle values unavailable.
	let log_required_domain_size =
		(log_n + coset_bits).max((PB::LOG_WIDTH + 1).saturating_sub(log_batch_size));
	if log_required_domain_size > log_domain_size {
		return Err(Error::DomainTooSmall {
			log_required_domain_size,
		});
	}

	Ok(())
}

#[inline]
fn calculate_twiddle<P>(s_evals: &impl TwiddleAccess<P::Scalar>, log_block_len: usize) -> P
where
	P: PackedField<Scalar: BinaryField>,
{
	let log_blocks_count = P::LOG_WIDTH - log_block_len - 1;

	let mut twiddle = P::default();
	for k in 0..1 << log_blocks_count {
		let (subblock_twiddle_0, subblock_twiddle_1) = s_evals.get_pair(log_blocks_count, k);
		let idx0 = k << (log_block_len + 1);
		let idx1 = idx0 | 1 << log_block_len;

		for l in 0..1 << log_block_len {
			twiddle.set(idx0 | l, subblock_twiddle_0);
			twiddle.set(idx1 | l, subblock_twiddle_1);
		}
	}
	twiddle
}

#[cfg(test)]
mod tests {
	use assert_matches::assert_matches;
	use binius_field::{
		BinaryField16b, BinaryField8b, PackedBinaryField8x16b, PackedFieldIndexable,
	};
	use binius_math::Error as MathError;
	use rand::{rngs::StdRng, SeedableRng};

	use super::*;

	#[test]
	fn test_additive_ntt_fails_with_field_too_small() {
		assert_matches!(
			SingleThreadedNTT::<BinaryField8b>::new(10),
			Err(Error::MathError(MathError::DomainSizeTooLarge))
		);
	}

	#[test]
	fn test_subspace_size_agrees_with_domain_size() {
		let ntt = SingleThreadedNTT::<BinaryField16b>::new(10).expect("msg");
		assert_eq!(ntt.subspace(0).dim(), 10);
		assert_eq!(ntt.subspace(9).dim(), 1);
	}

	#[test]
	fn one_packed_field_forward() {
		let s = SingleThreadedNTT::<BinaryField16b>::new(10).expect("msg");
		let mut packed = [PackedBinaryField8x16b::random(StdRng::from_entropy())];

		let mut packed_copy = packed;

		let unpacked = PackedBinaryField8x16b::unpack_scalars_mut(&mut packed_copy);

		let _ = s.forward_transform(&mut packed, 3, 0, 3);

		let _ = s.forward_transform(unpacked, 3, 0, 3);

		for (i, unpacked_item) in unpacked.iter().enumerate().take(8) {
			assert_eq!(packed[0].get(i), *unpacked_item);
		}
	}

	#[test]
	fn one_packed_field_inverse() {
		let s = SingleThreadedNTT::<BinaryField16b>::new(10).expect("msg");
		let mut packed = [PackedBinaryField8x16b::random(StdRng::from_entropy())];

		let mut packed_copy = packed;

		let unpacked = PackedBinaryField8x16b::unpack_scalars_mut(&mut packed_copy);

		let _ = s.inverse_transform(&mut packed, 3, 0, 3);

		let _ = s.inverse_transform(unpacked, 3, 0, 3);

		for (i, unpacked_item) in unpacked.iter().enumerate().take(8) {
			assert_eq!(packed[0].get(i), *unpacked_item);
		}
	}

	#[test]
	fn smaller_embedded_batch_forward() {
		let s = SingleThreadedNTT::<BinaryField16b>::new(10).expect("msg");
		let mut packed = [PackedBinaryField8x16b::random(StdRng::from_entropy())];

		let mut packed_copy = packed;

		let unpacked = &mut PackedBinaryField8x16b::unpack_scalars_mut(&mut packed_copy)[0..4];

		let _ = forward_transform(s.log_domain_size(), &s.s_evals, &mut packed, 3, 0, 2);

		let _ = s.forward_transform(unpacked, 3, 0, 2);

		for (i, unpacked_item) in unpacked.iter().enumerate().take(4) {
			assert_eq!(packed[0].get(i), *unpacked_item);
		}
	}

	#[test]
	fn smaller_embedded_batch_inverse() {
		let s = SingleThreadedNTT::<BinaryField16b>::new(10).expect("msg");
		let mut packed = [PackedBinaryField8x16b::random(StdRng::from_entropy())];

		let mut packed_copy = packed;

		let unpacked = &mut PackedBinaryField8x16b::unpack_scalars_mut(&mut packed_copy)[0..4];

		let _ = inverse_transform(s.log_domain_size(), &s.s_evals, &mut packed, 3, 0, 2);

		let _ = s.inverse_transform(unpacked, 3, 0, 2);

		for (i, unpacked_item) in unpacked.iter().enumerate().take(4) {
			assert_eq!(packed[0].get(i), *unpacked_item);
		}
	}

	// TODO: Write test that compares polynomial evaluation via additive NTT with naive Lagrange
	// polynomial interpolation. A randomized test should suffice for larger NTT sizes.
}
