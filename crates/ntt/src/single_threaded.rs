// Copyright 2024 Ulvetanna Inc.

use super::{additive_ntt::AdditiveNTT, error::Error, twiddle::TwiddleAccess};
use crate::twiddle::{expand_subspace_evals, OnTheFlyTwiddleAccess, PrecomputedTwiddleAccess};
use binius_field::{BinaryField, PackedField, PackedFieldIndexable};
use std::marker::PhantomData;

/// Implementation of `AdditiveNTT` that performs the computation single-threaded.
#[derive(Debug)]
pub struct SingleThreadedNTT<F: BinaryField, TA: TwiddleAccess<F> = OnTheFlyTwiddleAccess<F>> {
	s_evals: Vec<TA>,
	_marker: PhantomData<F>,
}

impl<F: BinaryField> SingleThreadedNTT<F> {
	/// Default constructor constructs an NTT over the canonical subspace for the field using
	/// on-the-fly computed twiddle factors.
	pub fn new(log_domain_size: usize) -> Result<Self, Error> {
		Self::with_domain_field::<F>(log_domain_size)
	}

	/// Constructs an NTT over an isomorphic subspace for the given domain field using on-the-fly
	/// computed twiddle factors.
	pub fn with_domain_field<DomainField: BinaryField + Into<F>>(
		log_domain_size: usize,
	) -> Result<Self, Error> {
		let twiddle_access = OnTheFlyTwiddleAccess::generate::<DomainField>(log_domain_size)?;
		Ok(Self::with_twiddle_access(twiddle_access))
	}

	pub fn precompute_twiddles(&self) -> SingleThreadedNTT<F, PrecomputedTwiddleAccess<F>> {
		SingleThreadedNTT::with_twiddle_access(expand_subspace_evals(&self.s_evals))
	}
}

impl<F: BinaryField, TA: TwiddleAccess<F>> SingleThreadedNTT<F, TA> {
	pub fn with_twiddle_access(twiddle_access: Vec<TA>) -> Self {
		Self {
			s_evals: twiddle_access,
			_marker: PhantomData,
		}
	}

	/// Base-2 logarithm of the size of the NTT domain.
	pub fn log_domain_size(&self) -> usize {
		self.s_evals.len()
	}

	/// Get the normalized subspace polynomial evaluation $\hat{W}_i(\beta_j)$.
	///
	/// ## Preconditions
	///
	/// * `i` must be less than `self.log_domain_size()`
	/// * `j` must be less than `self.log_domain_size() - i`
	pub fn get_subspace_eval(&self, i: usize, j: usize) -> F {
		self.s_evals[i].get(j)
	}
}

impl<F: BinaryField, TA: TwiddleAccess<F>> SingleThreadedNTT<F, TA> {
	pub fn twiddles(&self) -> &[TA] {
		&self.s_evals
	}
}

impl<F, TA: TwiddleAccess<F>, P> AdditiveNTT<P> for SingleThreadedNTT<F, TA>
where
	F: BinaryField,
	TA: TwiddleAccess<F>,
	P: PackedFieldIndexable<Scalar = F>,
{
	fn log_domain_size(&self) -> usize {
		self.log_domain_size()
	}

	fn get_subspace_eval(&self, i: usize, j: usize) -> F {
		self.get_subspace_eval(i, j)
	}

	fn forward_transform(
		&self,
		data: &mut [P],
		coset: u32,
		log_batch_size: usize,
	) -> Result<(), Error> {
		forward_transform(self.log_domain_size(), &self.s_evals, data, coset, log_batch_size)
	}

	fn inverse_transform(
		&self,
		data: &mut [P],
		coset: u32,
		log_batch_size: usize,
	) -> Result<(), Error> {
		inverse_transform(self.log_domain_size(), &self.s_evals, data, coset, log_batch_size)
	}
}

#[derive(Debug)]
pub struct NTTParams {
	pub log_n: usize,
	pub log_w: usize,
}

pub fn forward_transform<F: BinaryField, P: PackedFieldIndexable<Scalar = F>>(
	log_domain_size: usize,
	s_evals: &[impl TwiddleAccess<F>],
	data: &mut [P],
	coset: u32,
	log_batch_size: usize,
) -> Result<(), Error> {
	match data.len() {
		0 => return Ok(()),
		1 => {
			return match P::WIDTH {
				1 => Ok(()),
				_ => forward_transform(
					log_domain_size,
					s_evals,
					P::unpack_scalars_mut(data),
					coset,
					log_batch_size,
				),
			};
		}
		_ => {}
	};

	let log_b = log_batch_size;

	let NTTParams { log_n, log_w } =
		check_batch_transform_inputs(log_domain_size, data, coset, log_b)?;

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

	for i in (0..cutoff).rev() {
		let coset_twiddle = s_evals[i].coset(log_domain_size - log_n, coset as usize);

		// A block is a block of butterfly units that all have the same twiddle factor. Since we
		// are below the cutoff round, the block length is less than the packing width, and
		// therefore each packed multiplication is with a non-uniform twiddle. Since the subspace
		// polynomials are linear, we can calculate an additive factor that can be added to the
		// packed twiddles for all packed butterfly units.
		let log_block_len = i + log_b;
		let block_twiddle = calculate_twiddle::<P>(
			s_evals[i].coset(log_domain_size - 1 - cutoff, 0),
			log_block_len,
		);

		for j in 0..1 << (log_n - 1 - cutoff) {
			let twiddle = P::broadcast(coset_twiddle.get(j << (cutoff - i))) + block_twiddle;
			let (mut u, mut v) = data[j << 1].interleave(data[j << 1 | 1], log_block_len);
			u += v * twiddle;
			v += u;
			(data[j << 1], data[j << 1 | 1]) = u.interleave(v, log_block_len);
		}
	}

	Ok(())
}

fn inverse_transform<F: BinaryField, P: PackedFieldIndexable<Scalar = F>>(
	log_domain_size: usize,
	s_evals: &[impl TwiddleAccess<F>],
	data: &mut [P],
	coset: u32,
	log_batch_size: usize,
) -> Result<(), Error> {
	match data.len() {
		0 => return Ok(()),
		1 => {
			return match P::WIDTH {
				1 => Ok(()),
				_ => inverse_transform(
					log_domain_size,
					s_evals,
					P::unpack_scalars_mut(data),
					coset,
					log_batch_size,
				),
			};
		}
		_ => {}
	};

	let log_b = log_batch_size;

	let NTTParams { log_n, log_w } =
		check_batch_transform_inputs(log_domain_size, data, coset, log_b)?;

	// Cutoff is the stage of the NTT where each the butterfly units are contained within
	// packed base field elements.
	let cutoff = log_w.saturating_sub(log_b);

	#[allow(clippy::needless_range_loop)]
	for i in 0..cutoff {
		let coset_twiddle = s_evals[i].coset(log_domain_size - log_n, coset as usize);

		// A block is a block of butterfly units that all have the same twiddle factor. Since we
		// are below the cutoff round, the block length is less than the packing width, and
		// therefore each packed multiplication is with a non-uniform twiddle. Since the subspace
		// polynomials are linear, we can calculate an additive factor that can be added to the
		// packed twiddles for all packed butterfly units.
		let log_block_len = i + log_b;
		let block_twiddle = calculate_twiddle::<P>(
			s_evals[i].coset(log_domain_size - 1 - cutoff, 0),
			log_block_len,
		);

		for j in 0..1 << (log_n - 1 - cutoff) {
			let twiddle = P::broadcast(coset_twiddle.get(j << (cutoff - i))) + block_twiddle;
			let (mut u, mut v) = data[j << 1].interleave(data[j << 1 | 1], log_block_len);
			v += u;
			u += v * twiddle;
			(data[j << 1], data[j << 1 | 1]) = u.interleave(v, log_block_len);
		}
	}

	#[allow(clippy::needless_range_loop)]
	for i in cutoff..log_n {
		let coset_twiddle = s_evals[i].coset(log_domain_size - log_n, coset as usize);

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

pub fn check_batch_transform_inputs<PB: PackedField>(
	log_domain_size: usize,
	data: &[PB],
	coset: u32,
	log_batch_size: usize,
) -> Result<NTTParams, Error> {
	if !data.len().is_power_of_two() {
		return Err(Error::PowerOfTwoLengthRequired);
	}
	if !PB::WIDTH.is_power_of_two() {
		return Err(Error::PackingWidthMustDivideDimension);
	}

	let n = (data.len() * PB::WIDTH) >> log_batch_size;
	if n == 0 {
		return Err(Error::BatchTooLarge);
	}

	let log_n = n.trailing_zeros() as usize;
	let log_w = PB::WIDTH.trailing_zeros() as usize;

	let coset_bits = 32 - coset.leading_zeros() as usize;
	if log_n + coset_bits > log_domain_size {
		return Err(Error::DomainTooSmall {
			log_required_domain_size: log_n + coset_bits,
		});
	}

	Ok(NTTParams { log_n, log_w })
}

#[inline]
fn calculate_twiddle<P>(s_evals: impl TwiddleAccess<P::Scalar>, log_block_len: usize) -> P
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
	use super::*;
	use crate::reference::{forward_transform_simple, inverse_transform_simple};
	use assert_matches::assert_matches;
	use binius_field::{
		arch::packed_32::PackedBinaryField1x32b,
		packed_binary_field::{PackedBinaryField16x8b, PackedBinaryField4x32b},
		AESTowerField8b, BinaryField32b, BinaryField8b, ExtensionField, Field,
		PackedBinaryField8x16b, PackedExtension,
	};
	use rand::{rngs::StdRng, thread_rng, SeedableRng};
	use std::{array, iter::repeat_with};

	trait SimpleAdditiveNTT<F: BinaryField> {
		fn forward_transform_simple<FF>(&self, data: &mut [FF], coset: u32) -> Result<(), Error>
		where
			FF: ExtensionField<F>;

		fn inverse_transform_simple<FF>(&self, data: &mut [FF], coset: u32) -> Result<(), Error>
		where
			FF: ExtensionField<F>;
	}

	impl<F: BinaryField, TA: TwiddleAccess<F>> SimpleAdditiveNTT<F> for SingleThreadedNTT<F, TA> {
		fn forward_transform_simple<FF>(&self, data: &mut [FF], coset: u32) -> Result<(), Error>
		where
			FF: ExtensionField<F>,
		{
			forward_transform_simple(self.log_domain_size(), &self.s_evals, data, coset)
		}

		fn inverse_transform_simple<FF>(&self, data: &mut [FF], coset: u32) -> Result<(), Error>
		where
			FF: ExtensionField<F>,
		{
			inverse_transform_simple(self.log_domain_size(), &self.s_evals, data, coset)
		}
	}

	#[test]
	fn test_additive_ntt_fails_with_field_too_small() {
		assert_matches!(
			SingleThreadedNTT::<BinaryField8b>::new(10),
			Err(Error::FieldTooSmall {
				log_domain_size: 10
			})
		);
	}

	#[test]
	fn test_additive_ntt_transform() {
		let ntt = SingleThreadedNTT::<BinaryField8b>::new(8).unwrap();
		let data = (0..1 << 6)
			.map(|i| BinaryField8b::new(i as u8))
			.collect::<Vec<_>>();

		let mut result = data.clone();
		for coset in 0..4 {
			ntt.inverse_transform_simple(&mut result, coset).unwrap();
			ntt.forward_transform_simple(&mut result, coset).unwrap();
			assert_eq!(result, data);
		}
	}

	#[test]
	fn test_additive_ntt_with_precompute_matches() {
		let ntt = SingleThreadedNTT::<BinaryField8b>::new(8).unwrap();
		let ntt_with_precompute = ntt.precompute_twiddles();
		let data = (0..1 << 6)
			.map(|i| BinaryField8b::new(i as u8))
			.collect::<Vec<_>>();

		let mut result1 = data.clone();
		let mut result2 = data;
		for coset in 0..4 {
			ntt.inverse_transform_simple(&mut result1, coset).unwrap();
			ntt_with_precompute
				.inverse_transform_simple(&mut result2, coset)
				.unwrap();
			assert_eq!(result1, result2);

			ntt.forward_transform_simple(&mut result1, coset).unwrap();
			ntt_with_precompute
				.forward_transform_simple(&mut result2, coset)
				.unwrap();
			assert_eq!(result1, result2);
		}
	}

	#[test]
	fn test_additive_ntt_with_transform() {
		let ntt_binary8b = <SingleThreadedNTT<BinaryField8b>>::new(8).unwrap();
		let ntt_aes8b = <SingleThreadedNTT<AESTowerField8b>>::new(8).unwrap();
		let ntt_aes8b_change_of_bases =
			<SingleThreadedNTT<AESTowerField8b, _>>::with_domain_field::<BinaryField8b>(8).unwrap();

		let mut rng = thread_rng();

		let data: [BinaryField8b; 64] =
			array::from_fn(|_| <BinaryField8b as Field>::random(&mut rng));
		let data_as_aes: [AESTowerField8b; 64] = array::from_fn(|i| data[i].into());

		for coset in 0..4 {
			let mut result_bin = data;
			let mut result_aes = data_as_aes;
			let mut result_aes_cob = data_as_aes;
			ntt_binary8b
				.forward_transform_simple(&mut result_bin, coset)
				.unwrap();
			ntt_aes8b
				.forward_transform_simple(&mut result_aes, coset)
				.unwrap();
			ntt_aes8b_change_of_bases
				.forward_transform_simple(&mut result_aes_cob, coset)
				.unwrap();

			let result_bin_to_aes = result_bin.map(AESTowerField8b::from);

			assert_eq!(result_bin_to_aes, result_aes_cob);
			assert_ne!(result_bin_to_aes, result_aes);

			ntt_binary8b
				.inverse_transform_simple(&mut result_bin, coset)
				.unwrap();
			ntt_aes8b
				.inverse_transform_simple(&mut result_aes, coset)
				.unwrap();
			ntt_aes8b_change_of_bases
				.inverse_transform_simple(&mut result_aes_cob, coset)
				.unwrap();

			let result_bin_to_aes = result_bin.map(AESTowerField8b::from);

			assert_eq!(result_bin_to_aes, result_aes_cob);
		}
	}

	#[test]
	fn test_additive_ntt_transform_over_larger_field() {
		let mut rng = StdRng::seed_from_u64(0);

		let ntt = <SingleThreadedNTT<BinaryField8b, _>>::new(8).unwrap();
		let data = repeat_with(|| <BinaryField32b as Field>::random(&mut rng))
			.take(1 << 6)
			.collect::<Vec<_>>();

		let mut result = data.clone();
		for coset in 0..4 {
			ntt.inverse_transform_simple(&mut result, coset).unwrap();
			ntt.forward_transform_simple(&mut result, coset).unwrap();
			assert_eq!(result, data);
		}
	}

	#[test]
	fn test_packed_ntt_on_scalars() {
		type Packed = PackedBinaryField16x8b;

		let mut rng = StdRng::seed_from_u64(0);

		let ntt = <SingleThreadedNTT<BinaryField8b, _>>::new(8).unwrap();
		let mut data = repeat_with(|| Packed::random(&mut rng))
			.take(1 << 2)
			.collect::<Vec<_>>();
		let mut data_copy = data.clone();

		ntt.inverse_transform_simple(Packed::unpack_scalars_mut(&mut data), 2)
			.unwrap();
		AdditiveNTT::<Packed>::inverse_transform_ext(&ntt, &mut data_copy, 2).unwrap();
		assert_eq!(data, data_copy);

		ntt.forward_transform_simple(Packed::unpack_scalars_mut(&mut data), 3)
			.unwrap();
		AdditiveNTT::<Packed>::forward_transform_ext(&ntt, &mut data_copy, 3).unwrap();
		assert_eq!(data, data_copy);
	}

	fn check_packed_ntt_on_scalars_with_message_size_one<P, S, NTT>(ntt: NTT)
	where
		S: Field,
		P: PackedFieldIndexable<Scalar = S> + PackedExtension<S, PackedSubfield = P>,
		P::Scalar: BinaryField + ExtensionField<S>,
		NTT: AdditiveNTT<P> + SimpleAdditiveNTT<P::Scalar>,
	{
		let mut rng = StdRng::seed_from_u64(0);

		let mut data = repeat_with(|| <P as PackedField>::random(&mut rng))
			.take(1 << 0)
			.collect::<Vec<_>>();
		let mut data_copy = data.clone();

		ntt.inverse_transform_simple(P::unpack_scalars_mut(&mut data), 2)
			.unwrap();
		AdditiveNTT::<P>::inverse_transform_ext::<P>(&ntt, &mut data_copy, 2).unwrap();
		assert_eq!(data, data_copy);

		ntt.forward_transform_simple(P::unpack_scalars_mut(&mut data), 3)
			.unwrap();
		AdditiveNTT::<P>::forward_transform_ext::<P>(&ntt, &mut data_copy, 3).unwrap();
		assert_eq!(data, data_copy);
	}

	#[test]
	fn test_packed_ntt_with_otf_compute_on_scalars_with_message_size_one() {
		check_packed_ntt_on_scalars_with_message_size_one::<PackedBinaryField4x32b, _, _>(
			SingleThreadedNTT::new(8).unwrap(),
		);
		check_packed_ntt_on_scalars_with_message_size_one::<PackedBinaryField8x16b, _, _>(
			SingleThreadedNTT::new(8).unwrap(),
		);
		check_packed_ntt_on_scalars_with_message_size_one::<PackedBinaryField16x8b, _, _>(
			SingleThreadedNTT::new(8).unwrap(),
		);
		check_packed_ntt_on_scalars_with_message_size_one::<PackedBinaryField1x32b, _, _>(
			SingleThreadedNTT::new(8).unwrap(),
		);
	}

	#[test]
	fn test_packed_ntt_with_precompute_on_scalars_with_message_size_one() {
		check_packed_ntt_on_scalars_with_message_size_one::<PackedBinaryField4x32b, _, _>(
			SingleThreadedNTT::new(8).unwrap().precompute_twiddles(),
		);
		check_packed_ntt_on_scalars_with_message_size_one::<PackedBinaryField8x16b, _, _>(
			SingleThreadedNTT::new(8).unwrap().precompute_twiddles(),
		);
		check_packed_ntt_on_scalars_with_message_size_one::<PackedBinaryField16x8b, _, _>(
			SingleThreadedNTT::new(8).unwrap().precompute_twiddles(),
		);
		check_packed_ntt_on_scalars_with_message_size_one::<PackedBinaryField1x32b, _, _>(
			SingleThreadedNTT::new(8).unwrap().precompute_twiddles(),
		);
	}

	#[test]
	fn test_packed_ntt_over_larger_field() {
		type Packed = PackedBinaryField4x32b;

		let mut rng = StdRng::seed_from_u64(0);

		let ntt = <SingleThreadedNTT<BinaryField8b>>::new(8).unwrap();
		let ntt_with_precompute = <SingleThreadedNTT<BinaryField8b>>::new(8)
			.unwrap()
			.precompute_twiddles();
		let mut data = repeat_with(|| Packed::random(&mut rng))
			.take(1 << 4)
			.collect::<Vec<_>>();

		let mut data_copy = data.clone();
		let mut data_copy_2 = data.clone();

		ntt.inverse_transform_simple(PackedFieldIndexable::unpack_scalars_mut(&mut data), 2)
			.unwrap();
		AdditiveNTT::<PackedBinaryField16x8b>::inverse_transform_ext(&ntt, &mut data_copy, 2)
			.unwrap();
		AdditiveNTT::<PackedBinaryField16x8b>::inverse_transform_ext(
			&ntt_with_precompute,
			&mut data_copy_2,
			2,
		)
		.unwrap();
		assert_eq!(data, data_copy);
		assert_eq!(data, data_copy_2);

		ntt.forward_transform_simple(PackedFieldIndexable::unpack_scalars_mut(&mut data), 3)
			.unwrap();
		AdditiveNTT::<PackedBinaryField16x8b>::forward_transform_ext(&ntt, &mut data_copy, 3)
			.unwrap();
		AdditiveNTT::<PackedBinaryField16x8b>::forward_transform_ext(
			&ntt_with_precompute,
			&mut data_copy_2,
			3,
		)
		.unwrap();
		assert_eq!(data, data_copy);
		assert_eq!(data, data_copy_2);
	}

	// TODO: Write test that compares polynomial evaluation via additive NTT with naive Lagrange
	// polynomial interpolation. A randomized test should suffice for larger NTT sizes.
}
