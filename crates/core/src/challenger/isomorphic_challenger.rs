// Copyright 2024 Ulvetanna Inc.

use binius_field::{packed::iter_packed_slice, BinaryField, ExtensionField, PackedExtension};
use p3_challenger::{CanObserve, CanSample, CanSampleBits};
use std::{iter::repeat_with, marker::PhantomData, slice};

/// A wrapper over the challenger api that [`CanSample`] and [`CanObserve`] over the [`BinaryField`]
/// `F2` where the internal challenger needs to sample and observe over the BinaryField `F1` which
/// must be isomorphic to `F2`
#[derive(Debug, Clone)]
pub struct IsomorphicChallenger<F1: BinaryField, Challenger: Clone, F2: BinaryField> {
	challenger: Challenger,
	_marker: PhantomData<(F1, F2)>,
}

impl<F1, Challenger, F2> IsomorphicChallenger<F1, Challenger, F2>
where
	F1: BinaryField + From<F2> + Into<F2>,
	F2: BinaryField,
	Challenger: Clone,
{
	pub fn new(challenger: Challenger) -> Self {
		Self {
			challenger,
			_marker: PhantomData,
		}
	}
}

impl<F1, Challenger, F2, PF2> CanObserve<PF2> for IsomorphicChallenger<F1, Challenger, F2>
where
	F1: BinaryField + From<F2> + Into<F2>,
	F2: BinaryField,
	PF2: PackedExtension<F2>,
	PF2::Scalar: ExtensionField<F2>,
	Challenger: CanObserve<F1> + Clone,
{
	fn observe(&mut self, value: PF2) {
		self.observe_slice(slice::from_ref(&value));
	}

	fn observe_slice(&mut self, values: &[PF2]) {
		let repacked_size =
			values.len() * PF2::WIDTH * PF2::Scalar::DEGREE * F2::N_BITS / F1::N_BITS;
		let mut values_converted: Vec<F1> = Vec::with_capacity(repacked_size);
		for e in iter_packed_slice(PF2::cast_bases(values)) {
			values_converted.push(e.into());
		}
		self.challenger.observe_slice(&values_converted);
	}
}

impl<F1, Challenger, F2, F2E> CanSample<F2E> for IsomorphicChallenger<F1, Challenger, F2>
where
	F1: BinaryField + From<F2> + Into<F2>,
	F2: BinaryField,
	F2E: ExtensionField<F2>,
	Challenger: CanSample<F1> + Clone,
{
	fn sample(&mut self) -> F2E {
		let bases = repeat_with(|| self.challenger.sample().into())
			.take(F2E::DEGREE)
			.collect::<Vec<F2>>();

		F2E::from_bases(&bases).expect("the size of bases is F2E::DEGREE")
	}
}

impl<F1, Challenger, F2> CanSampleBits<usize> for IsomorphicChallenger<F1, Challenger, F2>
where
	F1: BinaryField,
	F2: BinaryField,
	Challenger: CanSampleBits<usize> + Clone,
{
	fn sample_bits(&mut self, bits: usize) -> usize {
		self.challenger.sample_bits(bits)
	}
}

#[cfg(test)]
mod test {
	use super::*;
	use crate::challenger::{new_duplex_challenger, new_hasher_challenger};
	use binius_field::{
		AESTowerField32b, AESTowerField8b, BinaryField128b, BinaryField128bPolyval, BinaryField32b,
		BinaryField8b, Field,
	};
	use binius_hash::{Groestl256, GroestlHasher, Vision32bPermutation};
	use rand::thread_rng;
	use std::array;

	#[test]
	fn test_isomorphic_hasher_challenger_sampling() {
		let challenger_over_aes =
			new_hasher_challenger::<AESTowerField8b, Groestl256<AESTowerField8b, _>>();
		let mut challenger_over_bin =
			new_hasher_challenger::<BinaryField8b, GroestlHasher<BinaryField8b>>();
		let mut isomorphic_challenger_over_bin: IsomorphicChallenger<
			AESTowerField32b,
			_,
			BinaryField32b,
		> = IsomorphicChallenger::new(challenger_over_aes);

		for _ in 0..20 {
			let as_b32_1: BinaryField32b = isomorphic_challenger_over_bin.sample();
			let as_b32_2: BinaryField32b = challenger_over_bin.sample();
			assert_eq!(as_b32_1, as_b32_2);
		}

		let mut challenger_over_bin =
			new_hasher_challenger::<BinaryField8b, GroestlHasher<BinaryField8b>>();
		let mut isomorphic_challenger_over_bin: IsomorphicChallenger<
			BinaryField128b,
			_,
			BinaryField128bPolyval,
		> = IsomorphicChallenger::new(challenger_over_bin.clone());

		for _ in 0..20 {
			let as_b128: BinaryField128b = challenger_over_bin.sample();
			let as_polyval128: BinaryField128bPolyval = isomorphic_challenger_over_bin.sample();

			assert_eq!(as_b128, as_polyval128.into());
		}
	}

	#[test]
	fn test_isomorphic_duplex_challenger_sampling_and_observing() {
		let mut challenger_over_bin =
			new_duplex_challenger::<BinaryField32b, Vision32bPermutation, 20, 24>();
		let mut isomorphic_challenger_over_aes: IsomorphicChallenger<
			BinaryField32b,
			_,
			AESTowerField32b,
		> = IsomorphicChallenger::new(challenger_over_bin.clone());

		const N: usize = 20;
		let mut rng = thread_rng();
		let bin32_observations: [BinaryField32b; N] =
			array::from_fn(|_| <BinaryField32b as Field>::random(&mut rng));
		let aes32_observations: [AESTowerField32b; N] =
			array::from_fn(|i| bin32_observations[i].into());

		for i in 0..N {
			isomorphic_challenger_over_aes.observe(aes32_observations[i]);
			challenger_over_bin.observe(bin32_observations[i]);
			let as_b32: BinaryField32b = challenger_over_bin.sample();
			let as_aes32: AESTowerField32b = isomorphic_challenger_over_aes.sample();
			assert_eq!(as_b32, as_aes32.into());
		}
	}

	#[test]
	fn test_isomorphic_hasher_challenger_sampling_and_observing() {
		let mut challenger_over_bin =
			new_hasher_challenger::<BinaryField8b, GroestlHasher<BinaryField8b>>();
		let mut isomorphic_challenger_over_bin: IsomorphicChallenger<
			BinaryField128b,
			_,
			BinaryField128bPolyval,
		> = IsomorphicChallenger::new(challenger_over_bin.clone());

		const N: usize = 20;
		let mut rng = thread_rng();
		let observable: [BinaryField128b; N] =
			array::from_fn(|_| <BinaryField128b as Field>::random(&mut rng));
		let observable_polyval: [BinaryField128bPolyval; N] =
			array::from_fn(|i| observable[i].into());

		for i in 0..20 {
			challenger_over_bin.observe(observable[i]);
			isomorphic_challenger_over_bin.observe(observable_polyval[i]);
			let as_b128: BinaryField128b = challenger_over_bin.sample();
			let as_polyval128: BinaryField128bPolyval = isomorphic_challenger_over_bin.sample();

			assert_eq!(as_b128, as_polyval128.into());
		}

		challenger_over_bin.observe_slice(&observable);
		isomorphic_challenger_over_bin.observe_slice(&observable_polyval);
		let as_b128: BinaryField128b = challenger_over_bin.sample();
		let as_polyval128: BinaryField128bPolyval = isomorphic_challenger_over_bin.sample();
		assert_eq!(as_b128, as_polyval128.into());
	}
}
