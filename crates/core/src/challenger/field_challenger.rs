// Copyright 2024 Irreducible Inc.

use binius_field::{
	BinaryField, BinaryField1b, ExtensionField, Field, PackedExtension, PackedFieldIndexable,
};
use p3_challenger::{CanObserve, CanSample, CanSampleBits};
use std::{cmp, slice};

/// A Fiat-Shamir challenger that can observe and sample field elements.
///
/// This challenger type can observe and sample elements of extension fields of some field `F`.
/// When the field `F` is a [`BinaryField`], the challenger can sample bits uniformly.
#[derive(Debug, Clone)]
pub struct FieldChallenger<F, Impl: FieldChallengerHelper<F>> {
	/// The buffer index of the next field element that will be sampled.
	index: usize,
	/// The bit index into the current field element being sampled.
	bit_index: usize,
	/// Buffered elements that may be returned when the verifier samples.
	buffer: Box<[F]>,
	helper: Impl,
}

// TODO: Construct with domain separation
impl<F, Impl> Default for FieldChallenger<F, Impl>
where
	F: Clone + Default,
	Impl: FieldChallengerHelper<F> + Default,
{
	fn default() -> Self {
		assert_ne!(Impl::RATE, 0);

		Self {
			index: Impl::RATE,
			bit_index: 0,
			buffer: vec![F::default(); Impl::RATE].into_boxed_slice(),
			helper: Impl::default(),
		}
	}
}

impl<F, Impl> FieldChallenger<F, Impl>
where
	F: Clone,
	Impl: FieldChallengerHelper<F>,
{
	fn sample_elems(&mut self, n: usize) -> Vec<F> {
		if self.bit_index != 0 {
			self.index += 1;
			self.bit_index = 0;
		}

		let mut elems = Vec::with_capacity(n);
		while elems.len() < n {
			if self.index == Impl::RATE {
				self.helper.sample(&mut self.buffer);
				self.index = 0;
			}
			let bases_remaining = n - elems.len();
			let buffer_remaining = Impl::RATE - self.index;
			let incr = cmp::min(bases_remaining, buffer_remaining);
			elems.extend_from_slice(&self.buffer[self.index..self.index + incr]);
			self.index += incr;
		}
		elems
	}

	fn observe_elems(&mut self, input: &[F])
	where
		F: BinaryField,
	{
		// Allow to avoid discrepancies when calls sample less than the buffer.len() times.
		if self.index != 0 && self.index != Impl::RATE {
			let mut index = self.index;
			let mut index_1b = Vec::new();

			while index > 0 {
				if index & 1 == 1 {
					index_1b.push(BinaryField1b::ONE)
				} else {
					index_1b.push(BinaryField1b::ZERO)
				}
				index >>= 1;
			}

			let index_fields = index_1b
				.chunks(<F as ExtensionField<BinaryField1b>>::DEGREE)
				.map(|base_elems| F::from_bases(base_elems).unwrap_or(F::ZERO))
				.collect::<Vec<_>>();

			self.helper.observe(&index_fields)
		}

		// Any buffered output is now invalid.
		self.index = Impl::RATE;
		self.bit_index = 0;

		self.helper.observe(input);
	}
}

/// A helper trait for implementing `FieldChallenger`.
///
/// The helper trait provides methods for sampling new challenges and observing data.
pub trait FieldChallengerHelper<F> {
	/// The number of elements returned by a single call to [`Self::sample`].
	const RATE: usize;

	/// Samples `RATE` new challenges.
	///
	/// ## Preconditions
	///
	/// - Length of output buffer must be RATE
	fn sample(&mut self, output: &mut [F]);

	/// Observes a slice of field elements sent by the prover.
	fn observe(&mut self, input: &[F]);
}

impl<F, PE, Impl> CanObserve<PE> for FieldChallenger<F, Impl>
where
	F: BinaryField,
	PE: PackedExtension<F, PackedSubfield: PackedFieldIndexable>,
	PE::Scalar: ExtensionField<F>,
	Impl: FieldChallengerHelper<F>,
{
	fn observe(&mut self, value: PE) {
		self.observe_slice(slice::from_ref(&value))
	}

	fn observe_slice(&mut self, values: &[PE]) {
		self.observe_elems(PackedFieldIndexable::unpack_scalars(PE::cast_bases(values)))
	}
}

impl<F, FE, Impl> CanSample<FE> for FieldChallenger<F, Impl>
where
	F: Field,
	FE: ExtensionField<F>,
	Impl: FieldChallengerHelper<F>,
{
	fn sample(&mut self) -> FE {
		let bases = self.sample_elems(FE::DEGREE);
		FE::from_bases(&bases).expect("the size of bases is FE::DEGREE")
	}
}

impl<F, Impl> CanSampleBits<usize> for FieldChallenger<F, Impl>
where
	F: BinaryField,
	Impl: FieldChallengerHelper<F>,
{
	fn sample_bits(&mut self, bits: usize) -> usize {
		let bits = bits.min(usize::BITS as usize);

		let mut sampled = 0;
		let mut bits_sampled = 0;
		while bits_sampled < bits {
			if self.bit_index == F::N_BITS {
				self.index += 1;
				self.bit_index = 0;
			}
			if self.index == Impl::RATE {
				self.helper.sample(&mut self.buffer);
				self.index = 0;
			}

			let output_remaining = bits - bits_sampled;
			let buffer_remaining = (Impl::RATE - self.index) * F::N_BITS - self.bit_index;
			let incr = cmp::min(output_remaining, buffer_remaining);

			let packed_1b = self.buffer[self.index..]
				.iter()
				.flat_map(|elem| elem.iter_bases());
			for bit in packed_1b.skip(self.bit_index).take(output_remaining) {
				if bit == BinaryField1b::ONE {
					sampled |= 1 << bits_sampled;
				}
				bits_sampled += 1;
			}

			let bit_index_incr = self.bit_index + incr;
			self.bit_index = bit_index_incr % F::N_BITS;
			self.index += bit_index_incr / F::N_BITS;
		}
		sampled
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use binius_field::BinaryField8b;

	#[derive(Default)]
	struct MockFieldChallengerHelper {
		index: usize,
	}

	impl FieldChallengerHelper<BinaryField8b> for MockFieldChallengerHelper {
		const RATE: usize = 32;

		fn sample(&mut self, output: &mut [BinaryField8b]) {
			for elem in output {
				*elem = BinaryField8b::new(self.index as u8);
				self.index += 1;
			}
		}

		fn observe(&mut self, input: &[BinaryField8b]) {
			for _elem in input {
				self.index += 1;
			}
		}
	}

	#[test]
	fn test_sample() {
		let mut challenger = <FieldChallenger<BinaryField8b, MockFieldChallengerHelper>>::default();

		// The first 3 sample calls will sample 32 elements from the helper.
		assert_eq!(CanSample::<BinaryField8b>::sample(&mut challenger), BinaryField8b::new(0));
		assert_eq!(CanSample::<BinaryField8b>::sample(&mut challenger), BinaryField8b::new(1));
		assert_eq!(CanSample::<BinaryField8b>::sample(&mut challenger), BinaryField8b::new(2));

		// bump the index by 2, becasue self.index ne 0 and ne RATE
		challenger.observe(BinaryField8b::ZERO);
		// bump the index by 1
		challenger.observe(BinaryField8b::ZERO);

		for i in 0..33 {
			assert_eq!(
				CanSample::<BinaryField8b>::sample(&mut challenger),
				BinaryField8b::new(35 + i)
			);
		}
	}
}
