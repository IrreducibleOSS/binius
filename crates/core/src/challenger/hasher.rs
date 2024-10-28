// Copyright 2024 Irreducible Inc.

use super::field_challenger::{FieldChallenger, FieldChallengerHelper};
use binius_field::{Field, PackedField, PackedFieldIndexable};
use binius_hash::Hasher;
use std::{marker::PhantomData, slice};

#[derive(Debug, Clone)]
struct HashChallenger<F, H: Hasher<F>> {
	hasher: H,
	_marker: PhantomData<F>,
}

impl<F, H: Hasher<F>> Default for HashChallenger<F, H> {
	fn default() -> Self {
		Self {
			hasher: H::new(),
			_marker: PhantomData,
		}
	}
}

impl<F, H> FieldChallengerHelper<F> for HashChallenger<F, H>
where
	F: Field,
	H: Hasher<F>,
	H::Digest: PackedFieldIndexable<Scalar = F>,
{
	const RATE: usize = H::Digest::WIDTH;

	fn sample(&mut self, output: &mut [F]) {
		let digest = self.hasher.finalize_reset();
		let elems = H::Digest::unpack_scalars(slice::from_ref(&digest));

		// Chain values for the next sample call.
		self.hasher.update(elems);

		output.copy_from_slice(elems);
	}

	fn observe(&mut self, input: &[F]) {
		self.hasher.update(input);
	}
}

/// Construct a Fiat-Shamir challenger from a normal, collision-resistant hash function.
pub fn new<F, H>() -> FieldChallenger<F, impl FieldChallengerHelper<F> + Clone>
where
	F: Field,
	H: Hasher<F> + Clone,
	H::Digest: PackedFieldIndexable<Scalar = F>,
{
	FieldChallenger::<F, HashChallenger<F, H>>::default()
}

#[cfg(test)]
mod tests {
	use super::*;
	use binius_field::{BinaryField128b, BinaryField64b, BinaryField8b};
	use binius_hash::GroestlHasher;
	use p3_challenger::CanSample;

	#[test]
	fn test_groestl_challenger_can_sample_ext_field() {
		let mut challenger = new::<BinaryField8b, GroestlHasher<BinaryField8b>>();
		let _: BinaryField64b = challenger.sample();
		let _: BinaryField128b = challenger.sample();
		// This sample triggers a flush
		let _: BinaryField128b = challenger.sample();
	}
}
