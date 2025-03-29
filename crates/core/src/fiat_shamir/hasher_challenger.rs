// Copyright 2024-2025 Irreducible Inc.

use std::{cmp::min, mem};

use bytes::{buf::UninitSlice, Buf, BufMut};
use digest::{
	core_api::{Block, BlockSizeUser},
	Digest, FixedOutputReset, Output,
};

use super::Challenger;

/// Challenger type which implements `[Buf]` that has similar functionality as `[CanSample]`
#[derive(Debug, Default)]
pub struct HasherSampler<H: Digest> {
	sample_count: usize,
	index: usize,
	buffer: Output<H>,
	hasher: H,
}

/// Challenger type which implements `[BufMut]` that has similar functionality as `[CanObserve]`
#[derive(Debug, Default)]
pub struct HasherObserver<H: Digest + BlockSizeUser> {
	observation_count: usize,
	index: usize,
	buffer: Block<H>,
	hasher: H,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Interaction {
	Observe(usize),
	Sample(usize),
}

#[derive(Debug)]
pub enum ObserverOrSampler<H: Digest + BlockSizeUser> {
	Observer(HasherObserver<H>),
	Sampler(HasherSampler<H>),
}

impl<H: Digest + BlockSizeUser + Default> Default for ObserverOrSampler<H> {
	fn default() -> Self {
		Self::Sampler(HasherSampler::default())
	}
}

/// Challenger interface over hashes that implement `[Digest]` trait,
///
/// This challenger works over bytes instead of Field elements
#[derive(Debug)]
pub struct HasherChallenger<H: Digest + BlockSizeUser> {
	interactions: Vec<Interaction>,
	observer_or_sampler: ObserverOrSampler<H>,
}

impl<H> Default for HasherChallenger<H>
where
	H: Digest + BlockSizeUser + FixedOutputReset,
{
	fn default() -> Self {
		Self::new(H::digest([]))
	}
}

/// Initialize the `HasherChallenger` with an `initial_digest` that reflects both the statement and the constraint system,
/// e.g. had by hashing the hash of the statement and the hash of the serialized constraint system.
impl<H> HasherChallenger<H>
where
	H: Digest + BlockSizeUser,
{
	fn new(initial_digest: Output<H>) -> Self {
		let mut hasher = H::new();
		Digest::update(&mut hasher, &initial_digest);

		Self {
			interactions: Vec::new(),
			observer_or_sampler: ObserverOrSampler::Sampler(HasherSampler {
				sample_count: 0,
				hasher,
				index: 0,
				buffer: initial_digest,
			}),
		}
	}
}

impl<H: Digest + BlockSizeUser + FixedOutputReset + Default> Challenger for HasherChallenger<H> {
	/// This returns the inner challenger which implements `[BufMut]`
	fn observer(&mut self) -> &mut impl BufMut {
		let observer = match mem::take(&mut self.observer_or_sampler) {
			ObserverOrSampler::Observer(observer) => observer,
			ObserverOrSampler::Sampler(sampler) => {
				let (observer, sample_count) = sampler.into_observer();
				if sample_count > 0 {
					self.interactions.push(Interaction::Sample(sample_count));
				}
				observer
			}
		};
		self.observer_or_sampler = ObserverOrSampler::Observer(observer);
		match &mut self.observer_or_sampler {
			ObserverOrSampler::Observer(observer) => observer,
			_ => unreachable!(),
		}
	}

	/// This returns the inner challenger which implements `[Buf]`
	fn sampler(&mut self) -> &mut impl Buf {
		let sampler = match mem::take(&mut self.observer_or_sampler) {
			ObserverOrSampler::Observer(observer) => {
				let (sampler, observation_count) = observer.into_sampler();
				if observation_count > 0 {
					self.interactions
						.push(Interaction::Observe(observation_count));
				}
				sampler
			}
			ObserverOrSampler::Sampler(sampler) => sampler,
		};
		self.observer_or_sampler = ObserverOrSampler::Sampler(sampler);
		match &mut self.observer_or_sampler {
			ObserverOrSampler::Sampler(sampler) => sampler,
			_ => unreachable!(),
		}
	}

	fn finalize(mut self) -> Vec<Interaction> {
		match self.observer_or_sampler {
			ObserverOrSampler::Observer(observer) => {
				if observer.observation_count > 0 {
					self.interactions
						.push(Interaction::Observe(observer.observation_count))
				}
			}
			ObserverOrSampler::Sampler(sampler) => {
				if sampler.sample_count > 0 {
					self.interactions
						.push(Interaction::Sample(sampler.sample_count))
				}
			}
		}
		self.interactions
	}
}

impl<H> HasherSampler<H>
where
	H: Digest + Default + BlockSizeUser,
{
	fn into_observer(self) -> (HasherObserver<H>, usize) {
		let observer = HasherObserver {
			observation_count: 0,
			hasher: self.hasher,
			index: 0,
			buffer: Block::<H>::default(),
		};
		(observer, self.sample_count)
	}
}

impl<H> HasherSampler<H>
where
	H: Digest + FixedOutputReset,
{
	fn fill_buffer(&mut self) {
		let digest = self.hasher.finalize_reset();

		// feed forward to the empty state
		Digest::update(&mut self.hasher, &digest);

		self.buffer = digest;
		self.index = 0
	}
}

impl<H> HasherObserver<H>
where
	H: Digest + BlockSizeUser + Default,
{
	fn into_sampler(mut self) -> (HasherSampler<H>, usize) {
		self.flush();

		let sampler = HasherSampler {
			sample_count: 0,
			hasher: self.hasher,
			index: <H as Digest>::output_size(),
			buffer: Output::<H>::default(),
		};
		(sampler, self.observation_count)
	}
}

impl<H> HasherObserver<H>
where
	H: Digest + BlockSizeUser,
{
	fn flush(&mut self) {
		self.hasher.update(&self.buffer[..self.index]);
		self.index = 0
	}
}

impl<H> Buf for HasherSampler<H>
where
	H: Digest + FixedOutputReset + Default,
{
	fn remaining(&self) -> usize {
		usize::MAX
	}

	fn chunk(&self) -> &[u8] {
		&self.buffer[self.index..]
	}

	fn advance(&mut self, mut cnt: usize) {
		self.sample_count += cnt;

		// Must handle the case when `cnt` is 0
		if self.index == <H as Digest>::output_size() {
			self.fill_buffer();
		}

		while cnt > 0 {
			let remaining = min(<H as Digest>::output_size() - self.index, cnt);
			if remaining == 0 {
				self.fill_buffer();
				continue;
			}
			cnt -= remaining;
			self.index += remaining;
		}
	}
}

unsafe impl<H> BufMut for HasherObserver<H>
where
	H: Digest + BlockSizeUser,
{
	fn remaining_mut(&self) -> usize {
		usize::MAX
	}

	unsafe fn advance_mut(&mut self, mut cnt: usize) {
		self.observation_count += cnt;

		while cnt > 0 {
			let remaining = min(<H as BlockSizeUser>::block_size() - self.index, cnt);
			cnt -= remaining;
			self.index += remaining;
			if self.index == <H as BlockSizeUser>::block_size() {
				self.flush();
			}
		}
	}

	fn chunk_mut(&mut self) -> &mut UninitSlice {
		let buffer = &mut self.buffer[self.index..];
		buffer.into()
	}
}

#[cfg(test)]
mod tests {
	use binius_hash::groestl::Groestl256;
	use rand::{thread_rng, RngCore};

	use super::*;

	#[test]
	fn test_starting_sampler() {
		let mut hasher = Groestl256::default();
		let empty_string_digest = Groestl256::digest([]);
		Digest::update(&mut hasher, empty_string_digest);

		let mut challenger = HasherChallenger::<Groestl256>::default();

		// first sampling
		let mut out = [0u8; 8];
		challenger.sampler().copy_to_slice(&mut out);

		let first_hash_out = empty_string_digest;

		assert_eq!(first_hash_out[0..8], out);

		// second sampling
		let mut out = [0u8; 24];
		challenger.sampler().copy_to_slice(&mut out);
		assert_eq!(first_hash_out[8..], out);

		// first observing
		challenger.observer().put_slice(&[0x48, 0x55]);
		hasher.update([0x48, 0x55]);

		// third sampling
		let mut out_after_observe = [0u8; 2];
		challenger.sampler().copy_to_slice(&mut out_after_observe);

		let second_hash_out = hasher.finalize_reset();

		assert_eq!(out_after_observe, second_hash_out[..2]);

		assert_eq!(
			challenger.finalize(),
			vec![
				Interaction::Sample(32),
				Interaction::Observe(2),
				Interaction::Sample(2)
			]
		);
	}

	#[test]
	fn test_starting_observer() {
		let mut hasher = Groestl256::default();
		Digest::update(&mut hasher, Groestl256::digest([]));

		let mut challenger = HasherChallenger::<Groestl256>::default();

		// first observing
		let mut observable = [0u8; 1019];
		thread_rng().fill_bytes(&mut observable);
		challenger.observer().put_slice(&observable[..39]);
		challenger.observer().put_slice(&observable[39..300]);
		challenger.observer().put_slice(&observable[300..987]);
		challenger.observer().put_slice(&observable[987..]);
		hasher.update(observable);

		// first sampling
		let mut out = [0u8; 7];
		challenger.sampler().copy_to_slice(&mut out);

		let first_hash_out = hasher.finalize_reset();
		hasher.update(first_hash_out);

		assert_eq!(first_hash_out[..7], out);

		// second observing
		thread_rng().fill_bytes(&mut observable);
		challenger.observer().put_slice(&observable[..128]);

		hasher.update(&observable[..128]);

		// second sampling
		let mut out = [0u8; 32];
		challenger.sampler().copy_to_slice(&mut out);

		let second_hasher_out = hasher.finalize_reset();
		hasher.update(second_hasher_out);

		assert_eq!(second_hasher_out[..], out);

		// third observing
		challenger.observer().put_slice(&observable[128..]);
		hasher.update(&observable[128..]);

		// third sampling
		let mut out_again = [0u8; 7];
		challenger.sampler().copy_to_slice(&mut out_again);

		let final_hasher_out = hasher.finalize_reset();
		assert_eq!(final_hasher_out[..7], out_again);

		assert_eq!(
			challenger.finalize(),
			vec![
				Interaction::Observe(1019),
				Interaction::Sample(7),
				Interaction::Observe(128),
				Interaction::Sample(32),
				Interaction::Observe(891),
				Interaction::Sample(7),
			]
		);
	}

	#[test]
	fn test_empty_interactions() {
		let mut challenger = HasherChallenger::<Groestl256>::default();
		challenger.observer();
		challenger.observer();
		challenger.sampler();
		challenger.observer();
		assert_eq!(challenger.finalize(), vec![]);
	}
}
