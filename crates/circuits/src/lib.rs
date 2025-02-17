// Copyright 2024-2025 Irreducible Inc.

//! The Binius frontend library, along with useful gadgets and examples.
//!
//! The frontend library provides high-level interfaces for constructing constraint systems in the
//! [`crate::builder`] module. Most other modules contain circuit gadgets that can be used to build
//! more complex constraint systems.

#![feature(array_try_map, array_try_from_fn)]
#![allow(clippy::module_inception)]

pub mod arithmetic;
pub mod bitwise;
pub mod builder;
pub mod collatz;
pub mod keccakf;
pub mod lasso;
mod pack;
pub mod plain_lookup;
pub mod sha256;
pub mod transparent;
pub mod u32fib;
pub mod unconstrained;
pub mod vision;

#[cfg(test)]
mod tests {
	use binius_core::{
		constraint_system::{
			self,
			channel::{Boundary, FlushDirection},
		},
		fiat_shamir::HasherChallenger,
		tower::CanonicalTowerFamily,
	};
	use binius_field::{
		as_packed_field::PackedType, underlier::WithUnderlier, BinaryField8b, Field,
	};
	use binius_hal::make_portable_backend;
	use binius_hash::compress::Groestl256ByteCompression;
	use binius_math::DefaultEvaluationDomainFactory;
	use groestl_crypto::Groestl256;

	use crate::builder::{
		types::{F, U},
		ConstraintSystemBuilder,
	};

	#[test]
	fn test_boundaries() {
		// Proving Collatz Orbits
		let allocator = bumpalo::Bump::new();
		let mut builder = ConstraintSystemBuilder::new_with_witness(&allocator);

		let log_size = PackedType::<U, BinaryField8b>::LOG_WIDTH + 2;

		let channel_id = builder.add_channel();

		let push_boundaries = Boundary {
			values: vec![F::from_underlier(6)],
			channel_id,
			direction: FlushDirection::Push,
			multiplicity: 1,
		};

		let pull_boundaries = Boundary {
			values: vec![F::ONE],
			channel_id,
			direction: FlushDirection::Pull,
			multiplicity: 1,
		};

		let boundaries = vec![pull_boundaries, push_boundaries];

		let even = builder.add_committed("even", log_size, 3);

		let half = builder.add_committed("half", log_size, 3);

		let odd = builder.add_committed("odd", log_size, 3);

		let output = builder.add_committed("output", log_size, 3);

		let mut even_counter = 0;

		let mut odd_counter = 0;

		if let Some(witness) = builder.witness() {
			let mut current = 6;

			let mut even = witness.new_column::<BinaryField8b>(even);

			let even_u8 = even.as_mut_slice::<u8>();

			let mut half = witness.new_column::<BinaryField8b>(half);

			let half_u8 = half.as_mut_slice::<u8>();

			let mut odd = witness.new_column::<BinaryField8b>(odd);

			let odd_u8 = odd.as_mut_slice::<u8>();

			let mut output = witness.new_column::<BinaryField8b>(output);

			let output_u8 = output.as_mut_slice::<u8>();

			while current != 1 {
				if current & 1 == 0 {
					even_u8[even_counter] = current;
					half_u8[even_counter] = current / 2;
					current = half_u8[even_counter];
					even_counter += 1;
				} else {
					odd_u8[odd_counter] = current;
					output_u8[odd_counter] = 3 * current + 1;
					current = output_u8[odd_counter];
					odd_counter += 1;
				}
			}
		}

		builder
			.flush(FlushDirection::Pull, channel_id, even_counter, [even])
			.unwrap();
		builder
			.flush(FlushDirection::Push, channel_id, even_counter, [half])
			.unwrap();
		builder
			.flush(FlushDirection::Pull, channel_id, odd_counter, [odd])
			.unwrap();
		builder
			.flush(FlushDirection::Push, channel_id, odd_counter, [output])
			.unwrap();

		let witness = builder
			.take_witness()
			.expect("builder created with witness");

		let constraint_system = builder.build().unwrap();

		let domain_factory = DefaultEvaluationDomainFactory::default();
		let backend = make_portable_backend();

		let proof = constraint_system::prove::<
			U,
			CanonicalTowerFamily,
			_,
			Groestl256,
			Groestl256ByteCompression,
			HasherChallenger<Groestl256>,
			_,
		>(&constraint_system, 1, 10, &boundaries, witness, &domain_factory, &backend)
		.unwrap();

		constraint_system::verify::<
			U,
			CanonicalTowerFamily,
			Groestl256,
			Groestl256ByteCompression,
			HasherChallenger<Groestl256>,
		>(&constraint_system, 1, 10, &boundaries, proof)
		.unwrap();
	}
}
