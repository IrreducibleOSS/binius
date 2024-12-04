// Copyright 2024 Irreducible Inc.

#![feature(array_try_map, array_try_from_fn)]

pub mod bitwise;
pub mod builder;
pub mod groestl;
pub mod keccakf;
pub mod lasso;
mod pack;
pub mod sha256;
pub mod transparent;
pub mod u32add;
pub mod u32fib;
pub mod unconstrained;
pub mod vision;

#[cfg(test)]
mod tests {
	use crate::{
		bitwise,
		builder::ConstraintSystemBuilder,
		groestl::groestl_p_permutation,
		keccakf::keccakf,
		lasso::{self, batch::LookupBatch, lookups, u32add::SeveralU32add},
		sha256::sha256,
		u32add::u32add_committed,
		u32fib::u32fib,
		unconstrained::unconstrained,
		vision::vision_permutation,
	};
	use binius_core::{
		constraint_system::{
			self,
			channel::{Boundary, FlushDirection},
			validate::validate_witness,
		},
		fiat_shamir::HasherChallenger,
		oracle::OracleId,
		tower::CanonicalTowerFamily,
	};
	use binius_field::{
		arch::OptimalUnderlier, as_packed_field::PackedType, underlier::WithUnderlier,
		AESTowerField16b, BinaryField128b, BinaryField1b, BinaryField32b, BinaryField64b,
		BinaryField8b, Field, TowerField,
	};
	use binius_hal::make_portable_backend;
	use binius_hash::{GroestlDigestCompression, GroestlHasher};
	use binius_math::DefaultEvaluationDomainFactory;
	use groestl_crypto::Groestl256;
	use std::array;

	type U = OptimalUnderlier;
	type F = BinaryField128b;

	#[test]
	fn test_lasso_u8mul() {
		let allocator = bumpalo::Bump::new();
		let mut builder = ConstraintSystemBuilder::<U, F>::new_with_witness(&allocator);
		let log_size = 10;

		let mult_a =
			unconstrained::<_, _, BinaryField8b>(&mut builder, "mult_a", log_size).unwrap();
		let mult_b =
			unconstrained::<_, _, BinaryField8b>(&mut builder, "mult_b", log_size).unwrap();

		let mul_lookup_table =
			lookups::u8_arithmetic::mul_lookup(&mut builder, "mul table").unwrap();

		let mut lookup_batch = LookupBatch::new(mul_lookup_table);

		let _product = lasso::u8mul(
			&mut builder,
			&mut lookup_batch,
			"lasso_u8mul",
			mult_a,
			mult_b,
			1 << log_size,
		)
		.unwrap();

		lookup_batch
			.execute::<_, _, BinaryField32b, BinaryField32b>(&mut builder)
			.unwrap();

		let witness = builder.take_witness().unwrap();
		let constraint_system = builder.build().unwrap();
		let boundaries = vec![];
		validate_witness(&constraint_system, &boundaries, &witness).unwrap();
	}

	#[test]
	fn test_lasso_batched_u8mul() {
		let allocator = bumpalo::Bump::new();
		let mut builder = ConstraintSystemBuilder::<U, F>::new_with_witness(&allocator);
		let log_size = 10;
		let mul_lookup_table =
			lookups::u8_arithmetic::mul_lookup(&mut builder, "mul table").unwrap();

		let mut lookup_batch = LookupBatch::new(mul_lookup_table);

		for _ in 0..10 {
			let mult_a =
				unconstrained::<_, _, BinaryField8b>(&mut builder, "mult_a", log_size).unwrap();
			let mult_b =
				unconstrained::<_, _, BinaryField8b>(&mut builder, "mult_b", log_size).unwrap();

			let _product = lasso::u8mul(
				&mut builder,
				&mut lookup_batch,
				"lasso_u8mul",
				mult_a,
				mult_b,
				1 << log_size,
			)
			.unwrap();
		}

		lookup_batch
			.execute::<_, _, BinaryField32b, BinaryField32b>(&mut builder)
			.unwrap();

		let witness = builder.take_witness().unwrap();
		let constraint_system = builder.build().unwrap();
		let boundaries = vec![];
		validate_witness(&constraint_system, &boundaries, &witness).unwrap();
	}

	#[test]
	fn test_lasso_batched_u8mul_rejects() {
		let allocator = bumpalo::Bump::new();
		let mut builder = ConstraintSystemBuilder::<U, F>::new_with_witness(&allocator);
		let log_size = 10;

		// We try to feed in the add table instead
		let mul_lookup_table =
			lookups::u8_arithmetic::add_lookup(&mut builder, "mul table").unwrap();

		let mut lookup_batch = LookupBatch::new(mul_lookup_table);

		// TODO?: Make this test fail 100% of the time, even though its almost impossible with rng
		for _ in 0..10 {
			let mult_a =
				unconstrained::<_, _, BinaryField8b>(&mut builder, "mult_a", log_size).unwrap();
			let mult_b =
				unconstrained::<_, _, BinaryField8b>(&mut builder, "mult_b", log_size).unwrap();

			let _product = lasso::u8mul(
				&mut builder,
				&mut lookup_batch,
				"lasso_u8mul",
				mult_a,
				mult_b,
				1 << log_size,
			)
			.unwrap();
		}

		lookup_batch
			.execute::<_, _, BinaryField32b, BinaryField32b>(&mut builder)
			.unwrap();

		let witness = builder.take_witness().unwrap();
		let constraint_system = builder.build().unwrap();
		let boundaries = vec![];
		validate_witness(&constraint_system, &boundaries, &witness)
			.expect_err("Channels should be unbalanced");
	}

	#[test]
	fn test_several_lasso_u32add() {
		let allocator = bumpalo::Bump::new();
		let mut builder = ConstraintSystemBuilder::<U, F>::new_with_witness(&allocator);

		let mut several_u32_add = SeveralU32add::new(&mut builder).unwrap();

		[11, 12, 13].into_iter().for_each(|log_size| {
			// BinaryField8b is used here because we utilize an 8x8x1→8 table
			let add_a_u8 =
				unconstrained::<_, _, BinaryField8b>(&mut builder, "add_a", log_size).unwrap();
			let add_b_u8 =
				unconstrained::<_, _, BinaryField8b>(&mut builder, "add_b", log_size).unwrap();
			let _sum = several_u32_add
				.u32add::<BinaryField8b, BinaryField8b>(
					&mut builder,
					"lasso_u32add",
					add_a_u8,
					add_b_u8,
				)
				.unwrap();
		});

		several_u32_add
			.finalize(&mut builder, "lasso_u32add")
			.unwrap();

		let witness = builder.take_witness().unwrap();
		let constraint_system = builder.build().unwrap();
		let boundaries = vec![];
		validate_witness(&constraint_system, &boundaries, &witness).unwrap();
	}

	#[test]
	fn test_lasso_u32add() {
		let allocator = bumpalo::Bump::new();
		let mut builder = ConstraintSystemBuilder::<U, F>::new_with_witness(&allocator);
		let log_size = 14;

		let add_a = unconstrained::<_, _, BinaryField1b>(&mut builder, "add_a", log_size).unwrap();
		let add_b = unconstrained::<_, _, BinaryField1b>(&mut builder, "add_b", log_size).unwrap();
		let _sum = lasso::u32add::<_, _, BinaryField1b, BinaryField1b>(
			&mut builder,
			"lasso_u32add",
			add_a,
			add_b,
		)
		.unwrap();

		let witness = builder.take_witness().unwrap();
		let constraint_system = builder.build().unwrap();
		let boundaries = vec![];
		validate_witness(&constraint_system, &boundaries, &witness).unwrap();
	}

	#[test]
	fn test_u32add() {
		let allocator = bumpalo::Bump::new();
		let mut builder = ConstraintSystemBuilder::<U, F>::new_with_witness(&allocator);
		let log_size = 14;
		let a = unconstrained::<_, _, BinaryField1b>(&mut builder, "a", log_size).unwrap();
		let b = unconstrained::<_, _, BinaryField1b>(&mut builder, "b", log_size).unwrap();
		let _c = u32add_committed(&mut builder, "u32add", a, b).unwrap();

		let witness = builder.take_witness().unwrap();
		let constraint_system = builder.build().unwrap();
		let boundaries = vec![];
		validate_witness(&constraint_system, &boundaries, &witness).unwrap();
	}

	#[test]
	fn test_u32fib() {
		let allocator = bumpalo::Bump::new();
		let mut builder = ConstraintSystemBuilder::<U, F>::new_with_witness(&allocator);
		let log_size_1b = 14;
		let _ = u32fib(&mut builder, "u32fib", log_size_1b).unwrap();

		let witness = builder.take_witness().unwrap();
		let constraint_system = builder.build().unwrap();
		let boundaries = vec![];
		validate_witness(&constraint_system, &boundaries, &witness).unwrap();
	}

	#[test]
	fn test_bitwise() {
		let allocator = bumpalo::Bump::new();
		let mut builder = ConstraintSystemBuilder::<U, F>::new_with_witness(&allocator);
		let log_size = 6;
		let a = unconstrained::<_, _, BinaryField1b>(&mut builder, "a", log_size).unwrap();
		let b = unconstrained::<_, _, BinaryField1b>(&mut builder, "b", log_size).unwrap();
		let _and = bitwise::and(&mut builder, "and", a, b).unwrap();
		let _xor = bitwise::xor(&mut builder, "xor", a, b).unwrap();
		let _or = bitwise::or(&mut builder, "or", a, b).unwrap();

		let witness = builder.take_witness().unwrap();
		let constraint_system = builder.build().unwrap();
		let boundaries = vec![];
		validate_witness(&constraint_system, &boundaries, &witness).unwrap();
	}

	#[test]
	fn test_keccakf() {
		let allocator = bumpalo::Bump::new();
		let mut builder = ConstraintSystemBuilder::<U, BinaryField1b>::new_with_witness(&allocator);
		let log_size = 12;
		let input = array::from_fn(|_| {
			unconstrained::<_, _, BinaryField1b>(&mut builder, "input", log_size).unwrap()
		});

		let _state_out = keccakf(&mut builder, input, log_size);

		let witness = builder.take_witness().unwrap();
		let constraint_system = builder.build().unwrap();
		let boundaries = vec![];
		validate_witness(&constraint_system, &boundaries, &witness).unwrap();
	}

	#[test]
	fn test_sha256() {
		let allocator = bumpalo::Bump::new();
		let mut builder = ConstraintSystemBuilder::<U, BinaryField1b>::new_with_witness(&allocator);
		let log_size = PackedType::<U, BinaryField1b>::LOG_WIDTH;
		let input: [OracleId; 16] = array::from_fn(|i| {
			unconstrained::<_, _, BinaryField1b>(&mut builder, i, log_size).unwrap()
		});
		let _state_out = sha256(&mut builder, input, log_size);

		let witness = builder.take_witness().unwrap();
		let constraint_system = builder.build().unwrap();
		let boundaries = vec![];
		validate_witness(&constraint_system, &boundaries, &witness).unwrap();
	}

	#[test]
	fn test_sha256_lasso() {
		let allocator = bumpalo::Bump::new();
		let mut builder =
			ConstraintSystemBuilder::<U, BinaryField32b>::new_with_witness(&allocator);
		let log_size = PackedType::<U, BinaryField1b>::LOG_WIDTH + BinaryField8b::TOWER_LEVEL;
		let input: [OracleId; 16] = array::from_fn(|i| {
			unconstrained::<_, _, BinaryField1b>(&mut builder, i, log_size).unwrap()
		});
		let _state_out = lasso::sha256(&mut builder, input, log_size);

		let witness = builder.take_witness().unwrap();
		let constraint_system = builder.build().unwrap();
		let boundaries = vec![];
		validate_witness(&constraint_system, &boundaries, &witness).unwrap();
	}

	#[test]
	fn test_groestl() {
		let allocator = bumpalo::Bump::new();
		let mut builder =
			ConstraintSystemBuilder::<OptimalUnderlier, AESTowerField16b>::new_with_witness(
				&allocator,
			);
		let log_size = 9;
		let _state_out = groestl_p_permutation(&mut builder, log_size).unwrap();

		let witness = builder.take_witness().unwrap();
		let constraint_system = builder.build().unwrap();
		let boundaries = vec![];
		validate_witness(&constraint_system, &boundaries, &witness).unwrap();
	}

	#[test]
	fn test_vision32b() {
		let allocator = bumpalo::Bump::new();
		let mut builder =
			ConstraintSystemBuilder::<OptimalUnderlier, BinaryField64b>::new_with_witness(
				&allocator,
			);
		let log_size = 8;
		let state_in: [OracleId; 24] = array::from_fn(|i| {
			unconstrained::<_, _, BinaryField32b>(&mut builder, format!("p_in[{i}]"), log_size)
				.unwrap()
		});
		let _state_out = vision_permutation(&mut builder, log_size, state_in).unwrap();

		let witness = builder.take_witness().unwrap();
		let constraint_system = builder.build().unwrap();
		let boundaries = vec![];
		validate_witness(&constraint_system, &boundaries, &witness).unwrap();
	}

	#[test]
	fn test_boundaries() {
		// Proving Collatz Orbits
		let allocator = bumpalo::Bump::new();
		let mut builder = ConstraintSystemBuilder::<U, F>::new_with_witness(&allocator);

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

		builder.flush(FlushDirection::Pull, channel_id, even_counter, [even]);
		builder.flush(FlushDirection::Push, channel_id, even_counter, [half]);
		builder.flush(FlushDirection::Pull, channel_id, odd_counter, [odd]);
		builder.flush(FlushDirection::Push, channel_id, odd_counter, [output]);

		let witness = builder
			.take_witness()
			.expect("builder created with witness");

		let constraint_system = builder.build().unwrap();

		let domain_factory = DefaultEvaluationDomainFactory::default();
		let backend = make_portable_backend();

		let proof = constraint_system::prove::<
			U,
			CanonicalTowerFamily,
			BinaryField64b,
			_,
			_,
			GroestlHasher<BinaryField128b>,
			GroestlDigestCompression<BinaryField8b>,
			HasherChallenger<groestl_crypto::Groestl256>,
			_,
		>(&constraint_system, 1, 10, witness, &domain_factory, &backend)
		.unwrap();

		constraint_system::verify::<
			U,
			CanonicalTowerFamily,
			_,
			_,
			GroestlHasher<BinaryField128b>,
			GroestlDigestCompression<BinaryField8b>,
			HasherChallenger<Groestl256>,
		>(
			&constraint_system,
			1,
			10,
			&domain_factory,
			vec![pull_boundaries, push_boundaries],
			proof,
		)
		.unwrap();
	}
}
