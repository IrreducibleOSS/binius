// Copyright 2024-2025 Irreducible Inc.

//! The Binius frontend library, along with useful gadgets and examples.
//!
//! The frontend library provides high-level interfaces for constructing constraint systems in the
//! [`crate::builder`] module. Most other modules contain circuit gadgets that can be used to build
//! more complex constraint systems.

#![deprecated = "use binius_m3 instead"]
// This is because there are quite some arith_expr! in this codebase and it's acceptable to blanket
// allow(deprecated) here since it's going away anyway.
#![allow(deprecated)]
#![allow(clippy::module_inception)]

pub mod arithmetic;
pub mod bitwise;
pub mod blake3;
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
			channel::{Boundary, FlushDirection, OracleOrConst, validate_witness},
		},
		fiat_shamir::HasherChallenger,
		oracle::{OracleId, ShiftVariant},
	};
	use binius_fast_compute::arith_circuit::ArithCircuitPoly;
	use binius_field::{
		BinaryField1b, BinaryField8b, BinaryField64b, BinaryField128b, Field, TowerField,
		arch::OptimalUnderlier, as_packed_field::PackedType, tower::CanonicalTowerFamily,
		underlier::WithUnderlier,
	};
	use binius_hal::make_portable_backend;
	use binius_hash::groestl::{Groestl256, Groestl256ByteCompression};
	use binius_macros::arith_expr;
	use binius_math::CompositionPoly;
	use rand::{seq::SliceRandom, thread_rng};

	type B128 = BinaryField128b;
	type B64 = BinaryField64b;

	use crate::{
		builder::{
			ConstraintSystemBuilder,
			test_utils::test_circuit,
			types::{F, U},
		},
		unconstrained::unconstrained,
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
			.flush(FlushDirection::Pull, channel_id, even_counter, [OracleOrConst::Oracle(even)])
			.unwrap();
		builder
			.flush(FlushDirection::Push, channel_id, even_counter, [OracleOrConst::Oracle(half)])
			.unwrap();
		builder
			.flush(FlushDirection::Pull, channel_id, odd_counter, [OracleOrConst::Oracle(odd)])
			.unwrap();
		builder
			.flush(FlushDirection::Push, channel_id, odd_counter, [OracleOrConst::Oracle(output)])
			.unwrap();

		let witness = builder
			.take_witness()
			.expect("builder created with witness");

		let constraint_system = builder.build().unwrap();

		let backend = make_portable_backend();

		let proof = constraint_system::prove::<
			U,
			CanonicalTowerFamily,
			Groestl256,
			Groestl256ByteCompression,
			HasherChallenger<Groestl256>,
			_,
		>(&constraint_system, 1, 10, &boundaries, witness, &backend)
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

	#[test]
	#[ignore]
	fn test_composite_circuit() {
		let backend = make_portable_backend();
		let allocator = bumpalo::Bump::new();
		let mut builder = ConstraintSystemBuilder::new_with_witness(&allocator);
		let n_vars = 8;
		let log_inv_rate = 1;
		let security_bits = 30;
		let comp_1 = arith_expr!(B128[x, y] = x*y*y*0x85 +x*x*y*0x9 + y + 0x123);
		let comp_2 =
			arith_expr!(B128[x, y, z] = x*z*y*0x81115 +x*y*0x98888 + y*z + z*z*z*z*z*z + 0x155523);
		let comp_3 = arith_expr!(B128[a, b, c, d, e, f] = e*f*f + a*b*c*2 + d*0x999 + 0x123);
		let comp_4 = arith_expr!(B128[a, b] = a*(b+a));

		let column_x = builder.add_committed("x", n_vars, 7);
		let column_y = builder.add_committed("y", n_vars, 7);
		let column_comp_1 = builder
			.add_composite_mle("comp1", n_vars, [column_x, column_y], comp_1.clone())
			.unwrap();

		let column_shift = builder
			.add_shifted(
				"shift",
				column_comp_1,
				(1 << n_vars) - 1,
				n_vars,
				ShiftVariant::CircularLeft,
			)
			.unwrap();

		let column_comp_2 = builder
			.add_composite_mle(
				"comp2",
				n_vars,
				[column_y, column_comp_1, column_shift],
				comp_2.clone(),
			)
			.unwrap();

		let column_z = builder.add_committed("z", n_vars + 1, 6);
		let column_packed = builder.add_packed("packed", column_z, 1).unwrap();

		let column_comp_3 = builder
			.add_composite_mle(
				"comp3",
				n_vars,
				[
					column_x,
					column_x,
					column_comp_1,
					column_shift,
					column_comp_2,
					column_packed,
				],
				comp_3.clone(),
			)
			.unwrap();

		let column_comp_4 = builder
			.add_composite_mle(
				"comp4",
				n_vars,
				[
					column_comp_2,
					column_comp_3,
					column_x,
					column_shift,
					column_y,
				],
				comp_4.clone(),
			)
			.unwrap();

		// dummy channel
		let channel = builder.add_channel();
		builder
			.send(
				channel,
				1 << n_vars,
				vec![
					OracleOrConst::Oracle(column_y),
					OracleOrConst::Oracle(column_x),
					OracleOrConst::Oracle(column_comp_1),
					OracleOrConst::Oracle(column_shift),
					OracleOrConst::Oracle(column_comp_2),
					OracleOrConst::Oracle(column_packed),
					OracleOrConst::Oracle(column_comp_3),
				],
			)
			.unwrap();
		builder
			.receive(
				channel,
				1 << n_vars,
				vec![
					OracleOrConst::Oracle(column_x),
					OracleOrConst::Oracle(column_y),
					OracleOrConst::Oracle(column_comp_1),
					OracleOrConst::Oracle(column_shift),
					OracleOrConst::Oracle(column_comp_2),
					OracleOrConst::Oracle(column_packed),
					OracleOrConst::Oracle(column_comp_3),
				],
			)
			.unwrap();

		let values_x = (0..(1 << n_vars))
			.map(|i| B128::from(i as u128))
			.collect::<Vec<_>>();
		let values_y = (0..(1 << n_vars))
			.map(|i| B128::from(i * i))
			.collect::<Vec<_>>();

		let arith_poly_1 = ArithCircuitPoly::new(comp_1);
		let values_comp_1 = (0..(1 << n_vars))
			.map(|i| arith_poly_1.evaluate(&[values_x[i], values_y[i]]).unwrap())
			.collect::<Vec<_>>();

		let mut values_shift = values_comp_1.clone();
		let first = values_shift.remove(0);
		values_shift.push(first);

		let arith_poly_2 = ArithCircuitPoly::new(comp_2);
		let values_comp_2 = (0..(1 << n_vars))
			.map(|i| {
				arith_poly_2
					.evaluate(&[values_y[i], values_comp_1[i], values_shift[i]])
					.unwrap()
			})
			.collect::<Vec<_>>();

		let values_z = (0..(1 << (n_vars + 1)))
			.map(|i| B64::from(i * i / 8 + i % 10_u64))
			.collect::<Vec<_>>();
		let values_packed = (0..(1 << n_vars))
			.map(|i| {
				B128::from(
					((values_z[2 * i + 1].val() as u128) << 64) + values_z[2 * i].val() as u128,
				)
			})
			.collect::<Vec<_>>();

		let arith_poly_3 = ArithCircuitPoly::new(comp_3);
		let values_comp_3 = (0..(1 << n_vars))
			.map(|i| {
				arith_poly_3
					.evaluate(&[
						values_x[i],
						values_x[i],
						values_comp_1[i],
						values_shift[i],
						values_comp_2[i],
						values_packed[i],
					])
					.unwrap()
			})
			.collect::<Vec<_>>();

		let arith_poly_4 = ArithCircuitPoly::new(comp_4);
		let values_comp_4 = (0..(1 << n_vars))
			.map(|i| {
				arith_poly_4
					.evaluate(&[values_comp_2[i], values_comp_3[i]])
					.unwrap()
			})
			.collect::<Vec<_>>();

		let mut add_witness_col_b128 = |oracle_id: OracleId, values: &[B128]| {
			builder
				.witness()
				.unwrap()
				.new_column::<B128>(oracle_id)
				.as_mut_slice()
				.copy_from_slice(values);
		};
		add_witness_col_b128(column_x, &values_x);
		add_witness_col_b128(column_y, &values_y);
		add_witness_col_b128(column_comp_1, &values_comp_1);
		add_witness_col_b128(column_shift, &values_shift);
		add_witness_col_b128(column_comp_2, &values_comp_2);
		add_witness_col_b128(column_packed, &values_packed);
		add_witness_col_b128(column_comp_3, &values_comp_3);
		add_witness_col_b128(column_comp_4, &values_comp_4);
		builder
			.witness()
			.unwrap()
			.new_column::<B64>(column_z)
			.as_mut_slice()
			.copy_from_slice(&values_z);

		let witness = builder.take_witness().unwrap();
		let constraint_system = builder.build().unwrap();

		validate_witness(&witness, &[], &[], 1).unwrap();

		let proof = binius_core::constraint_system::prove::<
			OptimalUnderlier,
			CanonicalTowerFamily,
			Groestl256,
			Groestl256ByteCompression,
			HasherChallenger<Groestl256>,
			_,
		>(&constraint_system, log_inv_rate, security_bits, &[], witness, &backend)
		.unwrap();

		binius_core::constraint_system::verify::<
			OptimalUnderlier,
			CanonicalTowerFamily,
			Groestl256,
			Groestl256ByteCompression,
			HasherChallenger<Groestl256>,
		>(&constraint_system, log_inv_rate, security_bits, &[], proof)
		.unwrap();
	}

	#[test]
	fn test_flush_with_const() {
		test_circuit(|builder| {
			let channel_id = builder.add_channel();
			let oracle = unconstrained::<BinaryField1b>(builder, "oracle", 1)?;
			builder
				.flush(
					FlushDirection::Push,
					channel_id,
					1,
					vec![
						OracleOrConst::Oracle(oracle),
						OracleOrConst::Const {
							base: F::ONE,
							tower_level: BinaryField1b::TOWER_LEVEL,
						},
					],
				)
				.unwrap();

			builder
				.flush(
					FlushDirection::Pull,
					channel_id,
					1,
					vec![
						OracleOrConst::Oracle(oracle),
						OracleOrConst::Const {
							base: F::ONE,
							tower_level: BinaryField1b::TOWER_LEVEL,
						},
					],
				)
				.unwrap();

			Ok(vec![])
		})
		.unwrap()
	}

	//Testing with larger oracles, and random constants, in a random order. To see if given
	// appropriate flushes with constants the channel balances.
	#[test]
	fn test_flush_with_const_large() {
		test_circuit(|builder| {
			let channel_id = builder.add_channel();
			let mut rng = thread_rng();
			let oracles = (0..5)
				.map(|i| unconstrained::<BinaryField128b>(builder, format!("oracle {i}"), 5))
				.collect::<Result<Vec<_>, _>>()?;
			let random_consts = (0..5).map(|_| OracleOrConst::Const {
				base: BinaryField128b::random(&mut rng),
				tower_level: BinaryField128b::TOWER_LEVEL,
			});
			//Places the oracles and consts in a random order
			//This is not a cryptographic random order, but it is good enough for testing
			let mut random_order = oracles
				.iter()
				.copied()
				.map(OracleOrConst::Oracle)
				.chain(random_consts)
				.collect::<Vec<_>>();
			random_order.shuffle(&mut rng);

			let random_order_iterator = random_order.iter().copied();
			for i in 0..1 << 5 {
				builder
					.flush(FlushDirection::Push, channel_id, i, random_order_iterator.clone())
					.unwrap();

				builder
					.flush(FlushDirection::Pull, channel_id, i, random_order_iterator.clone())
					.unwrap();
			}

			Ok(vec![])
		})
		.unwrap()
	}
}
