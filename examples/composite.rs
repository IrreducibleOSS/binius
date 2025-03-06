use binius_circuits::builder::ConstraintSystemBuilder;
use binius_core::{
	constraint_system::channel::validate_witness, fiat_shamir::HasherChallenger,
	oracle::ShiftVariant, polynomial::ArithCircuitPoly, tower::CanonicalTowerFamily,
};
use binius_field::{arch::OptimalUnderlier, BinaryField128b, BinaryField64b, BinaryField8b};
use binius_hal::make_portable_backend;
use binius_hash::compress::Groestl256ByteCompression;
use binius_macros::arith_expr;
use binius_math::{CompositionPoly, IsomorphicEvaluationDomainFactory};

type B128 = BinaryField128b;
type B64 = BinaryField64b;

fn main() {
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

	let column_x = builder.add_committed("x", n_vars, 7);
	let column_y = builder.add_committed("y", n_vars, 7);
	let column_comp_1 = builder
		.add_composite("comp1", n_vars, [column_x, column_y], comp_1.clone())
		.unwrap();

	let column_shift = builder
		.add_shifted("shift", column_comp_1, (1 << n_vars) - 1, n_vars, ShiftVariant::CircularLeft)
		.unwrap();

	let column_comp_2 = builder
		.add_composite("comp2", n_vars, [column_y, column_comp_1, column_shift], comp_2.clone())
		.unwrap();

	let column_z = builder.add_committed("z", n_vars + 1, 6);
	let column_packed = builder.add_packed("packed", column_z, 1).unwrap();

	let column_comp_3 = builder
		.add_composite(
			"comp3",
			n_vars,
			[
				column_x,
				column_y,
				column_comp_1,
				column_shift,
				column_comp_2,
				column_packed,
			],
			comp_3.clone(),
		)
		.unwrap();

	// dummy channel
	let channel = builder.add_channel();
	builder
		.send(
			channel,
			1 << n_vars,
			vec![
				column_x,
				column_y,
				column_comp_1,
				column_shift,
				column_comp_2,
				column_packed,
				column_comp_3,
			],
		)
		.unwrap();
	builder
		.receive(
			channel,
			1 << n_vars,
			vec![
				column_x,
				column_y,
				column_comp_1,
				column_shift,
				column_comp_2,
				column_packed,
				column_comp_3,
			],
		)
		.unwrap();

	let values_x = (0..(1 << n_vars))
		.map(|i| B128::from(i as u128))
		.collect::<Vec<_>>();
	let values_y = (0..(1 << n_vars))
		.map(|i| B128::from(i * i as u128))
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
		.map(|i| B64::from(i * i / 8 + i % 10 as u64))
		.collect::<Vec<_>>();
	let values_packed = (0..(1 << n_vars))
		.map(|i| {
			B128::from(((values_z[2 * i + 1].val() as u128) << 64) + values_z[2 * i].val() as u128)
		})
		.collect::<Vec<_>>();

	let arith_poly_3 = ArithCircuitPoly::new(comp_3);
	let values_comp_3 = (0..(1 << n_vars))
		.map(|i| {
			arith_poly_3
				.evaluate(&[
					values_x[i],
					values_y[i],
					values_comp_1[i],
					values_shift[i],
					values_comp_2[i],
					values_packed[i],
				])
				.unwrap()
		})
		.collect::<Vec<_>>();

	let mut add_witness_col_b128 = |oracle_id: usize, values: &[B128]| {
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
	builder
		.witness()
		.unwrap()
		.new_column::<B64>(column_z)
		.as_mut_slice()
		.copy_from_slice(&values_z);

	let witness = builder.take_witness().unwrap();
	let constraint_system = builder.build().unwrap();

	validate_witness(&witness, &[], &[], 1).unwrap();

	let domain_factory = IsomorphicEvaluationDomainFactory::<BinaryField8b>::default();
	let proof =
		binius_core::constraint_system::prove::<
			OptimalUnderlier,
			CanonicalTowerFamily,
			_,
			groestl_crypto::Groestl256,
			Groestl256ByteCompression,
			HasherChallenger<groestl_crypto::Groestl256>,
			_,
		>(
			&constraint_system, log_inv_rate, security_bits, &[], witness, &domain_factory, &backend
		)
		.unwrap();

	binius_core::constraint_system::verify::<
		OptimalUnderlier,
		CanonicalTowerFamily,
		groestl_crypto::Groestl256,
		Groestl256ByteCompression,
		HasherChallenger<groestl_crypto::Groestl256>,
	>(&constraint_system, log_inv_rate, security_bits, &[], proof)
	.unwrap();
}
