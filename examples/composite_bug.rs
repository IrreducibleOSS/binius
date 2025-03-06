use binius_circuits::builder::ConstraintSystemBuilder;
use binius_core::{
	constraint_system::channel::validate_witness, fiat_shamir::HasherChallenger,
	oracle::ShiftVariant, polynomial::ArithCircuitPoly, tower::CanonicalTowerFamily,
};
use binius_field::{arch::OptimalUnderlier, BinaryField128b, BinaryField8b};
use binius_hal::make_portable_backend;
use binius_hash::compress::Groestl256ByteCompression;
use binius_macros::arith_expr;
use binius_math::{CompositionPoly, IsomorphicEvaluationDomainFactory};

type B128 = BinaryField128b;

fn main() {
	let backend = make_portable_backend();
	let allocator = bumpalo::Bump::new();
	let mut builder = ConstraintSystemBuilder::new_with_witness(&allocator);
	let n_vars = 8;
	let log_inv_rate = 1;
	let security_bits = 30;
	let comp = arith_expr!(B128[x, y] = x*y*y*0x85 +x*x*0x9 + 0x123);

	let column_x = builder.add_committed("x", n_vars, 7);
	let column_y = builder.add_committed("y", n_vars, 7);
	let column_comp = builder
		.add_composite("comp", n_vars, [column_x, column_y], comp.clone())
		.unwrap();

	let column_shift = builder
		.add_shifted("shift", column_comp, (1 << n_vars) - 1, n_vars, ShiftVariant::CircularLeft)
		.unwrap();

	// dummy channel
	let channel = builder.add_channel();
	builder
		.send(channel, 1 << n_vars, vec![column_x, column_y, column_comp, column_shift])
		.unwrap();
	builder
		.receive(channel, 1 << n_vars, vec![column_x, column_y, column_comp, column_shift])
		.unwrap();

	let values_x = (0..(1 << n_vars))
		.map(|i| B128::from(i as u128))
		.collect::<Vec<_>>();
	let values_y = (0..(1 << n_vars))
		.map(|i| B128::from(i * i as u128))
		.collect::<Vec<_>>();

	let arith_poly = ArithCircuitPoly::new(comp);
	let values_comp = (0..(1 << n_vars))
		.map(|i| {
			arith_poly
				.evaluate(&[B128::from(values_x[i]), values_y[i]])
				.unwrap()
		})
		.collect::<Vec<_>>();

	let mut values_shift = values_comp.clone();
	let first = values_shift.remove(0);
	values_shift.push(first);

	let mut add_witness_col = |oracle_id: usize, values: &[B128]| {
		builder
			.witness()
			.unwrap()
			.new_column::<B128>(oracle_id)
			.as_mut_slice()
			.copy_from_slice(values);
	};
	add_witness_col(column_x, &values_x);
	add_witness_col(column_y, &values_y);
	add_witness_col(column_comp, &values_comp);
	add_witness_col(column_shift, &values_shift);

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
