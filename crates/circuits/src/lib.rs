// Copyright 2024 Irreducible Inc.

pub mod bitwise;
pub mod builder;
pub mod step_down;
pub mod u32add;
pub mod u32fib;
pub mod unconstrained;

#[cfg(test)]
mod tests {
	use crate::{
		bitwise, builder::ConstraintSystemBuilder, u32add::u32add, u32fib::u32fib,
		unconstrained::unconstrained,
	};
	use binius_core::constraint_system::validate::validate_witness;
	use binius_field::{arch::OptimalUnderlier, BinaryField128b};

	type U = OptimalUnderlier;
	type F = BinaryField128b;

	#[test]
	fn test_u32add() {
		let mut builder = ConstraintSystemBuilder::<U, F>::new_with_witness();
		let log_size = 14;
		let a = unconstrained(&mut builder, log_size).unwrap();
		let b = unconstrained(&mut builder, log_size).unwrap();
		let _c = u32add(&mut builder, log_size, a, b).unwrap();

		let witness = builder.take_witness().unwrap();
		let constraint_system = builder.build().unwrap();
		let boundaries = vec![];
		validate_witness(&constraint_system, boundaries, witness).unwrap();
	}

	#[test]
	fn test_u32fib() {
		let mut builder = ConstraintSystemBuilder::<U, F>::new_with_witness();
		let log_size_1b = 14;
		let _ = u32fib(&mut builder, log_size_1b).unwrap();

		let witness = builder.take_witness().unwrap();
		let constraint_system = builder.build().unwrap();
		let boundaries = vec![];
		validate_witness(&constraint_system, boundaries, witness).unwrap();
	}

	#[test]
	fn test_bitwise() {
		let mut builder = ConstraintSystemBuilder::<U, F>::new_with_witness();
		let log_size = 14;
		let a = unconstrained(&mut builder, log_size).unwrap();
		let b = unconstrained(&mut builder, log_size).unwrap();
		let _and = bitwise::and(&mut builder, log_size, a, b).unwrap();
		let _xor = bitwise::xor(&mut builder, log_size, a, b).unwrap();
		let _or = bitwise::or(&mut builder, log_size, a, b).unwrap();

		let witness = builder.take_witness().unwrap();
		let constraint_system = builder.build().unwrap();
		let boundaries = vec![];
		validate_witness(&constraint_system, boundaries, witness).unwrap();
	}
}
