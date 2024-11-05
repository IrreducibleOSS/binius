// Copyright 2024 Irreducible Inc.

pub mod bitwise;
pub mod builder;
pub mod groestl;
pub mod keccakf;
pub mod lasso;
pub mod step_down;
pub mod u32add;
pub mod u32fib;
pub mod unconstrained;

#[cfg(test)]
mod tests {
	use crate::{
		bitwise, builder::ConstraintSystemBuilder, groestl::groestl_p_permutation,
		keccakf::keccakf, lasso, u32add::u32add, u32fib::u32fib, unconstrained::unconstrained,
	};
	use binius_core::constraint_system::validate::validate_witness;
	use binius_field::{
		arch::OptimalUnderlier, AESTowerField16b, BinaryField128b, BinaryField1b, BinaryField8b,
	};

	type U = OptimalUnderlier;
	type F = BinaryField128b;

	#[test]
	fn test_lasso() {
		let mut builder = ConstraintSystemBuilder::<U, F>::new_with_witness();
		let log_size = 14;

		let mult_a =
			unconstrained::<_, _, BinaryField8b>(&mut builder, "mult_a", log_size).unwrap();
		let mult_b =
			unconstrained::<_, _, BinaryField8b>(&mut builder, "mult_b", log_size).unwrap();
		let _product = lasso::u8mul(&mut builder, "lasso_u8mul", mult_a, mult_b, log_size).unwrap();

		let witness = builder.take_witness().unwrap();
		let constraint_system = builder.build().unwrap();
		let boundaries = vec![];
		validate_witness(&constraint_system, boundaries, witness).unwrap();
	}

	#[test]
	fn test_u32add() {
		let mut builder = ConstraintSystemBuilder::<U, F>::new_with_witness();
		let log_size = 14;
		let a = unconstrained::<_, _, BinaryField1b>(&mut builder, "a", log_size).unwrap();
		let b = unconstrained::<_, _, BinaryField1b>(&mut builder, "b", log_size).unwrap();
		let _c = u32add(&mut builder, "u32add", log_size, a, b).unwrap();

		let witness = builder.take_witness().unwrap();
		let constraint_system = builder.build().unwrap();
		let boundaries = vec![];
		validate_witness(&constraint_system, boundaries, witness).unwrap();
	}

	#[test]
	fn test_u32fib() {
		let mut builder = ConstraintSystemBuilder::<U, F>::new_with_witness();
		let log_size_1b = 14;
		let _ = u32fib(&mut builder, "u32fib", log_size_1b).unwrap();

		let witness = builder.take_witness().unwrap();
		let constraint_system = builder.build().unwrap();
		let boundaries = vec![];
		validate_witness(&constraint_system, boundaries, witness).unwrap();
	}

	#[test]
	fn test_bitwise() {
		let mut builder = ConstraintSystemBuilder::<U, F>::new_with_witness();
		let log_size = 14;
		let a = unconstrained::<_, _, BinaryField1b>(&mut builder, "a", log_size).unwrap();
		let b = unconstrained::<_, _, BinaryField1b>(&mut builder, "b", log_size).unwrap();
		let _and = bitwise::and(&mut builder, "and", log_size, a, b).unwrap();
		let _xor = bitwise::xor(&mut builder, "xor", log_size, a, b).unwrap();
		let _or = bitwise::or(&mut builder, "or", log_size, a, b).unwrap();

		let witness = builder.take_witness().unwrap();
		let constraint_system = builder.build().unwrap();
		let boundaries = vec![];
		validate_witness(&constraint_system, boundaries, witness).unwrap();
	}

	#[test]
	fn test_keccakf() {
		let mut builder = ConstraintSystemBuilder::<U, BinaryField1b>::new_with_witness();
		let log_size = 12;
		let _state_out = keccakf(&mut builder, log_size);

		let witness = builder.take_witness().unwrap();
		let constraint_system = builder.build().unwrap();
		let boundaries = vec![];
		validate_witness(&constraint_system, boundaries, witness).unwrap();
	}

	#[test]
	fn test_groestl() {
		let mut builder =
			ConstraintSystemBuilder::<OptimalUnderlier, AESTowerField16b>::new_with_witness();
		let log_size = 9;
		let _state_out = groestl_p_permutation(&mut builder, log_size).unwrap();

		let witness = builder.take_witness().unwrap();
		let constraint_system = builder.build().unwrap();
		let boundaries = vec![];
		validate_witness(&constraint_system, boundaries, witness).unwrap();
	}
}
