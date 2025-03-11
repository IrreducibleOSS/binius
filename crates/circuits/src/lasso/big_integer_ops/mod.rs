// Copyright 2024-2025 Irreducible Inc.

pub mod byte_sliced_add;
pub mod byte_sliced_add_carryfree;
pub mod byte_sliced_double_conditional_increment;
pub mod byte_sliced_modular_mul;
pub mod byte_sliced_mul;
pub mod byte_sliced_test_utils;

pub use byte_sliced_add::byte_sliced_add;
pub use byte_sliced_add_carryfree::byte_sliced_add_carryfree;
pub use byte_sliced_double_conditional_increment::byte_sliced_double_conditional_increment;
pub use byte_sliced_modular_mul::byte_sliced_modular_mul;
pub use byte_sliced_mul::byte_sliced_mul;

#[cfg(test)]
mod tests {
	use binius_field::tower_levels::{
		TowerLevel1, TowerLevel16, TowerLevel2, TowerLevel4, TowerLevel8,
	};

	use super::byte_sliced_test_utils::{
		test_bytesliced_add, test_bytesliced_add_carryfree,
		test_bytesliced_double_conditional_increment, test_bytesliced_modular_mul,
		test_bytesliced_mul,
	};

	#[test]
	fn test_lasso_add_bytesliced() {
		test_bytesliced_add::<1, TowerLevel1>();
		test_bytesliced_add::<2, TowerLevel2>();
		test_bytesliced_add::<4, TowerLevel4>();
		test_bytesliced_add::<8, TowerLevel8>();
	}

	#[test]
	fn test_lasso_mul_bytesliced() {
		test_bytesliced_mul::<1, TowerLevel2>();
		test_bytesliced_mul::<2, TowerLevel4>();
		test_bytesliced_mul::<4, TowerLevel8>();
		test_bytesliced_mul::<8, TowerLevel16>();
	}

	#[test]
	fn test_lasso_modular_mul_bytesliced_level_2() {
		test_bytesliced_modular_mul::<1, TowerLevel2>();
	}

	#[test]
	fn test_lasso_modular_mul_bytesliced_level_4() {
		test_bytesliced_modular_mul::<2, TowerLevel4>();
	}

	#[test]
	fn test_lasso_modular_mul_bytesliced_level_8() {
		test_bytesliced_modular_mul::<4, TowerLevel8>();
	}

	#[test]
	fn test_lasso_modular_mul_bytesliced_level_16() {
		test_bytesliced_modular_mul::<8, TowerLevel16>();
	}

	#[test]
	fn test_lasso_bytesliced_double_conditional_increment() {
		test_bytesliced_double_conditional_increment::<1, TowerLevel1>();
		test_bytesliced_double_conditional_increment::<2, TowerLevel2>();
		test_bytesliced_double_conditional_increment::<4, TowerLevel4>();
		test_bytesliced_double_conditional_increment::<8, TowerLevel8>();
	}

	#[test]
	fn test_lasso_bytesliced_add_carryfree() {
		test_bytesliced_add_carryfree::<1, TowerLevel1>();
		test_bytesliced_add_carryfree::<2, TowerLevel2>();
		test_bytesliced_add_carryfree::<4, TowerLevel4>();
		test_bytesliced_add_carryfree::<8, TowerLevel8>();
	}
}
