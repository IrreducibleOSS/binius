// Copyright 2025 Irreducible Inc.

use binius_core::constraint_system::{channel::Boundary, validate::validate_witness};

use super::{ConstraintSystemBuilder, types::F};

pub fn test_circuit(
	build_circuit: fn(&mut ConstraintSystemBuilder) -> Result<Vec<Boundary<F>>, anyhow::Error>,
) -> Result<(), anyhow::Error> {
	let mut verifier_builder = ConstraintSystemBuilder::new();
	let verifier_boundaries = build_circuit(&mut verifier_builder)?;
	let verifier_constraint_system = verifier_builder.build()?;

	let allocator = bumpalo::Bump::new();
	let mut prover_builder = ConstraintSystemBuilder::new_with_witness(&allocator);
	let prover_boundaries = build_circuit(&mut prover_builder)?;
	let prover_witness = prover_builder.take_witness()?;
	let _prover_constraint_system = prover_builder.build()?;

	assert_eq!(verifier_boundaries, prover_boundaries);
	validate_witness(&verifier_constraint_system, &verifier_boundaries, &prover_witness)?;
	Ok(())
}
