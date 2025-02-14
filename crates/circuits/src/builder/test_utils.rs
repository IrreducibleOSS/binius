use binius_core::constraint_system::{channel::Boundary, validate::validate_witness};

use super::{types::F, ConstraintSystemBuilder};

pub fn test_circuit(
	build_circuit: fn(&mut ConstraintSystemBuilder) -> Result<Vec<Boundary<F>>, anyhow::Error>,
) -> Result<(), anyhow::Error> {
	let allocator = bumpalo::Bump::new();
	let mut builder = ConstraintSystemBuilder::new_with_witness(&allocator);
	let boundaries = build_circuit(&mut builder)?;
	let witness = builder.take_witness()?;
	let constraint_system = builder.build()?;
	validate_witness(&constraint_system, &boundaries, &witness)?;
	Ok(())
}
