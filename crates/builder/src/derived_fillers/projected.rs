// Copyright 2024-2025 Irreducible Inc.

use bytemuck::must_cast_slice;

use crate::constraint_system::{Filler, ProjectionVariant, U};

pub fn projected(
	tower_level: usize,
	values: Vec<binius_field::BinaryField1b>,
	variant: ProjectionVariant,
) -> Filler {
	Filler::new(move |inputs: &[&[U]], output: &mut [U]| {
		let input = inputs[0];
		let captured_var = values.as_slice();
		match variant {
			ProjectionVariant::FirstVars => {
				let input = must_cast_slice::<U, u32>(input);
			}
			ProjectionVariant::LastVars => {
				//
			}
		}
	})
}
