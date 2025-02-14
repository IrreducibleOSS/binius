// Copyright 2024-2025 Irreducible Inc.

use bytemuck::must_cast_slice;

use crate::constraint_system::{Filler, U};

pub fn shifted(
	tower_level: usize,
	offset: usize,
	block_size: Option<usize>,
	variant: crate::constraint_system::ShiftVariant,
) -> Filler {
	Filler::new(move |inputs: &[&[U]], output: &mut [U]| {
		let input = inputs[0];
		let captured_var = offset;
		match tower_level {
			0 => {
				let input = must_cast_slice::<U, u32>(input);
			}
			_ => panic!(),
		}
	})
}
