// Copyright 2024-2025 Irreducible Inc.
use super::multiply::mul_alpha;
use crate::{
	tower_levels::{TowerLevel, TowerLevelWithArithOps},
	underlier::WithUnderlier,
	AESTowerField8b, PackedAESBinaryField32x8b, PackedField,
};

#[inline(always)]
pub fn square<Level: TowerLevel<PackedAESBinaryField32x8b>>(
	field_element: &Level::Data,
	destination: &mut Level::Data,
) {
	let base_alpha =
		PackedAESBinaryField32x8b::from_scalars([AESTowerField8b::from_underlier(0xd3); 32]);
	square_main::<true, Level>(field_element, destination, base_alpha);
}

#[inline(always)]
pub fn square_main<const WRITING_TO_ZEROS: bool, Level: TowerLevel<PackedAESBinaryField32x8b>>(
	field_element: &Level::Data,
	destination: &mut Level::Data,
	base_alpha: PackedAESBinaryField32x8b,
) {
	if Level::WIDTH == 1 {
		if WRITING_TO_ZEROS {
			destination.as_mut()[0] = field_element.as_ref()[0].square();
		} else {
			destination.as_mut()[0] += field_element.as_ref()[0].square();
		}
		return;
	}

	let (a0, a1) = Level::split(field_element);

	let (result0, result1) = Level::split_mut(destination);
	let mut a1_squared = <<Level as TowerLevel<PackedAESBinaryField32x8b>>::Base as TowerLevel<
		PackedAESBinaryField32x8b,
	>>::default();

	square_main::<true, Level::Base>(a1, &mut a1_squared, base_alpha);

	mul_alpha::<WRITING_TO_ZEROS, Level::Base>(&a1_squared, result1, base_alpha);

	square_main::<WRITING_TO_ZEROS, Level::Base>(a0, result0, base_alpha);

	Level::Base::add_into(&a1_squared, result0);
}
