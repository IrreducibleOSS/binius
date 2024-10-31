// Copyright 2024 Irreducible Inc.
use super::{
	multiply::{mul_alpha, mul_main},
	tower_levels::{TowerLevel, TowerLevel16},
};
use crate::{underlier::WithUnderlier, AESTowerField8b, PackedAESBinaryField32x8b, PackedField};

#[inline(always)]
pub fn invert_or_zero(
	field_element: &[PackedAESBinaryField32x8b; 16],
	destination: &mut [PackedAESBinaryField32x8b; 16],
) {
	let base_alpha =
		PackedAESBinaryField32x8b::from_scalars([AESTowerField8b::from_underlier(0xd3); 32]);

	inv_main::<TowerLevel16>(field_element, destination, base_alpha);
}

#[inline(always)]
fn inv_main<Level: TowerLevel>(
	field_element: &Level::Data,
	destination: &mut Level::Data,
	base_alpha: PackedAESBinaryField32x8b,
) {
	if Level::WIDTH == 1 {
		destination.as_mut()[0] = field_element.as_ref()[0].invert_or_zero();
		return;
	}

	let (a0, a1) = Level::split(field_element);

	let (result0, result1) = Level::split_mut(destination);

	let mut intermediate = <<Level as TowerLevel>::Base as TowerLevel>::Data::default();

	// intermediate = subfield_alpha*a1
	mul_alpha::<true, Level::Base>(a1, &mut intermediate, base_alpha);

	// intermediate = a0 + subfield_alpha*a1
	Level::Base::add_into(a0, &mut intermediate);

	let mut delta = <<Level as TowerLevel>::Base as TowerLevel>::Data::default();

	// delta = intermediate * a0
	mul_main::<true, Level::Base>(&intermediate, a0, &mut delta, base_alpha);

	// delta = intermediate * a0 + a1^2
	mul_main::<false, Level::Base>(a1, a1, &mut delta, base_alpha);

	let mut delta_inv = <<Level as TowerLevel>::Base as TowerLevel>::Data::default();

	// delta_inv = 1/delta
	inv_main::<Level::Base>(&delta, &mut delta_inv, base_alpha);

	// result0 = delta_inv*intermediate
	mul_main::<true, Level::Base>(&delta_inv, &intermediate, result0, base_alpha);

	// result1 = delta_inv*intermediate
	mul_main::<true, Level::Base>(&delta_inv, a1, result1, base_alpha);
}
