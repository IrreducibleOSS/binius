// Copyright 2024 Ulvetanna Inc.

#[rustfmt::skip]
pub const TOWER_TO_GFNI_MAP: i64 = u64::from_le_bytes(
	[
		0b00111110,
		0b10011000,
		0b01001110,
		0b10010110,
		0b11101010,
		0b01101010,
		0b01010000,
		0b00110001,
	]
) as i64;

#[rustfmt::skip]
pub const GFNI_TO_TOWER_MAP: i64 = u64::from_le_bytes(
	[
		0b00001100,
		0b01110000,
		0b10100010,
		0b01110010,
		0b00111110,
		0b10000110,
		0b11101000,
		0b11010001,
	]
) as i64;
