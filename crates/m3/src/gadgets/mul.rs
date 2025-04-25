// Copyright 2025 Irreducible Inc.

use std::{array, marker::PhantomData};

use anyhow::Result;
use binius_field::{
	ext_basis, packed::set_packed_slice, BinaryField, ExtensionField, Field, PackedExtension,
	PackedField, PackedSubfield, TowerField,
};
use itertools::izip;

use crate::{
	builder::{upcast_col, Col, Expr, TableBuilder, TableWitnessSegment, B1, B128, B32, B64},
	gadgets::u32::add::{Incr, UnsignedAddPrimitives},
};

/// Helper trait to create Multiplication gadgets for unsigned integers of different bit lengths.
pub trait UnsignedMulPrimitives {
	type FP: TowerField;
	type FExpBase: TowerField + ExtensionField<Self::FP>;

	const BIT_LENGTH: usize;

	/// Computes the unsigned primitive mul of `x*y = z` and returns the tuple (z_high, z_low)
	/// representing the high and low bits respectively.
	fn mul(x: Self::FP, y: Self::FP) -> (Self::FP, Self::FP);

	fn is_bit_set_at(a: Self::FP, index: usize) -> bool {
		<Self::FP as ExtensionField<B1>>::get_base(&a, index) == B1::ONE
	}

	fn generator() -> Self::FExpBase;

	/// Returns the generator shifted by the bit length of `Self::FP`.
	fn shifted_generator() -> Self::FExpBase;
}

impl UnsignedMulPrimitives for u32 {
	type FP = B32;
	type FExpBase = B64;

	const BIT_LENGTH: usize = 32;

	fn mul(x: B32, y: B32) -> (B32, B32) {
		let res = x.val() as u64 * y.val() as u64;
		let low = B32::new(res as u32);
		let high = B32::new((res >> 32) as u32);
		(high, low)
	}

	fn generator() -> B64 {
		B64::MULTIPLICATIVE_GENERATOR
	}

	fn shifted_generator() -> B64 {
		let mut g = B64::MULTIPLICATIVE_GENERATOR;
		for _ in 0..32 {
			g *= g
		}
		g
	}
}

impl UnsignedMulPrimitives for u64 {
	type FP = B64;
	type FExpBase = B128;

	const BIT_LENGTH: usize = 64;

	fn mul(x: B64, y: B64) -> (B64, B64) {
		let res = x.val() as u128 * y.val() as u128;
		let low = B64::new(res as u64);
		let high = B64::new((res >> 64) as u64);
		(high, low)
	}

	fn generator() -> Self::FExpBase {
		B128::MULTIPLICATIVE_GENERATOR
	}

	fn shifted_generator() -> B128 {
		let mut g = Self::generator();
		for _ in 0..64 {
			g *= g
		}
		g
	}
}

// Internally used to have deduplicated implementations.
#[derive(Debug)]
struct Mul<UX: UnsignedMulPrimitives, const BIT_LENGTH: usize> {
	x_in_bits: [Col<B1>; BIT_LENGTH],
	y_in_bits: [Col<B1>; BIT_LENGTH],
	out_high_bits: [Col<B1>; BIT_LENGTH],
	out_low_bits: [Col<B1>; BIT_LENGTH],

	pub x_in: Col<UX::FP>,
	pub y_in: Col<UX::FP>,
	pub out_high: Col<UX::FP>,
	pub out_low: Col<UX::FP>,

	_marker: PhantomData<UX>,
}

impl<
		FExpBase: TowerField,
		FP: TowerField,
		UX: UnsignedMulPrimitives<FP = FP, FExpBase = FExpBase>,
		const BIT_LENGTH: usize,
	> Mul<UX, BIT_LENGTH>
where
	FExpBase: ExtensionField<FP> + ExtensionField<B1>,
	B128: ExtensionField<FExpBase> + ExtensionField<FP> + ExtensionField<B1>,
{
	pub fn new(table: &mut TableBuilder) -> Self {
		let x_in_bits = table.add_committed_multiple("x_in_bits");
		let y_in_bits = table.add_committed_multiple("y_in_bits");

		Self::with_inputs(table, x_in_bits, y_in_bits)
	}

	pub fn with_inputs(
		table: &mut TableBuilder,
		x_in_bits: [Col<B1>; BIT_LENGTH],
		y_in_bits: [Col<B1>; BIT_LENGTH],
	) -> Self {
		assert_eq!(FExpBase::TOWER_LEVEL, FP::TOWER_LEVEL + 1);
		assert_eq!(BIT_LENGTH, 1 << FP::TOWER_LEVEL);
		assert_eq!(BIT_LENGTH, UX::BIT_LENGTH);
		// These are currently the only bit lengths I've tested
		assert!(BIT_LENGTH == 32 || BIT_LENGTH == 64);

		let x_in = table.add_computed("x_in", pack_fp(x_in_bits));
		let y_in = table.add_computed("y_in", pack_fp(y_in_bits));

		let generator = UX::generator().into();
		let generator_pow_bit_len = UX::shifted_generator().into();

		let g_pow_x = table.add_static_exp::<FExpBase>("g^x", &x_in_bits, generator);
		let g_pow_xy = table.add_dynamic_exp::<FExpBase>("(g^x)^y", &y_in_bits, g_pow_x);

		let out_high_bits = table.add_committed_multiple("out_high");
		let out_low_bits = table.add_committed_multiple("out_low");

		let out_high = table.add_computed("out_high", pack_fp(out_high_bits));
		let out_low = table.add_computed("out_low", pack_fp(out_low_bits));

		let g_pow_out_low: Col<FExpBase> =
			table.add_static_exp("g^(out_low)", &out_low_bits, generator);
		let g_pow_out_high: Col<FExpBase> = table.add_static_exp(
			"(g^(2^BIT_LENGTH))^(out_high)",
			&out_high_bits,
			generator_pow_bit_len,
		);

		table.assert_zero("order_non_wrapping", x_in_bits[0] * y_in_bits[0] - out_low_bits[0]);
		table.assert_zero("exponentiation_equality", g_pow_xy - g_pow_out_low * g_pow_out_high);

		Self {
			x_in_bits,
			y_in_bits,
			out_high_bits,
			out_low_bits,
			x_in,
			y_in,
			out_high,
			out_low,
			_marker: PhantomData,
		}
	}

	#[inline]
	fn populate_internal<P>(
		&self,
		index: &mut TableWitnessSegment<P>,
		x_vals: impl IntoIterator<Item = FP>,
		y_vals: impl IntoIterator<Item = FP>,
		fill_input_bits: bool,
	) -> Result<()>
	where
		P: PackedField<Scalar = B128> + PackedExtension<B1> + PackedExtension<FP>,
	{
		let mut x_in_bits = array_util::try_map(self.x_in_bits, |bit| index.get_mut(bit))?;
		let mut y_in_bits = array_util::try_map(self.y_in_bits, |bit| index.get_mut(bit))?;
		let mut out_low_bits = array_util::try_map(self.out_low_bits, |bit| index.get_mut(bit))?;
		let mut out_high_bits = array_util::try_map(self.out_high_bits, |bit| index.get_mut(bit))?;

		let mut x_in = index.get_mut(self.x_in)?;
		let mut y_in = index.get_mut(self.y_in)?;
		let mut out_low = index.get_mut(self.out_low)?;
		let mut out_high = index.get_mut(self.out_high)?;

		for (i, (x, y)) in x_vals.into_iter().zip(y_vals.into_iter()).enumerate() {
			let (res_high, res_low) = UX::mul(x, y);
			set_packed_slice(&mut x_in, i, x);
			set_packed_slice(&mut y_in, i, y);
			set_packed_slice(&mut out_low, i, res_low);
			set_packed_slice(&mut out_high, i, res_high);

			for bit_idx in 0..BIT_LENGTH {
				if fill_input_bits {
					set_packed_slice(
						&mut x_in_bits[bit_idx],
						i,
						B1::from(UX::is_bit_set_at(x, bit_idx)),
					);
					set_packed_slice(
						&mut y_in_bits[bit_idx],
						i,
						B1::from(UX::is_bit_set_at(y, bit_idx)),
					);
				}
				set_packed_slice(
					&mut out_low_bits[bit_idx],
					i,
					B1::from(UX::is_bit_set_at(res_low, bit_idx)),
				);
				set_packed_slice(
					&mut out_high_bits[bit_idx],
					i,
					B1::from(UX::is_bit_set_at(res_high, bit_idx)),
				);
			}
		}

		Ok(())
	}

	pub fn populate_with_inputs<P>(
		&self,
		index: &mut TableWitnessSegment<P>,
		x_vals: impl IntoIterator<Item = FP>,
		y_vals: impl IntoIterator<Item = FP>,
	) -> Result<()>
	where
		P: PackedField<Scalar = B128> + PackedExtension<B1> + PackedExtension<FP>,
	{
		self.populate_internal(index, x_vals, y_vals, true)
	}

	pub fn populate<P>(
		&self,
		index: &mut TableWitnessSegment<P>,
		x_vals: impl IntoIterator<Item = FP>,
		y_vals: impl IntoIterator<Item = FP>,
	) -> Result<()>
	where
		P: PackedField<Scalar = B128> + PackedExtension<B1> + PackedExtension<FP>,
	{
		self.populate_internal(index, x_vals, y_vals, false)
	}
}

#[derive(Debug)]
pub struct MulUU32 {
	inner: Mul<u32, 32>,

	pub x_in: Col<B32>,
	pub y_in: Col<B32>,
	pub out_high: Col<B32>,
	pub out_low: Col<B32>,
	pub out_high_bits: [Col<B1>; 32],
	pub out_low_bits: [Col<B1>; 32],
}

impl MulUU32 {
	/// Constructor for `u32` multiplication gadget that creates the columns for inputs.
	/// You must call `MulUU32::populate` to fill the witness data.
	pub fn new(table: &mut TableBuilder) -> Self {
		let inner = Mul::new(table);

		Self {
			x_in: inner.x_in,
			y_in: inner.y_in,
			out_high: inner.out_high,
			out_low: inner.out_low,
			out_high_bits: inner.out_high_bits,
			out_low_bits: inner.out_low_bits,
			inner,
		}
	}

	/// Constructor for `u32` multiplication gadget that uses the provided columns for inputs.
	/// You must call `MulUU32::populate_with_inputs` to fill the witness data.
	pub fn with_inputs(
		table: &mut TableBuilder,
		x_in_bits: [Col<B1>; 32],
		y_in_bits: [Col<B1>; 32],
	) -> Self {
		let inner = Mul::with_inputs(table, x_in_bits, y_in_bits);

		Self {
			x_in: inner.x_in,
			y_in: inner.y_in,
			out_high: inner.out_high,
			out_low: inner.out_low,
			out_high_bits: inner.out_high_bits,
			out_low_bits: inner.out_low_bits,
			inner,
		}
	}

	pub fn populate_with_inputs<P>(
		&self,
		index: &mut TableWitnessSegment<P>,
		x_vals: impl IntoIterator<Item = B32>,
		y_vals: impl IntoIterator<Item = B32>,
	) -> Result<()>
	where
		P: PackedField<Scalar = B128> + PackedExtension<B1> + PackedExtension<B32>,
	{
		self.inner.populate_with_inputs(index, x_vals, y_vals)
	}

	pub fn populate<P>(
		&self,
		index: &mut TableWitnessSegment<P>,
		x_vals: impl IntoIterator<Item = B32>,
		y_vals: impl IntoIterator<Item = B32>,
	) -> Result<()>
	where
		P: PackedField<Scalar = B128> + PackedExtension<B1> + PackedExtension<B32>,
	{
		self.inner.populate(index, x_vals, y_vals)
	}
}

#[derive(Debug)]
pub struct MulUU64 {
	inner: Mul<u64, 64>,

	pub x_in: Col<B64>,
	pub y_in: Col<B64>,
	pub out_high: Col<B64>,
	pub out_low: Col<B64>,
	pub out_high_bits: [Col<B1>; 64],
	pub out_low_bits: [Col<B1>; 64],
}

impl MulUU64 {
	/// Constructor for `u64` multiplication gadget that creates the columns for inputs.
	/// You must call `MulUU64::populate` to fill the witness data.
	pub fn new(table: &mut TableBuilder) -> Self {
		let inner = Mul::new(table);

		Self {
			x_in: inner.x_in,
			y_in: inner.y_in,
			out_high: inner.out_high,
			out_low: inner.out_low,
			out_high_bits: inner.out_high_bits,
			out_low_bits: inner.out_low_bits,
			inner,
		}
	}

	/// Constructor for `u64` multiplication gadget that uses the provided columns for inputs.
	/// You must call `MulUU64::populate_with_inputs` to fill the witness data.
	pub fn with_inputs(
		table: &mut TableBuilder,
		x_in_bits: [Col<B1>; 64],
		y_in_bits: [Col<B1>; 64],
	) -> Self {
		let inner = Mul::with_inputs(table, x_in_bits, y_in_bits);

		Self {
			x_in: inner.x_in,
			y_in: inner.y_in,
			out_high: inner.out_high,
			out_low: inner.out_low,
			out_high_bits: inner.out_high_bits,
			out_low_bits: inner.out_low_bits,
			inner,
		}
	}

	pub fn populate_with_inputs<P>(
		&self,
		index: &mut TableWitnessSegment<P>,
		x_vals: impl IntoIterator<Item = B64>,
		y_vals: impl IntoIterator<Item = B64>,
	) -> Result<()>
	where
		P: PackedField<Scalar = B128> + PackedExtension<B1> + PackedExtension<B64>,
	{
		self.inner.populate_with_inputs(index, x_vals, y_vals)
	}

	pub fn populate<P>(
		&self,
		index: &mut TableWitnessSegment<P>,
		x_vals: impl IntoIterator<Item = B64>,
		y_vals: impl IntoIterator<Item = B64>,
	) -> Result<()>
	where
		P: PackedField<Scalar = B128> + PackedExtension<B1> + PackedExtension<B64>,
	{
		self.inner.populate(index, x_vals, y_vals)
	}
}

pub(crate) fn pack_fp<FP: TowerField, const BIT_LENGTH: usize>(
	bits: [Col<B1>; BIT_LENGTH],
) -> Expr<FP, 1> {
	assert_eq!(BIT_LENGTH, 1 << FP::TOWER_LEVEL);
	let basis: [_; BIT_LENGTH] = array::from_fn(ext_basis::<FP, B1>);
	bits.into_iter()
		.enumerate()
		.map(|(i, bit)| upcast_col(bit) * basis[i])
		.reduce(|a, b| a + b)
		.expect("bit has length checked above")
}

#[derive(Debug)]
pub struct MulSS32 {
	mul_inner: MulUU32,
	x_in_bits: [Col<B1>; 32],
	y_in_bits: [Col<B1>; 32],
	x_abs_value: SignConverter<u32, 32>,
	y_abs_value: SignConverter<u32, 32>,
	out_signed_value: SignConverter<u64, 64>,
	x_abs_bits: [Col<B1>; 32],
	y_abs_bits: [Col<B1>; 32],

	// Outputs
	pub out_bits: [Col<B1>; 64],
	pub x_in: Col<B32>,
	pub y_in: Col<B32>,
	pub out_high: Col<B32>,
	pub out_low: Col<B32>,
}

impl MulSS32 {
	pub fn new(table: &mut TableBuilder) -> Self {
		let x_in_bits = table.add_committed_multiple("x_in_bits");
		let y_in_bits = table.add_committed_multiple("y_in_bits");

		Self::with_input(table, x_in_bits, y_in_bits)
	}

	pub fn with_input(
		table: &mut TableBuilder,
		x_in_bits: [Col<B1>; 32],
		y_in_bits: [Col<B1>; 32],
	) -> Self {
		let x_in = table.add_computed("SS::x_in", pack_fp(x_in_bits));
		let y_in = table.add_computed("SS::y_in", pack_fp(y_in_bits));

		// Convert x and y to |x| and |y| via two's complement
		let x_is_negative = x_in_bits[31]; // Will be 1 if negative
		let y_is_negative = y_in_bits[31]; // Will be 1 if negative
		let x_abs_value = SignConverter::new(table, "x_abs_bits", x_in_bits, x_is_negative.into());
		let y_abs_value = SignConverter::new(table, "y_abs_bits", y_in_bits, y_is_negative.into());

		let x_abs_bits = x_abs_value.converted_bits;
		let y_abs_bits = y_abs_value.converted_bits;

		let mul_inner = MulUU32::with_inputs(table, x_abs_bits, y_abs_bits);

		let abs_mul_out_bits = array::from_fn(|i| {
			if i < 32 {
				mul_inner.out_low_bits[i]
			} else {
				mul_inner.out_high_bits[i - 32]
			}
		});

		let product_is_negative = x_in_bits[31] + y_in_bits[31]; // Will be 1 if the product is negative
		let out_signed_value =
			SignConverter::new(table, "out_bits", abs_mul_out_bits, product_is_negative);

		let out_bits = out_signed_value.converted_bits;
		let out_low_bits: [_; 32] = array::from_fn(|i| out_bits[i]);
		let out_high_bits: [_; 32] = array::from_fn(|i| out_bits[i + 32]);
		let out_high = table.add_computed("out_high", pack_fp(out_high_bits));
		let out_low = table.add_computed("out_low", pack_fp(out_low_bits));

		Self {
			x_abs_value,
			y_abs_value,
			out_signed_value,
			x_in_bits,
			y_in_bits,
			x_abs_bits,
			y_abs_bits,
			out_bits,
			mul_inner,
			x_in,
			y_in,
			out_low,
			out_high,
		}
	}

	pub fn populate_with_inputs<P>(
		&self,
		index: &mut TableWitnessSegment<P>,
		x_vals: impl IntoIterator<Item = B32>,
		y_vals: impl IntoIterator<Item = B32>,
	) -> Result<()>
	where
		P: PackedField<Scalar = B128> + PackedExtension<B1> + PackedExtension<B32>,
	{
		let mut inner_mul_x = Vec::new();
		let mut inner_mul_y = Vec::new();
		// For interior mutability we need scoped refs.
		{
			let mut x_in_bits = array_util::try_map(self.x_in_bits, |bit| index.get_mut(bit))?;
			let mut y_in_bits = array_util::try_map(self.y_in_bits, |bit| index.get_mut(bit))?;
			let mut out_bits = array_util::try_map(self.out_bits, |bit| index.get_mut(bit))?;
			let mut x_abs_bits = array_util::try_map(self.x_abs_bits, |bit| index.get_mut(bit))?;
			let mut y_abs_bits = array_util::try_map(self.y_abs_bits, |bit| index.get_mut(bit))?;

			let mut x_in = index.get_mut(self.x_in)?;
			let mut y_in = index.get_mut(self.y_in)?;
			let mut out_low = index.get_mut(self.out_low)?;
			let mut out_high = index.get_mut(self.out_high)?;

			for (i, (x, y)) in x_vals.into_iter().zip(y_vals.into_iter()).enumerate() {
				let x_i32 = x.val() as i32;
				let y_i32 = y.val() as i32;
				let x_abs = B32::new(abs(x_i32));
				let y_abs = B32::new(abs(y_i32));
				let res = x_i32 as i64 * y_i32 as i64;
				let res_high = B32::new((res >> 32) as u32);
				let res_low = B32::new(res as u32);
				inner_mul_x.push(x_abs);
				inner_mul_y.push(y_abs);
				set_packed_slice(&mut x_in, i, x);
				set_packed_slice(&mut y_in, i, y);
				set_packed_slice(&mut out_low, i, res_low);
				set_packed_slice(&mut out_high, i, res_high);

				for bit_idx in 0..32 {
					set_packed_slice(
						&mut x_abs_bits[bit_idx],
						i,
						B1::from(u32::is_bit_set_at(x_abs, bit_idx)),
					);
					set_packed_slice(
						&mut y_abs_bits[bit_idx],
						i,
						B1::from(u32::is_bit_set_at(y_abs, bit_idx)),
					);
					set_packed_slice(
						&mut x_in_bits[bit_idx],
						i,
						B1::from(u32::is_bit_set_at(x, bit_idx)),
					);
					set_packed_slice(
						&mut y_in_bits[bit_idx],
						i,
						B1::from(u32::is_bit_set_at(y, bit_idx)),
					);
					set_packed_slice(
						&mut out_bits[bit_idx + 32],
						i,
						B1::from(u32::is_bit_set_at(res_high, bit_idx)),
					);
					set_packed_slice(
						&mut out_bits[bit_idx],
						i,
						B1::from(u32::is_bit_set_at(res_low, bit_idx)),
					);
				}
			}
		}

		self.x_abs_value.populate(index)?;
		self.y_abs_value.populate(index)?;
		self.mul_inner.populate(index, inner_mul_x, inner_mul_y)?;
		self.out_signed_value.populate(index)?;

		Ok(())
	}
}

#[derive(Debug)]
pub struct MulSU32 {
	mul_inner: MulUU32,
	x_in_bits: [Col<B1>; 32],
	y_in_bits: [Col<B1>; 32],
	x_abs_value: SignConverter<u32, 32>,
	out_signed_value: SignConverter<u64, 64>,
	x_abs_bits: [Col<B1>; 32],
	out_bits: [Col<B1>; 64],

	pub x_in: Col<B32>,
	pub y_in: Col<B32>,
	pub out_high: Col<B32>,
	pub out_low: Col<B32>,
}

impl MulSU32 {
	pub fn new(table: &mut TableBuilder) -> Self {
		let x_in_bits = table.add_committed_multiple("x_in_bits");
		let y_in_bits = table.add_committed_multiple("y_in_bits");

		let x_in = table.add_computed("x_in", pack_fp(x_in_bits));
		let y_in = table.add_computed("y_in", pack_fp(y_in_bits));

		let x_is_negative = x_in_bits[31];
		let x_abs_value = SignConverter::new(table, "x_abs_bits", x_in_bits, x_is_negative.into());
		let x_abs_bits = x_abs_value.converted_bits;
		let mul_inner = MulUU32::with_inputs(table, x_abs_value.converted_bits, y_in_bits);
		let abs_mul_out_bits = array::from_fn(|i| {
			if i < 32 {
				mul_inner.out_low_bits[i]
			} else {
				mul_inner.out_high_bits[i - 32]
			}
		});

		let product_is_negative = x_in_bits[31];
		let out_signed_value =
			SignConverter::new(table, "out_bits", abs_mul_out_bits, product_is_negative.into());

		let out_bits = out_signed_value.converted_bits;
		let out_low_bits: [_; 32] = array::from_fn(|i| out_bits[i]);
		let out_high_bits: [_; 32] = array::from_fn(|i| out_bits[i + 32]);

		let out_high = table.add_computed("out_high", pack_fp(out_high_bits));
		let out_low = table.add_computed("out_low", pack_fp(out_low_bits));

		Self {
			mul_inner,
			x_in_bits,
			y_in_bits,
			x_abs_value,
			out_signed_value,
			x_abs_bits,
			out_bits,
			x_in,
			y_in,
			out_low,
			out_high,
		}
	}

	pub fn populate_with_inputs<P>(
		&self,
		index: &mut TableWitnessSegment<P>,
		x_vals: impl IntoIterator<Item = B32>,
		y_vals: impl IntoIterator<Item = B32>,
	) -> Result<()>
	where
		P: PackedField<Scalar = B128> + PackedExtension<B1> + PackedExtension<B32>,
	{
		let mut mul_inner_x = Vec::new();
		let mut mul_inner_y = Vec::new();
		{
			let mut x_in_bits = array_util::try_map(self.x_in_bits, |bit| index.get_mut(bit))?;
			let mut y_in_bits = array_util::try_map(self.y_in_bits, |bit| index.get_mut(bit))?;
			let mut out_bits = array_util::try_map(self.out_bits, |bit| index.get_mut(bit))?;
			let mut x_abs_bits = array_util::try_map(self.x_abs_bits, |bit| index.get_mut(bit))?;

			let mut x_in = index.get_mut(self.x_in)?;
			let mut y_in = index.get_mut(self.y_in)?;
			let mut out_low = index.get_mut(self.out_low)?;
			let mut out_high = index.get_mut(self.out_high)?;

			for (i, (x, y)) in x_vals.into_iter().zip(y_vals.into_iter()).enumerate() {
				let x_i32 = i32::from_le_bytes(x.val().to_le_bytes());
				let x_abs = B32::new(abs(x_i32));
				let res = x_i32 as i64 * y.val() as i64;
				let res_high = B32::new((res >> 32) as u32);
				let res_low = B32::new(res as u32);
				mul_inner_x.push(x_abs);
				mul_inner_y.push(y);
				set_packed_slice(&mut x_in, i, x);
				set_packed_slice(&mut y_in, i, y);
				set_packed_slice(&mut out_low, i, res_low);
				set_packed_slice(&mut out_high, i, res_high);

				for bit_idx in 0..32 {
					set_packed_slice(
						&mut x_abs_bits[bit_idx],
						i,
						B1::from(u32::is_bit_set_at(x_abs, bit_idx)),
					);
					set_packed_slice(
						&mut x_in_bits[bit_idx],
						i,
						B1::from(u32::is_bit_set_at(x, bit_idx)),
					);
					set_packed_slice(
						&mut y_in_bits[bit_idx],
						i,
						B1::from(u32::is_bit_set_at(y, bit_idx)),
					);
					set_packed_slice(
						&mut out_bits[bit_idx + 32],
						i,
						B1::from(u32::is_bit_set_at(res_high, bit_idx)),
					);
					set_packed_slice(
						&mut out_bits[bit_idx],
						i,
						B1::from(u32::is_bit_set_at(res_low, bit_idx)),
					);
				}
			}
		}

		self.x_abs_value.populate(index)?;
		self.mul_inner.populate(index, mul_inner_x, mul_inner_y)?;
		self.out_signed_value.populate(index)?;

		Ok(())
	}
}

/// Simple struct to convert to and from Two's complement representation based on bits. See [`SignConverter::new`]
///
/// NOTE: *We do not handle witness generation for the `converted_bits` and should be handled by caller*
#[derive(Debug)]
pub struct SignConverter<UPrimitive: UnsignedAddPrimitives, const BIT_LENGTH: usize> {
	twos_complement: TwosComplement<UPrimitive, BIT_LENGTH>,

	// Output,
	pub converted_bits: [Col<B1>; BIT_LENGTH],
}

impl<UPrimitive: UnsignedAddPrimitives, const BIT_LENGTH: usize>
	SignConverter<UPrimitive, BIT_LENGTH>
{
	/// Used to conditionally select bit representation based on the MSB bit (sign bit)
	///
	/// ## Parameters
	/// `name`: Name for the new column that will be created
	/// `in_bits`: The input bits from MSB to LSB
	/// `conditional`: The conditional bit to choose input bits, or it's two's complement
	///
	/// ## Example
	/// - If the conditional is zero, the output will be the input bits.
	/// - If the conditional is one, the output will be the two's complement of input bits.
	///
	pub fn new(
		table: &mut TableBuilder,
		name: &str,
		in_bits: [Col<B1>; BIT_LENGTH],
		conditional: Expr<B1, 1>,
	) -> Self {
		let twos_complement = TwosComplement::new(table, in_bits);
		let converted_bits = array::from_fn(|bit| {
			table.add_computed(
				format!("{name}[{bit}]"),
				twos_complement.result_bits[bit] * conditional.clone()
					+ (conditional.clone() + B1::ONE) * in_bits[bit],
			)
		});
		Self {
			twos_complement,
			converted_bits,
		}
	}

	pub fn populate<P>(&self, index: &mut TableWitnessSegment<P>) -> Result<()>
	where
		P: PackedField<Scalar = B128> + PackedExtension<B1>,
	{
		self.twos_complement.populate(index)
	}
}

#[derive(Debug)]
pub struct TwosComplement<UPrimitive: UnsignedAddPrimitives, const BIT_LENGTH: usize> {
	inverted: [Col<B1>; BIT_LENGTH],
	inner_incr: Incr<UPrimitive, BIT_LENGTH>,

	pub x_in: [Col<B1>; BIT_LENGTH],
	pub result_bits: [Col<B1>; BIT_LENGTH],
}

impl<UPrimitive: UnsignedAddPrimitives, const BIT_LENGTH: usize>
	TwosComplement<UPrimitive, BIT_LENGTH>
{
	pub fn new(table: &mut TableBuilder, x_in: [Col<B1>; BIT_LENGTH]) -> Self {
		let inverted = array::from_fn(|i| {
			table.add_computed(format!("TwosComplement::inverted[{i}]"), x_in[i] + B1::ONE)
		});
		let inner_incr = Incr::new(table, inverted);

		Self {
			inverted,
			result_bits: inner_incr.zout,
			inner_incr,
			x_in,
		}
	}

	pub fn populate<P>(&self, index: &mut TableWitnessSegment<P>) -> Result<(), anyhow::Error>
	where
		P: PackedField<Scalar = B128> + PackedExtension<B1>,
	{
		let one = PackedSubfield::<P, B1>::broadcast(B1::ONE);
		for (inverted, xin) in izip!(self.inverted.iter(), self.x_in.iter()) {
			let inp = index.get(*xin)?;
			let mut inverted = index.get_mut(*inverted)?;
			for (inp, value) in izip!(inp.iter(), inverted.iter_mut()) {
				*value = *inp + one;
			}
		}

		self.inner_incr.populate(index)?;

		Ok(())
	}
}

fn abs(x: i32) -> u32 {
	if x < 0 {
		(-x) as u32
	} else {
		x as u32
	}
}
