// Copyright 2025 Irreducible Inc.

use std::{array, marker::PhantomData, ops::MulAssign};

use anyhow::Result;
use binius_field::{
	ext_basis, packed::set_packed_slice, BinaryField, ExtensionField, Field, PackedExtension,
	PackedField, TowerField,
};

use crate::builder::{
	upcast_col, Col, Expr, TableBuilder, TableWitnessSegment, B1, B128, B32, B64,
};

// TODO: Should we add u8 and u16 as well?

trait UnsignedMulPrimitives {
	type FP: TowerField;
	type FPExt: TowerField + ExtensionField<Self::FP>;
	fn get_bit_length() -> usize;

	// Does primitive mul of x*y = z => (z_high, z_low)
	fn mul(x: Self::FP, y: Self::FP) -> (Self::FP, Self::FP);

	fn is_bit_set_at(a: Self::FP, index: usize) -> bool;

	fn get_generator() -> Self::FPExt;

	// Returns the generator shifted by the bit length of `Self::FP`.
	fn get_shifted_generator() -> Self::FPExt;
}

impl UnsignedMulPrimitives for u32 {
	type FP = B32;
	type FPExt = B64;

	fn get_bit_length() -> usize {
		32
	}

	fn mul(x: B32, y: B32) -> (B32, B32) {
		let res = x.val() as u64 * y.val() as u64;
		let low = B32::new(res as u32);
		let high = B32::new((res >> 32) as u32);
		(high, low)
	}

	fn is_bit_set_at(a: B32, index: usize) -> bool {
		((a.val() >> index) & 1) == 1
	}

	fn get_generator() -> B64 {
		B64::MULTIPLICATIVE_GENERATOR
	}

	fn get_shifted_generator() -> B64 {
		B64::MULTIPLICATIVE_GENERATOR.pow(1 << Self::get_bit_length())
	}
}

impl UnsignedMulPrimitives for u64 {
	type FP = B64;
	type FPExt = B128;

	fn get_bit_length() -> usize {
		64
	}

	fn mul(x: B64, y: B64) -> (B64, B64) {
		let res = x.val() as u128 * y.val() as u128;
		let low = B64::new(res as u64);
		let high = B64::new((res >> 64) as u64);
		(high, low)
	}

	fn is_bit_set_at(a: B64, index: usize) -> bool {
		((a.val() >> index) & 1) == 1
	}

	fn get_generator() -> Self::FPExt {
		B128::MULTIPLICATIVE_GENERATOR
	}

	fn get_shifted_generator() -> B128 {
		let base = Self::get_generator();
		let exp: u128 = 1 << Self::get_bit_length();
		let mut res = B128::one();
		for i in (0..128).rev() {
			res = res.square();
			if ((exp >> i) & 1) == 1 {
				res.mul_assign(base)
			}
		}
		res
	}
}

// Internally used to have deduplicated implementations.
#[derive(Debug)]
struct Mul<
	FPExt: TowerField,
	FP: TowerField,
	UX: UnsignedMulPrimitives<FP = FP, FPExt = FPExt>,
	const BIT_LENGTH: usize,
> {
	x_in_bits: [Col<B1>; BIT_LENGTH],
	y_in_bits: [Col<B1>; BIT_LENGTH],
	out_high_bits: [Col<B1>; BIT_LENGTH],
	out_low_bits: [Col<B1>; BIT_LENGTH],

	pub x_in: Col<FP>,
	pub y_in: Col<FP>,
	pub out_high: Col<FP>,
	pub out_low: Col<FP>,

	_marker: PhantomData<(FPExt, UX)>,
}

impl<
		FPExt: TowerField,
		FP: TowerField,
		UX: UnsignedMulPrimitives<FP = FP, FPExt = FPExt>,
		const BIT_LENGTH: usize,
	> Mul<FPExt, FP, UX, BIT_LENGTH>
where
	FPExt: ExtensionField<FP> + ExtensionField<B1>,
	B128: ExtensionField<FPExt> + ExtensionField<FP> + ExtensionField<B1>,
{
	// TODO: Figure out if we need to create a separate table for this gadget.

	pub fn new(table: &mut TableBuilder, name: impl ToString) -> Self {
		assert_eq!(FPExt::TOWER_LEVEL, FP::TOWER_LEVEL + 1);
		assert_eq!(BIT_LENGTH, 1 << FP::TOWER_LEVEL);
		assert_eq!(BIT_LENGTH, UX::get_bit_length());
		// These are currently the only bit lengths I've tested
		assert!(BIT_LENGTH == 32 || BIT_LENGTH == 64);

		table.with_namespace(format!("{}::U{}", name.to_string(), BIT_LENGTH));
		let x_in_bits = array::from_fn(|i| table.add_committed(format!("x_in_bits[{i}]")));
		let y_in_bits = array::from_fn(|i| table.add_committed(format!("y_in_bits[{i}]")));

		let x_in = table.add_computed("x_in", pack_fp(x_in_bits));
		let y_in = table.add_computed("y_in", pack_fp(y_in_bits));

		let generator = UX::get_generator().into();
		let generator_pow_bit_len = UX::get_shifted_generator().into();

		let g_pow_x: Col<FPExt> = table.add_static_exp("g^x", &x_in_bits, generator);
		let g_pow_xy: Col<FPExt> = table.add_dynamic_exp("(g^x)^y", &y_in_bits, g_pow_x);

		let out_high_bits = array::from_fn(|i| table.add_committed(format!("out_high[{i}]")));
		let out_low_bits = array::from_fn(|i| table.add_committed(format!("out_low[{i}]")));

		let out_high = table.add_computed("out_high", pack_fp(out_high_bits));
		let out_low = table.add_computed("out_low", pack_fp(out_low_bits));

		let g_pow_out_low: Col<FPExt> =
			table.add_static_exp("g^(out_low)", &out_low_bits, generator);
		let g_pow_out_high: Col<FPExt> = table.add_static_exp(
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

	pub fn populate_with_inputs<P>(
		&self,
		index: &mut TableWitnessSegment<P>,
		x_vals: impl IntoIterator<Item = FP>,
		y_vals: impl IntoIterator<Item = FP>,
	) -> Result<()>
	where
		P: PackedField<Scalar = B128> + PackedExtension<B1> + PackedExtension<FP>,
	{
		let mut x_in_bits = self
			.x_in_bits
			.iter()
			.map(|bit| index.get_mut(*bit))
			.collect::<Result<Vec<_>, _>>()?;
		let mut y_in_bits = self
			.y_in_bits
			.iter()
			.map(|bit| index.get_mut(*bit))
			.collect::<Result<Vec<_>, _>>()?;
		let mut out_low_bits = self
			.out_low_bits
			.iter()
			.map(|bit| index.get_mut(*bit))
			.collect::<Result<Vec<_>, _>>()?;
		let mut out_high_bits = self
			.out_high_bits
			.iter()
			.map(|bit| index.get_mut(*bit))
			.collect::<Result<Vec<_>, _>>()?;

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
				if UX::is_bit_set_at(x, bit_idx) {
					set_packed_slice(&mut x_in_bits[bit_idx], i, B1::ONE);
				}
				if UX::is_bit_set_at(y, bit_idx) {
					set_packed_slice(&mut y_in_bits[bit_idx], i, B1::ONE);
				}
				if UX::is_bit_set_at(res_low, bit_idx) {
					set_packed_slice(&mut out_low_bits[bit_idx], i, B1::ONE);
				}
				if UX::is_bit_set_at(res_high, bit_idx) {
					set_packed_slice(&mut out_high_bits[bit_idx], i, B1::ONE);
				}
			}
		}

		Ok(())
	}
}

#[derive(Debug)]
pub struct MulUU32 {
	inner: Mul<B64, B32, u32, 32>,

	pub x_in: Col<B32>,
	pub y_in: Col<B32>,
	pub out_high: Col<B32>,
	pub out_low: Col<B32>,
}

impl MulUU32 {
	pub fn new(table: &mut TableBuilder, name: impl ToString) -> Self {
		let inner = Mul::new(table, name);

		Self {
			x_in: inner.x_in,
			y_in: inner.y_in,
			out_high: inner.out_high,
			out_low: inner.out_low,
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
}

#[derive(Debug)]
pub struct MulUU64 {
	inner: Mul<B128, B64, u64, 64>,

	pub x_in: Col<B64>,
	pub y_in: Col<B64>,
	pub out_high: Col<B64>,
	pub out_low: Col<B64>,
}

impl MulUU64 {
	pub fn new(table: &mut TableBuilder, name: impl ToString) -> Self {
		let inner = Mul::new(table, name);

		Self {
			x_in: inner.x_in,
			y_in: inner.y_in,
			out_high: inner.out_high,
			out_low: inner.out_low,
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
}

fn pack_fp<FP: TowerField, const BIT_LENGTH: usize>(bits: [Col<B1>; BIT_LENGTH]) -> Expr<FP, 1> {
	assert_eq!(BIT_LENGTH, 1 << FP::TOWER_LEVEL);
	let basis: [_; BIT_LENGTH] = array::from_fn(ext_basis::<FP, B1>);
	bits.into_iter()
		.enumerate()
		.map(|(i, bit)| upcast_col(bit) * basis[i])
		.reduce(|a, b| a + b)
		.expect("bit has length checked above")
}
