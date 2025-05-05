// Copyright 2025 Irreducible Inc.

use std::{array, marker::PhantomData};

use anyhow::Result;
use binius_field::{
	packed::set_packed_slice, BinaryField, ExtensionField, Field, PackedExtension, PackedField,
	PackedSubfield, TowerField,
};
use itertools::izip;

use crate::{
	builder::{Col, Expr, TableBuilder, TableWitnessSegment, B1, B128, B32, B64},
	gadgets::{
		u32::{
			add::{Incr, UnsignedAddPrimitives},
			sub::WideU32Sub,
			U32SubFlags,
		},
		util::pack_fp,
	},
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
			g = g.square();
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
			g = g.square();
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

	pub xin: Col<UX::FP>,
	pub yin: Col<UX::FP>,
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
		xin_bits: [Col<B1>; BIT_LENGTH],
		yin_bits: [Col<B1>; BIT_LENGTH],
	) -> Self {
		assert_eq!(FExpBase::TOWER_LEVEL, FP::TOWER_LEVEL + 1);
		assert_eq!(BIT_LENGTH, 1 << FP::TOWER_LEVEL);
		assert_eq!(BIT_LENGTH, UX::BIT_LENGTH);
		// These are currently the only bit lengths I've tested
		assert!(BIT_LENGTH == 32 || BIT_LENGTH == 64);

		let x_in = table.add_computed("x_in", pack_fp(xin_bits));
		let y_in = table.add_computed("y_in", pack_fp(yin_bits));

		let generator = UX::generator().into();
		let generator_pow_bit_len = UX::shifted_generator().into();

		let g_pow_x = table.add_static_exp::<FExpBase>("g^x", &xin_bits, generator);
		let g_pow_xy = table.add_dynamic_exp::<FExpBase>("(g^x)^y", &yin_bits, g_pow_x);

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

		table.assert_zero("order_non_wrapping", xin_bits[0] * yin_bits[0] - out_low_bits[0]);
		table.assert_zero("exponentiation_equality", g_pow_xy - g_pow_out_low * g_pow_out_high);

		Self {
			x_in_bits: xin_bits,
			y_in_bits: yin_bits,
			out_high_bits,
			out_low_bits,
			xin: x_in,
			yin: y_in,
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

		let mut x_in = index.get_mut(self.xin)?;
		let mut y_in = index.get_mut(self.yin)?;
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

	pub xin: Col<B32>,
	pub yin: Col<B32>,
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
			xin: inner.xin,
			yin: inner.yin,
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
		xin_bits: [Col<B1>; 32],
		yin_bits: [Col<B1>; 32],
	) -> Self {
		let inner = Mul::with_inputs(table, xin_bits, yin_bits);

		Self {
			xin: inner.xin,
			yin: inner.yin,
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

	pub xin: Col<B64>,
	pub yin: Col<B64>,
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
			xin: inner.xin,
			yin: inner.yin,
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
		xin_bits: [Col<B1>; 64],
		yin_bits: [Col<B1>; 64],
	) -> Self {
		let inner = Mul::with_inputs(table, xin_bits, yin_bits);

		Self {
			xin: inner.xin,
			yin: inner.yin,
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

#[derive(Debug)]
pub struct MulSS32 {
	mul_inner: MulUU32,
	x_in_bits: [Col<B1>; 32],
	y_in_bits: [Col<B1>; 32],
	y_sub: WideU32Sub,
	new_prod_high_bits: [Col<B1>; 32],
	x_sub: WideU32Sub,

	// Outputs
	pub out_bits: [Col<B1>; 64],
	pub xin: Col<B32>,
	pub yin: Col<B32>,
	pub out_high: Col<B32>,
	pub out_low: Col<B32>,
}

impl MulSS32 {
	/// Create the gadget by automatically committing the required input columns.
	pub fn new(table: &mut TableBuilder) -> Self {
		let x_in_bits = table.add_committed_multiple("x_in_bits");
		let y_in_bits = table.add_committed_multiple("y_in_bits");

		Self::with_input(table, x_in_bits, y_in_bits)
	}

	/// Create the gadget with the supplied `xin_bits` and `yin_bits` columns.
	pub fn with_input(
		table: &mut TableBuilder,
		xin_bits: [Col<B1>; 32],
		yin_bits: [Col<B1>; 32],
	) -> Self {
		let xin = table.add_computed("x_in", pack_fp(xin_bits));
		let yin = table.add_computed("y_in", pack_fp(yin_bits));

		let x_is_negative = xin_bits[31]; // Will be 1 if negative
		let y_is_negative = yin_bits[31]; // Will be 1 if negative

		let mut inner_mul_table = table.with_namespace("MulUU32");
		let mul_inner = MulUU32::with_inputs(&mut inner_mul_table, xin_bits, yin_bits);

		let out_low_bits: [_; 32] = array::from_fn(|i| mul_inner.out_low_bits[i]);

		let prod_high_bits: [_; 32] = array::from_fn(|i| mul_inner.out_high_bits[i]);

		let mut inner_y_sub_table = table.with_namespace("X_less_than_zero");
		let y_sub = WideU32Sub::new(
			&mut inner_y_sub_table,
			prod_high_bits,
			yin_bits,
			U32SubFlags {
				commit_zout: true,
				..Default::default()
			},
		);

		let new_prod_high_bits = array::from_fn(|bit| {
			table.add_computed(
				format!("new_prod_high[{bit}]"),
				prod_high_bits[bit] + x_is_negative * (prod_high_bits[bit] + y_sub.zout[bit]),
			)
		});

		let mut inner_x_sub_table = table.with_namespace("Y_less_than_zero");
		let x_sub = WideU32Sub::new(
			&mut inner_x_sub_table,
			new_prod_high_bits,
			xin_bits,
			U32SubFlags {
				commit_zout: true,
				..Default::default()
			},
		);

		let out_high_bits: [_; 32] = array::from_fn(|bit| {
			table.add_computed(
				format!("out_high[{bit}]"),
				new_prod_high_bits[bit]
					+ y_is_negative * (new_prod_high_bits[bit] + x_sub.zout[bit]),
			)
		});

		let out_high = table.add_computed("out_high", pack_fp(out_high_bits));
		let out_low = table.add_computed("out_low", pack_fp(out_low_bits));
		let out_bits: [_; 64] = array::from_fn(|i| {
			if i < 32 {
				out_low_bits[i]
			} else {
				out_high_bits[i - 32]
			}
		});

		Self {
			y_sub,
			new_prod_high_bits,
			x_sub,
			x_in_bits: xin_bits,
			y_in_bits: yin_bits,
			out_bits,
			mul_inner,
			xin,
			yin,
			out_low,
			out_high,
		}
	}

	pub fn populate_with_inputs<P>(
		&self,
		index: &mut TableWitnessSegment<P>,
		x_vals: impl IntoIterator<Item = B32> + Clone,
		y_vals: impl IntoIterator<Item = B32> + Clone,
	) -> Result<()>
	where
		P: PackedField<Scalar = B128> + PackedExtension<B1> + PackedExtension<B32>,
	{
		// For interior mutability we need scoped refs.
		{
			let mut x_in_bits = array_util::try_map(self.x_in_bits, |bit| index.get_mut(bit))?;
			let mut y_in_bits = array_util::try_map(self.y_in_bits, |bit| index.get_mut(bit))?;
			let mut out_bits = array_util::try_map(self.out_bits, |bit| index.get_mut(bit))?;
			let mut new_prod_high_bits =
				array_util::try_map(self.new_prod_high_bits, |bit| index.get_mut(bit))?;

			let mut x_in = index.get_mut(self.xin)?;
			let mut y_in = index.get_mut(self.yin)?;
			let mut out_low = index.get_mut(self.out_low)?;
			let mut out_high = index.get_mut(self.out_high)?;

			for (i, (x, y)) in x_vals
				.clone()
				.into_iter()
				.zip(y_vals.clone().into_iter())
				.enumerate()
			{
				let res = x.val() as u64 * y.val() as u64;
				let prod_hi = B32::new((res >> 32) as u32);
				let prod_lo = B32::new(res as u32);
				set_packed_slice(&mut x_in, i, x);
				set_packed_slice(&mut y_in, i, y);
				set_packed_slice(&mut out_low, i, prod_lo);
				let new_prod_hi = if (x.val() as i32) < 0 {
					B32::new(prod_hi.val().wrapping_sub(y.val()))
				} else {
					prod_hi
				};
				let out_hi = if (y.val() as i32) < 0 {
					B32::new(new_prod_hi.val().wrapping_sub(x.val()))
				} else {
					new_prod_hi
				};
				set_packed_slice(&mut out_high, i, out_hi);

				for bit_idx in 0..32 {
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
						B1::from(u32::is_bit_set_at(out_hi, bit_idx)),
					);
					set_packed_slice(
						&mut new_prod_high_bits[bit_idx],
						i,
						B1::from(u32::is_bit_set_at(new_prod_hi, bit_idx)),
					);
				}
			}
		}

		self.mul_inner.populate(index, x_vals, y_vals)?;
		self.y_sub.populate(index)?;
		self.x_sub.populate(index)?;

		Ok(())
	}
}

/// A gadget that computes Signed x Unsigned multiplication with the full 64-bit signed result
#[derive(Debug)]
pub struct MulSU32 {
	mul_inner: MulUU32,
	x_in_bits: [Col<B1>; 32],
	y_in_bits: [Col<B1>; 32],
	out_high_bits: [Col<B1>; 32],
	y_sub: WideU32Sub,

	/// Output columns
	pub xin: Col<B32>,
	pub yin: Col<B32>,
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
		let mul_inner = MulUU32::with_inputs(table, x_in_bits, y_in_bits);
		let prod_high_bits: [_; 32] = array::from_fn(|i| mul_inner.out_high_bits[i]);

		let mut inner_y_sub_table = table.with_namespace("X_less_than_zero");
		let y_sub = WideU32Sub::new(
			&mut inner_y_sub_table,
			prod_high_bits,
			y_in_bits,
			U32SubFlags {
				commit_zout: true,
				..Default::default()
			},
		);
		let out_high_bits = array::from_fn(|bit| {
			table.add_computed(
				format!("out_high[{bit}]"),
				prod_high_bits[bit] + x_is_negative * (prod_high_bits[bit] + y_sub.zout[bit]),
			)
		});

		let out_low_bits: [_; 32] = array::from_fn(|i| mul_inner.out_low_bits[i]);

		let out_high = table.add_computed("out_high", pack_fp(out_high_bits));
		let out_low = table.add_computed("out_low", pack_fp(out_low_bits));

		Self {
			mul_inner,
			x_in_bits,
			y_in_bits,
			y_sub,
			out_high_bits,
			xin: x_in,
			yin: y_in,
			out_low,
			out_high,
		}
	}

	pub fn populate_with_inputs<P>(
		&self,
		index: &mut TableWitnessSegment<P>,
		x_vals: impl IntoIterator<Item = B32> + Clone,
		y_vals: impl IntoIterator<Item = B32> + Clone,
	) -> Result<()>
	where
		P: PackedField<Scalar = B128> + PackedExtension<B1> + PackedExtension<B32>,
	{
		{
			let mut x_in_bits = array_util::try_map(self.x_in_bits, |bit| index.get_mut(bit))?;
			let mut y_in_bits = array_util::try_map(self.y_in_bits, |bit| index.get_mut(bit))?;
			let mut out_high_bits =
				array_util::try_map(self.out_high_bits, |bit| index.get_mut(bit))?;

			let mut x_in = index.get_mut(self.xin)?;
			let mut y_in = index.get_mut(self.yin)?;
			let mut out_low = index.get_mut(self.out_low)?;
			let mut out_high = index.get_mut(self.out_high)?;

			for (i, (x, y)) in x_vals
				.clone()
				.into_iter()
				.zip(y_vals.clone().into_iter())
				.enumerate()
			{
				let res = x.val() as u64 * y.val() as u64;
				let prod_hi = B32::new((res >> 32) as u32);
				let prod_lo = B32::new(res as u32);
				set_packed_slice(&mut x_in, i, x);
				set_packed_slice(&mut y_in, i, y);
				set_packed_slice(&mut out_low, i, prod_lo);
				let out_hi = if (x.val() as i32) < 0 {
					B32::new(prod_hi.val().wrapping_sub(y.val()))
				} else {
					prod_hi
				};
				set_packed_slice(&mut out_high, i, out_hi);

				for bit_idx in 0..32 {
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
						&mut out_high_bits[bit_idx],
						i,
						B1::from(u32::is_bit_set_at(out_hi, bit_idx)),
					);
				}
			}
		}

		self.mul_inner.populate(index, x_vals, y_vals)?;
		self.y_sub.populate(index)?;

		Ok(())
	}
}

/// Simple struct to convert to and from Two's complement representation based on bits. See [`SignConverter::new`]
///
/// NOTE: *We do not handle witness generation for the `converted_bits` and should be handled by caller*
#[derive(Debug)]
pub struct SignConverter<UPrimitive: UnsignedAddPrimitives, const BIT_LENGTH: usize> {
	twos_complement: TwosComplement<UPrimitive, BIT_LENGTH>,

	// Output columns
	pub converted_bits: [Col<B1>; BIT_LENGTH],
}

impl<UPrimitive: UnsignedAddPrimitives, const BIT_LENGTH: usize>
	SignConverter<UPrimitive, BIT_LENGTH>
{
	/// Used to conditionally select bit representation based on the MSB bit (sign bit)
	///
	/// ## Parameters
	/// * `in_bits`: The input bits from MSB to LSB
	/// * `conditional`: The conditional bit to choose input bits, or it's two's complement
	///
	/// ## Example
	/// - If the conditional is zero, the output will be the input bits.
	/// - If the conditional is one, the output will be the two's complement of input bits.
	///
	pub fn new(
		table: &mut TableBuilder,
		xin: [Col<B1>; BIT_LENGTH],
		conditional: Expr<B1, 1>,
	) -> Self {
		let twos_complement = TwosComplement::new(table, xin);
		let converted_bits = array::from_fn(|bit| {
			table.add_computed(
				format!("converted_bits[{bit}]"),
				twos_complement.result_bits[bit] * conditional.clone()
					+ (conditional.clone() + B1::ONE) * xin[bit],
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

/// Simple gadget that's used to convert to and from two's complement binary representations
#[derive(Debug)]
pub struct TwosComplement<UPrimitive: UnsignedAddPrimitives, const BIT_LENGTH: usize> {
	inverted: [Col<B1>; BIT_LENGTH],
	inner_incr: Incr<UPrimitive, BIT_LENGTH>,

	// Input columns
	pub xin: [Col<B1>; BIT_LENGTH],
	// Output columns
	pub result_bits: [Col<B1>; BIT_LENGTH],
}

impl<UPrimitive: UnsignedAddPrimitives, const BIT_LENGTH: usize>
	TwosComplement<UPrimitive, BIT_LENGTH>
{
	pub fn new(table: &mut TableBuilder, xin: [Col<B1>; BIT_LENGTH]) -> Self {
		let inverted =
			array::from_fn(|i| table.add_computed(format!("inverted[{i}]"), xin[i] + B1::ONE));
		let mut inner_table = table.with_namespace("Increment");
		let inner_incr = Incr::new(&mut inner_table, inverted);

		Self {
			inverted,
			result_bits: inner_incr.zout,
			inner_incr,
			xin,
		}
	}

	pub fn populate<P>(&self, index: &mut TableWitnessSegment<P>) -> Result<(), anyhow::Error>
	where
		P: PackedField<Scalar = B128> + PackedExtension<B1>,
	{
		let one = PackedSubfield::<P, B1>::broadcast(B1::ONE);
		for (inverted, xin) in izip!(self.inverted.iter(), self.xin.iter()) {
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
