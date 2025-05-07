// Copyright 2025 Irreducible Inc.

use std::array;

use binius_field::{packed::set_packed_slice, Field, PackedExtension, PackedField};
use itertools::izip;

use crate::{
	builder::{Col, TableBuilder, TableWitnessSegment, B1, B128, B32, B64},
	gadgets::{
		mul::{MulSS32, MulUU32, SignConverter, UnsignedMulPrimitives},
		u32::{add::WideAdd, sub::WideSub, U32AddFlags, U32SubFlags},
		util::pack_fp,
	},
};

/// Gadget for unsigned division of two u32s.
///
/// `p = q*a + r`
#[derive(Debug)]
pub struct DivUU32 {
	mul_inner: MulUU32,
	sum: WideAdd<u64, 64>,
	sub: WideSub<u64, 64>,

	p_in_bits: [Col<B1>; 32],
	q_in_bits: [Col<B1>; 32],
	out_div_bits: [Col<B1>; 32],
	out_rem_bits: [Col<B1>; 32],

	pub p_in: Col<B32>,
	pub q_in: Col<B32>,
	pub out_div: Col<B32>,
	pub out_rem: Col<B32>,
}

impl DivUU32 {
	pub fn new(table: &mut TableBuilder) -> Self {
		let zero = table.add_constant("zero", [B1::ZERO]);
		let p_in_bits = table.add_committed_multiple("p_in_bits");
		let q_in_bits = table.add_committed_multiple("q_in_bits");

		let p_in = table.add_computed("p_in", pack_fp(p_in_bits));
		let q_in = table.add_computed("q_in", pack_fp(q_in_bits));

		let zero_extend_p: [_; 64] = array::from_fn(|i| if i < 32 { p_in_bits[i] } else { zero });

		let out_div_bits = table.add_committed_multiple("out_div_bits");
		let out_rem_bits = table.add_committed_multiple("out_rem_bits");

		let zero_extend_q: [_; 64] = array::from_fn(|i| if i < 32 { q_in_bits[i] } else { zero });
		let zero_extend_rem = array::from_fn(|i| if i < 32 { out_rem_bits[i] } else { zero });

		let out_div = table.add_computed("out_div", pack_fp(out_div_bits));
		let out_rem = table.add_computed("out_rem", pack_fp(out_rem_bits));

		let mul_inner = MulUU32::with_inputs(table, q_in_bits, out_div_bits);

		// Check q is non-zero
		table.assert_nonzero(q_in);

		let product_cols = array::from_fn(|i| {
			if i < 32 {
				mul_inner.out_low_bits[i]
			} else {
				mul_inner.out_high_bits[i - 32]
			}
		});

		// Check p = q * a + r in 64 bits
		let sum = WideAdd::<u64, 64>::new(
			table,
			product_cols,
			zero_extend_rem,
			U32AddFlags {
				commit_zout: true,
				expose_final_carry: true,
				..Default::default()
			},
		);

		#[allow(clippy::needless_range_loop)]
		for bit in 0..64 {
			table.assert_zero(
				format!("division_satisfied[{bit}]"),
				zero_extend_p[bit] - sum.z_out[bit],
			);
		}

		// Add constraint to make sure that r < q by computing s = r - q in a larger bit length.
		// There maybe a better way to do it with channels and simpler comparator logic.
		let mut inner_comparator = table.with_namespace("sign_comparator");
		let sub = WideSub::<u64, 64>::new(
			&mut inner_comparator,
			zero_extend_rem,
			zero_extend_q,
			U32SubFlags {
				commit_zout: true,
				..Default::default()
			},
		);
		// Check that the sign bit is set
		table.assert_zero("less_than", sub.zout[63] + B1::ONE);

		Self {
			mul_inner,
			sum,
			sub,

			p_in_bits,
			q_in_bits,
			out_div_bits,
			out_rem_bits,

			p_in,
			q_in,
			out_div,
			out_rem,
		}
	}

	pub fn populate_with_inputs<P>(
		&self,
		index: &mut TableWitnessSegment<P>,
		p_vals: impl IntoIterator<Item = B32>,
		q_vals: impl IntoIterator<Item = B32> + Clone,
	) -> anyhow::Result<()>
	where
		P: PackedField<Scalar = B128>
			+ PackedExtension<B1>
			+ PackedExtension<B32>
			+ PackedExtension<B64>,
	{
		let mut inner_div = Vec::new();
		{
			let mut p_in_bits = array_util::try_map(self.p_in_bits, |bit| index.get_mut(bit))?;
			let mut q_in_bits = array_util::try_map(self.q_in_bits, |bit| index.get_mut(bit))?;
			let mut out_div_bits =
				array_util::try_map(self.out_div_bits, |bit| index.get_mut(bit))?;
			let mut out_rem_bits =
				array_util::try_map(self.out_rem_bits, |bit| index.get_mut(bit))?;

			let mut p_in = index.get_mut(self.p_in)?;
			let mut q_in = index.get_mut(self.q_in)?;
			let mut out_div = index.get_mut(self.out_div)?;
			let mut out_rem = index.get_mut(self.out_rem)?;

			for (i, (p, q)) in izip!(p_vals, q_vals.clone()).enumerate() {
				let div = p.val() / q.val();
				let rem = p.val() % q.val();
				set_packed_slice(&mut p_in, i, p);
				set_packed_slice(&mut q_in, i, q);
				let div = B32::new(div);
				let rem = B32::new(rem);
				set_packed_slice(&mut out_div, i, div);
				set_packed_slice(&mut out_rem, i, rem);

				inner_div.push(div);

				for bit_idx in 0..32 {
					set_packed_slice(
						&mut p_in_bits[bit_idx],
						i,
						B1::from(u32::is_bit_set_at(p, bit_idx)),
					);
					set_packed_slice(
						&mut q_in_bits[bit_idx],
						i,
						B1::from(u32::is_bit_set_at(q, bit_idx)),
					);
					set_packed_slice(
						&mut out_div_bits[bit_idx],
						i,
						B1::from(u32::is_bit_set_at(div, bit_idx)),
					);
					set_packed_slice(
						&mut out_rem_bits[bit_idx],
						i,
						B1::from(u32::is_bit_set_at(rem, bit_idx)),
					);
				}
			}
		}

		self.mul_inner.populate(index, q_vals, inner_div)?;
		self.sum.populate(index)?;
		self.sub.populate(index)?;

		Ok(())
	}
}

/// Gadget for signed division of two i32s.
///
/// `p = q*a + r` where p and r have the same sign bit.
#[derive(Debug)]
pub struct DivSS32 {
	mul_inner: MulSS32,
	sum: WideAdd<u64, 64>,
	sub: WideAdd<u64, 64>,
	abs_r_value: SignConverter<u64, 64>,
	neg_abs_q_value: SignConverter<u64, 64>,
	abs_r_bits: [Col<B1>; 64],
	neg_abs_q_bits: [Col<B1>; 64],

	pub p_in_bits: [Col<B1>; 32],
	pub q_in_bits: [Col<B1>; 32],
	pub out_div_bits: [Col<B1>; 32],
	pub out_rem_bits: [Col<B1>; 32],

	pub p_in: Col<B32>,
	pub q_in: Col<B32>,
	pub out_div: Col<B32>,
	pub out_rem: Col<B32>,
}

impl DivSS32 {
	pub fn new(table: &mut TableBuilder) -> Self {
		let p_in_bits = table.add_committed_multiple("p_in_bits");
		let q_in_bits = table.add_committed_multiple("q_in_bits");

		let p_in = table.add_computed("p_in", pack_fp(p_in_bits));
		let q_in = table.add_computed("q_in", pack_fp(q_in_bits));

		let sign_extend_p: [_; 64] =
			array::from_fn(|i| if i < 32 { p_in_bits[i] } else { p_in_bits[31] });

		let out_div_bits = table.add_committed_multiple("out_div_bits");
		let out_rem_bits = table.add_committed_multiple("out_rem_bits");

		// Check sign(p) == sign(r)
		table.assert_zero("sign_dividend_eq_sign_rem", p_in_bits[31] - out_rem_bits[31]);

		let sign_extend_q: [_; 64] =
			array::from_fn(|i| if i < 32 { q_in_bits[i] } else { q_in_bits[31] });
		let sign_extend_rem = array::from_fn(|i| {
			if i < 32 {
				out_rem_bits[i]
			} else {
				out_rem_bits[31]
			}
		});

		let out_div = table.add_computed("out_div", pack_fp(out_div_bits));
		let out_rem = table.add_computed("out_rem", pack_fp(out_rem_bits));

		let mul_inner = MulSS32::with_input(table, q_in_bits, out_div_bits);

		// Check q is non-zero
		table.assert_nonzero(q_in);

		// Check p = q * a + r in 64 bits
		let sum = WideAdd::<u64, 64>::new(
			table,
			mul_inner.out_bits,
			sign_extend_rem,
			U32AddFlags {
				commit_zout: true,
				expose_final_carry: true,
				..Default::default()
			},
		);

		#[allow(clippy::needless_range_loop)]
		for bit in 0..64 {
			table.assert_zero(
				format!("division_satisfied_{bit}"),
				sign_extend_p[bit] - sum.z_out[bit],
			);
		}

		// Add constraint to make sure that |r| < |q| by computing s = |r| - |q| in a larger bit length.
		// There maybe a better way to do it with channels and simpler comparator logic.
		let r_is_negative = out_rem_bits[31];
		let mut inner_abs_rem_table = table.with_namespace("rem_abs_value");
		let abs_r_value =
			SignConverter::new(&mut inner_abs_rem_table, sign_extend_rem, r_is_negative.into());
		let abs_r_bits = abs_r_value.converted_bits;

		let q_is_positive = q_in_bits[31] + B1::ONE;
		let mut inner_neg_abs_q_table = table.with_namespace("neg_abs_q");
		let neg_abs_q_value =
			SignConverter::new(&mut inner_neg_abs_q_table, sign_extend_q, q_is_positive);
		let neg_abs_q_bits = neg_abs_q_value.converted_bits;

		let sub = WideAdd::<u64, 64>::new(
			table,
			abs_r_bits,
			neg_abs_q_bits,
			U32AddFlags {
				commit_zout: true,
				..Default::default()
			},
		);
		// Check that the sign bit is set
		table.assert_zero("less_than", sub.z_out[63] + B1::ONE);

		Self {
			mul_inner,
			sum,
			sub,
			abs_r_value,
			neg_abs_q_value,

			abs_r_bits,
			neg_abs_q_bits,
			p_in_bits,
			q_in_bits,
			out_div_bits,
			out_rem_bits,

			p_in,
			q_in,
			out_div,
			out_rem,
		}
	}

	pub fn populate_with_inputs<P>(
		&self,
		index: &mut TableWitnessSegment<P>,
		p_vals: impl IntoIterator<Item = B32>,
		q_vals: impl IntoIterator<Item = B32> + Clone,
	) -> anyhow::Result<()>
	where
		P: PackedField<Scalar = B128> + PackedExtension<B1> + PackedExtension<B32>,
	{
		// This vector holds the witness data for the inner multiplication gadget
		let mut inner_div = Vec::new();

		{
			let mut p_in_bits = array_util::try_map(self.p_in_bits, |bit| index.get_mut(bit))?;
			let mut q_in_bits = array_util::try_map(self.q_in_bits, |bit| index.get_mut(bit))?;
			let mut out_div_bits =
				array_util::try_map(self.out_div_bits, |bit| index.get_mut(bit))?;
			let mut out_rem_bits =
				array_util::try_map(self.out_rem_bits, |bit| index.get_mut(bit))?;
			let mut abs_r_bits = array_util::try_map(self.abs_r_bits, |bit| index.get_mut(bit))?;
			let mut neg_abs_q_bits =
				array_util::try_map(self.neg_abs_q_bits, |bit| index.get_mut(bit))?;

			let mut p_in = index.get_mut(self.p_in)?;
			let mut q_in = index.get_mut(self.q_in)?;
			let mut out_div = index.get_mut(self.out_div)?;
			let mut out_rem = index.get_mut(self.out_rem)?;

			for (i, (p, q)) in izip!(p_vals, q_vals.clone()).enumerate() {
				let p_i32 = p.val() as i32;
				let q_i32 = q.val() as i32;
				let div = p_i32 / q_i32;
				let rem = p_i32 % q_i32;
				let abs_rem_b64 = if rem < 0 {
					B64::new((-rem) as u64)
				} else {
					B64::new(rem as u64)
				};
				let neg_abs_q_b64 = if q_i32 < 0 {
					B64::new(q_i32 as i64 as u64)
				} else {
					B64::new((-q_i32) as i64 as u64)
				};
				set_packed_slice(&mut p_in, i, p);
				set_packed_slice(&mut q_in, i, q);
				let div = B32::new(div as u32);
				let rem = B32::new(rem as u32);
				set_packed_slice(&mut out_div, i, div);
				set_packed_slice(&mut out_rem, i, rem);
				inner_div.push(div);

				for bit_idx in 0..32 {
					set_packed_slice(
						&mut p_in_bits[bit_idx],
						i,
						B1::from(u32::is_bit_set_at(p, bit_idx)),
					);
					set_packed_slice(
						&mut q_in_bits[bit_idx],
						i,
						B1::from(u32::is_bit_set_at(q, bit_idx)),
					);
					set_packed_slice(
						&mut out_div_bits[bit_idx],
						i,
						B1::from(u32::is_bit_set_at(div, bit_idx)),
					);
					set_packed_slice(
						&mut out_rem_bits[bit_idx],
						i,
						B1::from(u32::is_bit_set_at(rem, bit_idx)),
					);
				}
				for bit_idx in 0..64 {
					set_packed_slice(
						&mut abs_r_bits[bit_idx],
						i,
						B1::from(u64::is_bit_set_at(abs_rem_b64, bit_idx)),
					);
					set_packed_slice(
						&mut neg_abs_q_bits[bit_idx],
						i,
						B1::from(u64::is_bit_set_at(neg_abs_q_b64, bit_idx)),
					);
				}
			}
		}

		self.mul_inner
			.populate_with_inputs(index, q_vals, inner_div)?;
		self.sum.populate(index)?;
		self.abs_r_value.populate(index)?;
		self.neg_abs_q_value.populate(index)?;
		self.sub.populate(index)?;

		Ok(())
	}
}
