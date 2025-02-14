// Copyright 2024-2025 Irreducible Inc.

use binius_maybe_rayon::prelude::*;
use bytemuck::{must_cast_slice, must_cast_slice_mut};

use crate::constraint_system::{Filler, OracleId, U};

pub fn u32_add_carry_out(_xin: OracleId, _yin: OracleId) -> Filler {
	Filler::new(|inputs, output| {
		let xin = must_cast_slice::<U, u32>(inputs[0]);
		let yin = must_cast_slice::<U, u32>(inputs[1]);
		let cout = must_cast_slice_mut::<U, u32>(output);
		(xin, yin, cout)
			.into_par_iter()
			.for_each(|(xin, yin, cout)| {
				let (zout, carry) = xin.overflowing_add(*yin);
				let cin = xin ^ yin ^ zout;
				*cout = ((carry as u32) << 31) | (cin >> 1);
			})
	})
}

pub fn u32_sub_carry_out(_zin: OracleId, _yin: OracleId) -> Filler {
	Filler::new(|inputs, output| {
		let zin = must_cast_slice::<U, u32>(inputs[0]);
		let yin = must_cast_slice::<U, u32>(inputs[1]);
		let zout = must_cast_slice_mut::<U, u32>(output);
		(zin, yin, zout)
			.into_par_iter()
			.for_each(|(zin, yin, cout)| {
				let (xin, carry) = zin.overflowing_sub(*yin);
				let cin = xin ^ yin ^ zin;
				*cout = ((carry as u32) << 31) | (cin >> 1);
			})
	})
}

pub fn u32_sum_with_cin(_xin: OracleId, _yin: OracleId, _cin: OracleId) -> Filler {
	Filler::new(|inputs, output| {
		let xin = must_cast_slice::<U, u32>(inputs[0]);
		let yin = must_cast_slice::<U, u32>(inputs[1]);
		let cin = must_cast_slice::<U, u32>(inputs[2]);
		let zout = must_cast_slice_mut::<U, u32>(output);
		(xin, yin, cin, zout)
			.into_par_iter()
			.for_each(|(xin, yin, cin, zout)| {
				*zout = xin ^ yin ^ cin;
			})
	})
}

pub fn constant(value: u32) -> Filler {
	Filler::new(move |_inputs, output| {
		must_cast_slice_mut::<U, u32>(output).fill(value);
	})
}
