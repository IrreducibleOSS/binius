// Copyright 2024 Irreducible Inc.

use super::*;
use crate::arch::x86_64::m128::M128;
use core::arch::x86_64::*;
use gfni_arithmetics::GfniType;

impl GfniType for M128 {
	#[inline(always)]
	fn gf2p8affine_epi64_epi8(x: Self, a: Self) -> Self {
		unsafe { _mm_gf2p8affine_epi64_epi8::<0>(x.0, a.0) }.into()
	}

	#[inline(always)]
	fn gf2p8mul_epi8(a: Self, b: Self) -> Self {
		unsafe { _mm_gf2p8mul_epi8(a.0, b.0) }.into()
	}

	#[inline(always)]
	fn gf2p8affineinv_epi64_epi8(x: Self, a: Self) -> Self {
		unsafe { _mm_gf2p8affineinv_epi64_epi8::<0>(x.0, a.0) }.into()
	}
}
