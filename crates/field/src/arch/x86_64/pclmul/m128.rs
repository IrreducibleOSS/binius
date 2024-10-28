// Copyright 2024 Irreducible Inc.

use super::montgomery_mul::PolyvalSimdType;
use crate::arch::x86_64::m128::M128;
use std::arch::x86_64::*;

impl PolyvalSimdType for M128 {
	#[inline(always)]
	unsafe fn shuffle_epi32<const IMM8: i32>(a: Self) -> Self {
		_mm_shuffle_epi32::<IMM8>(a.0).into()
	}

	#[inline(always)]
	unsafe fn xor(a: Self, b: Self) -> Self {
		_mm_xor_si128(a.0, b.0).into()
	}

	#[inline(always)]
	unsafe fn clmul_epi64<const IMM8: i32>(a: Self, b: Self) -> Self {
		_mm_clmulepi64_si128::<IMM8>(a.0, b.0).into()
	}

	#[inline(always)]
	unsafe fn srli_epi64<const IMM8: i32>(a: Self) -> Self {
		_mm_srli_epi64::<IMM8>(a.0).into()
	}

	#[inline(always)]
	unsafe fn slli_epi64<const IMM8: i32>(a: Self) -> Self {
		_mm_slli_epi64::<IMM8>(a.0).into()
	}

	#[inline(always)]
	unsafe fn unpacklo_epi64(a: Self, b: Self) -> Self {
		_mm_unpacklo_epi64(a.0, b.0).into()
	}
}
