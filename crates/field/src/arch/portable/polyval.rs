// Copyright 2023 Ulvetanna Inc.
// Copyright (c) 2019-2023 RustCrypto Developers
// Copyright (c) 2016 Thomas Pornin <pornin@bolet.org>

//! Constant-time software implementation of GF(2^128) operations for 64-bit architectures.
//! Adapted from RustCrypto/universal-hashes, which is itself adapted from BearSSL's
//! `ghash_ctmul64.c`:
//!
//! <https://bearssl.org/gitweb/?p=BearSSL;a=blob;f=src/hash/ghash_ctmul64.c;hb=4b6046412>

use core::num::Wrapping;
use subtle::CtOption;

/// The POLYVAL "dot" operation defined in [`RFC 8542`][1], Section 3.
///
/// This is the operation $a b x^{-128}$ in the field
/// $F_2\[X\] / (X^128 + X^127 + X^126 + X^121 + 1)$, which is a Montgomery multiplication of
/// polynomials with an R value of $X^128$.
///
/// [1]: <https://datatracker.ietf.org/doc/html/rfc8452>
pub fn montgomery_multiply(a: u128, b: u128) -> u128 {
	let h0 = a as u64;
	let h1 = (a >> 64) as u64;
	let h0r = rev64(h0);
	let h1r = rev64(h1);
	let h2 = h0 ^ h1;
	let h2r = h0r ^ h1r;

	let y0 = b as u64;
	let y1 = (b >> 64) as u64;
	let y0r = rev64(y0);
	let y1r = rev64(y1);
	let y2 = y0 ^ y1;
	let y2r = y0r ^ y1r;
	let z0 = bmul64(y0, h0);
	let z1 = bmul64(y1, h1);

	let mut z2 = bmul64(y2, h2);
	let mut z0h = bmul64(y0r, h0r);
	let mut z1h = bmul64(y1r, h1r);
	let mut z2h = bmul64(y2r, h2r);

	z2 ^= z0 ^ z1;
	z2h ^= z0h ^ z1h;
	z0h = rev64(z0h) >> 1;
	z1h = rev64(z1h) >> 1;
	z2h = rev64(z2h) >> 1;

	let v0 = z0;
	let mut v1 = z0h ^ z2;
	let mut v2 = z1 ^ z2h;
	let mut v3 = z1h;

	v2 ^= v0 ^ (v0 >> 1) ^ (v0 >> 2) ^ (v0 >> 7);
	v1 ^= (v0 << 63) ^ (v0 << 62) ^ (v0 << 57);
	v3 ^= v1 ^ (v1 >> 1) ^ (v1 >> 2) ^ (v1 >> 7);
	v2 ^= (v1 << 63) ^ (v1 << 62) ^ (v1 << 57);

	v2 as u128 | (v3 as u128) << 64
}

pub fn montgomery_square(x: u128) -> u128 {
	// TODO: Optimize this using the squaring tricks in characteristic-2 fields
	montgomery_multiply(x, x)
}

pub fn invert(_x: u128) -> CtOption<u128> {
	todo!()
}

/// Multiplication in GF(2)[X], truncated to the low 64-bits, with “holes”
/// (sequences of zeroes) to avoid carry spilling.
///
/// When carries do occur, they wind up in a "hole" and are subsequently masked
/// out of the result.
fn bmul64(x: u64, y: u64) -> u64 {
	let x0 = Wrapping(x & 0x1111_1111_1111_1111);
	let x1 = Wrapping(x & 0x2222_2222_2222_2222);
	let x2 = Wrapping(x & 0x4444_4444_4444_4444);
	let x3 = Wrapping(x & 0x8888_8888_8888_8888);
	let y0 = Wrapping(y & 0x1111_1111_1111_1111);
	let y1 = Wrapping(y & 0x2222_2222_2222_2222);
	let y2 = Wrapping(y & 0x4444_4444_4444_4444);
	let y3 = Wrapping(y & 0x8888_8888_8888_8888);

	let mut z0 = ((x0 * y0) ^ (x1 * y3) ^ (x2 * y2) ^ (x3 * y1)).0;
	let mut z1 = ((x0 * y1) ^ (x1 * y0) ^ (x2 * y3) ^ (x3 * y2)).0;
	let mut z2 = ((x0 * y2) ^ (x1 * y1) ^ (x2 * y0) ^ (x3 * y3)).0;
	let mut z3 = ((x0 * y3) ^ (x1 * y2) ^ (x2 * y1) ^ (x3 * y0)).0;

	z0 &= 0x1111_1111_1111_1111;
	z1 &= 0x2222_2222_2222_2222;
	z2 &= 0x4444_4444_4444_4444;
	z3 &= 0x8888_8888_8888_8888;

	z0 | z1 | z2 | z3
}

/// Bit-reverse a `u64` in constant time
fn rev64(mut x: u64) -> u64 {
	x = ((x & 0x5555_5555_5555_5555) << 1) | ((x >> 1) & 0x5555_5555_5555_5555);
	x = ((x & 0x3333_3333_3333_3333) << 2) | ((x >> 2) & 0x3333_3333_3333_3333);
	x = ((x & 0x0f0f_0f0f_0f0f_0f0f) << 4) | ((x >> 4) & 0x0f0f_0f0f_0f0f_0f0f);
	x = ((x & 0x00ff_00ff_00ff_00ff) << 8) | ((x >> 8) & 0x00ff_00ff_00ff_00ff);
	x = ((x & 0xffff_0000_ffff) << 16) | ((x >> 16) & 0xffff_0000_ffff);
	(x << 32) | (x >> 32)
}
