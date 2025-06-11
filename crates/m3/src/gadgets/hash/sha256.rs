// Copyright 2025 Irreducible Inc.

//! SHA-256 compression function arithmetisation gadget for the M3 framework.
//!
//! This models a single SHA-256 compression (ignoring padding and parsing),
//! following the style of the Groestl and Keccak gadgets.

use std::array;

use anyhow::Result;
use binius_core::oracle::ShiftVariant;
use binius_field::Field;

use crate::builder::{B1, B32, Col, TableBuilder, TableWitnessSegment};

// SHA-256 round constants, K
pub const ROUND_CONSTS_K: [u32; 64] = [
	0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
	0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
	0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
	0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
	0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
	0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
	0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
	0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

pub const ROUND_CONSTS_B1: [B1; 2048] = [B1::ZERO; 2048];

pub const INIT: [u32; 8] = [
	0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

pub struct Sha256 {
	/// SHA-256 state in: 8 32-bit words
	pub state_in: Col<B1, { 32 * 16 }>,
	/// SHA-256 state out: 8 32-bit words
	pub state_out: [Col<B32>; 8],
	/// SHA-256 message schedule: 64 32-bit words
	pub w: Col<B1, { 32 * 64 }>,

	/// Working variables
	a: Col<B32>,
	b: Col<B32>,
	c: Col<B32>,
	d: Col<B32>,
	e: Col<B32>,
	f: Col<B32>,
	g: Col<B32>,
	h: Col<B32>,

	/// Round constants: 64 32-bit words
	k: Col<B32, 64>,

	/// Temporary variables for the SHA-256 compression function
	t1: Col<B32>,
	t2: Col<B32>,
}

impl Sha256 {
	pub fn new(table: &mut TableBuilder) {
		// Initialize the message schedule w in batches of 16, the first batch is the state_in
		let w: Col<B1, { 32 * 64 }> = table.add_committed("w");

		let w_minus_16 =
			table.add_shifted("w_minus_16", w, 11, 32 * 16, ShiftVariant::LogicalRight);

		let w_minus_7 =
			table.add_shifted("w_minus_7", w, 11, 32 * (16 - 7), ShiftVariant::LogicalRight);

		let sigma0 = Sigma0::new(table, w);

		let sigma1 = Sigma1::new(table, w);

		let sigma0_minus_15 =
			table.add_shifted("sigma0_minus_15", sigma0.out, 11, 32, ShiftVariant::LogicalRight);
		let sigma1_minus_2 = table.add_shifted(
			"sigma1_minus_2",
			sigma0.out,
			11,
			32 * 14,
			ShiftVariant::LogicalRight,
		);

		table.assert_zero("message_schedule", w_minus_16 + w_minus_7 + sigma0.out + sigma1.out + w);

		let a: Col<B1, { 32 * 64 }> = table.add_committed("a");
		let b: Col<B1, { 32 * 64 }> = table.add_committed("b");
		let c: Col<B1, { 32 * 64 }> = table.add_committed("c");
		let d: Col<B1, { 32 * 64 }> = table.add_committed("d");
		let e: Col<B1, { 32 * 64 }> = table.add_committed("e");
		let f: Col<B1, { 32 * 64 }> = table.add_committed("f");
		let g: Col<B1, { 32 * 64 }> = table.add_committed("g");
		let h: Col<B1, { 32 * 64 }> = table.add_committed("h");
		let k: Col<B1, { 32 * 64 }> = table.add_constant("k", ROUND_CONSTS_B1);
		let ch: Col<B1, { 32 * 64 }> = table.add_committed("ch");
		let maj: Col<B1, { 32 * 64 }> = table.add_committed("maj");
        

	}
}

pub struct Sigma0 {
	rotr_7: Col<B1, { 32 * 64 }>,
	rotr_18: Col<B1, { 32 * 64 }>,
	shr_3: Col<B1, { 32 * 64 }>,
	pub out: Col<B1, { 32 * 64 }>,
}

impl Sigma0 {
	pub fn new(table: &mut TableBuilder, state_in: Col<B1, { 32 * 64 }>) -> Self {
		let rotr_7 = table.add_shifted("rotr_7", state_in, 5, 32 - 7, ShiftVariant::CircularLeft);
		let rotr_18 =
			table.add_shifted("rotr_18", state_in, 5, 32 - 18, ShiftVariant::CircularLeft);
		let shr_3 = table.add_shifted("shr_3", state_in, 5, 3, ShiftVariant::LogicalRight);
		let out = table.add_computed("sigma_0", rotr_7 + rotr_18 + shr_3);
		Self {
			rotr_7,
			rotr_18,
			shr_3,
			out,
		}
	}
}
pub struct Sigma1 {
	rotr_17: Col<B1, { 32 * 64 }>,
	rotr_19: Col<B1, { 32 * 64 }>,
	shr_10: Col<B1, { 32 * 64 }>,
	pub out: Col<B1, { 32 * 64 }>,
}

impl Sigma1 {
	pub fn new(table: &mut TableBuilder, state_in: Col<B1, { 32 * 64 }>) -> Self {
		let rotr_17 =
			table.add_shifted("rotr_17", state_in, 5, 32 - 17, ShiftVariant::CircularLeft);
		let rotr_19 =
			table.add_shifted("rotr_19", state_in, 5, 32 - 19, ShiftVariant::CircularLeft);
		let shr_10 = table.add_shifted("shr_10", state_in, 5, 10, ShiftVariant::LogicalRight);
		let out = table.add_computed("sigma_1", rotr_17 + rotr_19 + shr_10);

		Self {
			rotr_17,
			rotr_19,
			shr_10,
			out,
		}
	}
}

pub struct BigSigma0 {
	rotr_2: Col<B1, { 32 * 64 }>,
	rotr_13: Col<B1, { 32 * 64 }>,
	rotr_22: Col<B1, { 32 * 64 }>,
	pub out: Col<B1, { 32 * 64 }>,
}

impl BigSigma0 {
	pub fn new(table: &mut TableBuilder, state_in: Col<B1, { 32 * 64 }>) -> Self {
		let rotr_2 = table.add_shifted("rotr_2", state_in, 5, 32 - 2, ShiftVariant::CircularLeft);
		let rotr_13 =
			table.add_shifted("rotr_13", state_in, 5, 32 - 13, ShiftVariant::CircularLeft);
		let rotr_22 =
			table.add_shifted("rotr_22", state_in, 5, 32 - 22, ShiftVariant::CircularLeft);
		let out = table.add_computed("big_sigma_0", rotr_2 + rotr_13 + rotr_22);

		Self {
			rotr_2,
			rotr_13,
			rotr_22,
			out,
		}
	}
}
pub struct BigSigma1 {
	rotr_6: Col<B1, { 32 * 64 }>,
	rotr_11: Col<B1, { 32 * 64 }>,
	rotr_25: Col<B1, { 32 * 64 }>,
	pub out: Col<B1, { 32 * 64 }>,
}

impl BigSigma1 {
	pub fn new(table: &mut TableBuilder, state_in: Col<B1, { 32 * 64 }>) -> Self {
		let rotr_6 = table.add_shifted("rotr_6", state_in, 5, 32 - 6, ShiftVariant::CircularLeft);
		let rotr_11 =
			table.add_shifted("rotr_11", state_in, 5, 32 - 11, ShiftVariant::CircularLeft);
		let rotr_25 =
			table.add_shifted("rotr_25", state_in, 5, 32 - 25, ShiftVariant::CircularLeft);
		let out = table.add_computed("big_sigma_1", rotr_6 + rotr_11 + rotr_25);

		Self {
			rotr_6,
			rotr_11,
			rotr_25,
			out,
		}
	}
}
