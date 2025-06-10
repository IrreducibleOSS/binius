// Copyright 2025 Irreducible Inc.

//! SHA-256 compression function arithmetisation gadget for the M3 framework.
//!
//! This models a single SHA-256 compression (ignoring padding and parsing),
//! following the style of the Groestl and Keccak gadgets.

use std::array;

use anyhow::Result;

use crate::builder::{B1, B32, Col, TableBuilder, TableWitnessSegment};

/// SHA-256 state: 8 32-bit words
pub type State = [Col<B32>; 8];
/// SHA-256 message schedule: 64 32-bit words
pub type Schedule = [Col<B32>; 64];

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

pub const INIT: [u32; 8] = [
	0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

pub struct Sha256Gadget {
	pub state_in: State,
	pub state_out: State,
	pub schedule: Schedule,
}

impl Sha256Gadget {
	pub fn new(table: &mut TableBuilder) -> Self {
		// Input state
		let state_in: State = table.add_committed_multiple::<B32, 8>("state_in");
		// Message schedule
		let schedule: Schedule = table.add_committed_multiple::<B32, 64>("w");
		// Output state
		let state_out: State =
			array::from_fn(|i| table.add_computed(format!("state_out[{i}]"), state_in[i]));

		// Working variables
		let mut a = state_in[0];
		let mut b = state_in[1];
		let mut c = state_in[2];
		let mut d = state_in[3];
		let mut e = state_in[4];
		let mut f = state_in[5];
		let mut g = state_in[6];
		let mut h = state_in[7];

		// Round constants as columns
		let round_consts: [Col<B32>; 64] =
			table.add_constant_multiple::<B32, 64>("k", &ROUND_CONSTS_K.map(B32::from));

		for i in 0..64 {
			let k = round_consts[i];
			let w = schedule[i];
			// SHA-256 round function
			let s1 = table.add_computed(
				format!("s1[{i}]"),
				e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25),
			);
			let ch = table.add_computed(format!("ch[{i}]"), (e & f) ^ ((!e) & g));
			let temp1 = table.add_computed(format!("temp1[{i}]"), h + s1 + ch + k + w);
			let s0 = table.add_computed(
				format!("s0[{i}]"),
				a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22),
			);
			let maj = table.add_computed(format!("maj[{i}]"), (a & b) ^ (a & c) ^ (b & c));
			let temp2 = table.add_computed(format!("temp2[{i}]"), s0 + maj);

			h = g;
			g = f;
			f = e;
			e = d + temp1;
			d = c;
			c = b;
			b = a;
			a = temp1 + temp2;
		}

		// Output state = input state + working variables
		let state_out = array::from_fn(|i| {
			table.add_computed(
				format!("state_out[{i}]"),
				state_in[i]
					+ match i {
						0 => a,
						1 => b,
						2 => c,
						3 => d,
						4 => e,
						5 => f,
						6 => g,
						7 => h,
						_ => unreachable!(),
					},
			)
		});

		Self {
			state_in,
			state_out,
			schedule,
		}
	}

	pub fn populate<P>(&self, _index: &mut TableWitnessSegment<P>) -> Result<()> {
		// TODO: Implement witness population for SHA-256
		Ok(())
	}
}



#[cfg(test)]
mod tests {
	use bumpalo::Bump;

	use super::*;
	use crate::builder::{ConstraintSystem, Statement, WitnessIndex};

	#[test]
	fn test_sha256_gadget_structure() {
		let mut cs = ConstraintSystem::new();
		let mut table = cs.add_table("sha256");
		let _gadget = Sha256Gadget::new(&mut table);
	}
}
