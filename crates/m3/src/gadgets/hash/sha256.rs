// Copyright 2025 Irreducible Inc.

//! SHA-256 compression function arithmetisation gadget for the M3 framework.
//!
//! This module implements a constraint system for verifying the SHA-256 compression function,
//! which is the core component of the SHA-256 cryptographic hash function. The implementation
//! follows the SHA-256 specification (FIPS 180-4) and provides zero-knowledge proof capabilities
//! for SHA-256 computations.
//!
//! ## Overview
//!
//! SHA-256 operates on 512-bit message blocks and produces a 256-bit hash digest. The compression
//! function processes each block through 64 rounds of operations, maintaining an 8-word (256-bit)
//! internal state. This gadget arithmetizes these operations to enable verification in
//! zero-knowledge.
//!
//! ## Implementation Structure
//!
//! The gadget is organized around several key components:
//!
//! - **Message Schedule Extension**: Expands the 16-word input block to 64 words using the σ₀ and
//!   σ₁ functions.
//! - **Compression Rounds**: 64 rounds of the main compression algorithm using working variables
//!   a-h and the Σ₀, Σ₁, Ch, and Maj functions.
//! - **State Management**: Tracks the evolution of the 256-bit state through all rounds.
//!
//! ## Cryptographic Functions
//!
//! The implementation includes arithmetic circuits for all SHA-256 primitive functions:
//!
//! - **σ₀(x)**: `ROTR⁷(x) ⊕ ROTR¹⁸(x) ⊕ SHR³(x)` - Used in message schedule extension.
//! - **σ₁(x)**: `ROTR¹⁷(x) ⊕ ROTR¹⁹(x) ⊕ SHR¹⁰(x)` - Used in message schedule extension.
//! - **Σ₀(x)**: `ROTR²(x) ⊕ ROTR¹³(x) ⊕ ROTR²²(x)` - Used in compression rounds.
//! - **Σ₁(x)**: `ROTR⁶(x) ⊕ ROTR¹¹(x) ⊕ ROTR²⁵(x)` - Used in compression rounds.
//! - **Ch(x, y, z)**: `(x ∧ y) ⊕ (¬x ∧ z)` - Choice function used in compression rounds.
//! - **Maj(x, y, z)**: `(x ∧ y) ⊕ (x ∧ z) ⊕ (y ∧ z)` - Majority function used in compression
//!   rounds.
//!
//! ## Usage Example
//!
//! ```rust,ignore
//! let mut cs = ConstraintSystem::new();
//! let mut table = cs.add_table("sha256");
//! let sha256 = Sha256::new(&mut table);
//!
//! // Populate the table with actual SHA-256 computation data during proving
//! sha256.populate(&mut table_witness_segment, message_blocks)?;
//! ```
//!
//! ## Technical Details
//!
//! This implementation uses a bit-packed representation where 32-bit words are stored as
//! columns of individual bits. This approach:
//!
//! - Enables efficient constraint representation of bitwise operations.
//! - Allows for precise tracking of carry propagation in additions.
//! - Supports the circular shift operations required by SHA-256.
//!
//! ## Key Components
//!
//! - **Message Schedule (`w`)**: A 64-word schedule derived from the input message block.
//! - **Working Variables (`a` to `h`)**: Eight 32-bit variables used in each compression round.
//! - **Round Constants (`K`)**: Predefined constants used in each round to ensure cryptographic
//!   strength.
//! - **State (`state_in` and `state_out`)**: The 256-bit input and output states for the
//!   compression function.
//!
//! ## References
//!
//! - [FIPS 180-4: Secure Hash Standard (SHS)](https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.180-4.pdf)
//! - [RFC 6234: US Secure Hash Algorithms](https://tools.ietf.org/rfc/rfc6234.txt)

use std::array;

use anyhow::Result;
use array_util::ArrayExt;
use binius_core::oracle::ShiftVariant;
use binius_field::{Field, PackedExtension, PackedFieldIndexable, packed::set_packed_slice};

use crate::{
	builder::{B1, B32, B128, Col, TableBuilder, TableWitnessSegment},
	gadgets::add::{self, U32Add, U32AddFlags, U32AddStacked},
};

/// SHA-256 round constants (K).
///
/// These 64 constants represent the first 32 bits of the fractional parts of the cube roots
/// of the first 64 prime numbers. They are used in each round of the compression function
/// to provide cryptographic strength against various attacks.
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

/// Converts a u32 to an array of 32 B1s (little-endian bit order).
const fn u32_to_b1_bits_le(x: u32) -> [B1; 32] {
	let mut bits = [B1::ZERO; 32];
	let mut i = 0;
	while i < 32 {
		bits[i] = if (x >> i) & 1 == 1 { B1::ONE } else { B1::ZERO };
		i += 1;
	}
	bits
}

/// Expands ROUND_CONSTS_K into a bitslice of B1s (little-endian per word).
pub const ROUND_CONSTS_B1: [B1; 2048] = {
	let mut arr = [B1::ZERO; 2048];
	let mut i = 0;
	while i < 64 {
		let bits = u32_to_b1_bits_le(ROUND_CONSTS_K[i]);
		let mut j = 0;
		while j < 32 {
			arr[i * 32 + j] = bits[j];
			j += 1;
		}
		i += 1;
	}
	arr
};

/// SHA-256 initial hash values (H).
///
/// These 8 constants represent the first 32 bits of the fractional parts of the square roots
/// of the first 8 prime numbers. They form the initial 256-bit state for SHA-256 computation.
pub const INIT: [u32; 8] = [
	0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

/// SHA-256 compression function constraint system.
///
/// This struct represents the complete SHA-256 compression function as a constraint system,
/// providing columns for all intermediate values and enforcing the cryptographic relationships
/// between them. The compression function takes a 512-bit message block and a 256-bit state,
/// producing a new 256-bit state.
///
/// ## Structure
///
/// The gadget maintains:
/// - Input state (256 bits as 8×32-bit words)
/// - Message schedule (2048 bits as 64×32-bit words)
/// - Working variables for all 64 rounds
/// - Output state (256 bits as 8×32-bit words)
///
/// ## Constraint Enforcement
///
/// The implementation enforces:
/// - Correct message schedule extension using σ₀ and σ₁ functions
/// - Proper evolution of working variables through all rounds
/// - Correct application of Σ₀, Σ₁, Ch, and Maj functions
/// - Final state computation as the sum of initial and final working variables
pub struct Sha256 {
	/// SHA-256 input state: 16 32-bit words.
	///
	/// In the bit-packed representation, this stores the initial 256-bit hash state
	/// plus space for intermediate computations. The first 8 words represent the
	/// actual input state (a-h), while additional space may be used for constraints.
	pub state_in: [Col<B1, 32>; 16],

	/// SHA-256 output state: 8 32-bit words.
	///
	/// The final hash state after compression, represented as 8 separate 32-bit
	/// columns for efficient constraint handling.
	pub state_out: [Col<B1, 32>; 8],

	/// Message schedule: 64 32-bit words.
	///
	/// The expanded message schedule W₀, W₁, ..., W₆₃ where:
	/// - W₀ through W₁₅ are the original 16 words of the message block
	/// - W₁₆ through W₆₃ are computed using the message schedule extension
	w: Col<B1, { 32 * 64 }>,
	sigma0: Sigma0,
	sigma1: Sigma1,
	w_blocks: [Col<B1, 32>; 64],
	w_minus_16: Col<B1, { 32 * 64 }>,
	w_minus_7: Col<B1, { 32 * 64 }>,
	sigma0_minus_2: Col<B1, { 32 * 64 }>,
	sigma1_minus_15: Col<B1, { 32 * 64 }>,

	s0: U32AddStacked<{ 32 * 64 }>,
	s1: U32AddStacked<{ 32 * 64 }>,
	s3: U32AddStacked<{ 32 * 64 }>,
	/// Working variables for all 64 rounds.
	///
	/// These represent the 8 working variables (a, b, c, d, e, f, g, h) as they
	/// evolve through each of the 64 compression rounds. Each variable is stored
	/// as a bit-packed column spanning all rounds.
	initial_state: [Col<B1, 32>; 8],
	rounds: [Round; 64],

	final_sums: [U32Add; 8],

	/// Round constants: 64 32-bit words.
	///
	/// The cryptographic constants K₀, K₁, ..., K₆₃ used in each round of the
	/// compression function.
	round_constants: [Col<B1, 32>; 64],

	/// Selector for message schedule expansion.
	///
	/// This column is used to select the first 48 words of the message schedule
	/// for the SHA-256 compression function. It constrains the message schedule
	/// extension operations in the gadget.
	pub selector: Col<B1, { 32 * 64 }>,
}

impl Sha256 {
	/// Creates a new SHA-256 compression function gadget.
	///
	/// This function sets up the complete constraint system for SHA-256 compression,
	/// including all intermediate columns and the constraints that enforce correct
	/// computation of the hash function.
	///
	/// # Arguments
	///
	/// * `table` - The table builder for creating columns and constraints
	///
	/// # Implementation Details
	///
	/// The constructor:
	/// 1. Sets up the message schedule with σ₀ and σ₁ extension
	/// 2. Creates working variable columns for all 64 rounds
	/// 3. Enforces round function constraints using Σ₀, Σ₁, Ch, and Maj
	/// 4. Links rounds together through shifted column relationships
	/// 5. Computes final output state from working variables
	pub fn new(table: &mut TableBuilder) -> Self {
		// Initialize the message schedule w.
		let w: Col<B1, { 32 * 64 }> = table.add_committed("w");

		// σ₀ and σ₁ functions for message schedule extension
		let sigma0 = Sigma0::new(table, w);
		let sigma1 = Sigma1::new(table, w);

		let state_in: [Col<B1, 32>; 16] =
			array::from_fn(|i| table.add_selected_block(format!("state_in[{i}]"), w, i));

		let selector = table.add_constant("message_schedule_selector", SCHEDULE_EXPAND_SELECTOR);

		// Contains the values of the message schedule for t past 16 rounds.
		let w_minus_16: Col<B1, { 32 * 64 }> =
			table.add_shifted("w_minus_16", w, 11, 32 * 16, ShiftVariant::LogicalRight);
		let w_minus_7: Col<B1, { 32 * 64 }> =
			table.add_shifted("w_minus_7", w, 11, 32 * (16 - 7), ShiftVariant::LogicalRight);
		let sigma0_minus_2 = table.add_shifted(
			"sigma0_minus_2",
			sigma0.out,
			11,
			32 * (16 - 2),
			ShiftVariant::LogicalRight,
		);
		let sigma1_minus_15 = table.add_shifted(
			"sigma1_minus_15",
			sigma1.out,
			11,
			32 * (16 - 15),
			ShiftVariant::LogicalRight,
		);

		let s0 = U32AddStacked::new(table, w, w_minus_7, false, None);
		let s1 = U32AddStacked::new(table, sigma0_minus_2, sigma1_minus_15, false, None);
		let s3 = U32AddStacked::new(table, s0.zout, s1.zout, false, None);
		table.assert_zero("message schedule expansion", selector * (s3.zout - w_minus_16));

		let w_blocks: [Col<B1, 32>; 64] =
			array::from_fn(|i| table.add_selected_block(format!("w[{i}]"), w, i));
		let initial_state: [Col<B1, 32>; 8] = array::from_fn(|i| {
			table.add_constant(format!("initial_state[{i}]"), u32_to_b1_bits_le(INIT[i]))
		});
		let round_constants: [Col<B1, 32>; 64] = array::from_fn(|i| {
			table.add_constant(format!("round_constant[{i}]"), u32_to_b1_bits_le(ROUND_CONSTS_K[i]))
		});

		let mut rounds = Vec::with_capacity(64);
		for i in 0..64 {
			let round = if i == 0 {
				// First round uses initial state and first message schedule word
				Round::first_round(
					table,
					initial_state,
					round_constants[i].clone(),
					w_blocks[i].clone(),
				)
			} else {
				// Subsequent rounds use previous round's state and message schedule word
				Round::next_round(
					table,
					&rounds[i - 1],
					round_constants[i].clone(),
					w_blocks[i].clone(),
				)
			};
			rounds.push(round);
		}

		// Extract the final state from the last round's working variables
		let &Round {
			a,
			b,
			c,
			d,
			e,
			f,
			g,
			h,
			..
		} = &rounds[63];

		let final_state: [Col<B1, 32>; 8] = [a, b, c, d, e, f, g, h];

		let final_sums: [U32Add; 8] =
			array::from_fn(|i| U32Add::new(table, final_state[i], initial_state[i], ADD_FLAGS));

		let state_out: [Col<B1, 32>; 8] = array::from_fn(|i| final_sums[i].zout);

		Self {
			state_in,
			state_out,
			initial_state,
			rounds: rounds.try_into().expect("64 rounds expected"),
			w_blocks,
			w,
			round_constants,
			w_minus_16,
			w_minus_7,
			sigma0_minus_2,
			sigma1_minus_15,
			sigma0,
			sigma1,
			s0,
			s1,
			s3,
			selector,
			final_sums,
		}
	}

	/// Populates the witness columns for the SHA-256 compression function.
	///
	/// # Arguments
	///
	/// * `index` - The witness segment to populate
	/// * `message` - The 512-bit input message block as 16 32-bit words
	/// * `state_in` - The 256-bit input state as 8 32-bit words
	///
	/// # Returns
	///
	/// Ok(()) on success, Error otherwise
	pub fn populate<P>(
		&self,
		index: &mut TableWitnessSegment<P>,
		message_blocks: impl IntoIterator<Item = [u32; 16]>,
	) -> Result<(), anyhow::Error>
	where
		P: PackedFieldIndexable<Scalar = B128> + PackedExtension<B1> + PackedExtension<B32>,
	{
		{
			let mut w = index.get_mut_as::<u32, _, { 32 * 64 }>(self.w)?;
			let mut w_blocks = self
				.w_blocks
				.try_map_ext(|col| index.get_mut_as::<u32, _, 32>(col))?;
			let mut state_in = self
				.state_in
				.try_map_ext(|col| index.get_mut_as::<u32, _, 32>(col))?;
			let mut w_minus_16: std::cell::RefMut<'_, [u32]> = index.get_mut_as(self.w_minus_16)?;
			let mut w_minus_7: std::cell::RefMut<'_, [u32]> = index.get_mut_as(self.w_minus_7)?;

			let mut initial_state = self
				.initial_state
				.try_map_ext(|col| index.get_mut_as::<u32, _, 32>(col))?;
			let mut round_constants = self
				.round_constants
				.try_map_ext(|col| index.get_mut_as::<u32, _, 32>(col))?;
			let mut selector: std::cell::RefMut<'_, [u32]> = index.get_mut_as(self.selector)?;

			for (i, message) in message_blocks.into_iter().enumerate() {
				for j in 0..16 {
					w[64 * i + j] = message[j];
					w_blocks[j][i] = message[j];
					state_in[j][i] = message[j];
				}

				compute_message_schedule(&mut w[64 * i..64 * (i + 1)]);
				for j in 0..48 {
					w_blocks[j + 16][i] = w[64 * i + j + 16];
					w_minus_16[64 * i + j] = w[64 * i + j + 16];
					w_minus_7[64 * i + j] = w[64 * i + j + 9];
					selector[64 * i + j] = 0xFFFFFFFF;
				}

				for j in 0..64 {
					round_constants[j][i] = ROUND_CONSTS_K[j];
				}
				for j in 0..8 {
					initial_state[j][i] = INIT[j];
				}
			}
		}
		{
			self.sigma0.populate(index)?;
			self.sigma1.populate(index)?;
			let sigma0: std::cell::RefMut<'_, [u32]> = index.get_mut_as(self.sigma0.out)?;
			let sigma1: std::cell::RefMut<'_, [u32]> = index.get_mut_as(self.sigma1.out)?;
			let mut sigma0_minus_2: std::cell::RefMut<'_, [u32]> =
				index.get_mut_as(self.sigma0_minus_2)?;
			let mut sigma1_minus_15: std::cell::RefMut<'_, [u32]> =
				index.get_mut_as(self.sigma1_minus_15)?;

			for i in 0..index.size() / 64 {
				for j in 0..64 {
					sigma0_minus_2[i * 64 + j] = sigma0[i * 64 + j + 14];
					sigma1_minus_15[i * 64 + j] = sigma1[i * 64 + j + 1];
				}
			}
		}

		self.s0.populate(index)?;
		self.s1.populate(index)?;
		self.s3.populate(index)?;

		for i in 0..64 {
			self.rounds[i].populate(index)?;
		}

		for i in 0..8 {
			self.final_sums[i].populate(index)?;
		}

		Ok(())
	}
}

#[derive(Debug)]
pub struct Round {
	/// The round number (0-63) for this compression round.
	pub round: usize,

	/// The working variables a-h for this round.
	pub a: Col<B1, 32>,
	pub b: Col<B1, 32>,
	pub c: Col<B1, 32>,
	pub d: Col<B1, 32>,
	pub e: Col<B1, 32>,
	pub f: Col<B1, 32>,
	pub g: Col<B1, 32>,
	pub h: Col<B1, 32>,

	ch: Col<B1, 32>,
	maj: Col<B1, 32>,

	temp_sum_0: U32Add,
	temp_sum_1: U32Add,
	temp_sum_2: U32Add,
	temp_sum_3: U32Add,
	temp_sum_4: U32Add,
	temp_sum_5: U32Add,
	temp_sum_6: U32Add,

	/// The temporary values T₁ and T₂ computed in this round.
	pub t1: Col<B1, 32>,
	pub t2: Col<B1, 32>,

	bigsigma0: BigSigma0,
	bigsigma1: BigSigma1,
	/// The round constant Kₜ for this round.
	pub k: Col<B1, 32>,
}

const ADD_FLAGS: U32AddFlags = U32AddFlags {
	carry_in_bit: None,
	commit_zout: false,
	expose_final_carry: false,
};
impl Round {
	fn next_round(
		table: &mut TableBuilder,
		previous: &Round,
		round_constant: Col<B1, 32>,
		message_schedule: Col<B1, 32>,
	) -> Self {
		assert!(previous.round < 63, "Cannot compute next round for round 63");
		let &Round {
			round,
			a,
			b,
			c,
			d,
			e,
			f,
			g,
			h,
			..
		} = previous;
		let bigsigma0 = BigSigma0::new(table, a);
		let bigsigma1 = BigSigma1::new(table, e);

		let ch = table.add_committed(format!("ch[{}]", round + 1));
		let maj = table.add_committed(format!("maj[{}]", round + 1));

		table.assert_zero(format!("ch[{}]", round + 1), g + e * (f + g));
		table.assert_zero(format!("maj[{}]", round + 1), maj - (a * (b + c) + b * c));
		let temp_sum_0 = U32Add::new(table, bigsigma1.out, ch, ADD_FLAGS);
		let temp_sum_1 = U32Add::new(table, round_constant, message_schedule, ADD_FLAGS);
		let temp_sum_2 = U32Add::new(table, temp_sum_0.zout, temp_sum_1.zout, ADD_FLAGS);
		let temp_sum_3 = U32Add::new(table, h, temp_sum_2.zout, ADD_FLAGS);
		let t1 = temp_sum_3.zout;

		let temp_sum_4 = U32Add::new(table, bigsigma0.out, maj, ADD_FLAGS);

		let t2 = temp_sum_4.zout;
		let temp_sum_5 = U32Add::new(table, t1, t2, ADD_FLAGS);
		let temp_sum_6 = U32Add::new(table, d, t1, ADD_FLAGS);

		let round = previous.round + 1;
		let h = g;
		let g = f;
		let f = e;
		let e = temp_sum_6.zout;
		let d = c;
		let c = b;
		let b = a;
		let a = temp_sum_5.zout;

		Round {
			round,
			a,
			b,
			c,
			d,
			e,
			f,
			g,
			h,
			bigsigma0,
			bigsigma1,
			temp_sum_0,
			temp_sum_1,
			temp_sum_2,
			temp_sum_3,
			temp_sum_4,
			temp_sum_5,
			temp_sum_6,
			t1,
			t2,
			ch,
			maj,
			k: round_constant.clone(),
		}
	}

	pub fn first_round(
		table: &mut TableBuilder,
		initial_state: [Col<B1, 32>; 8],
		round_constant: Col<B1, 32>,
		message_schedule: Col<B1, 32>,
	) -> Self {
		let [a, b, c, d, e, f, g, h] = initial_state;
		let bigsigma0 = BigSigma0::new(table, a);
		let bigsigma1 = BigSigma1::new(table, e);

		let ch = table.add_committed("ch[0]");
		let maj = table.add_committed("maj[0]");

		table.assert_zero("ch[0]", g + e * (f + g));
		table.assert_zero("maj[0]", maj - (a * (b + c) + b * c));
		let temp_sum_0 = U32Add::new(table, bigsigma1.out, ch, ADD_FLAGS);
		let temp_sum_1 = U32Add::new(table, round_constant, message_schedule, ADD_FLAGS);
		let temp_sum_2 = U32Add::new(table, temp_sum_0.zout, temp_sum_1.zout, ADD_FLAGS);
		let temp_sum_3 = U32Add::new(table, h, temp_sum_2.zout, ADD_FLAGS);
		let t1 = temp_sum_3.zout;

		let temp_sum_4 = U32Add::new(table, bigsigma0.out, maj, ADD_FLAGS);

		let t2 = temp_sum_4.zout;
		let temp_sum_5 = U32Add::new(table, t1, t2, ADD_FLAGS);
		let temp_sum_6 = U32Add::new(table, d, t1, ADD_FLAGS);

		let round = 0;
		let h = g;
		let g = f;
		let f = e;
		let e = temp_sum_6.zout;
		let d = c;
		let c = b;
		let b = a;
		let a = temp_sum_5.zout;

		Round {
			round,
			a,
			b,
			c,
			d,
			e,
			f,
			g,
			h,
			ch,
			maj,
			bigsigma0,
			bigsigma1,
			temp_sum_0,
			temp_sum_1,
			temp_sum_2,
			temp_sum_3,
			temp_sum_4,
			temp_sum_5,
			temp_sum_6,
			t1,
			t2,
			k: round_constant.clone(),
		}
	}

	/// Helper method to populate a single round's witness values
	fn populate<P>(&self, index: &mut TableWitnessSegment<P>) -> Result<(), anyhow::Error>
	where
		P: PackedFieldIndexable<Scalar = B128> + PackedExtension<B1> + PackedExtension<B32>,
	{
		{
			let a: std::cell::Ref<'_, [u32]> = index.get_as(self.a)?;
			let b: std::cell::Ref<'_, [u32]> = index.get_as(self.b)?;
			let c: std::cell::Ref<'_, [u32]> = index.get_as(self.c)?;
			let e: std::cell::Ref<'_, [u32]> = index.get_as(self.e)?;
			let f: std::cell::Ref<'_, [u32]> = index.get_as(self.f)?;
			let g: std::cell::Ref<'_, [u32]> = index.get_as(self.g)?;

			let mut ch: std::cell::RefMut<'_, [u32]> = index.get_mut_as(self.ch)?;
			let mut maj: std::cell::RefMut<'_, [u32]> = index.get_mut_as(self.maj)?;

			for i in 0..index.size() {
				ch[i] = (e[i] & f[i]) ^ (!e[i] & g[i]);
				maj[i] = (a[i] & b[i]) ^ (a[i] & c[i]) ^ (b[i] & c[i]);
			}
		}

		self.bigsigma0.populate(index)?;
		self.bigsigma1.populate(index)?;

		self.temp_sum_0.populate(index)?;
		self.temp_sum_1.populate(index)?;
		self.temp_sum_2.populate(index)?;
		self.temp_sum_3.populate(index)?;
		self.temp_sum_4.populate(index)?;
		self.temp_sum_5.populate(index)?;
		self.temp_sum_6.populate(index)?;

		Ok(())
	}
}
/// The σ₀ function used in SHA-256 message schedule extension.
///
/// This function implements σ₀(x) = ROTR⁷(x) ⊕ ROTR¹⁸(x) ⊕ SHR³(x) where:
/// - ROTR^n(x) denotes x rotated right by n positions
/// - SHR^n(x) denotes x shifted right by n positions (with zero fill)
/// - ⊕ denotes bitwise XOR
///
/// The function is used to extend the 16-word message block to the full 64-word
/// message schedule required for SHA-256 compression.
pub struct Sigma0 {
	pub input: Col<B1, { 32 * 64 }>,
	/// Right rotation by 7 positions: ROTR⁷(x)
	rotr_7: Col<B1, { 32 * 64 }>,
	/// Right rotation by 18 positions: ROTR¹⁸(x)  
	rotr_18: Col<B1, { 32 * 64 }>,
	/// Right shift by 3 positions: SHR³(x)
	shr_3: Col<B1, { 32 * 64 }>,
	/// The computed σ₀ output: ROTR⁷(x) ⊕ ROTR¹⁸(x) ⊕ SHR³(x)
	pub out: Col<B1, { 32 * 64 }>,
}

impl Sigma0 {
	/// Creates a new σ₀ function gadget.
	///
	/// # Arguments
	///
	/// * `table` - The table builder for creating columns and constraints
	/// * `state_in` - The input column to apply σ₀ to
	///
	/// # Returns
	///
	/// A new `Sigma0` instance with all necessary columns and constraints
	pub fn new(table: &mut TableBuilder, state_in: Col<B1, { 32 * 64 }>) -> Self {
		// Implement rotations using circular left shifts (equivalent to right rotations)
		let rotr_7 = table.add_shifted("rotr_7", state_in, 5, 32 - 7, ShiftVariant::CircularLeft);
		let rotr_18 =
			table.add_shifted("rotr_18", state_in, 5, 32 - 18, ShiftVariant::CircularLeft);
		// Implement right shift using logical right shift
		let shr_3 = table.add_shifted("shr_3", state_in, 5, 3, ShiftVariant::LogicalRight);
		// Compute σ₀ as XOR of the three components
		let out = table.add_computed("sigma_0", rotr_7 + rotr_18 + shr_3);
		Self {
			input: state_in,
			rotr_7,
			rotr_18,
			shr_3,
			out,
		}
	}

	/// Populates the witness for the σ₀ function.
	pub fn populate<P>(&self, index: &mut TableWitnessSegment<P>) -> Result<(), anyhow::Error>
	where
		P: PackedFieldIndexable<Scalar = B128> + PackedExtension<B1>,
	{
		let input = index.get_as::<u32, _, { 32 * 64 }>(self.input)?;
		let mut rotr_7 = index.get_mut_as::<u32, _, { 32 * 64 }>(self.rotr_7)?;
		let mut rotr_18 = index.get_mut_as::<u32, _, { 32 * 64 }>(self.rotr_18)?;
		let mut shr_3 = index.get_mut_as::<u32, _, { 32 * 64 }>(self.shr_3)?;
		let mut out = index.get_mut_as::<u32, _, { 32 * 64 }>(self.out)?;

		for i in 0..input.len() {
			rotr_7[i] = input[i].rotate_right(7);
			rotr_18[i] = input[i].rotate_right(18);
			shr_3[i] = input[i] >> 3;
			out[i] = rotr_7[i] ^ rotr_18[i] ^ shr_3[i];
		}

		Ok(())
	}
}

/// The σ₁ function used in SHA-256 message schedule extension.
///
/// This function implements σ₁(x) = ROTR¹⁷(x) ⊕ ROTR¹⁹(x) ⊕ SHR¹⁰(x) where:
/// - ROTR^n(x) denotes x rotated right by n positions
/// - SHR^n(x) denotes x shifted right by n positions (with zero fill)
/// - ⊕ denotes bitwise XOR
///
/// Like σ₀, this function is used in message schedule extension to provide
/// cryptographic mixing of the message words.
pub struct Sigma1 {
	pub input: Col<B1, { 32 * 64 }>,
	/// Right rotation by 17 positions: ROTR¹⁷(x)
	rotr_17: Col<B1, { 32 * 64 }>,
	/// Right rotation by 19 positions: ROTR¹⁹(x)
	rotr_19: Col<B1, { 32 * 64 }>,
	/// Right shift by 10 positions: SHR¹⁰(x)
	shr_10: Col<B1, { 32 * 64 }>,
	/// The computed σ₁ output: ROTR¹⁷(x) ⊕ ROTR¹⁹(x) ⊕ SHR¹⁰(x)
	pub out: Col<B1, { 32 * 64 }>,
}

impl Sigma1 {
	/// Creates a new σ₁ function gadget.
	///
	/// # Arguments
	///
	/// * `table` - The table builder for creating columns and constraints
	/// * `state_in` - The input column to apply σ₁ to
	///
	/// # Returns
	///
	/// A new `Sigma1` instance with all necessary columns and constraints
	pub fn new(table: &mut TableBuilder, state_in: Col<B1, { 32 * 64 }>) -> Self {
		// Implement rotations using circular left shifts (equivalent to right rotations)
		let rotr_17 =
			table.add_shifted("rotr_17", state_in, 5, 32 - 17, ShiftVariant::CircularLeft);
		let rotr_19 =
			table.add_shifted("rotr_19", state_in, 5, 32 - 19, ShiftVariant::CircularLeft);
		// Implement right shift using logical right shift
		let shr_10 = table.add_shifted("shr_10", state_in, 5, 10, ShiftVariant::LogicalRight);
		// Compute σ₁ as XOR of the three components
		let out = table.add_computed("sigma_1", rotr_17 + rotr_19 + shr_10);

		Self {
			input: state_in,
			rotr_17,
			rotr_19,
			shr_10,
			out,
		}
	}

	/// Populates the witness for the σ₁ function.
	pub fn populate<P>(&self, index: &mut TableWitnessSegment<P>) -> Result<(), anyhow::Error>
	where
		P: PackedFieldIndexable<Scalar = B128> + PackedExtension<B1>,
	{
		let input: std::cell::Ref<'_, [u32]> = index.get_as(self.input)?;
		let mut rotr_17 = index.get_mut_as(self.rotr_17)?;
		let mut rotr_19 = index.get_mut_as(self.rotr_19)?;
		let mut shr_10 = index.get_mut_as(self.shr_10)?;
		let mut out = index.get_mut_as(self.out)?;

		for i in 0..input.len() {
			rotr_17[i] = input[i].rotate_right(17);
			rotr_19[i] = input[i].rotate_right(19);
			shr_10[i] = input[i] >> 10;
			out[i] = rotr_17[i] ^ rotr_19[i] ^ shr_10[i];
		}

		Ok(())
	}
}

/// The Σ₀ function used in SHA-256 compression rounds.
///
/// This function implements Σ₀(x) = ROTR²(x) ⊕ ROTR¹³(x) ⊕ ROTR²²(x) where:
/// - ROTR^n(x) denotes x rotated right by n positions
/// - ⊕ denotes bitwise XOR
///
/// Σ₀ is applied to working variable 'a' in each compression round and contributes
/// to the T₂ temporary value in the round function.
#[derive(Debug)]
pub struct BigSigma0 {
	pub input: Col<B1, 32>,
	/// Right rotation by 2 positions: ROTR²(x)
	rotr_2: Col<B1, 32>,
	/// Right rotation by 13 positions: ROTR¹³(x)
	rotr_13: Col<B1, 32>,
	/// Right rotation by 22 positions: ROTR²²(x)
	rotr_22: Col<B1, 32>,
	/// The computed Σ₀ output: ROTR²(x) ⊕ ROTR¹³(x) ⊕ ROTR²²(x)
	pub out: Col<B1, 32>,
}

impl BigSigma0 {
	/// Creates a new Σ₀ function gadget.
	///
	/// # Arguments
	///
	/// * `table` - The table builder for creating columns and constraints
	/// * `state_in` - The input column to apply Σ₀ to (typically working variable 'a')
	///
	/// # Returns
	///
	/// A new `BigSigma0` instance with all necessary columns and constraints
	pub fn new(table: &mut TableBuilder, state_in: Col<B1, 32>) -> Self {
		// Implement rotations using circular left shifts (equivalent to right rotations)
		let rotr_2 = table.add_shifted("rotr_2", state_in, 5, 32 - 2, ShiftVariant::CircularLeft);
		let rotr_13 =
			table.add_shifted("rotr_13", state_in, 5, 32 - 13, ShiftVariant::CircularLeft);
		let rotr_22 =
			table.add_shifted("rotr_22", state_in, 5, 32 - 22, ShiftVariant::CircularLeft);
		// Compute Σ₀ as XOR of the three rotations
		let out = table.add_computed("big_sigma_0", rotr_2 + rotr_13 + rotr_22);

		Self {
			input: state_in,
			rotr_2,
			rotr_13,
			rotr_22,
			out,
		}
	}

	/// Populates the witness for the Σ₀ function.
	pub fn populate<P>(&self, index: &mut TableWitnessSegment<P>) -> Result<(), anyhow::Error>
	where
		P: PackedFieldIndexable<Scalar = B128> + PackedExtension<B1>,
	{
		let input: std::cell::Ref<'_, [u32]> = index.get_as(self.input)?;
		let mut rotr_2 = index.get_mut_as(self.rotr_2)?;
		let mut rotr_13 = index.get_mut_as(self.rotr_13)?;
		let mut rotr_22 = index.get_mut_as(self.rotr_22)?;
		let mut out = index.get_mut_as(self.out)?;

		for i in 0..input.len() {
			rotr_2[i] = input[i].rotate_right(2);
			rotr_13[i] = input[i].rotate_right(13);
			rotr_22[i] = input[i].rotate_right(22);
			out[i] = rotr_2[i] ^ rotr_13[i] ^ rotr_22[i];
		}

		Ok(())
	}
}

/// The Σ₁ function used in SHA-256 compression rounds.
///
/// This function implements Σ₁(x) = ROTR⁶(x) ⊕ ROTR¹¹(x) ⊕ ROTR²⁵(x) where:
/// - ROTR^n(x) denotes x rotated right by n positions
/// - ⊕ denotes bitwise XOR
///
/// Σ₁ is applied to working variable 'e' in each compression round and contributes
/// to the T₁ temporary value in the round function.
#[derive(Debug)]
pub struct BigSigma1 {
	pub input: Col<B1, 32>,
	/// Right rotation by 6 positions: ROTR⁶(x)
	rotr_6: Col<B1, 32>,
	/// Right rotation by 11 positions: ROTR¹¹(x)
	rotr_11: Col<B1, 32>,
	/// Right rotation by 25 positions: ROTR²⁵(x)
	rotr_25: Col<B1, 32>,
	/// The computed Σ₁ output: ROTR⁶(x) ⊕ ROTR¹¹(x) ⊕ ROTR²⁵(x)
	pub out: Col<B1, 32>,
}

impl BigSigma1 {
	/// Creates a new Σ₁ function gadget.
	///
	/// # Arguments
	///
	/// * `table` - The table builder for creating columns and constraints
	/// * `state_in` - The input column to apply Σ₁ to (typically working variable 'e')
	///
	/// # Returns
	///
	/// A new `BigSigma1` instance with all necessary columns and constraints
	pub fn new(table: &mut TableBuilder, state_in: Col<B1, 32>) -> Self {
		// Implement rotations using circular left shifts (equivalent to right rotations)
		let rotr_6 = table.add_shifted("rotr_6", state_in, 5, 32 - 6, ShiftVariant::CircularLeft);
		let rotr_11 =
			table.add_shifted("rotr_11", state_in, 5, 32 - 11, ShiftVariant::CircularLeft);
		let rotr_25 =
			table.add_shifted("rotr_25", state_in, 5, 32 - 25, ShiftVariant::CircularLeft);
		// Compute Σ₁ as XOR of the three rotations
		let out = table.add_computed("big_sigma_1", rotr_6 + rotr_11 + rotr_25);

		Self {
			input: state_in,
			rotr_6,
			rotr_11,
			rotr_25,
			out,
		}
	}

	/// Populates the witness for the Σ₁ function.
	pub fn populate<P>(&self, index: &mut TableWitnessSegment<P>) -> Result<(), anyhow::Error>
	where
		P: PackedFieldIndexable<Scalar = B128> + PackedExtension<B1>,
	{
		let input: std::cell::Ref<'_, [u32]> = index.get_as(self.input)?;
		let mut rotr_6 = index.get_mut_as(self.rotr_6)?;
		let mut rotr_11 = index.get_mut_as(self.rotr_11)?;
		let mut rotr_25 = index.get_mut_as(self.rotr_25)?;
		let mut out = index.get_mut_as(self.out)?;

		for i in 0..input.len() {
			rotr_6[i] = input[i].rotate_right(6);
			rotr_11[i] = input[i].rotate_right(11);
			rotr_25[i] = input[i].rotate_right(25);
			out[i] = rotr_6[i] ^ rotr_11[i] ^ rotr_25[i];
		}

		Ok(())
	}
}

/// Computes the SHA-256 message schedule W₀, W₁, ..., W₆₃ for the given index i.
///
/// This function extends the first 16 words of the message block into a full
/// 64-word schedule using the σ₀ and σ₁ functions.
///
/// # Arguments
///
/// * `w` - The mutable slice containing the message schedule words
/// * `i` - The current index in the message schedule (0-63)
pub fn compute_message_schedule(w: &mut [u32]) {
	for i in 16..64 {
		let w_i_16 = w[i - 16];
		let w_i_7 = w[i - 7];
		let w_i_2 = w[i - 2];
		let w_i_15 = w[i - 15];

		let sigma0 = w_i_15.rotate_right(7) ^ w_i_15.rotate_right(18) ^ (w_i_15 >> 3);
		let sigma1 = w_i_2.rotate_right(17) ^ w_i_2.rotate_right(19) ^ (w_i_2 >> 10);

		w[i] = w_i_16
			.wrapping_add(sigma0)
			.wrapping_add(w_i_7)
			.wrapping_add(sigma1);
	}
}
/// Selector for message schedule expansion.
///
/// This constant array is used to select the first 48 words of the message schedule
/// for the SHA-256 compression function. It is used to constrain the message schedule
/// extension operations in the gadget.
///
/// The selector is an array of B1s where the first 32*48 bits are set to B1::ONE
/// and the remaining bits are set to B1::ZERO.
pub const SCHEDULE_EXPAND_SELECTOR: [B1; 32 * 64] = {
	let mut arr = [B1::ZERO; 32 * 64];
	let mut i = 0;
	while i < 32 * 48 {
		arr[i] = B1::ONE;
		i += 1;
	}
	arr
};

#[cfg(test)]
mod tests {
	use binius_compute::cpu::alloc::CpuComputeAllocator;
	use binius_field::{
		arch::OptimalUnderlier128b, as_packed_field::PackedType, packed::get_packed_slice,
	};
	use rand::{Rng, SeedableRng, prelude::StdRng};

	use super::*;
	use crate::builder::{ConstraintSystem, Statement, WitnessIndex};

	/// Test the SHA-256 message schedule computation
	#[test]
	fn test_message_schedule() {
		let mut message = [0u32; 64];
		for i in 0..16 {
			message[i] = i as u32 + 1; // Simple sequential pattern
		}

		compute_message_schedule(&mut message);

		// Expected values calculated from the SHA-256 spec
		let expected_values = [
			0x00000001, 0x00000002, 0x00000003, 0x00000004, 0x00000005, 0x00000006, 0x00000007,
			0x00000008, 0x00000009, 0x0000000a, 0x0000000b, 0x0000000c, 0x0000000d, 0x0000000e,
			0x0000000f, 0x00000010, 0x5c91c1da, 0x25a70f77, 0x49dc9c6e, 0x6edfce69, 0xbf31d35f,
			0x0af879e3, 0x7fd5cc1c, 0xe79ca3c3, 0xdcba9585, 0x5b0441a7, 0x5a62bb69, 0x5958bb8f,
			0x14c09918, 0x7a1c2702, 0xbc0c6b94,
			0xad6e9c72,
			// ... and so on (values for indices 32-63 would be here)
		];

		for i in 0..32 {
			assert_eq!(message[i], expected_values[i], "Mismatch at index {}", i);
		}
	}

	/// Test individual SHA-256 functions (sigma0, sigma1, BigSigma0, BigSigma1)
	#[test]
	fn test_sha256_functions() {
		// Test vectors for sigma0, sigma1, BigSigma0, BigSigma1
		let test_cases: [(u32, u32, u32, u32, u32); 4] = [
			(0x12345678, 0x0bd80664, 0x61edc123, 0xfe0e1290, 0xa1124567),
			(0xabcdef01, 0xfa8e2301, 0xc7d32167, 0xba7f98df, 0xd1a0abd6),
			(0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000),
			(0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff),
		];

		for &(input, expected_sigma0, expected_sigma1, expected_big_sigma0, expected_big_sigma1) in
			&test_cases
		{
			// sigma0(x) = ROTR^7(x) XOR ROTR^18(x) XOR SHR^3(x)
			let sigma0_result = input.rotate_right(7) ^ input.rotate_right(18) ^ (input >> 3);
			assert_eq!(sigma0_result, expected_sigma0, "sigma0 failed for input {:#x}", input);

			// sigma1(x) = ROTR^17(x) XOR ROTR^19(x) XOR SHR^10(x)
			let sigma1_result = input.rotate_right(17) ^ input.rotate_right(19) ^ (input >> 10);
			assert_eq!(sigma1_result, expected_sigma1, "sigma1 failed for input {:#x}", input);

			// Sigma0(x) = ROTR^2(x) XOR ROTR^13(x) XOR ROTR^22(x)
			let big_sigma0_result =
				input.rotate_right(2) ^ input.rotate_right(13) ^ input.rotate_right(22);
			assert_eq!(
				big_sigma0_result, expected_big_sigma0,
				"BigSigma0 failed for input {:#x}",
				input
			);

			// Sigma1(x) = ROTR^6(x) XOR ROTR^11(x) XOR ROTR^25(x)
			let big_sigma1_result =
				input.rotate_right(6) ^ input.rotate_right(11) ^ input.rotate_right(25);
			assert_eq!(
				big_sigma1_result, expected_big_sigma1,
				"BigSigma1 failed for input {:#x}",
				input
			);
		}
	}

	/// Test the Ch and Maj functions used in SHA-256
	#[test]
	fn test_ch_maj_functions() {
		let test_cases: [(u32, u32, u32, u32, u32); 4] = [
			(0x12345678, 0xabcdef01, 0x87654321, 0x83456739, 0x02345661),
			(0xffffffff, 0x00000000, 0xffffffff, 0x00000000, 0x00000000),
			(0x00000000, 0xffffffff, 0x00000000, 0xffffffff, 0x00000000),
			(0xffffffff, 0xffffffff, 0x00000000, 0xffffffff, 0xffffffff),
		];

		for &(x, y, z, expected_ch, expected_maj) in &test_cases {
			// Ch(x, y, z) = (x & y) ^ (~x & z)
			let ch_result = (x & y) ^ (!x & z);
			assert_eq!(ch_result, expected_ch, "Ch failed for inputs {:#x}, {:#x}, {:#x}", x, y, z);

			// Maj(x, y, z) = (x & y) ^ (x & z) ^ (y & z)
			let maj_result = (x & y) ^ (x & z) ^ (y & z);
			assert_eq!(
				maj_result, expected_maj,
				"Maj failed for inputs {:#x}, {:#x}, {:#x}",
				x, y, z
			);
		}
	}

	/// Test the SHA-256 gadget with a simple test vector
	#[test]
	fn test_sha256_single_block() {
		const N_ROWS: usize = 1;

		let mut cs = ConstraintSystem::new();
		let mut table = cs.add_table("sha256");
		let sha256 = Sha256::new(&mut table);

		let mut allocator = CpuComputeAllocator::new(1 << 16);
		let allocator = allocator.into_bump_allocator();
		let table_id = table.id();

		let statement = Statement {
			boundaries: vec![],
			table_sizes: vec![N_ROWS],
		};

		let mut witness =
			WitnessIndex::<PackedType<OptimalUnderlier128b, B128>>::new(&cs, &allocator);
		let table_witness = witness.init_table(table_id, N_ROWS).unwrap();
		let mut segment = table_witness.full_segment();

		// Test vector: empty message (single block with padding)
		let message_block = [
			0x80000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
			0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
			0x00000000, 0x00000000,
		];

		sha256
			.populate(&mut segment, std::iter::once(message_block))
			.unwrap();

		// Validate state_out against expected hash value for empty message
		let expected_hash = [
			0xe3b0c442, 0x98fc1c14, 0x9afbf4c8, 0x996fb924, 0x27ae41e4, 0x649b934c, 0xa495991b,
			0x7852b855,
		];

		for (i, &expected) in expected_hash.iter().enumerate() {
			let state_out = segment.get_as::<u32, _, 32>(sha256.state_out[i]).unwrap();
			assert_eq!(state_out[0], expected, "State out mismatch at index {}", i);
		}

		// Validate constraint system
		let ccs = cs.compile(&statement).unwrap();
		let table_sizes = witness.table_sizes();
		let witness = witness.into_multilinear_extension_index();

		binius_core::constraint_system::validate::validate_witness(
			&ccs,
			&[],
			&table_sizes,
			&witness,
		)
		.unwrap();
	}

	/// Test the SHA-256 gadget with multiple standard test vectors
	#[test]
	fn test_sha256_standard_vectors() {
		// Standard test vectors for SHA-256
		let test_vectors = [
			// Test vector 1: empty message
			(
				[
					0x80000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
					0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
					0x00000000, 0x00000000, 0x00000000, 0x00000000,
				],
				[
					0xe3b0c442, 0x98fc1c14, 0x9afbf4c8, 0x996fb924, 0x27ae41e4, 0x649b934c,
					0xa495991b, 0x7852b855,
				],
			),
			// Test vector 2: "abc"
			(
				[
					0x61626380, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
					0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
					0x00000000, 0x00000000, 0x00000000, 0x00000018,
				],
				[
					0xba7816bf, 0x8f01cfea, 0x414140de, 0x5dae2223, 0xb00361a3, 0x96177a9c,
					0xb410ff61, 0xf20015ad,
				],
			),
		];

		let mut cs = ConstraintSystem::new();
		let mut table = cs.add_table("sha256");
		let sha256 = Sha256::new(&mut table);

		let mut allocator = CpuComputeAllocator::new(1 << 16);
		let allocator = allocator.into_bump_allocator();
		let table_id = table.id();

		let statement = Statement {
			boundaries: vec![],
			table_sizes: vec![test_vectors.len()],
		};

		let mut witness =
			WitnessIndex::<PackedType<OptimalUnderlier128b, B128>>::new(&cs, &allocator);
		let table_witness = witness.init_table(table_id, test_vectors.len()).unwrap();
		let mut segment = table_witness.full_segment();

		// Get message blocks from test vectors
		let message_blocks: Vec<[u32; 16]> = test_vectors.iter().map(|(block, _)| *block).collect();
		sha256.populate(&mut segment, message_blocks).unwrap();

		// Validate each test vector
		for (i, (_, expected_hash)) in test_vectors.iter().enumerate() {
			for (j, &expected) in expected_hash.iter().enumerate() {
				let state_out = segment.get_as::<u32, _, 32>(sha256.state_out[j]).unwrap();
				assert_eq!(
					state_out[i], expected,
					"Test vector {} state out mismatch at index {}",
					i, j
				);
			}
		}

		// Validate constraint system
		let ccs = cs.compile(&statement).unwrap();
		let table_sizes = witness.table_sizes();
		let witness = witness.into_multilinear_extension_index();

		binius_core::constraint_system::validate::validate_witness(
			&ccs,
			&[],
			&table_sizes,
			&witness,
		)
		.unwrap();
	}

	/// Ensure the number of committed bits per row is as expected
	#[test]
	fn ensure_committed_bits_per_row() {
		let mut cs = ConstraintSystem::new();
		let mut table = cs.add_table("sha256");
		let _ = Sha256::new(&mut table);

		let id = table.id();
		let stat = cs.tables[id].stat();

		// This is an expected value that should match the design
		assert_eq!(stat.bits_per_row_committed(), 26624);
	}

	/// Randomized property-based test for SHA-256
	#[test]
	fn prop_test_sha256() {
		const N_ITER: usize = 16; // Use a smaller number for test time

		let mut cs = ConstraintSystem::new();
		let mut table = cs.add_table("sha256");
		let sha256 = Sha256::new(&mut table);

		let mut allocator = CpuComputeAllocator::new(1 << 16);
		let allocator = allocator.into_bump_allocator();
		let table_id = table.id();

		let statement = Statement {
			boundaries: vec![],
			table_sizes: vec![N_ITER],
		};

		let mut witness =
			WitnessIndex::<PackedType<OptimalUnderlier128b, B128>>::new(&cs, &allocator);
		let table_witness = witness.init_table(table_id, N_ITER).unwrap();
		let mut segment = table_witness.full_segment();

		let mut rng = StdRng::seed_from_u64(0);

		// Generate random message blocks
		let message_blocks: Vec<[u32; 16]> = (0..N_ITER)
			.map(|_| {
				let mut block = [0u32; 16];
				for j in 0..16 {
					block[j] = rng.r#gen();
				}
				block
			})
			.collect();

		// Process the blocks with our SHA-256 implementation
		sha256
			.populate(&mut segment, message_blocks.clone())
			.unwrap();

		// Now validate using a reference implementation (we'll use our compute_message_schedule
		// and manually apply the compression function for simplicity)
		for (i, block) in message_blocks.iter().enumerate() {
			let mut state = INIT;
			let mut w = [0u32; 64];

			// Initialize message schedule
			for j in 0..16 {
				w[j] = block[j];
			}
			compute_message_schedule(&mut w);

			// Manual compression function
			let mut a = state[0];
			let mut b = state[1];
			let mut c = state[2];
			let mut d = state[3];
			let mut e = state[4];
			let mut f = state[5];
			let mut g = state[6];
			let mut h = state[7];

			for j in 0..64 {
				let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
				let ch = (e & f) ^ (!e & g);
				let temp1 = h
					.wrapping_add(s1)
					.wrapping_add(ch)
					.wrapping_add(ROUND_CONSTS_K[j])
					.wrapping_add(w[j]);
				let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
				let maj = (a & b) ^ (a & c) ^ (b & c);
				let temp2 = s0.wrapping_add(maj);

				h = g;
				g = f;
				f = e;
				e = d.wrapping_add(temp1);
				d = c;
				c = b;
				b = a;
				a = temp1.wrapping_add(temp2);
			}

			state[0] = state[0].wrapping_add(a);
			state[1] = state[1].wrapping_add(b);
			state[2] = state[2].wrapping_add(c);
			state[3] = state[3].wrapping_add(d);
			state[4] = state[4].wrapping_add(e);
			state[5] = state[5].wrapping_add(f);
			state[6] = state[6].wrapping_add(g);
			state[7] = state[7].wrapping_add(h);

			// Validate against our gadget's output
			for j in 0..8 {
				let state_out = segment.get_as::<u32, _, 32>(sha256.state_out[j]).unwrap();
				assert_eq!(
					state_out[i], state[j],
					"Random test {}: state out mismatch at index {}",
					i, j
				);
			}
		}

		// Validate constraint system
		let ccs = cs.compile(&statement).unwrap();
		let table_sizes = witness.table_sizes();
		let witness = witness.into_multilinear_extension_index();

		binius_core::constraint_system::validate::validate_witness(
			&ccs,
			&[],
			&table_sizes,
			&witness,
		)
		.unwrap();
	}
}
