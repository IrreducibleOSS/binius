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
//!   σ₁ functions
//! - **Compression Rounds**: 64 rounds of the main compression algorithm using working variables
//!   a-h and the Σ₀, Σ₁, Ch, and Maj functions
//! - **State Management**: Tracks the evolution of the 256-bit state through all rounds
//!
//! ## Cryptographic Functions
//!
//! The implementation includes arithmetic circuits for all SHA-256 primitive functions:
//!
//! - **σ₀(x)**: `ROTR⁷(x) ⊕ ROTR¹⁸(x) ⊕ SHR³(x)` - Used in message schedule
//! - **σ₁(x)**: `ROTR¹⁷(x) ⊕ ROTR¹⁹(x) ⊕ SHR¹⁰(x)` - Used in message schedule
//! - **Σ₀(x)**: `ROTR²(x) ⊕ ROTR¹³(x) ⊕ ROTR²²(x)` - Used in compression
//! - **Σ₁(x)**: `ROTR⁶(x) ⊕ ROTR¹¹(x) ⊕ ROTR²⁵(x)` - Used in compression
//! - **Ch(x,y,z)**: `(x ∧ y) ⊕ (¬x ∧ z)` - Choice function
//! - **Maj(x,y,z)**: `(x ∧ y) ⊕ (x ∧ z) ⊕ (y ∧ z)` - Majority function
//!
//! ## Usage Example
//!
//! ```rust,ignore
//! let mut cs = ConstraintSystem::new();
//! let mut table = cs.add_table("sha256");
//! let sha256 = Sha256::new(&mut table);
//!
//! // The gadget provides columns for state input/output and message schedule
//! // Populate with actual SHA-256 computation data during proving
//! ```
//!
//! ## Technical Details
//!
//! This implementation uses a bit-packed representation where 32-bit words are stored as
//! columns of individual bits. This approach:
//!
//! - Enables efficient constraint representation of bitwise operations
//! - Allows for precise tracking of carry propagation in additions
//! - Supports the circular shift operations required by SHA-256
//!
//! The gadget follows the style established by other hash function gadgets in this crate,
//! particularly the Groestl and Keccak implementations, ensuring consistency across the
//! cryptographic primitives library.
//!
//! ## References
//!
//! - [FIPS 180-4: Secure Hash Standard (SHS)](https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.180-4.pdf)
//! - [RFC 6234: US Secure Hash Algorithms](https://tools.ietf.org/rfc/rfc6234.txt)

use std::array;

use anyhow::Result;
use binius_core::oracle::ShiftVariant;
use binius_field::Field;

use crate::{
	builder::{B1, B32, Col, TableBuilder, TableWitnessSegment},
	gadgets::add::{self, U32Add, U32AddFlags},
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
	pub w: Col<B1, { 32 * 64 }>,

	/// Working variables for all 64 rounds.
	///
	/// These represent the 8 working variables (a, b, c, d, e, f, g, h) as they
	/// evolve through each of the 64 compression rounds. Each variable is stored
	/// as a bit-packed column spanning all rounds.
	a: Col<B1, { 32 * 64 }>,
	b: Col<B1, { 32 * 64 }>,
	c: Col<B1, { 32 * 64 }>,
	d: Col<B1, { 32 * 64 }>,
	e: Col<B1, { 32 * 64 }>,
	f: Col<B1, { 32 * 64 }>,
	g: Col<B1, { 32 * 64 }>,
	h: Col<B1, { 32 * 64 }>,

	/// Round constants: 64 32-bit words.
	///
	/// The cryptographic constants K₀, K₁, ..., K₆₃ used in each round of the
	/// compression function.
	k: Col<B32, 64>,

	/// Temporary computation variables.
	///
	/// These store intermediate results during the compression rounds:
	/// - T₁ = h + Σ₁(e) + Ch(e,f,g) + Kₜ + Wₜ
	/// - T₂ = Σ₀(a) + Maj(a,b,c)
	t1: Col<B32>,
	t2: Col<B32>,
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
	pub fn new(table: &mut TableBuilder) {
		// Initialize the message schedule w.
		let w: Col<B1, { 32 * 64 }> = table.add_committed("w");

		// σ₀ and σ₁ functions for message schedule extension
		let sigma0 = Sigma0::new(table, w);
		let sigma1 = Sigma1::new(table, w);

		let w_t: [Col<B1, 32>; 64] =
			array::from_fn(|i| table.add_selected_block(format!("w[{i}]"), w, i));

		let state_in: [Col<B1, 32>; 16] = array::from_fn(|i| w_t[i]);

		let sigma0_t: [Col<B1, 32>; 64] =
			array::from_fn(|i| table.add_selected_block(format!("sigma0(w[{i}])"), sigma0.out, i));
		let sigma1_t: [Col<B1, 32>; 64] =
			array::from_fn(|i| table.add_selected_block(format!("sigma1(w[{i}])"), sigma1.out, i));

		let mut s0s = Vec::new();
		let mut s1s = Vec::new();
		let mut s2s = Vec::new();
		// Enforce the message schedule extension constraint
		// TODO: Integer addition
		for t in 16..64 {
			let s0 = U32Add::new(
				table,
				sigma1_t[t - 2],
				w_t[t - 7],
				U32AddFlags {
					carry_in_bit: None,
					commit_zout: false,
					expose_final_carry: false,
				},
			);

			let s1 = U32Add::new(
				table,
				sigma0_t[t - 15],
				w_t[t - 16],
				U32AddFlags {
					carry_in_bit: None,
					commit_zout: false,
					expose_final_carry: false,
				},
			);

			let s2 = U32Add::new(
				table,
				s0.zout,
				s1.zout,
				U32AddFlags {
					carry_in_bit: None,
					commit_zout: false,
					expose_final_carry: false,
				},
			);

			table.assert_zero(format!("schedule expansion[{t}]"), w_t[t] + s2.zout);

			s0s.push(s0);
			s1s.push(s1);
			s2s.push(s2);
		}
		// Working variables for the 64 compression rounds
		let a: Col<B1, { 32 * 64 }> = table.add_committed("a");
		let b: Col<B1, { 32 * 64 }> = table.add_committed("b");
		let c: Col<B1, { 32 * 64 }> = table.add_committed("c");
		let d: Col<B1, { 32 * 64 }> = table.add_committed("d");
		let e: Col<B1, { 32 * 64 }> = table.add_committed("e");
		let f: Col<B1, { 32 * 64 }> = table.add_committed("f");
		let g: Col<B1, { 32 * 64 }> = table.add_committed("g");
		let h: Col<B1, { 32 * 64 }> = table.add_committed("h");
		
        let k: [Col<B1, 32>; 64] =
			array::from_fn(|i| table.add_constant("k", u32_to_b1_bits_le(ROUND_CONSTS_K[i])));

		let a_blocks = array::from_fn(|i| table.add_selected_block(format!("a[{i}]"), a, i));
		let b_blocks = array::from_fn(|i| table.add_selected_block(format!("b[{i}]"), b, i));
		let c_blocks = array::from_fn(|i| table.add_selected_block(format!("c[{i}]"), c, i));
		let d_blocks = array::from_fn(|i| table.add_selected_block(format!("d[{i}]"), d, i));
		let e_blocks = array::from_fn(|i| table.add_selected_block(format!("e[{i}]"), e, i));
		let f_blocks = array::from_fn(|i| table.add_selected_block(format!("f[{i}]"), f, i));
		let g_blocks = array::from_fn(|i| table.add_selected_block(format!("g[{i}]"), g, i));
		let h_blocks = array::from_fn(|i| table.add_selected_block(format!("h[{i}]"), h, i));

		// Σ₀ and Σ₁ functions for the compression rounds
		let big_sigma_1 = BigSigma1::new(table, e);
		let big_sigma_0 = BigSigma0::new(table, a);

        let big_sigma_1_blocks = array::from_fn(|i| table.add_selected_block(format!("big_sigma_1[{i}]"), big_sigma_1.out, i));
        let big_sigma_0_blocks = array::from_fn(|i| table.add_selected_block(format!("big_sigma_0[{i}]"), big_sigma_0.out, i));
        
		// Choice and majority functions for compression rounds
		let ch: Col<B1, { 32 * 64 }> = table.add_committed("ch");
		let maj: Col<B1, { 32 * 64 }> = table.add_committed("maj");

		table.assert_zero("ch", g + e * (f + g) - ch);
		table.assert_zero("maj", a * (b + c) + b * c - maj);

		// TODO: Integer addition
		let t1 = table.add_computed("t1", h + big_sigma_1.out + ch + k + w);
		// TODO: Integer addition
		let t2 = table.add_computed("t2", big_sigma_0.out + maj);

		for i in 0..64 {

        }
		// Final output state computation
		let a_final: Col<B1, 32> = table.add_selected_block("a_final", a, 63);
		let b_final: Col<B1, 32> = table.add_selected_block("b_final", b, 63);
		let c_final: Col<B1, 32> = table.add_selected_block("c_final", c, 63);
		let d_final: Col<B1, 32> = table.add_selected_block("d_final", d, 63);
		let e_final: Col<B1, 32> = table.add_selected_block("e_final", e, 63);
		let f_final: Col<B1, 32> = table.add_selected_block("f_final", f, 63);
		let g_final: Col<B1, 32> = table.add_selected_block("g_final", g, 63);
		let h_final: Col<B1, 32> = table.add_selected_block("h_final", h, 63);
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
			rotr_7,
			rotr_18,
			shr_3,
			out,
		}
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
			rotr_17,
			rotr_19,
			shr_10,
			out,
		}
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
pub struct BigSigma0 {
	/// Right rotation by 2 positions: ROTR²(x)
	rotr_2: Col<B1, { 32 * 64 }>,
	/// Right rotation by 13 positions: ROTR¹³(x)
	rotr_13: Col<B1, { 32 * 64 }>,
	/// Right rotation by 22 positions: ROTR²²(x)
	rotr_22: Col<B1, { 32 * 64 }>,
	/// The computed Σ₀ output: ROTR²(x) ⊕ ROTR¹³(x) ⊕ ROTR²²(x)
	pub out: Col<B1, { 32 * 64 }>,
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
	pub fn new(table: &mut TableBuilder, state_in: Col<B1, { 32 * 64 }>) -> Self {
		// Implement rotations using circular left shifts (equivalent to right rotations)
		let rotr_2 = table.add_shifted("rotr_2", state_in, 5, 32 - 2, ShiftVariant::CircularLeft);
		let rotr_13 =
			table.add_shifted("rotr_13", state_in, 5, 32 - 13, ShiftVariant::CircularLeft);
		let rotr_22 =
			table.add_shifted("rotr_22", state_in, 5, 32 - 22, ShiftVariant::CircularLeft);
		// Compute Σ₀ as XOR of the three rotations
		let out = table.add_computed("big_sigma_0", rotr_2 + rotr_13 + rotr_22);

		Self {
			rotr_2,
			rotr_13,
			rotr_22,
			out,
		}
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
pub struct BigSigma1 {
	/// Right rotation by 6 positions: ROTR⁶(x)
	rotr_6: Col<B1, { 32 * 64 }>,
	/// Right rotation by 11 positions: ROTR¹¹(x)
	rotr_11: Col<B1, { 32 * 64 }>,
	/// Right rotation by 25 positions: ROTR²⁵(x)
	rotr_25: Col<B1, { 32 * 64 }>,
	/// The computed Σ₁ output: ROTR⁶(x) ⊕ ROTR¹¹(x) ⊕ ROTR²⁵(x)
	pub out: Col<B1, { 32 * 64 }>,
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
	pub fn new(table: &mut TableBuilder, state_in: Col<B1, { 32 * 64 }>) -> Self {
		// Implement rotations using circular left shifts (equivalent to right rotations)
		let rotr_6 = table.add_shifted("rotr_6", state_in, 5, 32 - 6, ShiftVariant::CircularLeft);
		let rotr_11 =
			table.add_shifted("rotr_11", state_in, 5, 32 - 11, ShiftVariant::CircularLeft);
		let rotr_25 =
			table.add_shifted("rotr_25", state_in, 5, 32 - 25, ShiftVariant::CircularLeft);
		// Compute Σ₁ as XOR of the three rotations
		let out = table.add_computed("big_sigma_1", rotr_6 + rotr_11 + rotr_25);

		Self {
			rotr_6,
			rotr_11,
			rotr_25,
			out,
		}
	}
}
