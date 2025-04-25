// Copyright 2025 Irreducible Inc.

mod model {
	use binius_hash::groestl::{GroestlShortImpl, GroestlShortInternal};
	use binius_m3::{
		builder::{B32, B8},
		emulate::Channel,
	};
	use bytemuck::{cast_slice, must_cast_mut, must_cast_ref};

	// Channels:
	// - Hash verifications
	// - Hash state updates
	// - Memory channel
	//
	// 3 tables:
	// - InitHash
	// - CompressBlock
	// - FinalHash

	// See Grøstl specification, section 3.5.
	const INIT_STATE: [B32; 16] = [
		B32::new(0x00),
		B32::new(0x00),
		B32::new(0x00),
		B32::new(0x00),
		B32::new(0x00),
		B32::new(0x00),
		B32::new(0x00),
		B32::new(0x00),
		B32::new(0x00),
		B32::new(0x00),
		B32::new(0x00),
		B32::new(0x00),
		B32::new(0x00),
		B32::new(0x00),
		B32::new(0x00),
		B32::new(0x00000010),
	];

	/// Grøstl-256 block size, measured in 32-bit words.
	const BLOCK_SIZE: u16 = 16;

	type GroestlState = [B32; 16];

	/// The InitHash table pulls hash requests and pushes hash state tokens.
	pub struct InitHashEvent {
		pub call: GroestlCallToken,
	}

	impl InitHashEvent {
		pub fn fire(
			&self,
			hash_call_chan: &mut Channel<GroestlCallToken>,
			hash_state_chan: &mut Channel<GroestlStateToken>,
		) {
			let GroestlCallToken {
				data_ptr,
				data_len,
				hash_ptr,
			} = self.call.clone();
			hash_call_chan.pull(self.call.clone());
			hash_state_chan.push(GroestlStateToken {
				data_ptr,
				data_len,
				hash_ptr,
				state: INIT_STATE,
			});
		}
	}

	pub struct CompressBlockEvent {
		pub token: GroestlStateToken,
		pub block: GroestlState,
		pub out_state: GroestlState,
	}

	impl CompressBlockEvent {
		pub fn fire(
			&self,
			rom_chan: &mut Channel<ROMEntry>,
			hash_state_chan: &mut Channel<GroestlStateToken>,
		) {
			let GroestlStateToken {
				data_ptr,
				data_len,
				hash_ptr,
				state,
			} = self.token.clone();

			assert!(data_len >= BLOCK_SIZE);

			// Read block from ROM.
			for i in 0..BLOCK_SIZE as u32 {
				rom_chan.pull(ROMEntry {
					addr: B32::new(data_ptr.val() + i),
					value: self.block[i as usize],
				});
			}

			// Assert that the compression function is applied on token state, block and produces
			// out_state.
			let mut state =
				GroestlShortImpl::state_from_bytes(must_cast_ref::<GroestlState, [u8; 64]>(&state));
			GroestlShortImpl::compress(
				&mut state,
				must_cast_ref::<GroestlState, [u8; 64]>(&self.block),
			);
			assert_eq!(
				&GroestlShortImpl::state_to_bytes(&state),
				must_cast_ref::<GroestlState, [u8; 64]>(&self.out_state)
			);

			hash_state_chan.pull(self.token.clone());
			hash_state_chan.push(GroestlStateToken {
				data_ptr: B32::new(data_ptr.val() + BLOCK_SIZE as u32),
				data_len: data_len - BLOCK_SIZE,
				hash_ptr,
				state: self.out_state,
			});
		}
	}

	pub struct FinalizeHashEvent {
		pub token: GroestlStateToken,
		pub block: GroestlState,
		pub out_state: GroestlState,
	}

	impl FinalizeHashEvent {
		pub fn fire(
			&self,
			rom_chan: &mut Channel<ROMEntry>,
			hash_state_chan: &mut Channel<GroestlStateToken>,
		) {
			let GroestlStateToken {
				data_ptr,
				data_len,
				hash_ptr,
				state: token_state,
			} = self.token.clone();

			assert!(data_len < BLOCK_SIZE);

			// Read block from ROM.
			for i in 0..data_len as u32 {
				rom_chan.pull(ROMEntry {
					addr: B32::new(data_ptr.val() + i),
					value: self.block[i as usize],
				});
			}

			// Check the padding on block[data_ptr..]

			// Read digest from ROM.
			for i in 0..32 {
				rom_chan.pull(ROMEntry {
					addr: B32::new(hash_ptr.val() + i),
					value: self.out_state[8 + i as usize],
				});
			}

			// Assert that the compression function is applied on token state, block and produces
			// out_state.
			let mut state = GroestlShortImpl::state_from_bytes(must_cast_ref::<
				GroestlState,
				[u8; 64],
			>(&token_state));
			GroestlShortImpl::compress(
				&mut state,
				must_cast_ref::<GroestlState, [u8; 64]>(&self.block),
			);
			GroestlShortImpl::compress(
				&mut state,
				must_cast_ref::<GroestlState, [u8; 64]>(&self.block),
			);
			GroestlShortImpl::p_perm(&mut state);
			G::xor_state(&mut res, &self.state);

			assert_eq!(
				&GroestlShortImpl::state_to_bytes(&state),
				must_cast_ref::<GroestlState, [u8; 64]>(&self.out_state)
			);

			// TODO: Assert the groestl compression

			hash_state_chan.pull(self.token.clone());
		}
	}

	#[derive(Debug, Clone, PartialEq, Eq, Hash)]
	pub struct ROMEntry {
		addr: B32,
		value: B32,
	}

	pub struct GroestlHashTrace {
		pub init_hash: Vec<InitHashEvent>,
		pub compress_block: Vec<CompressBlockEvent>,
		pub rom: Vec<ROMEntry>,
	}

	/// A call to assert a Grøstl-256 hash.
	///
	/// Tokens are consumed from the channel only if the hash is valid.
	#[derive(Debug, Clone, PartialEq, Eq, Hash)]
	pub struct GroestlCallToken {
		/// The pointer to the data in the ROM to be hashed.
		pub data_ptr: B32,
		/// The length of the data in bytes, capped at 64 KiB.
		pub data_len: u16,
		/// The pointer to the 32-byte digest in the ROM.
		pub hash_ptr: B32,
	}

	#[derive(Debug, Clone, PartialEq, Eq, Hash)]
	pub struct GroestlStateToken {
		/// The pointer to the data in the ROM to be hashed.
		pub data_ptr: B32,
		/// The length of the data in bytes, capped at 64 KiB.
		pub data_len: u16,
		/// The pointer to the 32-byte digest in the ROM.
		pub hash_ptr: B32,
		/// The 64-byte Grøstl-256 state.
		pub state: GroestlState,
	}

	impl GroestlHashTrace {
		/// Validate the Groestl hash statements.
		///
		/// The boundary values are sets of Grøstl-256 hash calls and ROM reads.
		pub fn validate(&self, hash_calls: &[GroestlCallToken], rom_reads: &[ROMEntry]) {
			let mut rom_chan = Channel::default();
			let mut hash_call_chan = Channel::default();
			let mut hash_state_chan = Channel::default();

			for call in hash_calls {
				hash_call_chan.push(call.clone());
			}
			for entry in rom_reads {
				rom_chan.pull(entry.clone());
			}

			for event in &self.init_hash {
				event.fire(&mut sequence_chan);
			}
			assert!(rom_chan.is_balanced());
			assert!(hash_call_chan.is_balanced());
			assert!(hash_state_chan.is_balanced());
		}
	}
}
