// Copyright 2025 Irreducible Inc.

//! A lightweight version of the Grøstl-256 hash function specialized for 1024 byte long messages.

mod model {
	use std::{array, cell::Cell, collections::BTreeMap};

	use binius_hash::groestl::{GroestlShortImpl, GroestlShortInternal};
	use binius_m3::{builder::B32, emulate::Channel};
	use bytemuck::{must_cast, must_cast_ref};

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
	#[rustfmt::skip]
	const INIT_STATE: [u8; 64] = [
		0x00, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00,
	 	0x00, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x01, 0x00,
	];

	/// A word size in bytes that could be fetched from ROM per request.
	const WORD_SIZE: usize = 4;
	/// A number of bytes per Grøstl-256 block.
	const BLOCK_SIZE_BYTES: usize = 64;
	/// Grøstl-256 block size, measured in 32-bit words.
	const BLOCK_SIZE_WORDS: usize = BLOCK_SIZE_BYTES / WORD_SIZE;

	/// The number of bytes per the message to be hashed.
	const MESSAGE_SIZE_BYTES: usize = 1024;
	const MESSAGE_SIZE_WORDS: usize = MESSAGE_SIZE_BYTES / 4;
	const HASH_SIZE_WORDS: usize = 8;

	/// The number of full Grøstl-256 blocks required to fill the message.
	const MSG_BLOCK_COUNT: usize = MESSAGE_SIZE_BYTES / BLOCK_SIZE_BYTES;
	/// That is the total number of blocks including the padding block.
	const TOTAL_BLOCK_COUNT: usize = MSG_BLOCK_COUNT + 1;

	type GroestlState = [B32; 16];

	// Tokens: types that are pushed to/pulled from channels.

	/// A call to assert a Grøstl-256 hash.
	///
	/// Tokens are consumed from the channel only if the hash is valid.
	#[derive(Debug, Copy, Clone, PartialEq, Eq, Ord, PartialOrd)]
	pub struct GroestlCallToken {
		/// The pointer to the data in the ROM to be hashed.
		///
		/// This is assumed to be of fixed length.
		pub data_ptr: B32,
		/// The pointer to the 32-byte digest in the ROM.
		pub hash_ptr: B32,
	}

	#[derive(Debug, Clone, PartialEq, PartialOrd, Ord, Eq)]
	pub struct GroestlCompressCallToken {
		/// The pointer in the ROM to the next block of the message to be processed.
		pub data_ptr: B32,
		/// The index of this compression.
		pub compression_index: u16,
		/// The pointer to the 32-byte digest in the ROM.
		///
		/// This exists to plumb the hash ptr to the finalization table. It's not used in the
		/// compression table.
		pub hash_ptr: B32,
		/// The 64-byte Grøstl-256 state.
		pub in_state: GroestlState,
	}

	/// The InitHash table pulls hash requests and pushes hash state tokens.
	pub struct InitHashEvent {
		pub call: GroestlCallToken,
	}

	impl InitHashEvent {
		pub fn fire(
			&self,
			hash_call_chan: &mut Channel<GroestlCallToken>,
			hash_state_chan: &mut Channel<GroestlCompressCallToken>,
		) {
			let GroestlCallToken { data_ptr, hash_ptr } = self.call;
			hash_call_chan.pull(self.call);
			hash_state_chan.push(GroestlCompressCallToken {
				data_ptr,
				hash_ptr,
				compression_index: 0,
				in_state: must_cast(INIT_STATE),
			});
		}
	}

	/// A row in the CompressBlock table. Pushes and pulls the compression tokens.
	pub struct CompressEvent {
		pub token: GroestlCompressCallToken,
		pub block: GroestlState,
		pub out_state: GroestlState,
	}

	impl CompressEvent {
		pub fn fire(
			&self,
			rom_chan: &mut Channel<RomEntry>,
			hash_state_chan: &mut Channel<GroestlCompressCallToken>,
		) {
			let GroestlCompressCallToken {
				data_ptr,
				compression_index,
				hash_ptr,
				in_state,
			} = self.token.clone();

			assert!((compression_index as usize) < MSG_BLOCK_COUNT);

			// Read block from ROM.
			for i in 0..BLOCK_SIZE_WORDS as u32 {
				rom_chan.pull(RomEntry {
					addr: B32::new(data_ptr.val() + i),
					value: self.block[i as usize],
				});
			}

			// Assert that the compression function is applied on token state, block and produces
			// out_state.
			let out_state = Groestl::from_state(in_state).compress(self.block).state();
			assert_eq!(self.out_state, out_state);

			hash_state_chan.pull(self.token.clone());
			hash_state_chan.push(GroestlCompressCallToken {
				data_ptr: B32::new(data_ptr.val() + BLOCK_SIZE_WORDS as u32),
				compression_index: compression_index + 1,
				hash_ptr,
				in_state: out_state,
			});
		}
	}

	/// A row in the FinalHash table. Pulls a compression token.
	pub struct FinalizeHashEvent {
		pub token: GroestlCompressCallToken,
		/// The last block to apply compression function to.
		pub block: GroestlState,
		/// The final output of the compression function.
		pub out_state: GroestlState,
	}

	impl FinalizeHashEvent {
		pub fn fire(
			&self,
			rom_chan: &mut Channel<RomEntry>,
			hash_state_chan: &mut Channel<GroestlCompressCallToken>,
		) {
			let GroestlCompressCallToken {
				data_ptr,
				compression_index,
				hash_ptr,
				in_state,
			} = self.token.clone();

			// Assert that the compression index is the last block.
			assert_eq!(compression_index as usize, MSG_BLOCK_COUNT - 1);

			// Reads the last block from the ROM.
			for i in 0..BLOCK_SIZE_WORDS as u32 {
				rom_chan.pull(RomEntry {
					addr: B32::new(data_ptr.val() + i),
					value: self.block[i as usize],
				});
			}

			// Read digest from ROM.
			for i in 0..HASH_SIZE_WORDS as u32 {
				rom_chan.pull(RomEntry {
					addr: B32::new(hash_ptr.val() + i),
					// This is essentially truncation to the last 256 bits of the state.
					value: self.out_state[8 + i as usize],
				});
			}

			let out_state = Groestl::from_state(in_state)
				.compress(self.block)
				.compress_padding(TOTAL_BLOCK_COUNT)
				.transform_output()
				.state();
			assert_eq!(&out_state, &self.out_state);

			hash_state_chan.pull(self.token.clone());
		}
	}

	#[derive(Debug, Copy, Clone, PartialOrd, Ord, PartialEq, Eq, Hash)]
	pub struct RomEntry {
		/// The address of a word requested from the ROM.
		addr: B32,
		/// The associated value.
		value: B32,
	}

	/// 32-bit sparse read-only memory addressed by words.
	pub struct Rom {
		/// Maps from a 32-bit address to a 32-bit value and it the number of read accesses
		/// performed on it.
		address_space: BTreeMap<B32, (B32, Cell<u8>)>,
	}

	impl Rom {
		pub fn new() -> Self {
			Self {
				address_space: BTreeMap::new(),
			}
		}

		pub fn write_word(&mut self, addr: B32, word: B32) {
			self.address_space.insert(addr, (word, Cell::new(0)));
		}

		pub fn write_hash(&mut self, addr: B32, hash: [u8; 32]) {
			let hash: [u32; 8] = must_cast(hash);
			for (i, word) in hash.into_iter().enumerate() {
				self.write_word(B32::from(addr.val() + i as u32), B32::from(word));
			}
		}

		pub fn write_message(&mut self, addr: B32, message: [u8; MESSAGE_SIZE_BYTES]) {
			let message: [u32; MESSAGE_SIZE_WORDS] = must_cast(message);
			for (i, word) in message.into_iter().enumerate() {
				self.write_word(B32::from(addr.val() + i as u32), B32::from(word));
			}
		}

		/// Read a word from the given address and note it in the log.
		///
		/// Panics if reads from a previously unwritten address or if the maximum number of reads
		/// exceeded.
		pub fn read_word(&mut self, addr: B32) -> B32 {
			let Some((value, n_reads)) = self.address_space.get(&addr) else {
				panic!("{addr} was never written before")
			};
			let Some(v) = n_reads.get().checked_add(1) else {
				panic!("{addr} was read maximum number of times");
			};
			n_reads.set(v);
			*value
		}

		/// Read a (Grøstl-256) block and note the reads in the log.
		///
		/// Panics if reads from a previously unwritten address or if the maximum number of reads
		/// exceeded.
		pub fn read_block(&mut self, addr: B32) -> [B32; BLOCK_SIZE_WORDS] {
			array::from_fn(|i| {
				let addr = B32::from(addr.val() + i as u32);
				self.read_word(addr)
			})
		}

		/// Read a 256-bit digest and note the reads in the log.
		///
		/// Panics if reads from a previously unwritten address or if the maximum number of reads
		/// exceeded.
		pub fn read_digest(&mut self, addr: B32) -> [B32; HASH_SIZE_WORDS] {
			array::from_fn(|i| {
				let addr = B32::from(addr.val() + i as u32);
				self.read_word(addr)
			})
		}

		/// Returns the log of reads performed on this ROM.
		pub fn rom_reads<'a>(&'a self) -> impl Iterator<Item = RomEntry> + 'a {
			self.address_space
				.iter()
				.filter(|(_, (_, n_reads))| n_reads.get() > 0)
				.flat_map(|(&addr, (value, n_reads))| {
					std::iter::repeat_n(
						RomEntry {
							addr,
							value: *value,
						},
						n_reads.get() as usize,
					)
				})
		}
	}

	pub struct GroestlHashTrace {
		pub init_hash: Vec<InitHashEvent>,
		pub compress_block: Vec<CompressEvent>,
		pub finalize_hash: Vec<FinalizeHashEvent>,
	}

	impl GroestlHashTrace {
		pub fn generate(hash_calls: &[GroestlCallToken], rom: &mut Rom) -> Self {
			assert!(!hash_calls.is_empty(), "empty table is trivial");

			let mut me = Self {
				init_hash: Vec::new(),
				compress_block: Vec::new(),
				finalize_hash: Vec::new(),
			};

			for call @ &GroestlCallToken { data_ptr, hash_ptr } in hash_calls {
				me.init_hash.push(InitHashEvent { call: *call });

				let mut token = GroestlCompressCallToken {
					data_ptr,
					compression_index: 0,
					hash_ptr,
					in_state: must_cast(INIT_STATE),
				};

				for compression_index in 0..MSG_BLOCK_COUNT {
					let block = rom.read_block(token.data_ptr);
					let g = Groestl::from_state(token.in_state).compress(block);

					if compression_index < MSG_BLOCK_COUNT - 1 {
						// This is normal compression. Generate token for the next block.
						let out_state = g.state();
						let next_token = GroestlCompressCallToken {
							data_ptr: B32::from(token.data_ptr.val() + BLOCK_SIZE_WORDS as u32),
							compression_index: (compression_index + 1) as u16,
							hash_ptr,
							in_state: out_state,
						};
						me.compress_block.push(CompressEvent {
							token,
							block,
							out_state,
						});
						token = next_token;
					} else {
						// This is the last block. That means we should run the finalization.
						// At this point `g` already has the last block compressed, so we are
						// only left with compressing the padding block and transforming the
						// output.
						let out_state = g
							.compress_padding(TOTAL_BLOCK_COUNT)
							.transform_output()
							.state();

						// Note the digest read.
						let _digest = rom.read_digest(token.hash_ptr);

						me.finalize_hash.push(FinalizeHashEvent {
							token,
							block,
							out_state,
						});

						// Explicit break is required to please borrowck.
						break;
					}
				}
			}

			assert_eq!(me.init_hash.len(), hash_calls.len());
			assert_eq!(me.finalize_hash.len(), hash_calls.len());
			assert_eq!(me.compress_block.len(), hash_calls.len() * 15);

			me
		}

		/// Validate the Groestl hash statements.
		///
		/// The boundary values are sets of Grøstl-256 hash calls and ROM reads.
		pub fn validate(
			&self,
			hash_calls: &[GroestlCallToken],
			rom_reads: impl Iterator<Item = RomEntry>,
		) {
			let mut rom_chan = Channel::default();
			let mut hash_call_chan = Channel::default();
			let mut hash_state_chan = Channel::default();

			for call in hash_calls {
				hash_call_chan.push(*call);
			}
			for entry in rom_reads {
				rom_chan.push(entry);
			}

			for event in &self.init_hash {
				event.fire(&mut hash_call_chan, &mut hash_state_chan);
			}
			for event in &self.compress_block {
				event.fire(&mut rom_chan, &mut hash_state_chan);
			}
			for event in &self.finalize_hash {
				event.fire(&mut rom_chan, &mut hash_state_chan);
			}

			rom_chan.assert_balanced();
			hash_call_chan.assert_balanced();
			hash_state_chan.assert_balanced();
		}
	}

	/// Utility that defines Grøstl-256 on the B32 domain.
	struct Groestl {
		state: <GroestlShortImpl as GroestlShortInternal>::State,
	}

	impl Groestl {
		fn from_state(state: GroestlState) -> Self {
			let state = GroestlShortImpl::state_from_bytes(must_cast_ref(&state));
			Self { state }
		}

		fn compress(mut self, block: GroestlState) -> Self {
			GroestlShortImpl::compress(
				&mut self.state,
				must_cast_ref::<GroestlState, [u8; 64]>(&block),
			);
			self
		}

		fn compress_padding(mut self, block_len: usize) -> Self {
			// The padding block consists of:
			// | Example      | Name
			// | 0x80u8       | delimiter.
			// | [0x00u8; 55] | zero padding
			// | 0x11u64      | length, as u64 BE.
			let mut padding_block = [0u8; 64];
			padding_block[0] = 0x80;
			padding_block[56..64].copy_from_slice(&(block_len as u64).to_be_bytes());
			GroestlShortImpl::compress(
				&mut self.state,
				must_cast_ref::<[u8; 64], [u8; 64]>(&padding_block),
			);
			self
		}

		/// Perform the output transformation: P(H) ⊕ H.
		fn transform_output(mut self) -> Self {
			let mut p_perm = self.state;
			GroestlShortImpl::p_perm(&mut p_perm);
			GroestlShortImpl::xor_state(&mut self.state, &p_perm);
			self
		}

		fn state(&self) -> GroestlState {
			must_cast(GroestlShortImpl::state_to_bytes(&self.state))
		}
	}

	#[test]
	fn test_groestl_hash_high_level_model() {
		let mut rom = Rom::new();
		let mut hash_calls = Vec::new();

		let mut exercise_hash = |hash_ptr: u32, data_ptr: u32, message: [u8; 1024]| {
			use binius_hash::groestl::Groestl256;
			use digest::Digest;
			let hash_ptr = B32::from(hash_ptr);
			let data_ptr = B32::from(data_ptr);
			let hash = Groestl256::digest(message);
			rom.write_hash(hash_ptr, hash.into());
			rom.write_message(data_ptr, message);
			hash_calls.push(GroestlCallToken { data_ptr, hash_ptr });
		};

		// Test a simple message.
		exercise_hash(16, 24, [0x01u8; 1024]);
		// Exercise it once more. This will overwrite the memory for hash and message and add
		// a hash call. The former is idempotent but the latter is not, which we are after.
		// Specifically, we want to test multiplicites.
		exercise_hash(16, 24, [0x01u8; 1024]);
		// Exercise endianess.
		exercise_hash(1048, 1080, array::from_fn(|i| (i % 255) as u8));
		// Close to the end of heap.
		exercise_hash(u32::MAX - 1024 - 32, u32::MAX - 1024, [0u8; 1024]);

		let trace = GroestlHashTrace::generate(&hash_calls, &mut rom);
		trace.validate(&hash_calls, rom.rom_reads());
	}
}
