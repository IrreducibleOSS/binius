/// High-level model for binary Merkle trees using the Grøstl-256 output transformation as a 2-to-1
/// compression function.

mod model {
	use binius_core::constraint_system::channel::ChannelId;
	use binius_hash::{
		groestl::{Groestl256, Groestl256ByteCompression},
		PseudoCompressionFunction,
	};
	use binius_m3::{
		builder::{Col, ConstraintSystem, B1, B32, B8},
		emulate::Channel,
	};
	use digest::{generic_array::GenericArray, typenum::Gr};

	// Signature of the Nodes channel: (Root ID, Data, Depth, Index)
	type NodeFlush = (u8, [u8; 32], u32, u32);
	// Signature of the Roots channel: (Root ID, Root digest)
	type RootFlush = (u8, [u8; 32]);

	/// A struct whose fields contain the channels involved in the trace to verify merkle paths for
	/// a binary merkle tree
	pub struct MerkleTreeChannels {
		/// This channel gets flushed with tokens during "intermediate" steps of the verification
		/// where the tokens are the values of the parent digest along with associated position
		/// information such as the root it is associated to, the value of the digest, the depth
		/// and the index.
		nodes: Channel<NodeFlush>,

		///	This channel contains flushes that validate that the "final" digest obtained in a
		/// merkle path is matches that of one of the claimed roots, pushed as boundary values.
		roots: Channel<RootFlush>,
	}

	impl MerkleTreeChannels {
		pub fn new() -> Self {
			Self {
				nodes: Channel::default(),
				roots: Channel::default(),
			}
		}
	}
	/// A token representing a step in verifying a merkle path for inclusion.
	pub struct MerklePathToken {
		pub root_id: u8,
		pub left: [u8; 32],
		pub right: [u8; 32],
		pub parent: [u8; 32],
		pub parent_depth: u32,
		pub parent_index: u32,
		pub flush_left: bool,
		pub flush_right: bool,
	}

	/// A token representing the final step of comparing the claimed root.
	pub struct MerkleRootToken {
		pub root_id: u8,
		pub digest: [u8; 32],
	}

	impl MerklePathToken {
		pub fn new(
			root_id: u8,
			left: [u8; 32],
			right: [u8; 32],
			parent: [u8; 32],
			parent_depth: u32,
			parent_index: u32,
			flush_left: bool,
			flush_right: bool,
		) -> Self {
			Self {
				root_id,
				left,
				right,
				parent,
				parent_depth,
				parent_index,
				flush_left,
				flush_right,
			}
		}
	}

	pub struct MerkleTreeTrace {
		pub nodes: Vec<MerklePathToken>,
		pub root: Vec<MerkleRootToken>,
	}

	impl MerklePathToken {
		pub fn fire(&self, node_channel: &mut Channel<NodeFlush>) {
			let concatenation = [self.left.into(), self.right.into()];
			let compression = Groestl256ByteCompression;
			assert_eq!(
				compression.compress(concatenation),
				self.parent.into(),
				"Parent digest mismatch"
			);

			// Push the parent digest to the nodes channel and optionally pull the left or right
			// child depending on the flush flags.
			node_channel.push((self.root_id, self.parent, self.parent_depth, self.parent_index));
			if self.flush_left {
				node_channel.pull((
					self.root_id,
					self.left,
					self.parent_depth + 1,
					2 * self.parent_index,
				));
			} else if self.flush_right {
				node_channel.pull((
					self.root_id,
					self.right,
					self.parent_depth + 1,
					2 * self.parent_index + 1,
				));
			}
		}
	}

	impl MerkleRootToken {
		pub fn fire(
			&self,
			node_channel: &mut Channel<NodeFlush>,
			root_channel: &mut Channel<RootFlush>,
		) {
			//Pull the root node value presumed to have been pushed to the nodes channel from the
			// merkle path table.
			node_channel.pull((self.root_id, self.digest, 0, 0));
			//Pull the root node from the roots channel, presumed to have been pushed as a boundary
			// value.
			root_channel.pull((self.root_id, self.digest));
		}
	}

	#[test]
	fn test_high_level_model_inclusion() {}
}
