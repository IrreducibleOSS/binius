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
	pub struct MerkleTreeChannels {
		nodes: Channel<NodeFlush>,
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
	pub struct MerkleTreeToken {
		pub left: [u8; 32],
		pub right: [u8; 32],
		pub parent: [u8; 32],
		pub depth: u32,
		pub index: u32,
		pub flush_left: bool,
		pub flush_right: bool,
	}

	impl MerkleTreeToken {
		pub fn new(
			left: [u8; 32],
			right: [u8; 32],
			parent: [u8; 32],
			depth: u32,
			index: u32,
			flush_left: bool,
			flush_right: bool,
		) -> Self {
			Self {
				left,
				right,
				parent,
				depth,
				index,
				flush_left,
				flush_right,
			}
		}
	}

	pub struct MerkleTreeTrace {
		pub nodes: Vec<MerkleTreeToken>,
	}

	impl MerkleTreeToken {
		pub fn fire(
			&self,
			node_channel: &mut Channel<NodeFlush>,
			root_channel: &mut Channel<RootFlush>,
		) {
			let concatenation = [self.left.into(), self.right.into()];
			let compression = Groestl256ByteCompression;
			compression.compress(concatenation);
		}
	}
}
