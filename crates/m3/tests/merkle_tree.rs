/// High-level model for binary Merkle trees using the Grøstl-256 output transformation as a 2-to-1
/// compression function.

mod model {
	use binius_hash::groestl::{GroestlShortImpl, GroestlShortInternal};
	use binius_m3::emulate::Channel;
	use rand::{rngs::StdRng, Rng, SeedableRng};

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

	impl MerkleRootToken {
		pub fn new(root_id: u8, digest: [u8; 32]) -> Self {
			Self { root_id, digest }
		}
	}

	// Uses the Groestl256 compression function to compress two 32-byte inputs into a single 32-byte
	fn compress(left: &[u8], right: &[u8], output: &mut [u8]) {
		let mut state_bytes = [0u8; 64];
		let (half0, half1) = state_bytes.split_at_mut(32);
		half0.copy_from_slice(left);
		half1.copy_from_slice(right);
		let input = GroestlShortImpl::state_from_bytes(&state_bytes);
		let mut state = input;
		GroestlShortImpl::p_perm(&mut state);
		GroestlShortImpl::xor_state(&mut state, &input);
		state_bytes = GroestlShortImpl::state_to_bytes(&state);
		output.copy_from_slice(&state_bytes[32..]);
	}

	// Merkle tree implementation for the model, assumes the leaf layer consists of [u8;32] blobs.
	pub struct MerkleTree {
		depth: usize,
		nodes: Vec<[u8; 32]>,
		root: [u8; 32],
	}

	impl MerkleTree {
		/// Constructs a Merkle tree from the given leaf nodes that uses the Groestl output
		/// transformation (Groestl-P permutation + XOR) as a digest compression function.
		pub fn new(leafs: &[[u8; 32]]) -> Self {
			assert!(leafs.len().is_power_of_two(), "Length of leafs needs to be a power of 2.");
			let depth = leafs.len().ilog2() as usize;
			let mut nodes = vec![[0u8; 32]; 2 * leafs.len() - 1];

			// Fill the leaves in the flattened tree.
			nodes[0..leafs.len()].copy_from_slice(leafs);

			// Marks the beginning of the layer in the flattened tree.
			let mut current_depth_marker = 0;
			let mut parent_depth_marker = 0;
			// Build the tree from the leaves up to the root.
			for i in (0..depth).rev() {
				let level_size = 1 << i + 1;
				let next_level_size = 1 << i;
				parent_depth_marker += level_size;

				let (current_layer, parent_layer) = nodes
					[current_depth_marker..parent_depth_marker + next_level_size]
					.split_at_mut(level_size);

				for j in 0..next_level_size {
					let left = &current_layer[2 * j];
					let right = &current_layer[2 * j + 1];
					compress(left, right, &mut parent_layer[j])
				}
				// Move the marker to the next level.
				current_depth_marker = parent_depth_marker;
			}
			// The root of the tree is the last node in the flattened tree.
			let root = nodes
				.last()
				.expect("Merkle tree should not be empty")
				.clone();
			Self { depth, nodes, root }
		}

		/// Returns a merkle path for the given index.
		pub fn merkle_path(&self, index: usize) -> Vec<[u8; 32]> {
			assert!(index < 1 << self.depth, "Index out of range.");
			let path = (0..self.depth)
				.map(|j| {
					let node_index = (((1 << j) - 1) << (self.depth + 1 - j)) | (index >> j) ^ 1;
					self.nodes[node_index].clone()
				})
				.collect();
			path
		}

		/// Verifies a merkle path for inclusion in the tree.
		pub fn verify_path(path: &[[u8; 32]], root: [u8; 32], leaf: [u8; 32], index: usize) {
			assert!(index < 1 << path.len(), "Index out of range.");
			let mut current_hash = leaf;
			let mut next_hash = [0u8; 32];
			for i in 0..path.len() {
				if (index >> i) & 1 == 0 {
					compress(&current_hash, &path[i], &mut next_hash);
				} else {
					compress(&path[i], &current_hash, &mut next_hash);
				}
				current_hash = next_hash;
			}
			assert_eq!(current_hash, root);
		}
	}
	pub struct MerkleTreeTrace {
		pub nodes: Vec<MerklePathToken>,
		pub root: Vec<MerkleRootToken>,
	}

	impl MerklePathToken {
		pub fn fire(&self, node_channel: &mut Channel<NodeFlush>) {
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
			// Pull the root node value presumed to have been pushed to the nodes channel from the
			// merkle path table.
			node_channel.pull((self.root_id, self.digest, 0, 0));
			// Pull the root node from the roots channel, presumed to have been pushed as a boundary
			// value.
			root_channel.pull((self.root_id, self.digest));
		}
	}

	impl MerkleTreeTrace {
		/// Method to generate the trace given the witness values. The function assumes that the
		/// root_id is the index of the root in the roots vector and that the paths and leaves are
		/// passed in with their assigned root_id.
		fn generate(roots: Vec<[u8; 32]>, paths: &Vec<(u8, u32, [u8; 32], Vec<[u8; 32]>)>) -> Self {
			let mut path_vec: Vec<MerklePathToken> = Vec::new();
			let mut root_vec: Vec<MerkleRootToken> = Vec::new();
			// Number of times each root is referenced in the paths.
			let mut root_multiplicities = vec![0; roots.len()];

			for i in 0..paths.len() {
				let (root_id, index, leaf, path) = &paths[i];

				root_multiplicities[*root_id as usize] += 1;

				let mut leaf = *leaf;
				for (i, node) in path.iter().enumerate() {
					let mut parent = [0u8; 32];
					if (index >> i) & 1 == 0 {
						compress(&leaf, node, &mut parent);
						path_vec.push(MerklePathToken::new(
							*root_id as u8,
							leaf,
							*node,
							parent,
							(path.len() - i - 1) as u32,
							(index >> (i + 1)) as u32,
							true,
							false,
						));
					} else {
						compress(node, &leaf, &mut parent);
						path_vec.push(MerklePathToken::new(
							*root_id as u8,
							*node,
							leaf,
							parent,
							(path.len() - i - 1) as u32,
							(index >> (i + 1)) as u32,
							false,
							true,
						));
					}
					leaf = parent;
				}
			}

			for (i, root) in roots.iter().enumerate() {
				for _ in 0..root_multiplicities[i] {
					root_vec.push(MerkleRootToken::new(i as u8, *root));
				}
			}

			Self {
				nodes: path_vec,
				root: root_vec,
			}
		}

		fn validate(&self, channels: &mut MerkleTreeChannels) {
			// Push the roots to the roots channel.
			for root in &self.root {
				root.fire(&mut channels.nodes, &mut channels.roots);
			}

			// Push the nodes to the nodes channel.
			for node in &self.nodes {
				node.fire(&mut channels.nodes);
			}

			// Assert that the nodes and roots channels are balanced.
			channels.nodes.assert_balanced();
			channels.roots.assert_balanced();
		}
	}

	// Tests for the merkle tree implementation.
	#[test]
	fn test_merkle_tree() {
		let leaves = vec![
			[0u8; 32], [1u8; 32], [2u8; 32], [3u8; 32], [4u8; 32], [5u8; 32], [6u8; 32], [7u8; 32],
		];
		let tree = MerkleTree::new(&leaves);
		let path = tree.merkle_path(0);
		let root = tree.root;
		let leaf = leaves[0];
		MerkleTree::verify_path(&path, root, leaf, 0);

		assert_eq!(tree.depth, 3);
	}

	// Tests for the Merkle tree trace generation
	#[test]
	fn test_high_level_model_inclusion() {
		let mut rng = StdRng::from_seed([0; 32]);
		let path_index = rng.gen_range(0..1 << 10);
		let leaves = (0..1 << 10)
			.into_iter()
			.map(|_| rng.gen::<[u8; 32]>())
			.collect::<Vec<_>>();

		let tree = MerkleTree::new(&leaves);
		let root = tree.root;
		let path = tree.merkle_path(path_index);
		let path_root_id = 0;
		let mut channels = MerkleTreeChannels::new();
		// Boundary values: The leaf and the root.
		channels.nodes.push((
			path_root_id.into(),
			leaves[path_index],
			path.len() as u32,
			path_index as u32,
		));
		channels.roots.push((path_root_id.into(), root));
		let merkle_path = MerkleTreeTrace::generate(
			vec![root],
			&vec![(path_root_id, path_index as u32, leaves[path_index], path)],
		);
		merkle_path.validate(&mut channels);
	}

	#[test]
	fn test_high_level_model_inclusion_multiple_paths() {
		let mut rng = StdRng::from_seed([0; 32]);

		let leaves = (0..1 << 10)
			.into_iter()
			.map(|_| rng.gen::<[u8; 32]>())
			.collect::<Vec<_>>();

		let tree = MerkleTree::new(&leaves);
		let root = tree.root;
		let paths = (0..5)
			.map(|_| {
				let path_index = rng.gen_range(0..1 << 10);
				(0u8, path_index as u32, leaves[path_index], tree.merkle_path(path_index))
			})
			.collect::<Vec<_>>();
		let mut channels = MerkleTreeChannels::new();
		// Boundary values: The leaf and the root.
		for (root_id, path_index, leaf, path) in &paths {
			channels
				.nodes
				.push((*root_id, *leaf, path.len() as u32, *path_index as u32));
			channels.roots.push((*root_id, root));
		}
		let merkle_path = MerkleTreeTrace::generate(vec![root], &paths);
		merkle_path.validate(&mut channels);
	}

	#[test]
	fn test_high_level_model_inclusion_multiple_roots() {
		let mut rng = StdRng::from_seed([0; 32]);
		let path_index = rng.gen_range(0..1 << 10);
		let leaves = (0..3)
			.map(|_| {
				(0..1 << 10)
					.into_iter()
					.map(|_| rng.gen::<[u8; 32]>())
					.collect::<Vec<_>>()
			})
			.collect::<Vec<_>>();

		let trees = (0..3)
			.map(|i| MerkleTree::new(&leaves[i]))
			.collect::<Vec<_>>();
		let roots = (0..3).map(|i| trees[i].root).collect::<Vec<_>>();
		let paths = trees
			.iter()
			.enumerate()
			.map(|(i, tree)| {
				(i as u8, path_index as u32, leaves[i][path_index], tree.merkle_path(path_index))
			})
			.collect::<Vec<_>>();

		let mut channels = MerkleTreeChannels::new();

		for i in 0..3 {
			let (root_id, path_index, leaf, path) = &paths[i];
			// Boundary values: The leaf and the root.
			channels
				.nodes
				.push((*root_id as u8, *leaf, path.len() as u32, *path_index as u32));
			channels.roots.push((*root_id, roots[i]));
		}

		let merkle_path = MerkleTreeTrace::generate(roots, &paths);
		merkle_path.validate(&mut channels);
	}
}
