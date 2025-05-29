// Copyright 2025 Irreducible Inc.

/// High-level model for binary Merkle trees using the Grøstl-256 output transformation as a 2-to-1
/// compression function.
mod model {
	use std::{
		collections::{HashMap, HashSet},
		hash::Hash,
	};

	use binius_hash::groestl::{GroestlShortImpl, GroestlShortInternal};
	use binius_m3::emulate::Channel;
	use rand::{Rng, SeedableRng, rngs::StdRng};

	/// Signature of the Nodes channel: (Root ID, Data, Depth, Index)
	#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
	pub struct NodeFlushToken {
		pub root_id: u8,
		pub data: [u8; 32],
		pub depth: usize,
		pub index: usize,
	}

	/// Signature of the Roots channel: (Root ID, Root digest)
	#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
	pub struct RootFlushToken {
		pub root_id: u8,
		pub data: [u8; 32],
	}

	/// A type alias for the Merkle path, which is a vector of tuples containing the root ID, index,
	/// leaf, and the siblings on the path to the root from the leaf.

	#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
	pub struct MerklePath {
		pub root_id: u8,
		pub index: usize,
		pub leaf: [u8; 32],
		pub nodes: Vec<[u8; 32]>,
	}

	/// A struct whose fields contain the channels involved in the trace to verify merkle paths for
	/// a binary merkle tree
	#[allow(clippy::tabs_in_doc_comments)]
	pub struct MerkleTreeChannels {
		/// This channel gets flushed with tokens during "intermediate" steps of the verification
		/// where the tokens are the values of the parent digest of the claimed siblings along with
		/// associated position information such as the root it is associated to, the values of
		/// the child digests, the depth and the index.
		nodes: Channel<NodeFlushToken>,

		/// This channel contains flushes that validate that the "final" digest obtained in a
		/// merkle path is matches that of one of the claimed roots, pushed as boundary values.
		roots: Channel<RootFlushToken>,
	}

	impl MerkleTreeChannels {
		pub fn new() -> Self {
			Self {
				nodes: Channel::default(),
				roots: Channel::default(),
			}
		}
	}
	/// A table representing a step in verifying a merkle path for inclusion.

	#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
	pub struct MerklePathEvent {
		pub root_id: u8,
		pub left: [u8; 32],
		pub right: [u8; 32],
		pub parent: [u8; 32],
		pub parent_depth: usize,
		pub parent_index: usize,
		pub flush_left: bool,
		pub flush_right: bool,
	}

	/// A table representing the final step of comparing the claimed root.

	#[derive(Debug, Clone, PartialEq, Eq, Hash)]
	pub struct MerkleRootEvent {
		pub root_id: u8,
		pub digest: [u8; 32],
	}

	impl MerkleRootEvent {
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

	/// Merkle tree implementation for the model, assumes the leaf layer consists of [u8;32] blobs.
	/// The tree is built in a flattened manner, where the leaves are at the beginning of the vector
	/// and layers are placed adjacent to each other.
	pub struct MerkleTree {
		depth: usize,
		nodes: Vec<[u8; 32]>,
		pub root: [u8; 32],
	}

	impl MerkleTree {
		/// Constructs a Merkle tree from the given leaf nodes that uses the Groestl output
		/// transformation (Groestl-P permutation + XOR) as a digest compression function.
		pub fn new(leaves: &[[u8; 32]]) -> Self {
			assert!(leaves.len().is_power_of_two(), "Length of leaves needs to be a power of 2.");
			let depth = leaves.len().ilog2() as usize;
			let mut nodes = vec![[0u8; 32]; 2 * leaves.len() - 1];

			// Fill the leaves in the flattened tree.
			nodes[0..leaves.len()].copy_from_slice(leaves);

			// Marks the beginning of the layer in the flattened tree.
			let mut current_depth_marker = 0;
			let mut parent_depth_marker = 0;
			// Build the tree from the leaves up to the root.
			for i in (0..depth).rev() {
				let level_size = 1 << (i + 1);
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
			let root = *nodes.last().expect("Merkle tree should not be empty");
			Self { depth, nodes, root }
		}

		/// Returns a merkle path for the given index.
		pub fn merkle_path(&self, index: usize) -> Vec<[u8; 32]> {
			assert!(index < 1 << self.depth, "Index out of range.");
			(0..self.depth)
				.map(|j| {
					let node_index = (((1 << j) - 1) << (self.depth + 1 - j)) | (index >> j) ^ 1;
					self.nodes[node_index]
				})
				.collect()
		}

		/// Verifies a merkle path for inclusion in the tree.
		pub fn verify_path(path: &[[u8; 32]], root: [u8; 32], leaf: [u8; 32], index: usize) {
			assert!(index < 1 << path.len(), "Index out of range.");
			let mut current_hash = leaf;
			let mut next_hash = [0u8; 32];
			for (i, node) in path.iter().enumerate() {
				if (index >> i) & 1 == 0 {
					compress(&current_hash, node, &mut next_hash);
				} else {
					compress(node, &current_hash, &mut next_hash);
				}
				current_hash = next_hash;
			}
			assert_eq!(current_hash, root);
		}
	}

	impl MerklePathEvent {
		/// Method to fire the event, pushing the parent digest if the parent flag is set and
		/// pulling the left or right child depending on the flush flags.
		pub fn fire(&self, node_channel: &mut Channel<NodeFlushToken>) {
			// Push the parent digest to the nodes channel and optionally pull the left or right
			// child depending on the flush flags.
			node_channel.push(NodeFlushToken {
				root_id: self.root_id,
				data: self.parent,
				depth: self.parent_depth,
				index: self.parent_index,
			});

			if self.flush_left {
				node_channel.pull(NodeFlushToken {
					root_id: self.root_id,
					data: self.left,
					depth: self.parent_depth + 1,
					index: 2 * self.parent_index,
				});
			}
			if self.flush_right {
				node_channel.pull(NodeFlushToken {
					root_id: self.root_id,
					data: self.right,
					depth: self.parent_depth + 1,
					index: 2 * self.parent_index + 1,
				});
			}
		}
	}

	impl MerkleRootEvent {
		pub fn fire(
			&self,
			node_channel: &mut Channel<NodeFlushToken>,
			root_channel: &mut Channel<RootFlushToken>,
		) {
			// Pull the root node value presumed to have been pushed to the nodes channel from the
			// merkle path table.
			node_channel.pull(NodeFlushToken {
				root_id: self.root_id,
				data: self.digest,
				depth: 0,
				index: 0,
			});
			// Pull the root node from the roots channel, presumed to have been pushed as a boundary
			// value.
			root_channel.pull(RootFlushToken {
				root_id: self.root_id,
				data: self.digest,
			});
		}
	}

	/// Struct representing the boundary values of merkle tree inclusion proof statement.
	#[derive(Debug, Clone, PartialEq, Eq)]
	pub struct MerkleBoundaries {
		pub leaf: HashSet<NodeFlushToken>,
		pub root: HashSet<RootFlushToken>,
	}

	impl MerkleBoundaries {
		pub fn new() -> Self {
			Self {
				leaf: HashSet::new(),
				root: HashSet::new(),
			}
		}

		pub fn insert(&mut self, leaf: NodeFlushToken, root: RootFlushToken) {
			self.leaf.insert(leaf);
			self.root.insert(root);
		}
	}

	/// Struct representing the trace of the merkle tree inclusion proof statement.
	pub struct MerkleTreeTrace {
		pub boundaries: MerkleBoundaries,
		pub nodes: Vec<MerklePathEvent>,
		pub root: HashSet<MerkleRootEvent>,
	}
	impl MerkleTreeTrace {
		/// Method to generate the trace given the witness values. The function assumes that the
		/// root_id is the index of the root in the roots vector and that the paths and leaves are
		/// passed in with their assigned root_id.
		pub fn generate(roots: Vec<[u8; 32]>, paths: &[MerklePath]) -> Self {
			let mut path_nodes = Vec::new();
			let mut root_nodes = HashSet::new();
			let mut boundaries = MerkleBoundaries::new();
			// Number of times each root is referenced in the paths. Since internal nodes have been
			// deduped, these need to be pushed into the nodes channel as many times as they are
			// referenced in the paths.

			// Tracks the filled nodes in the tree
			let mut filled_nodes = HashMap::new();

			for path in paths.iter() {
				let MerklePath {
					root_id,
					index,
					leaf,
					nodes,
				} = path;

				let mut current_child = *leaf;

				// Populate the boundary values for the statement.
				boundaries.insert(
					NodeFlushToken {
						root_id: *root_id,
						data: current_child,
						depth: nodes.len(),
						index: *index,
					},
					RootFlushToken {
						root_id: *root_id,
						data: roots[*root_id as usize],
					},
				);

				// Populate the root table's events.
				root_nodes.insert(MerkleRootEvent::new(*root_id, roots[*root_id as usize]));

				let mut parent_node = [0u8; 32];
				for (i, &node) in nodes.iter().enumerate() {
					match filled_nodes.get_mut(&(*root_id, index >> (i + 1), nodes.len() - i - 1)) {
						Some((_, _, parent, flush_left, flush_right)) => {
							if (index >> i) & 1 == 0 {
								parent_node = *parent;
								*flush_left = true;
							} else {
								parent_node = *parent;
								*flush_right = true;
							}
						}
						None => {
							if (index >> i) & 1 == 0 {
								compress(&node, &current_child, &mut parent_node);
								filled_nodes.insert(
									(*root_id, index >> (i + 1), nodes.len() - i - 1),
									(current_child, node, parent_node, true, false),
								);
							} else {
								compress(&node, &current_child, &mut parent_node);
								filled_nodes.insert(
									(*root_id, index >> (i + 1), nodes.len() - i - 1),
									(node, current_child, parent_node, false, true),
								);
							}
						}
					}
					current_child = parent_node
				}
			}

			path_nodes.extend(filled_nodes.iter().map(|(key, value)| {
				let &(root_id, parent_index, parent_depth) = key;
				let &(left, right, parent, flush_left, flush_right) = value;
				MerklePathEvent {
					root_id,
					left,
					right,
					parent,
					parent_depth,
					parent_index,
					flush_left,
					flush_right,
				}
			}));

			Self {
				boundaries,
				nodes: path_nodes,
				root: root_nodes,
			}
		}

		fn validate(&self) {
			let mut channels = MerkleTreeChannels::new();

			// Push the boundary values to the nodes and roots channels.
			for leaf in &self.boundaries.leaf {
				channels.nodes.push(*leaf);
			}
			for root in &self.boundaries.root {
				channels.roots.push(*root);
			}

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
			.map(|_| rng.r#gen::<[u8; 32]>())
			.collect::<Vec<_>>();

		let tree = MerkleTree::new(&leaves);
		let root = tree.root;
		let path = tree.merkle_path(path_index);
		let path_root_id = 0;
		let merkle_tree_trace = MerkleTreeTrace::generate(
			vec![root],
			&[MerklePath {
				root_id: path_root_id,
				index: path_index,
				leaf: leaves[path_index],
				nodes: path,
			}],
		);
		merkle_tree_trace.validate();
	}

	#[test]
	fn test_high_level_model_inclusion_multiple_paths() {
		let mut rng = StdRng::from_seed([0; 32]);

		let leaves = (0..1 << 4)
			.map(|_| rng.r#gen::<[u8; 32]>())
			.collect::<Vec<_>>();

		let tree = MerkleTree::new(&leaves);
		let root = tree.root;
		let paths = (0..2)
			.map(|_| {
				let path_index = rng.gen_range(0..1 << 4);
				MerklePath {
					root_id: 0u8,
					index: path_index,
					leaf: leaves[path_index],
					nodes: tree.merkle_path(path_index),
				}
			})
			.collect::<Vec<_>>();
		let merkle_tree_trace = MerkleTreeTrace::generate(vec![root], &paths);
		merkle_tree_trace.validate();
	}

	#[test]
	fn test_high_level_model_inclusion_multiple_roots() {
		let mut rng = StdRng::from_seed([0; 32]);
		let path_index = rng.gen_range(0..1 << 10);
		let leaves = (0..3)
			.map(|_| {
				(0..1 << 10)
					.map(|_| rng.r#gen::<[u8; 32]>())
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
			.map(|(i, tree)| MerklePath {
				root_id: i as u8,
				index: path_index,
				leaf: leaves[i][path_index],
				nodes: tree.merkle_path(path_index),
			})
			.collect::<Vec<_>>();

		let merkle_tree_trace = MerkleTreeTrace::generate(roots, &paths);
		merkle_tree_trace.validate();
	}
}

mod arithmetisation {
	use std::{array, cell::RefMut, os::macos::raw::stat};

	use anyhow::anyhow;
	use array_util::ArrayExt;
	use binius_core::{constraint_system::channel::ChannelId, oracle::ShiftVariant, witness};
	use binius_field::{
		AESTowerField8b, BinaryField8b, ExtensionField, Field, PackedBinaryField4x8b,
		PackedBinaryField8x8b, PackedBinaryField32x8b, PackedExtension, PackedField,
		PackedFieldIndexable, PackedSubfield,
		arch::OptimalUnderlier128b,
		as_packed_field::{AsSinglePacked, PackedType},
		linear_transformation::PackedTransformationFactory,
		packed::{get_packed_slice, set_packed_slice},
		underlier::WithUnderlier,
	};
	use binius_hash::{
		compression,
		groestl::{Groestl256, GroestlShortImpl, GroestlShortInternal},
		permutation,
	};
	use binius_m3::{
		builder::{
			B1, B8, B32, B64, B128, Boundary, Col, ConstraintSystem, Error, Expr, FlushDirection,
			Table, TableBuilder, TableFiller, TableId, TableWitnessSegment, WitnessIndex,
			test_utils::validate_system_witness, upcast_col,
		},
		gadgets::hash::groestl::{Permutation, PermutationVariant},
	};
	use bumpalo::Bump;
	use either::Either::Left;
	use itertools::Itertools;
	use rand::{Rng, SeedableRng, rngs::StdRng};

	use super::*;
	use crate::model::{MerklePath, MerklePathEvent, MerkleRootEvent, MerkleTree, MerkleTreeTrace};

	/// A struct representing the constraint system for the Merkle tree. Like any M3 instance,
	/// it is characterized by the tables with the column constraints, and the channels with
	/// the flushing rules.
	pub struct MerkleTreeCS {
		pub merkle_path_table_left: NodesTable,
		pub merkle_path_table_right: NodesTable,
		pub merkle_path_table_both: NodesTable,
		pub root_table: RootTable,
		pub nodes_channel: ChannelId,
		pub roots_channel: ChannelId,
	}

	impl MerkleTreeCS {
		pub fn new(cs: &mut ConstraintSystem) -> Self {
			let nodes_channel = cs.add_channel("merkle_tree_nodes");
			let roots_channel = cs.add_channel("merkle_tree_roots");

			let merkle_path_table_left =
				NodesTable::new(cs, MerklePathPullChild::Left, nodes_channel);
			let merkle_path_table_right =
				NodesTable::new(cs, MerklePathPullChild::Right, nodes_channel);
			let merkle_path_table_both =
				NodesTable::new(cs, MerklePathPullChild::Both, nodes_channel);

			let root_table = RootTable::new(cs, nodes_channel, roots_channel);

			Self {
				merkle_path_table_left,
				merkle_path_table_right,
				merkle_path_table_both,
				root_table,
				nodes_channel,
				roots_channel,
			}
		}

		pub fn fill_tables(
			&self,
			trace: &MerkleTreeTrace,
			witness: &mut WitnessIndex<PackedType<OptimalUnderlier128b, B128>>,
		) {
			// Filter the MerklePathEvents into three iterators based on the pull child type.
			let left_events = trace
				.nodes
				.iter()
				.copied()
				.filter(|event| event.flush_left && !event.flush_right)
				.collect::<Vec<_>>();
			let right_events = trace
				.nodes
				.iter()
				.copied()
				.filter(|event| !event.flush_left && event.flush_right)
				.collect::<Vec<_>>();
			let both_events = trace
				.nodes
				.iter()
				.copied()
				.filter(|event| event.flush_left && event.flush_right)
				.collect::<Vec<_>>();

			// Fill the nodes tables based on the filtered events.
			witness
				.fill_table_sequential(&self.merkle_path_table_left, &left_events)
				.expect("Failed to fill left nodes table");
			witness
				.fill_table_sequential(&self.merkle_path_table_right, &right_events)
				.expect("Failed to fill right nodes table");
			witness
				.fill_table_sequential(&self.merkle_path_table_both, &both_events)
				.expect("Failed to fill both nodes table");

			// Fill the roots table.
			witness
				.fill_table_sequential(
					&self.root_table,
					&trace.root.clone().into_iter().collect::<Vec<_>>(),
				)
				.expect("Failed to fill roots table");
		}
	}

	/// We need to implement tables for the different cases of Merkle path pulls based on aggregated
	/// inclusion proofs. Which is, given a set of indexes to open, we may need to pull the left
	/// child, the right child, or both. As a table must contain columns of the same height, we
	/// need to implement three tables for the three cases.
	pub enum MerklePathPullChild {
		Left,
		Right,
		Both,
	}

	pub struct MerkleTreeTables {
		pull_left: NodesTable,
		pull_right: NodesTable,
		pull_both: NodesTable,
	}

	//TODO: Break into cases to reduce the number of packs.
	pub struct NodesTable {
		id: TableId,
		// The root id field is used to identify the root the node is associated with.
		root_id: Col<B8>,
		// Concatenated bytes of the left and right digests of the Merkle tree node. Organised
		// in a packed row of the Groestl-256 permutation state.
		state_in: [Col<B8, 8>; 8],
		state_out_shifted: [Col<B8, 8>; 8],
		// These are the selected columns that correspond to the bytes that would be obtained after
		// trimming the output transformation of Groestl-256 (P-permutation + XOR with input).
		permutation_output_columns: [Col<B8, 4>; 8],
		// Left, right and parent columns are used to store derive the values of the columns
		// of the permutation states corresponding to the left and right digests of the
		// Merkle tree nodes as the gadget packs rows of the state as opposed to the columns.
		left_columns: [Col<B8, 4>; 8],
		right_columns: [Col<B8, 4>; 8],
		parent_columns: [Col<B8, 4>; 8],
		pub parent_depth: Col<B32>,
		pub child_depth: Col<B32>,
		parent_index: Col<B1, 32>,
		left_index: Col<B1, 32>,
		right_index_packed: Col<B32>,
		/// A table representing the 2-1 digest compression function used in the Merkle tree.
		/// This table represents the output transformation of the Groestl-256 hash function,
		/// which is the P-permutation on the state followed by an XOR with the input and finally
		/// extracting the last 32 bytes of the state.
		permutation: Permutation,
		pub pull_child: MerklePathPullChild,
	}

	impl NodesTable {
		pub fn new(
			cs: &mut ConstraintSystem,
			pull_child: MerklePathPullChild,
			nodes_channel: ChannelId,
		) -> Self {
			let mut table = cs.add_table(format!("merkle_tree_nodes_{}", {
				match pull_child {
					MerklePathPullChild::Left => "left",
					MerklePathPullChild::Right => "right",
					MerklePathPullChild::Both => "both",
				}
			}));

			let id = table.id();
			// committed coluumns are used to store the values of the nodes.
			let root_id = table.add_committed("root_id");

			// The concatanated bytes of the left and right digests are used for the permutation
			// gadget. Note these are packed rows of the Groestl-P permutation state. Where as the
			// right and left digests would correspond to the first 4 and last 4 columns of the
			// state.

			let left_columns: [Col<BinaryField8b, 4>; 8] =
				array::from_fn(|i| table.add_committed(format!("left_columns_{i}")));
			let right_columns: [Col<BinaryField8b, 4>; 8] =
				array::from_fn(|i| table.add_committed(format!("right_columns_{i}")));
			let state_in = array::from_fn(|i| table.add_committed(format!("state_in_{i}")));

			let left_packed: [Col<B32>; 8] =
				array::from_fn(|i| table.add_packed(format!("left_packed{i}"), left_columns[i]));
			let right_packed: [Col<B32>; 8] =
				array::from_fn(|i| table.add_packed(format!("right_packed{i}"), right_columns[i]));
			let state_in_packed: [Col<B64>; 8] =
				array::from_fn(|i| table.add_packed(format!("state_in_packed_{i}"), state_in[i]));

			for i in 0..8 {
				table.assert_zero(
					format!("state_in_assert{i}"),
					state_in_packed[i]
						- upcast_col(left_packed[i])
						- upcast_col(right_packed[i]) * B64::from(1 << 32),
				);
			}
			let permutation = Permutation::new(
				&mut table.with_namespace("permutation"),
				PermutationVariant::P,
				state_in,
			);

			let state_out = permutation.state_out();

			let state_out_shifted: [Col<BinaryField8b, 8>; 8] = array::from_fn(|i| {
				table.add_shifted(
					format!("state_in_shifted_{i}"),
					state_out[i],
					3,
					4,
					ShiftVariant::LogicalRight,
				)
			});

			let permutation_output_columns: [Col<BinaryField8b, 4>; 8] = array::from_fn(|i| {
				table.add_selected_block(
					format!("permutation_output_columns_{i}"),
					state_out_shifted[i],
					4,
				)
			});

			let parent_columns: [Col<B8, 4>; 8] = array::from_fn(|i| {
				table.add_computed(
					format!("parent_columns_{i}"),
					permutation_output_columns[i] + right_columns[i],
				)
			});

			let parent_packed: [Col<B32>; 8] = array::from_fn(|i| {
				table.add_packed(format!("parent_packed{i}"), parent_columns[i])
			});

			let parent_depth = table.add_committed("parent_depth");
			let child_depth = table.add_committed("child_depth");
			let parent_index: Col<B1, 32> = table.add_committed("parent_index");
			let left_index: Col<B1, 32> =
				table.add_shifted("left_index", parent_index, 5, 1, ShiftVariant::LogicalLeft);
			// The indexes need to be packed and upcasted to agree with the flushing rules of the
			// channel.
			let left_index_packed = table.add_packed("left_index_packed", left_index);
			let right_index_packed =
				table.add_computed("right_index_packed", left_index_packed + B32::ONE);
			let parent_index_packed: Col<B32> =
				table.add_packed("parent_index_packed", parent_index);

			let left_index_upcasted = upcast_col(left_index_packed);
			let right_index_upcasted = upcast_col(right_index_packed);
			let parent_index_upcasted = upcast_col(parent_index_packed);
			let parent_depth_upcasted = upcast_col(parent_depth);
			let child_depth_upcasted = upcast_col(child_depth);
			let root_id_upcasted = upcast_col(root_id);

			table.push(
				nodes_channel,
				to_node_flush(
					root_id_upcasted,
					parent_packed,
					parent_depth_upcasted,
					parent_index_upcasted,
				),
			);

			match pull_child {
				MerklePathPullChild::Left => table.pull(
					nodes_channel,
					to_node_flush(
						root_id_upcasted,
						left_packed,
						child_depth_upcasted,
						left_index_upcasted,
					),
				),
				MerklePathPullChild::Right => table.pull(
					nodes_channel,
					to_node_flush(
						root_id_upcasted,
						right_packed,
						child_depth_upcasted,
						right_index_upcasted,
					),
				),
				MerklePathPullChild::Both => {
					table.pull(
						nodes_channel,
						to_node_flush(
							root_id_upcasted,
							left_packed,
							child_depth_upcasted,
							left_index_upcasted,
						),
					);
					table.pull(
						nodes_channel,
						to_node_flush(
							root_id_upcasted,
							right_packed,
							child_depth_upcasted,
							right_index_upcasted,
						),
					)
				}
			}
			Self {
				id,
				root_id,
				state_in,
				state_out_shifted,
				permutation_output_columns,
				left_columns,
				right_columns,
				parent_columns,
				parent_depth,
				child_depth,
				parent_index,
				left_index,
				right_index_packed,
				pull_child,
				permutation,
			}
		}
	}

	// Helper functions to convert the index and digest columns into a flushable format.
	fn to_node_flush(
		root_id: Col<B32>,
		digest: [Col<B32>; 8],
		depth: Col<B32>,
		index: Col<B32>,
	) -> [Col<B32>; 11] {
		[
			root_id, digest[0], digest[1], digest[2], digest[3], digest[4], digest[5], digest[6],
			digest[7], depth, index,
		]
	}

	fn to_root_flush(root_id: Col<B32>, digest: [Col<B32>; 8]) -> [Col<B32>; 9] {
		[
			root_id, digest[0], digest[1], digest[2], digest[3], digest[4], digest[5], digest[6],
			digest[7],
		]
	}

	pub struct RootTable {
		pub id: TableId,
		pub root_id: Col<B8>,
		pub digest: [Col<B32>; 8],
		pub nodes_channel: ChannelId,
		pub roots_channel: ChannelId,
	}

	impl RootTable {
		pub fn new(
			cs: &mut ConstraintSystem,
			nodes_channel: ChannelId,
			roots_channel: ChannelId,
		) -> Self {
			let mut table = cs.add_table("merkle_tree_roots");
			let id = table.id();
			let root_id = table.add_committed("root_id");
			let digest = array::from_fn(|i| table.add_committed(format!("digest_{i}")));

			let zero = table.add_constant("zero", [B32::ZERO]);
			let root_id_upcasted = upcast_col(root_id);
			table.pull(nodes_channel, to_node_flush(root_id_upcasted, digest, zero, zero));
			table.pull(roots_channel, to_root_flush(root_id_upcasted, digest));
			Self {
				id,
				root_id,
				digest,
				nodes_channel,
				roots_channel,
			}
		}
	}

	impl<P> TableFiller<P> for NodesTable
	where
		P: PackedFieldIndexable<Scalar = B128>
			+ PackedExtension<B1>
			+ PackedExtension<B8>
			+ PackedExtension<B32>
			+ PackedExtension<B64>,
		PackedSubfield<P, B8>:
			PackedTransformationFactory<PackedSubfield<P, B8>> + PackedFieldIndexable,
	{
		type Event = MerklePathEvent;

		fn id(&self) -> TableId {
			self.id
		}

		fn fill<'a>(
			&'a self,
			rows: impl Iterator<Item = &'a Self::Event> + Clone,
			witness: &'a mut TableWitnessSegment<P>,
		) -> anyhow::Result<()> {
			let state_ins = rows
				.clone()
				.map(|MerklePathEvent { left, right, .. }| {
					let mut digest = [B8::ZERO; 64];
					let (left_bytes, right_bytes) =
						(B8::from_underliers_arr_ref(&left), B8::from_underliers_arr_ref(&right));

					digest[..32].copy_from_slice(left_bytes);
					digest[32..].copy_from_slice(right_bytes);
					digest
				})
				.collect::<Vec<_>>();

			self.permutation.populate_state_in(witness, &state_ins)?;
			self.permutation.populate(witness)?;

			let mut witness_root_id: RefMut<'_, [u8]> = witness.get_mut_as(self.root_id)?;
			let mut witness_parent_depth: RefMut<'_, [u32]> =
				witness.get_mut_as(self.parent_depth)?;
			let mut witness_child_depth: RefMut<'_, [u32]> =
				witness.get_mut_as(self.child_depth)?;
			let mut witness_parent_index: RefMut<'_, [u32]> =
				witness.get_mut_as(self.parent_index)?;
			// The left and right indexes are packed into a single column, so we need to unpack them
			let mut witness_left_index: RefMut<'_, [u32]> = witness.get_mut_as(self.left_index)?;
			let mut witness_right_index_packed: RefMut<'_, [u32]> =
				witness.get_mut_as(self.right_index_packed)?;

			let mut witness_state_out: [RefMut<'_, [PackedBinaryField8x8b]>; 8] = self
				.permutation
				.state_out()
				.try_map_ext(|col| witness.get_mut_as(col))?;

			let mut witness_state_out_shifted: [RefMut<'_, [PackedBinaryField8x8b]>; 8] = self
				.state_out_shifted
				.try_map_ext(|col| witness.get_mut_as(col))?;

			let mut witness_permutation_output_columns: [RefMut<'_, [PackedBinaryField4x8b]>; 8] =
				self.permutation_output_columns
					.try_map_ext(|col| witness.get_mut_as(col))?;

			let mut left_columns: [RefMut<'_, [PackedBinaryField4x8b]>; 8] = self
				.left_columns
				.try_map_ext(|col| witness.get_mut_as(col))?;
			let mut right_columns: [RefMut<'_, [PackedBinaryField4x8b]>; 8] = self
				.right_columns
				.try_map_ext(|col| witness.get_mut_as(col))?;
			let mut parent_columns: [RefMut<'_, [PackedBinaryField4x8b]>; 8] = self
				.parent_columns
				.try_map_ext(|col| witness.get_mut_as(col))?;
			for (i, event) in rows.enumerate() {
				let MerklePathEvent {
					root_id,
					parent_depth,
					parent_index,
					left,
					right,
					parent,
					..
				} = event;

				witness_root_id[i] = *root_id;
				witness_parent_depth[i] = *parent_depth as u32;
				witness_parent_index[i] = *parent_index as u32;
				witness_child_depth[i] = *parent_depth as u32 + 1;
				witness_left_index[i] = (parent_index << 1) as u32;
				witness_right_index_packed[i] = (parent_index << 1 | 1) as u32;
				let left_bytes = B8::from_underliers_arr_ref(&left);
				let right_bytes = B8::from_underliers_arr_ref(&right);
				let parent_bytes = B8::from_underliers_arr_ref(&parent);

				for jk in 0..32 {
					// Row in the state
					let j = jk % 8;
					// Col in the state
					let k = jk / 8;

					// Set the packed slice for the rows.
					set_packed_slice(&mut left_columns[j], i * 4 + k, left_bytes[8 * k + j]);
					set_packed_slice(&mut right_columns[j], i * 4 + k, right_bytes[8 * k + j]);
					set_packed_slice(&mut parent_columns[j], i * 4 + k, parent_bytes[8 * k + j]);

					// Filling the shifted state output, and trimmed out columns.
					let permutation_output = get_packed_slice(&witness_state_out[j], i * 8 + 4 + k);
					set_packed_slice(
						&mut witness_state_out_shifted[j],
						i * 8 + k,
						permutation_output,
					);
					set_packed_slice(
						&mut witness_permutation_output_columns[j],
						i * 4 + k,
						permutation_output,
					);
				}
			}
			Ok(())
		}
	}

	impl<P> TableFiller<P> for RootTable
	where
		P: PackedFieldIndexable<Scalar = B128>
			+ PackedExtension<B1>
			+ PackedExtension<B8>
			+ PackedExtension<B32>
			+ PackedExtension<B64>,
		PackedSubfield<P, B8>: PackedFieldIndexable,
	{
		type Event = MerkleRootEvent;

		fn id(&self) -> TableId {
			self.id
		}

		fn fill<'a>(
			&'a self,
			rows: impl Iterator<Item = &'a Self::Event> + Clone,
			witness: &'a mut TableWitnessSegment<P>,
		) -> anyhow::Result<()> {
			let mut witness_root_id = witness.get_mut_as(self.root_id)?;
			let mut witness_root_digest: Vec<RefMut<'_, [PackedBinaryField4x8b]>> = (0..8)
				.map(|i| witness.get_mut_as(self.digest[i]))
				.collect::<Result<Vec<_>, _>>()?;

			for (i, event) in rows.enumerate() {
				let MerkleRootEvent { root_id, digest } = event;
				witness_root_id[i] = *root_id;
				let digest_as_field = B8::from_underliers_arr_ref(&digest);
				for jk in 0..32 {
					// Row in the state
					let j = jk % 8;
					// Col in the state
					let k = jk / 8;
					set_packed_slice(&mut witness_root_digest[j], i * 4 + k, digest_as_field[jk]);
				}
			}
			for j in 0..4 {
				println!("Root digest column {}: {:?}", j, witness_root_digest[j]);
			}
			Ok(())
		}
	}

	fn digest_to_state(digest: [u8; 32]) -> [B64; 4] {
		let state_field = B8::from_underliers_arr(digest);
		let mut out = [B64::ZERO; 4];
		for i in 0..4 {
			out[i] = <B64 as ExtensionField<B8>>::from_bases(
				state_field[i * 8..(i + 1) * 8].iter().copied(),
			)
			.expect("Failed to convert digest to state");
		}

		out
	}

	fn digest_to_cols(bytes: &[B8; 32]) -> [PackedBinaryField4x8b; 8] {
		let mut cols = [PackedBinaryField4x8b::zero(); 8];
		for ij in 0..32 {
			// Row in the state
			let i = ij % 8;
			// Col in the state
			let j = ij / 8;

			// Set the packed slice for the row.
			set_packed_slice(&mut cols, 8 * i + j, bytes[8 * j + i]);
		}
		cols
	}

	fn bytes_to_boundary(bytes: &[u8; 32]) -> [B32; 8] {
		let mut cols = [PackedBinaryField4x8b::zero(); 8];
		for ij in 0..32 {
			// Row in the state
			let i = ij % 8;
			// Col in the state
			let j = ij / 8;

			// Set the packed slice for the row.
			set_packed_slice(
				&mut cols,
				4 * i + j,
				B8::from(AESTowerField8b::new(bytes[8 * j + i])),
			);
		}
		cols.map(|col| B32::from(col.to_underlier()))
	}

	// A function to get the left and right columns of the Groestl-256 state from the
	// concatenated bytes of the left and right digests.

	fn bytes_to_cols(
		left_bytes: &[B8; 32],
		right_bytes: &[B8; 32],
	) -> ([PackedBinaryField4x8b; 8], [PackedBinaryField4x8b; 8]) {
		let (mut left_cols, mut right_cols) =
			([PackedBinaryField4x8b::zero(); 8], [PackedBinaryField4x8b::zero(); 8]);

		for ij in 0..32 {
			// Row in the state
			let i = ij % 8;
			// Col in the state
			let j = ij / 8;

			set_packed_slice(&mut left_cols, 8 * i + j, left_bytes[8 * j + i]);
			set_packed_slice(&mut right_cols, 8 * i + j, right_bytes[8 * j + i]);
		}
		(left_cols, right_cols)
	}

	#[test]
	fn check_ordering() {
		let mut rng = StdRng::seed_from_u64(0);
		let bytes: [u8; 64] = array::from_fn(|_| rng.r#gen());
		let state = GroestlShortImpl::state_from_bytes(&bytes);
		let mut state_out = GroestlShortImpl::state_from_bytes(&bytes);
		let packed_state = PackedBinaryField8x8b::from_underliers_arr(state_out);
		let gadget_state_out: [BinaryField8b; 64] = array::from_fn(|ij| {
			let i = ij % 8;
			let j = ij / 8;
			packed_state[i].get(j)
		});

		let bytes = GroestlShortImpl::state_to_bytes(&state_out);
		println!("state {:?}", bytes);
		println!("gadget_state {:?}", B8::to_underliers_arr(gadget_state_out));
	}

	#[test]
	fn check_state_orientation() {
		let mut bytes: [u8; 64] = [0u8; 64];

		bytes[32..].copy_from_slice(&[1u8; 32]);
		println!("Bytes: {:?}", bytes);
		let state = GroestlShortImpl::state_from_bytes(&bytes);
		for col in state.iter() {
			println!("{:?}", col.to_be_bytes());
		}
	}
	#[test]
	fn test_nodes_table_constructor() {
		let mut cs = ConstraintSystem::new();
		let nodes_channel = cs.add_channel("nodes");
		let pull_child = MerklePathPullChild::Left;
		let nodes_table = NodesTable::new(&mut cs, pull_child, nodes_channel);
		assert_eq!(nodes_table.left_columns.len(), 8);
		assert_eq!(nodes_table.right_columns.len(), 8);
		assert_eq!(nodes_table.parent_columns.len(), 8);
	}
	#[test]
	fn test_root_table_constructor() {
		let mut cs = ConstraintSystem::new();
		let nodes_channel = cs.add_channel("nodes");
		let roots_channel = cs.add_channel("roots");
		let root_table = RootTable::new(&mut cs, nodes_channel, roots_channel);
		assert_eq!(root_table.digest.len(), 8);
	}
	#[test]
	fn test_node_table_filling() {
		let mut cs = ConstraintSystem::new();
		let nodes_channel = cs.add_channel("nodes");
		let roots_channel = cs.add_channel("roots");
		// Create a nodes table with the left child pull.
		let pull_child = MerklePathPullChild::Left;
		let nodes_table = NodesTable::new(&mut cs, pull_child, nodes_channel);
		let root_table = RootTable::new(&mut cs, nodes_channel, roots_channel);
		let tree = MerkleTree::new(&[
			[0u8; 32], [1u8; 32], [2u8; 32], [3u8; 32], [4u8; 32], [5u8; 32], [6u8; 32], [7u8; 32],
		]);

		let index = 0;
		let path = tree.merkle_path(0);
		let trace = MerkleTreeTrace::generate(
			vec![tree.root],
			&[MerklePath {
				root_id: 0,
				index,
				leaf: [0u8; 32],
				nodes: path,
			}],
		);
		let allocator = Bump::new();
		let mut witness =
			WitnessIndex::<PackedType<OptimalUnderlier128b, B128>>::new(&cs, &allocator);

		witness
			.fill_table_sequential(&nodes_table, &trace.nodes)
			.unwrap();
	}

	#[test]
	fn test_root_table_filling() {
		let mut cs = ConstraintSystem::new();
		let nodes_channel = cs.add_channel("nodes");
		let roots_channel = cs.add_channel("roots");
		let root_table = RootTable::new(&mut cs, nodes_channel, roots_channel);
		let leaves = [
			[0u8; 32], [1u8; 32], [2u8; 32], [3u8; 32], [4u8; 32], [5u8; 32], [6u8; 32], [7u8; 32],
		];
		let tree = MerkleTree::new(&leaves);
		let path = tree.merkle_path(0);
		let trace = MerkleTreeTrace::generate(
			vec![tree.root],
			&[MerklePath {
				root_id: 0,
				index: 0,
				leaf: leaves[0],
				nodes: path,
			}],
		);
		let allocator = Bump::new();
		let mut witness =
			WitnessIndex::<PackedType<OptimalUnderlier128b, B128>>::new(&cs, &allocator);

		witness
			.fill_table_sequential(&root_table, &trace.root.into_iter().collect::<Vec<_>>())
			.unwrap();
	}

	#[test]
	fn test_merkle_tree_cs_fill_tables() {
		let mut cs = ConstraintSystem::new();
		let merkle_tree_cs = MerkleTreeCS::new(&mut cs);

		let tree = MerkleTree::new(&[
			[0u8; 32], [1u8; 32], [2u8; 32], [3u8; 32], [4u8; 32], [5u8; 32], [6u8; 32], [7u8; 32],
		]);
		let index = 0;
		let path = tree.merkle_path(index);
		let vals = B8::from_underliers_arr(tree.root);

		let trace = MerkleTreeTrace::generate(
			vec![tree.root],
			&[MerklePath {
				root_id: 0,
				index,
				leaf: [0u8; 32],
				nodes: path,
			}],
		);

		let allocator = Bump::new();
		let mut witness =
			WitnessIndex::<PackedType<OptimalUnderlier128b, B128>>::new(&cs, &allocator);

		merkle_tree_cs.fill_tables(&trace, &mut witness);
	}

	#[test]
	fn test_merkle_tree_cs_end_to_end() {
		let mut cs = ConstraintSystem::new();
		let merkle_tree_cs = MerkleTreeCS::new(&mut cs);

		// Create a Merkle tree with 8 leaves
		let leaves = vec![
			[0u8; 32], [1u8; 32], [2u8; 32], [3u8; 32], [4u8; 32], [5u8; 32], [6u8; 32], [7u8; 32],
		];
		let tree = MerkleTree::new(&leaves);

		// Generate a Merkle path for a specific index
		let index = 3;
		let path = tree.merkle_path(index);

		// Generate the trace for the Merkle tree
		let trace = MerkleTreeTrace::generate(
			vec![tree.root],
			&[MerklePath {
				root_id: 0,
				index,
				leaf: leaves[index],
				nodes: path,
			}],
		);

		// Allocate memory for the witness
		let allocator = Bump::new();
		let mut witness =
			WitnessIndex::<PackedType<OptimalUnderlier128b, B128>>::new(&cs, &allocator);

		// Fill the tables with the trace
		merkle_tree_cs.fill_tables(&trace, &mut witness);

		// Create boundary values based on the trace's boundaries
		let mut boundaries = Vec::new();

		// Add boundaries for leaf nodes
		for leaf in &trace.boundaries.leaf {
			let leaf_state = bytes_to_boundary(&leaf.data);
			let values = vec![
				B128::new(leaf.root_id as u128),
				B128::from(leaf_state[0]),
				B128::from(leaf_state[1]),
				B128::from(leaf_state[2]),
				B128::from(leaf_state[3]),
				B128::from(leaf_state[4]),
				B128::from(leaf_state[5]),
				B128::from(leaf_state[6]),
				B128::from(leaf_state[7]),
				B128::new(leaf.depth as u128),
				B128::new(leaf.index as u128),
			];
			boundaries.push(Boundary {
				values,
				channel_id: merkle_tree_cs.nodes_channel,
				direction: FlushDirection::Push,
				multiplicity: 1,
			});
		}

		// Add boundaries for roots
		for root in &trace.boundaries.root {
			let state = bytes_to_boundary(&root.data);
			let values = vec![
				B128::new(root.root_id as u128),
				B128::from(state[0]),
				B128::from(state[1]),
				B128::from(state[2]),
				B128::from(state[3]),
				B128::from(state[4]),
				B128::from(state[5]),
				B128::from(state[6]),
				B128::from(state[7]),
			];
			boundaries.push(Boundary {
				values,
				channel_id: merkle_tree_cs.roots_channel,
				direction: FlushDirection::Push,
				multiplicity: 1,
			});
		}

		// Validate the system and witness
		validate_system_witness::<OptimalUnderlier128b>(&cs, witness, boundaries);
	}
}
