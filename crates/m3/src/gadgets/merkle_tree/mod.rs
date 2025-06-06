// Copyright 2025 Irreducible Inc.

/// This module provides the arithmetisation of the Merkle tree inclusion proof.
///
/// It defines the constraint system, witness filling, and boundary construction for
/// verifying Merkle tree paths using Groestl-256 as the compression function.
///
/// The main components are:
/// - `MerkleTreeCS`: The constraint system for the Merkle tree, including all tables and channels.
/// - `NodesTable` and `RootTable`: Table definitions for Merkle path steps and root checks.
/// - Table filling logic for witness generation.
/// - Boundary construction for circuit input/output consistency.
///
/// The module is designed to be used in tests and as a reference for implementing Merkle
/// tree inclusion proofs in the Binius M3 framework.
pub mod trace;

use std::{array, cell::RefMut, cmp::Reverse};

use array_util::ArrayExt;
use binius_core::{constraint_system::channel::ChannelId, oracle::ShiftVariant};
use binius_field::{
	BinaryField8b, Field, PackedBinaryField4x8b, PackedBinaryField8x8b, PackedExtension,
	PackedField, PackedFieldIndexable, PackedSubfield,
	linear_transformation::PackedTransformationFactory,
	packed::{get_packed_slice, set_packed_slice},
	underlier::WithUnderlier,
};
use itertools::Itertools;
use trace::{MerklePathEvent, MerkleRootEvent, MerkleTreeTrace, NodeFlushToken, RootFlushToken};

use crate::{
	builder::{
		B1, B8, B32, B64, B128, Boundary, Col, ConstraintSystem, FlushDirection, TableBuilder,
		TableFiller, TableId, TableWitnessSegment, WitnessIndex, tally, upcast_col,
	},
	gadgets::{
		hash::groestl::{Permutation, PermutationVariant},
		indexed_lookup::incr::{Incr, IncrIndexedLookup, IncrLookup, merge_incr_vals},
	},
};
/// A struct representing the constraint system for the Merkle tree. Like any M3 instance,
/// it is characterized by the tables with the column constraints, and the channels with
/// the flushing rules.
pub struct MerkleTreeCS {
	/// The tables for the three cases of merkle path pulls i.e. left, right or both children
	/// being pulled.
	pub merkle_path_table_left: NodesTable,
	pub merkle_path_table_right: NodesTable,
	pub merkle_path_table_both: NodesTable,

	/// Table for reconciling the final values of the merkle paths with the roots.
	pub root_table: RootTable,

	pub incr_table: IncrLookup,
	/// Channel for all intermediate nodes in the merkle paths being verified.
	/// Follows format [Root ID, Digest, Depth, Index].
	pub nodes_channel: ChannelId,

	/// Channel for the roots for roots of the merkle paths being verified
	/// (deduped for multiple paths in the same tree).
	/// Follows format [Root ID, Digest].
	pub roots_channel: ChannelId,

	/// Channel for verifying that child depth is one more than parent depth
	/// has one value.
	pub lookup_channel: ChannelId,
}

impl MerkleTreeCS {
	pub fn new(cs: &mut ConstraintSystem) -> Self {
		let nodes_channel = cs.add_channel("merkle_tree_nodes");
		let roots_channel = cs.add_channel("merkle_tree_roots");
		let lookup_channel = cs.add_channel("incr_lookup");
		let permutation_channel = cs.add_channel("permutation");

		let merkle_path_table_left =
			NodesTable::new(cs, MerklePathPullChild::Left, nodes_channel, lookup_channel);
		let merkle_path_table_right =
			NodesTable::new(cs, MerklePathPullChild::Right, nodes_channel, lookup_channel);
		let merkle_path_table_both =
			NodesTable::new(cs, MerklePathPullChild::Both, nodes_channel, lookup_channel);

		let root_table = RootTable::new(cs, nodes_channel, roots_channel);

		let mut table = cs.add_table("incr_lookup_table");
		let incr_table = IncrLookup::new(&mut table, lookup_channel, permutation_channel, 20);
		Self {
			merkle_path_table_left,
			merkle_path_table_right,
			merkle_path_table_both,
			root_table,
			nodes_channel,
			roots_channel,
			lookup_channel,
			incr_table,
		}
	}

	pub fn fill_tables(
		&self,
		trace: &MerkleTreeTrace,
		cs: &ConstraintSystem,
		witness: &mut WitnessIndex,
	) -> anyhow::Result<()> {
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
		witness.fill_table_parallel(&self.merkle_path_table_left, &left_events)?;
		witness.fill_table_parallel(&self.merkle_path_table_right, &right_events)?;
		witness.fill_table_parallel(&self.merkle_path_table_both, &both_events)?;

		// Fill the roots table.
		witness.fill_table_parallel(
			&self.root_table,
			&trace.root.clone().into_iter().collect::<Vec<_>>(),
		)?;

		let lookup_counts = tally(cs, witness, &[], self.lookup_channel, &IncrIndexedLookup)?;

		// Fill the lookup table with the sorted counts
		let sorted_counts = lookup_counts
			.into_iter()
			.enumerate()
			.sorted_by_key(|(_, count)| Reverse(*count))
			.collect::<Vec<_>>();
		witness.fill_table_parallel(&self.incr_table, &sorted_counts)?;
		witness.fill_constant_cols()?;
		Ok(())
	}

	pub fn make_boundaries(&self, trace: &MerkleTreeTrace) -> Vec<Boundary<B128>> {
		let mut boundaries = Vec::new();
		// Add boundaries for leaf nodes
		for &NodeFlushToken {
			root_id,
			data,
			depth,
			index,
		} in &trace.boundaries.leaf
		{
			let leaf_state = bytes_to_boundary(&data);
			let values = vec![
				B128::from(root_id as u128),
				leaf_state[0],
				leaf_state[1],
				leaf_state[2],
				leaf_state[3],
				leaf_state[4],
				leaf_state[5],
				leaf_state[6],
				leaf_state[7],
				B128::from(depth as u128),
				B128::from(index as u128),
			];
			boundaries.push(Boundary {
				values,
				channel_id: self.nodes_channel,
				direction: FlushDirection::Push,
				multiplicity: 1,
			});
		}

		// Add boundaries for roots
		for &RootFlushToken { root_id, data } in &trace.boundaries.root {
			let state = bytes_to_boundary(&data);
			let values = vec![
				B128::new(root_id as u128),
				state[0],
				state[1],
				state[2],
				state[3],
				state[4],
				state[5],
				state[6],
				state[7],
			];
			boundaries.push(Boundary {
				values,
				channel_id: self.roots_channel,
				direction: FlushDirection::Push,
				multiplicity: 1,
			});
		}
		boundaries
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

// TODO: Break into cases to reduce the number of packs.
pub struct NodesTable {
	id: TableId,
	// The root id field is used to identify the root the node is associated with.
	root_id: Col<B8>,
	// Concatenated bytes of the left and right digests of the Merkle tree node. Organised
	// in a packed row of the Groestl-256 permutation state.
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
	pub parent_depth: Col<B8>,
	pub child_depth: Col<B8>,
	parent_index: Col<B1, 32>,
	left_index: Col<B1, 32>,
	right_index_packed: Col<B32>,
	/// A gadget representing the Groestl-256 P pemutation. The output transformation as
	/// per the Groestl-256 specification (https://www.groestl.info/Groestl.pdf) is being used as a digest compression function here.
	permutation: Permutation,
	/// A gadget for handling integer increments in the Merkle tree. It is used to
	/// constrain that the depth of the child node is one more than the parent node.
	increment: Incr,
	pub _pull_child: MerklePathPullChild,
}

impl NodesTable {
	pub fn new(
		cs: &mut ConstraintSystem,
		pull_child: MerklePathPullChild,
		nodes_channel_id: ChannelId,
		lookup_chan: ChannelId,
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

		let left_columns: [Col<BinaryField8b, 4>; 8] = table.add_committed_multiple("left_columns");
		let right_columns: [Col<BinaryField8b, 4>; 8] =
			table.add_committed_multiple("right_columns");
		let state_in = table.add_committed_multiple("state_in");

		let left_packed: [Col<B32>; 8] =
			array::from_fn(|i| table.add_packed(format!("left_packed[{i}]"), left_columns[i]));
		let right_packed: [Col<B32>; 8] =
			array::from_fn(|i| table.add_packed(format!("right_packed[{i}]"), right_columns[i]));
		let state_in_packed: [Col<B64>; 8] =
			array::from_fn(|i| table.add_packed(format!("state_in_packed[{i}]"), state_in[i]));

		for i in 0..8 {
			table.assert_zero(
				format!("state_in_assert[{i}]"),
				(state_in_packed[i]
					- upcast_col(left_packed[i])
					- upcast_col(right_packed[i]) * B64::from(1 << 32))
				.into(),
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
				format!("state_in_shifted[{i}]"),
				state_out[i],
				3,
				4,
				ShiftVariant::LogicalRight,
			)
		});

		let permutation_output_columns: [Col<BinaryField8b, 4>; 8] = array::from_fn(|i| {
			table.add_selected_block(
				format!("permutation_output_columns[{i}]"),
				state_out_shifted[i],
				4,
			)
		});

		let parent_columns: [Col<B8, 4>; 8] = array::from_fn(|i| {
			table.add_computed(
				format!("parent_columns[{i}]"),
				permutation_output_columns[i] + right_columns[i],
			)
		});

		let parent_packed: [Col<B32>; 8] =
			array::from_fn(|i| table.add_packed(format!("parent_packed[{i}]"), parent_columns[i]));

		let parent_depth = table.add_committed("parent_depth");

		let one = table.add_constant("one", [B1::ONE]);

		let increment = Incr::new(&mut table, lookup_chan, parent_depth, one);
		let child_depth = increment.output;

		let parent_index: Col<B1, 32> = table.add_committed("parent_index");
		let left_index: Col<B1, 32> =
			table.add_shifted("left_index", parent_index, 5, 1, ShiftVariant::LogicalLeft);
		// The indexes need to be packed and upcasted to agree with the flushing rules of the
		// channel.
		let left_index_packed = table.add_packed("left_index_packed", left_index);
		let right_index_packed =
			table.add_computed("right_index_packed", left_index_packed + B32::ONE);
		let parent_index_packed: Col<B32> = table.add_packed("parent_index_packed", parent_index);

		let left_index_upcasted = upcast_col(left_index_packed);
		let right_index_upcasted = upcast_col(right_index_packed);
		let parent_index_upcasted = upcast_col(parent_index_packed);
		let parent_depth_upcasted = upcast_col(parent_depth);
		let child_depth_upcasted = upcast_col(child_depth);
		let root_id_upcasted = upcast_col(root_id);

		let mut nodes_channel = NodesChannel::new(&mut table, nodes_channel_id);

		nodes_channel.push(
			root_id_upcasted,
			parent_packed,
			parent_depth_upcasted,
			parent_index_upcasted,
		);

		match pull_child {
			MerklePathPullChild::Left => nodes_channel.pull(
				root_id_upcasted,
				left_packed,
				child_depth_upcasted,
				left_index_upcasted,
			),
			MerklePathPullChild::Right => nodes_channel.pull(
				root_id_upcasted,
				right_packed,
				child_depth_upcasted,
				right_index_upcasted,
			),
			MerklePathPullChild::Both => {
				nodes_channel.pull(
					root_id_upcasted,
					left_packed,
					child_depth_upcasted,
					left_index_upcasted,
				);
				nodes_channel.pull(
					root_id_upcasted,
					right_packed,
					child_depth_upcasted,
					right_index_upcasted,
				);
			}
		}
		Self {
			id,
			root_id,
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
			_pull_child: pull_child,
			permutation,
			increment,
		}
	}
}

pub struct NodesChannel<'a> {
	table: &'a mut TableBuilder<'a>,
	channel_id: ChannelId,
}

impl<'a> NodesChannel<'a> {
	pub fn new(table: &'a mut TableBuilder<'a>, channel_id: ChannelId) -> Self {
		Self { table, channel_id }
	}

	pub fn push(
		&mut self,
		root_id: Col<B32>,
		digest: [Col<B32>; 8],
		depth: Col<B32>,
		index: Col<B32>,
	) {
		self.table
			.push(self.channel_id, to_node_flush(upcast_col(root_id), digest, depth, index));
	}

	pub fn pull(
		&mut self,
		root_id: Col<B32>,
		digest: [Col<B32>; 8],
		depth: Col<B32>,
		index: Col<B32>,
	) {
		self.table
			.pull(self.channel_id, to_node_flush(upcast_col(root_id), digest, depth, index));
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
}

impl RootTable {
	pub fn new(
		cs: &mut ConstraintSystem,
		nodes_channel_id: ChannelId,
		roots_channel_id: ChannelId,
	) -> Self {
		let mut table = cs.add_table("merkle_tree_roots");
		let id = table.id();
		let root_id = table.add_committed("root_id");
		let digest = table.add_committed_multiple("digest");

		let zero = table.add_constant("zero", [B32::ZERO]);
		let root_id_upcasted = upcast_col(root_id);
		table.pull(roots_channel_id, to_root_flush(root_id_upcasted, digest));
		let mut nodes_channel = NodesChannel::new(&mut table, nodes_channel_id);
		nodes_channel.pull(root_id_upcasted, digest, zero, zero);
		Self {
			id,
			root_id,
			digest,
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
					(B8::from_underliers_arr_ref(left), B8::from_underliers_arr_ref(right));

				digest[..32].copy_from_slice(left_bytes);
				digest[32..].copy_from_slice(right_bytes);
				digest
			})
			.collect::<Vec<_>>();
		self.permutation.populate_state_in(witness, &state_ins)?;
		self.permutation.populate(witness)?;

		let mut witness_root_id: RefMut<'_, [u8]> = witness.get_mut_as(self.root_id)?;
		let mut witness_parent_depth: RefMut<'_, [u8]> = witness.get_mut_as(self.parent_depth)?;
		let mut witness_child_depth: RefMut<'_, [u8]> = witness.get_mut_as(self.child_depth)?;
		let mut witness_parent_index: RefMut<'_, [u32]> = witness.get_mut_as(self.parent_index)?;
		// The left and right indexes are packed into a single column, so we need to unpack them
		let mut witness_left_index: RefMut<'_, [u32]> = witness.get_mut_as(self.left_index)?;
		let mut witness_right_index_packed: RefMut<'_, [u32]> =
			witness.get_mut_as(self.right_index_packed)?;

		let witness_state_out: [RefMut<'_, [PackedBinaryField8x8b]>; 8] = self
			.permutation
			.state_out()
			.try_map_ext(|col| witness.get_mut_as(col))?;

		let mut witness_state_out_shifted: [RefMut<'_, [PackedBinaryField8x8b]>; 8] = self
			.state_out_shifted
			.try_map_ext(|col| witness.get_mut_as(col))?;

		let mut witness_permutation_output_columns: [RefMut<'_, [PackedBinaryField4x8b]>; 8] = self
			.permutation_output_columns
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

		let mut increment_merged: RefMut<'_, [u32]> = witness.get_mut_as(self.increment.merged)?;

		{
			for (i, event) in rows.enumerate() {
				let &MerklePathEvent {
					root_id,
					parent_depth,
					parent_index,
					left,
					right,
					parent,
					..
				} = event;

				witness_root_id[i] = root_id;
				witness_parent_depth[i] = parent_depth
					.try_into()
					.expect("Parent depth must fit in u8");
				witness_parent_index[i] = parent_index as u32;
				witness_child_depth[i] = witness_parent_depth[i] + 1;
				witness_left_index[i] = 2 * parent_index as u32;
				witness_right_index_packed[i] = 2 * parent_index as u32 + 1;

				increment_merged[i] =
					merge_incr_vals(witness_parent_depth[i], true, witness_child_depth[i], false);
				let left_bytes: [BinaryField8b; 32] = B8::from_underliers_arr(left);
				let right_bytes: [BinaryField8b; 32] = B8::from_underliers_arr(right);
				let parent_bytes: [BinaryField8b; 32] = B8::from_underliers_arr(parent);

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
			let &MerkleRootEvent { root_id, digest } = event;
			witness_root_id[i] = root_id;
			let digest_as_field = B8::from_underliers_arr(digest);
			for (jk, &byte) in digest_as_field.iter().enumerate() {
				// Row in the state
				let j = jk % 8;
				// Col in the state
				let k = jk / 8;
				set_packed_slice(&mut witness_root_digest[j], i * 4 + k, byte);
			}
		}
		Ok(())
	}
}

fn bytes_to_boundary(bytes: &[u8; 32]) -> [B128; 8] {
	let mut cols = [PackedBinaryField4x8b::zero(); 8];
	for ij in 0..32 {
		// Row in the state
		let i = ij % 8;
		// Col in the state
		let j = ij / 8;

		// Set the packed slice for the row.
		set_packed_slice(&mut cols, 4 * i + j, B8::from(bytes[8 * j + i]));
	}
	cols.map(|col| B128::from(col.to_underlier() as u128))
}

#[cfg(test)]
mod tests {
	use binius_field::{arch::OptimalUnderlier, as_packed_field::PackedType};
	use bumpalo::Bump;
	use rand::{Rng, SeedableRng, rngs::StdRng};
	use trace::{MerklePath, MerkleTree};

	use super::*;
	use crate::builder::test_utils::validate_system_witness;
	#[test]
	fn test_nodes_table_constructor() {
		let mut cs = ConstraintSystem::new();
		let nodes_channel = cs.add_channel("nodes");
		let lookup_channel = cs.add_channel("lookup");
		let pull_child = MerklePathPullChild::Left;
		let nodes_table = NodesTable::new(&mut cs, pull_child, nodes_channel, lookup_channel);
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
		let lookup_channel = cs.add_channel("lookup");
		// Create a nodes table with the left child pull.
		let pull_child = MerklePathPullChild::Left;
		let nodes_table = NodesTable::new(&mut cs, pull_child, nodes_channel, lookup_channel);
		let tree = MerkleTree::new(&[
			[0u8; 32], [1u8; 32], [2u8; 32], [3u8; 32], [4u8; 32], [5u8; 32], [6u8; 32], [7u8; 32],
		]);

		let index = 0;
		let path = tree.merkle_path(0);
		let trace = MerkleTreeTrace::generate(
			vec![tree.root()],
			&[MerklePath {
				root_id: 0,
				index,
				leaf: [0u8; 32],
				nodes: path,
			}],
		);
		let allocator = Bump::new();
		let mut witness = WitnessIndex::<PackedType<OptimalUnderlier, B128>>::new(&cs, &allocator);

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
			vec![tree.root()],
			&[MerklePath {
				root_id: 0,
				index: 0,
				leaf: leaves[0],
				nodes: path,
			}],
		);
		let allocator = Bump::new();
		let mut witness = WitnessIndex::<PackedType<OptimalUnderlier, B128>>::new(&cs, &allocator);

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

		let trace = MerkleTreeTrace::generate(
			vec![tree.root()],
			&[MerklePath {
				root_id: 0,
				index,
				leaf: [0u8; 32],
				nodes: path,
			}],
		);

		let allocator = Bump::new();
		let mut witness = WitnessIndex::<PackedType<OptimalUnderlier, B128>>::new(&cs, &allocator);

		merkle_tree_cs
			.fill_tables(&trace, &cs, &mut witness)
			.unwrap();
	}

	#[test]
	fn test_merkle_tree_cs_end_to_end() {
		let mut cs = ConstraintSystem::new();
		let merkle_tree_cs = MerkleTreeCS::new(&mut cs);

		let mut rng = StdRng::seed_from_u64(0);
		// Create a Merkle tree with 8 leaves
		let index = rng.gen_range(0..1 << 10);
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
		let roots = (0..3).map(|i| trees[i].root()).collect::<Vec<_>>();
		let paths = trees
			.iter()
			.enumerate()
			.map(|(i, tree)| MerklePath {
				root_id: i as u8,
				index,
				leaf: leaves[i][index],
				nodes: tree.merkle_path(index),
			})
			.collect::<Vec<_>>();

		let trace = MerkleTreeTrace::generate(roots, &paths);

		// Allocate memory for the witness
		let allocator = Bump::new();
		let mut witness = WitnessIndex::new(&cs, &allocator);

		// Fill the tables with the trace
		merkle_tree_cs
			.fill_tables(&trace, &cs, &mut witness)
			.unwrap();

		// Create boundary values based on the trace's boundaries
		let boundaries = merkle_tree_cs.make_boundaries(&trace);

		// Validate the system and witness
		validate_system_witness::<OptimalUnderlier>(&cs, witness, boundaries);
	}
}
