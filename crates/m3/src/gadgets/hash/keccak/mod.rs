// Copyright 2025 Irreducible Inc.

//! SHA3-256 (Keccak) hash function verification gadgets.

use std::array;

use anyhow::Result;
use array_util::ArrayExt as _;
use binius_core::oracle::ShiftVariant;
use binius_field::{
	linear_transformation::PackedTransformationFactory, Field, PackedExtension,
	PackedFieldIndexable, PackedSubfield, TowerField,
};
pub use state::{StateMatrix, StateRow};
use trace::PermutationTrace;

use crate::builder::{Col, Expr, TableBuilder, TableWitnessSegment, B1, B128, B64, B8};

mod state;
mod test_vector;
mod trace;

// This implementation tries to be as close to the
// [Keccak Specification Summary][keccak_spec_summary] and as such it is highly recommended to
// get familiar with it. A lot of terminology is carried over from that spec.
//
// [keccak_spec_summary]: https://keccak.team/keccak_specs_summary.html

/// 8x 64-bit lanes packed[^packed] in the single column.
///
/// For the motivation see [`Keccakf`] documentation.
///
/// [^packed]: here it means in the SIMD sense, not as in packed columns.
pub type PackedLane8 = Col<B1, { 64 * 8 }>;

const ROUNDS_PER_PERMUTATION: usize = 24;
const BATCHES_PER_PERMUTATION: usize = 3;
const TRACKS_PER_BATCH: usize = 8;
const STATE_IN_TRACK: usize = 0;
const STATE_OUT_TRACK: usize = 7;

/// Keccak-f\[1600\] permutation function verification gadget.
///
/// This gadget consists of 3x horizontally combined batches of 8x rounds each, 24 rounds in total.
/// You can think about it as 8x wide SIMD performing one permutation per a table row. Below is
/// the graphical representation of the layout.
///
/// ```plain
/// | Batch 0  | Batch 1  | Batch 3  |
/// |----------|----------|----------|
/// | Round 00 | Round 01 | Round 02 |
/// | Round 03 | Round 04 | Round 05 |
/// | Round 06 | Round 07 | Round 08 |
/// | Round 09 | Round 10 | Round 11 |
/// | Round 12 | Round 13 | Round 14 |
/// | Round 15 | Round 16 | Round 17 |
/// | Round 18 | Round 19 | Round 20 |
/// | Round 21 | Round 22 | Round 23 |
/// ```
///
/// We refer to each individual round within a batch as a **track**. For example, the 7th (
/// zero-based here and henceforth) track of the 1st batch is responsible for the 22nd round.
///
/// Each batch exposes two notable columns: `state_in` and `state` which are inputs and outputs
/// respectively for the rounds in each batch. Both of those has the type of [`StateMatrix`]
/// containing [`PackedLane8`]. Let's break those down.
///
/// [`StateMatrix`] is a concept coming from the keccak which represents a 5x5 matrix. In keccak
/// each cell is a 64-bit integer called lane. In our case however, since the SIMD-like approach,
/// each cell is represented by a pack of columns - one for each track and this is what
/// [`PackedLane8`] represents.
///
/// To feed the input to permutation, you need to initialize the `state_in` column of the 0th batch
/// with the input state matrix. See [`Self::populate_state_in`] if you have values handy.
pub struct Keccakf {
	batches: [RoundBatch; BATCHES_PER_PERMUTATION],
	/// The lanes of the input and output state columns. These are exposed to make it convenient to
	/// use the gadget along with flushing.
	pub input: StateMatrix<Col<B64>>,
	pub output: StateMatrix<Col<B64>>,

	/// Represents a variation of the `state_in` state matrix of the 0th batch where each track is
	/// shifted in place of previous one, meaning that the 0th track will store the `state_in` for
	/// the 3rd round.
	///
	/// This is used for the state-in to state-out linking rule.
	next_state_in: StateMatrix<PackedLane8>,

	/// Link selector.
	///
	/// This is all ones for the first 7 tracks and all zeroes for the last one.
	///
	/// Used to turn off the state-in to state-out forwarding check for the last track.
	link_sel: PackedLane8,
}

impl Keccakf {
	/// Creates a new instance of the gadget.
	///
	/// See the struct documentation for more details.
	pub fn new(table: &mut TableBuilder) -> Self {
		let state_in: StateMatrix<PackedLane8> =
			StateMatrix::from_fn(|(x, y)| table.add_committed(format!("state_in[{x},{y}]")));

		let mut state = state_in;

		// Declaring packed state_in columns for exposing in the struct.
		let state_in_packed: StateMatrix<Col<B64, 8>> = StateMatrix::from_fn(|(x, y)| {
			table.add_packed(format!("state_in_packed[{x},{y}]"), state[(x, y)])
		});

		// Constructing the batches of rounds. The final value of `state` will be the permutation
		// output.
		let batches = array::from_fn(|batch_no| {
			let batch = RoundBatch::new(
				&mut table.with_namespace(format!("batch[{batch_no}]")),
				state.clone(),
				batch_no,
			);
			state = batch.state_out.clone();
			batch
		});

		// Declaring packed state_out columns to be exposed in the struct.
		let state_out_packed: StateMatrix<Col<B64, 8>> = StateMatrix::from_fn(|(x, y)| {
			table.add_packed(format!("state_out_packed[{x},{y}]"), state[(x, y)])
		});

		let input = StateMatrix::from_fn(|(x, y)| {
			table.add_selected(format!("input[{x},{y}]"), state_in_packed[(x, y)], 0)
		});

		let output = StateMatrix::from_fn(|(x, y)| {
			table.add_selected(format!("output[{x},{y}]"), state_out_packed[(x, y)], 7)
		});

		let link_sel = table.add_constant(
			"link_sel",
			array::from_fn(|bit_index| {
				if bit_index < 64 * 7 {
					B1::ONE
				} else {
					B1::ZERO
				}
			}),
		);
		let next_state_in = StateMatrix::from_fn(|(x, y)| {
			table.add_shifted(
				format!("next_state_in[{x},{y}]"),
				batches[0].state_in[(x, y)],
				9,
				64,
				ShiftVariant::LogicalRight,
			)
		});
		for x in 0..5 {
			for y in 0..5 {
				let state_out = &batches[2].state_out;
				table.assert_zero(
					"link_out_to_next_in",
					(state_out[(x, y)] - next_state_in[(x, y)]) * link_sel,
				);
			}
		}
		Self {
			batches,
			next_state_in,
			input,
			output,
			link_sel,
		}
	}

	/// Populate the gadget.
	///
	/// Requires state in already to be populated. To populate with known values use
	/// [`Self::populate_state_in`].
	pub fn populate<P>(&self, index: &mut TableWitnessSegment<P>) -> Result<()>
	where
		P: PackedFieldIndexable<Scalar = B128>
			+ PackedExtension<B1>
			+ PackedExtension<B8>
			+ PackedExtension<B64>,
		PackedSubfield<P, B8>: PackedTransformationFactory<PackedSubfield<P, B8>>,
	{
		// `state_in` for the first track of the first batch specifies the initial state for
		// permutation. Read it out, gather trace and populate each batch.
		let permutation_traces = self.batches[0]
			.read_state_ins(index, 0)?
			.map(trace::keccakf_trace)
			.collect::<Vec<PermutationTrace>>();
		for batch in &self.batches {
			batch.populate(index, &permutation_traces)?;
		}

		for k in 0..permutation_traces.len() {
			for x in 0..5 {
				for y in 0..5 {
					let mut next_state_in: std::cell::RefMut<'_, [u64]> =
						index.get_mut_as(self.next_state_in[(x, y)])?;
					let batch_0_state_in: std::cell::Ref<'_, [u64]> =
						index.get_as(self.batches[0].state_in[(x, y)])?;
					let batch_2_state_out: std::cell::Ref<'_, [u64]> =
						index.get_as(self.batches[2].state_out[(x, y)])?;
					for track in 0..TRACKS_PER_BATCH - 1 {
						next_state_in[TRACKS_PER_BATCH * k + track] =
							batch_0_state_in[TRACKS_PER_BATCH * k + track + 1];
						assert_eq!(
							next_state_in[TRACKS_PER_BATCH * k + track],
							batch_2_state_out[TRACKS_PER_BATCH * k + track]
						);
					}
					// Populating the packed and selected input and output columns.
					let mut input: std::cell::RefMut<'_, [u64]> =
						index.get_mut_as(self.input[(x, y)])?;
					let mut output: std::cell::RefMut<'_, [u64]> =
						index.get_mut_as(self.output[(x, y)])?;

					input[k] = permutation_traces[k].input()[(x, y)];
					output[k] = permutation_traces[k].output()[(x, y)];
				}
			}
		}

		{
			let mut link_sel: std::cell::RefMut<'_, [u64]> = index.get_mut_as(self.link_sel)?;
			assert!(link_sel.len() % TRACKS_PER_BATCH == 0);
			for link_sel_chunk in link_sel.chunks_exact_mut(TRACKS_PER_BATCH) {
				link_sel_chunk.copy_from_slice(&[
					u64::MAX,
					u64::MAX,
					u64::MAX,
					u64::MAX,
					u64::MAX,
					u64::MAX,
					u64::MAX,
					0,
				]);
			}
		}

		Ok(())
	}

	/// Returns the `state_in` column for the 0th batch. The input to the permutation is at the
	/// 0th track.
	pub fn packed_state_in(&self) -> &StateMatrix<PackedLane8> {
		&self.batches[0].state_in
	}

	/// Returns the `state_out` column for the 2nd batch. The output of the permutation is at the
	/// 7th track.
	pub fn packed_state_out(&self) -> &StateMatrix<PackedLane8> {
		&self.batches[2].state_out
	}

	/// Populate the input state of the permutation.
	pub fn populate_state_in<'a, P>(
		&self,
		index: &mut TableWitnessSegment<P>,
		state_ins: impl IntoIterator<Item = &'a StateMatrix<u64>>,
	) -> Result<()>
	where
		P: PackedFieldIndexable<Scalar = B128> + PackedExtension<B1> + PackedExtension<B8>,
		PackedSubfield<P, B8>: PackedTransformationFactory<PackedSubfield<P, B8>>,
	{
		self.batches[0].populate_state_in(index, STATE_IN_TRACK, state_ins)?;
		Ok(())
	}

	/// Read the resulting states of permutation, one item per row.
	///
	/// Only makes sense to call after [`Self::populate`] was called.
	pub fn read_state_outs<'a, P>(
		&self,
		index: &'a TableWitnessSegment<P>,
	) -> Result<impl Iterator<Item = StateMatrix<u64>> + 'a>
	where
		P: PackedFieldIndexable<Scalar = B128> + PackedExtension<B1> + PackedExtension<B8>,
		PackedSubfield<P, B8>: PackedTransformationFactory<PackedSubfield<P, B8>>,
	{
		self.batches[2].read_state_outs(index, STATE_OUT_TRACK)
	}
}

/// A gadget of a batch of keccak-f\[1600\] permutation rounds.
///
/// This batch runs 8 rounds of keccak-f. Since SHA3-256 is defined to have 24 rounds, you would
/// need to use 3 of these gadgets to implement a full permutation.
struct RoundBatch {
	batch_no: usize,
	state_in: StateMatrix<PackedLane8>,
	state_out: StateMatrix<PackedLane8>,
	c: StateRow<PackedLane8>,
	c_shift: StateRow<PackedLane8>,
	d: StateRow<PackedLane8>,
	a_theta: StateMatrix<PackedLane8>,
	b: StateMatrix<PackedLane8>,
	round_const: PackedLane8,
}

impl RoundBatch {
	fn new(table: &mut TableBuilder, state_in: StateMatrix<PackedLane8>, batch_no: usize) -> Self {
		assert!(batch_no < BATCHES_PER_PERMUTATION);
		let state_out =
			StateMatrix::from_fn(|(x, y)| table.add_committed(format!("state_out[{x},{y}]")));

		// # θ step
		//
		// for x in 0…4:
		//   C[x] = A[x,0] xor A[x,1] xor A[x,2] xor A[x,3] xor A[x,4],
		//   D[x] = C[x-1] xor rot(C[x+1],1),
		let c = StateRow::from_fn(|x| {
			table.add_computed(
				format!("c[{x}]"),
				sum_expr(array::from_fn::<_, 5, _>(|offset| state_in[(x, offset)])),
			)
		});
		let c_shift = StateRow::from_fn(|x| {
			table.add_shifted(format!("c[{x}]"), c[x], 6, 1, ShiftVariant::CircularLeft)
		});
		let d =
			StateRow::from_fn(|x| table.add_computed(format!("d[{x}]"), c[x + 4] + c_shift[x + 1]));

		// for (x,y) in (0…4,0…4):
		//   A[x,y] = A[x,y] xor D[x]
		let a_theta = StateMatrix::from_fn(|(x, y)| {
			table.add_computed(format!("a_theta[{x},{y}]"), state_in[(x, y)] + d[x])
		});

		// # ρ and π steps
		let b = StateMatrix::from_fn(|(x, y)| {
			if (x, y) == (0, 0) {
				a_theta[(0, 0)]
			} else {
				const INV2: usize = 3;
				let dy = x; // 0‥4
				let tmp = (y + 2 * x) % 5;
				let dx = (INV2 * tmp) % 5;
				let rot = RHO[dx][dy] as usize;
				const LOG2_64: usize = 6;
				table.add_shifted(
					format!("b[{x},{y}]"),
					a_theta[(dx, dy)],
					LOG2_64,
					rot,
					ShiftVariant::CircularLeft,
				)
			}
		});

		let round_const = table
			.add_constant(format!("round_const[{batch_no}]"), round_consts_for_batch(batch_no));

		for x in 0..5 {
			for y in 0..5 {
				let output = state_out[(x, y)];
				let b0 = b[(x, y)];
				let b1 = b[(x + 1, y)];
				let b2 = b[(x + 2, y)];
				// Constraint output to have the result from the chi and optionally iota steps.
				//
				// # χ step
				// A[x,y] = B[x,y] xor ((not B[x+1,y]) and B[x+2,y]),  for (x,y) in (0…4,0…4)
				//
				// # ι step
				// A[0,0] = A[0,0] xor RC
				if (x, y) == (0, 0) {
					table.assert_zero(
						format!("chi_iota[{x},{y}]"),
						output - (round_const + b0 + (b1 - B1::from(1)) * b2),
					);
				} else {
					table.assert_zero(
						format!("chi[{x},{y}]"),
						output - (b0 + (b1 - B1::from(1)) * b2),
					);
				}
			}
		}

		Self {
			batch_no,
			state_in,
			state_out,
			c,
			c_shift,
			d,
			a_theta,
			b,
			round_const,
		}
	}

	fn populate<P>(
		&self,
		index: &mut TableWitnessSegment<P>,
		permutation_traces: &[trace::PermutationTrace],
	) -> Result<()>
	where
		P: PackedFieldIndexable<Scalar = B128> + PackedExtension<B1> + PackedExtension<B8>,
		PackedSubfield<P, B8>: PackedTransformationFactory<PackedSubfield<P, B8>>,
	{
		for (k, trace) in permutation_traces.iter().enumerate() {
			// Gather all batch round traces for the batch number.
			let brt = trace.per_batch(self.batch_no);

			// Fill in round_const witness with the corresponding round constants
			let mut round_const: std::cell::RefMut<'_, [u64]> =
				index.get_mut_as(self.round_const)?;
			for track in 0..TRACKS_PER_BATCH {
				round_const[TRACKS_PER_BATCH * k + track] = brt[track].rc;
			}
			drop(round_const);

			for x in 0..5 {
				let mut c: std::cell::RefMut<'_, [u64]> = index.get_mut_as(self.c[x])?;
				let mut c_shift: std::cell::RefMut<'_, [u64]> =
					index.get_mut_as(self.c_shift[x])?;
				let mut d: std::cell::RefMut<'_, [u64]> = index.get_mut_as(self.d[x])?;

				for track in 0..TRACKS_PER_BATCH {
					let cell_pos = TRACKS_PER_BATCH * k + track;
					c[cell_pos] = brt[track].c[x];
					c_shift[cell_pos] = c[cell_pos].rotate_left(1);
					d[cell_pos] = brt[track].d[x];
				}

				for y in 0..5 {
					let mut state_in: std::cell::RefMut<'_, [u64]> =
						index.get_mut_as(self.state_in[(x, y)])?;
					let mut state_out: std::cell::RefMut<'_, [u64]> =
						index.get_mut_as(self.state_out[(x, y)])?;
					let mut a_theta: std::cell::RefMut<'_, [u64]> =
						index.get_mut_as(self.a_theta[(x, y)])?;

					// b[0,0] is defined as a_theta[0,0].
					//
					// That means two things:
					// 1. The value is assigned by `a_theta`.
					// 2. We have to skip mutably borrowing it here because that would overlap with
					//    the mutable borrow of `a_theta` above.
					let mut b: Option<std::cell::RefMut<'_, [u64]>> = if (x, y) != (0, 0) {
						Some(index.get_mut_as(self.b[(x, y)])?)
					} else {
						None
					};
					for track in 0..TRACKS_PER_BATCH {
						let cell_pos = TRACKS_PER_BATCH * k + track;
						state_in[cell_pos] = brt[track].state_in[(x, y)];
						state_out[cell_pos] = brt[track].state_out[(x, y)];
						a_theta[cell_pos] = brt[track].a_theta[(x, y)];
						if let Some(ref mut b) = b {
							b[cell_pos] = brt[track].b[(x, y)];
						}
					}
				}
			}
		}

		Ok(())
	}

	fn populate_state_in<'a, P>(
		&self,
		index: &mut TableWitnessSegment<P>,
		track: usize,
		state_ins: impl IntoIterator<Item = &'a StateMatrix<u64>>,
	) -> Result<()>
	where
		P: PackedFieldIndexable<Scalar = B128> + PackedExtension<B1> + PackedExtension<B8>,
		PackedSubfield<P, B8>: PackedTransformationFactory<PackedSubfield<P, B8>>,
	{
		for (k, state_in) in state_ins.into_iter().enumerate() {
			for x in 0..5 {
				for y in 0..5 {
					let mut state_in_data: std::cell::RefMut<'_, [u64]> =
						index.get_mut_as(self.state_in[(x, y)])?;
					state_in_data[TRACKS_PER_BATCH * k + track] = state_in[(x, y)];
				}
			}
		}
		Ok(())
	}

	fn read_state_ins<'a, P>(
		&self,
		index: &'a TableWitnessSegment<P>,
		track: usize,
	) -> Result<impl Iterator<Item = StateMatrix<u64>> + 'a>
	where
		P: PackedFieldIndexable<Scalar = B128> + PackedExtension<B1> + PackedExtension<B8>,
		PackedSubfield<P, B8>: PackedTransformationFactory<PackedSubfield<P, B8>>,
	{
		let state_in = self
			.state_in
			.as_inner()
			.try_map_ext(|col| index.get_mut_as(col))?;

		let iter = (0..index.size()).map(move |k| {
			StateMatrix::from_fn(|(x, y)| state_in[x + 5 * y][TRACKS_PER_BATCH * k + track])
		});
		Ok(iter)
	}

	fn read_state_outs<'a, P>(
		&self,
		index: &'a TableWitnessSegment<P>,
		track: usize,
	) -> Result<impl Iterator<Item = StateMatrix<u64>> + 'a>
	where
		P: PackedFieldIndexable<Scalar = B128> + PackedExtension<B1> + PackedExtension<B8>,
		PackedSubfield<P, B8>: PackedTransformationFactory<PackedSubfield<P, B8>>,
	{
		let state_out = self
			.state_out
			.as_inner()
			.try_map_ext(|col| index.get_mut_as(col))?;
		let iter = (0..index.size()).map(move |k| {
			StateMatrix::from_fn(|(x, y)| state_out[x + 5 * y][TRACKS_PER_BATCH * k + track])
		});
		Ok(iter)
	}
}

/// Returns an expression representing a sum over the given values.
fn sum_expr<F, const V: usize, const N: usize>(values: [Col<F, V>; N]) -> Expr<F, V>
where
	F: TowerField,
{
	assert!(!values.is_empty());
	let mut expr: Expr<F, V> = values[0].into();
	for value in &values[1..] {
		expr = expr + *value;
	}
	expr
}

/// Returns the RC for every round/track in the given batch.
///
/// The return type is basically 8 tracks of 64-bit round constants represented as bit patterns.
fn round_consts_for_batch(batch_no: usize) -> [B1; 64 * 8] {
	assert!(batch_no < BATCHES_PER_PERMUTATION);
	let mut batch_rc = [B1::from(0); 64 * 8];
	for track in 0..TRACKS_PER_BATCH {
		let rc = RC[nth_round_per_batch(batch_no, track)];
		for bit in 0..64 {
			let bit_value = ((rc >> bit) & 1) as u8;
			batch_rc[track * 64 + bit] = B1::from(bit_value);
		}
	}
	batch_rc
}

/// Calculates the round number that is performed at the `nth` track of the `batch_no` batch.
fn nth_round_per_batch(batch_no: usize, nth: usize) -> usize {
	assert!(batch_no < BATCHES_PER_PERMUTATION);
	assert!(nth < TRACKS_PER_BATCH);
	nth * BATCHES_PER_PERMUTATION + batch_no
}

/// Rotation offsets, laid out as [x][y].
const RHO: [[u32; 5]; 5] = [
	[0, 36, 3, 41, 18],
	[1, 44, 10, 45, 2],
	[62, 6, 43, 15, 61],
	[28, 55, 25, 21, 56],
	[27, 20, 39, 8, 14],
];

/// RC\[i\] is a round constant used in the ⍳ step at the ith round.
const RC: [u64; ROUNDS_PER_PERMUTATION] = [
	0x0000000000000001,
	0x0000000000008082,
	0x800000000000808A,
	0x8000000080008000,
	0x000000000000808B,
	0x0000000080000001,
	0x8000000080008081,
	0x8000000000008009,
	0x000000000000008A,
	0x0000000000000088,
	0x0000000080008009,
	0x000000008000000A,
	0x000000008000808B,
	0x800000000000008B,
	0x8000000000008089,
	0x8000000000008003,
	0x8000000000008002,
	0x8000000000000080,
	0x000000000000800A,
	0x800000008000000A,
	0x8000000080008081,
	0x8000000000008080,
	0x0000000080000001,
	0x8000000080008008,
];

#[cfg(test)]
mod tests {
	use binius_field::{arch::OptimalUnderlier128b, as_packed_field::PackedType};
	use bumpalo::Bump;

	use super::*;
	use crate::{
		builder::{ConstraintSystem, Statement, WitnessIndex},
		gadgets::hash::keccak::test_vector::TEST_VECTOR,
	};

	#[test]
	fn test_round_gadget() {
		const N_ROWS: usize = 1;

		let mut cs = ConstraintSystem::new();
		let mut table = cs.add_table("test");

		let state_in = StateMatrix::from_fn(|(x, y)| table.add_committed(format!("in[{x},{y}]")));
		let rb = RoundBatch::new(&mut table, state_in, 0);

		let allocator = Bump::new();
		let table_id = table.id();

		let statement = Statement {
			boundaries: vec![],
			table_sizes: vec![N_ROWS],
		};
		let mut witness =
			WitnessIndex::<PackedType<OptimalUnderlier128b, B128>>::new(&cs, &allocator);
		let table_witness = witness.init_table(table_id, N_ROWS).unwrap();
		let mut segment = table_witness.full_segment();

		let trace = trace::keccakf_trace(StateMatrix::default());
		rb.populate(&mut segment, &[trace]).unwrap();

		let ccs = cs.compile(&statement).unwrap();
		let witness = witness.into_multilinear_extension_index();

		binius_core::constraint_system::validate::validate_witness(&ccs, &[], &witness).unwrap();
	}

	#[test]
	fn test_permutation() {
		const N_ROWS: usize = TEST_VECTOR.len();

		let mut cs = ConstraintSystem::new();
		let mut table = cs.add_table("test");

		let keccakf = Keccakf::new(&mut table);

		let allocator = Bump::new();
		let table_id = table.id();

		let statement = Statement {
			boundaries: vec![],
			table_sizes: vec![N_ROWS],
		};
		let mut witness =
			WitnessIndex::<PackedType<OptimalUnderlier128b, B128>>::new(&cs, &allocator);
		let table_witness = witness.init_table(table_id, N_ROWS).unwrap();
		let mut segment = table_witness.full_segment();

		let state_ins = TEST_VECTOR
			.iter()
			.map(|&[state_in, _]| StateMatrix::from_values(state_in))
			.collect::<Vec<_>>();

		keccakf.populate_state_in(&mut segment, &state_ins).unwrap();
		keccakf.populate(&mut segment).unwrap();
		let state_outs = keccakf
			.read_state_outs(&segment)
			.unwrap()
			.collect::<Vec<_>>();
		for (i, actual_out) in state_outs.iter().enumerate() {
			let expected_out = StateMatrix::from_values(TEST_VECTOR[i][1]);
			if *actual_out != expected_out {
				panic!("Mismatch at index {i}: expected {expected_out:#?}, got {actual_out:#?}",);
			}
		}

		let ccs = cs.compile(&statement).unwrap();
		let witness = witness.into_multilinear_extension_index();

		binius_core::constraint_system::validate::validate_witness(&ccs, &[], &witness).unwrap();
	}
}
