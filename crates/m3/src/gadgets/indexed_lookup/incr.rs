use std::{iter, slice};

use binius_core::constraint_system::channel::ChannelId;
use binius_field::{
	PackedExtension, PackedFieldIndexable, ext_basis,
	packed::{get_packed_slice, set_packed_slice},
};
use binius_math::{ArithCircuit, ArithExpr};

use crate::{
	builder::{
		B1, B8, B32, B128, Col, IndexedLookup, TableBuilder, TableFiller, TableId,
		TableWitnessSegment, upcast_col,
	},
	gadgets::lookup::LookupProducer,
};

struct Incr {
	pub input: Col<B8>,
	pub carry_in: Col<B1>,
	pub output: Col<B8>,
	pub carry_out: Col<B1>,
	pub merged: Col<B32>,
}

impl Incr {
	fn new(
		table: &mut TableBuilder,
		lookup_chan: ChannelId,
		input: Col<B8>,
		carry_in: Col<B1>,
	) -> Self {
		let output = table.add_committed::<B8, 1>("output");
		let carry_out = table.add_committed::<B1, 1>("carry_out");
		let merged = merge_incr_cols(table, input, carry_in, output, carry_out);

		table.pull(lookup_chan, [merged]);

		Self {
			input,
			carry_in,
			output,
			carry_out,
			merged,
		}
	}

	fn populate<P>(&self, witness: &mut TableWitnessSegment<P>) -> anyhow::Result<()>
	where
		P: PackedFieldIndexable<Scalar = B128>
			+ PackedExtension<B1>
			+ PackedExtension<B8>
			+ PackedExtension<B32>,
	{
		let input = witness.get_as::<u8, _, 1>(self.input)?;
		let carry_in = witness.get(self.carry_in)?;
		let mut output = witness.get_mut_as::<u8, _, 1>(self.output)?;
		let mut carry_out = witness.get_mut(self.carry_out)?;
		let mut merged = witness.get_mut_as::<u32, _, 1>(self.merged)?;

		for i in 0..witness.size() {
			let input_i = input[i];
			let carry_in_bit = bool::from(get_packed_slice(&carry_in, i).val());

			let (output_i, carry_out_bit) = input_i.overflowing_add(carry_in_bit.into());
			output[i] = output_i;
			set_packed_slice(&mut carry_out, i, B1::from(carry_out_bit));
			merged[i] = ((carry_out_bit as u32) << 17)
				| ((carry_in_bit as u32) << 16)
				| ((output_i as u32) << 8)
				| input_i as u32;
		}

		Ok(())
	}
}

struct IncrLooker {
	pub input: Col<B8>,
	pub carry_in: Col<B1>,
	incr: Incr,
}

impl IncrLooker {
	fn new(table: &mut TableBuilder, lookup_chan: ChannelId) -> Self {
		let input = table.add_committed::<B8, 1>("input");
		let carry_in = table.add_committed::<B1, 1>("carry_in");
		let incr = Incr::new(table, lookup_chan, input, carry_in);

		Self {
			input,
			carry_in,
			incr,
		}
	}

	fn populate<'a, P>(
		&self,
		witness: &mut TableWitnessSegment<P>,
		events: impl Iterator<Item = &'a (u8, bool)>,
	) -> anyhow::Result<()>
	where
		P: PackedFieldIndexable<Scalar = B128>
			+ PackedExtension<B1>
			+ PackedExtension<B8>
			+ PackedExtension<B32>,
	{
		{
			let mut input = witness.get_mut_as::<u8, _, 1>(self.input)?;
			let mut carry_in = witness.get_mut(self.carry_in)?;

			for (i, &(input_i, carry_in_bit)) in events.enumerate() {
				input[i] = input_i;
				set_packed_slice(&mut carry_in, i, B1::from(carry_in_bit));
			}
		}

		self.incr.populate(witness)?;
		Ok(())
	}
}

struct IncrLookup {
	table_id: TableId,
	entries_ordered: Col<B32>,
	entries_sorted: Col<B32>,
	lookup_producer: LookupProducer,
}

impl IncrLookup {
	fn new(
		table: &mut TableBuilder,
		chan: ChannelId,
		permutation_chan: ChannelId,
		n_multiplicity_bits: usize,
	) -> Self {
		table.require_fixed_size(IncrIndexedLookup.log_size());

		// The entries_ordered column is the one that is filled with the lookup table entries.
		let entries_ordered = table.add_fixed("incr_lookup", incr_circuit());
		let entries_sorted = table.add_committed::<B32, 1>("entries_sorted");

		// Use flush to check that entries_sorted is a permutation of entries_ordered.
		table.push(permutation_chan, [entries_ordered]);
		table.pull(permutation_chan, [entries_sorted]);

		let lookup_producer =
			LookupProducer::new(table, chan, &[entries_sorted], n_multiplicity_bits);
		Self {
			table_id: table.id(),
			entries_ordered,
			entries_sorted,
			lookup_producer,
		}
	}
}

// TODO: It seems very possible to make a generic table filler for indexed lookup tables.
impl TableFiller for IncrLookup {
	// Tuple of index and count
	type Event = (usize, u32);

	fn id(&self) -> TableId {
		self.table_id
	}

	fn fill<'a>(
		&'a self,
		rows: impl Iterator<Item = &'a Self::Event> + Clone,
		witness: &'a mut TableWitnessSegment,
	) -> anyhow::Result<()> {
		// Fill the entries_ordered column
		{
			let mut col_data = witness.get_scalars_mut(self.entries_ordered)?;
			let start_index = witness.index() << witness.log_size();
			for (i, col_data_i) in col_data.iter_mut().enumerate() {
				let mut entry_128b = B128::default();
				IncrIndexedLookup.index_to_entry(start_index + i, slice::from_mut(&mut entry_128b));
				*col_data_i = B32::try_from(entry_128b).expect("guaranteed by IncrIndexedLookup");
			}
		}

		// Fill the entries_sorted column
		{
			let mut entries_sorted = witness.get_scalars_mut(self.entries_sorted)?;
			for (merged_i, &(index, _)) in iter::zip(&mut *entries_sorted, rows.clone()) {
				let mut entry_128b = B128::default();
				IncrIndexedLookup.index_to_entry(index, slice::from_mut(&mut entry_128b));
				*merged_i = B32::try_from(entry_128b).expect("guaranteed by IncrIndexedLookup");
			}
		}

		self.lookup_producer
			.populate(witness, rows.map(|&(_i, count)| count))?;
		Ok(())
	}
}

struct IncrIndexedLookup;

impl IndexedLookup<B128> for IncrIndexedLookup {
	fn log_size(&self) -> usize {
		// Input is an 8-bit value plus 1-bit carry-in
		8 + 1
	}

	fn entry_to_index(&self, entry: &[B128]) -> usize {
		debug_assert_eq!(entry.len(), 1);
		let merged_val = entry[0].val() as u32;
		let input = merged_val & 0xFF;
		let carry_in_bit = (merged_val >> 16) & 1 == 1;
		(carry_in_bit as usize) << 8 | input as usize
	}

	fn index_to_entry(&self, index: usize, entry: &mut [B128]) {
		debug_assert_eq!(entry.len(), 1);
		let input = (index % (1 << 8)) as u8;
		let carry_in_bit = (index >> 8) & 1 == 1;
		let (output, carry_out_bit) = input.overflowing_add(carry_in_bit.into());
		let entry_u32 = merge_incr_vals(input, carry_in_bit, output, carry_out_bit);
		entry[0] = B32::new(entry_u32).into();
	}
}

/// Returns a circuit that describes the carry-in for the i_th bit of incrementing an 8-bit
/// number by a carry-in bit. We describe the following polynomials
pub fn carry_in_circuit(i: usize) -> ArithExpr<B128> {
	// The circuit is a lookup table for the increment operation, which takes an 8-bit input and
	// returns an 8-bit output and a carry bit. The circuit is defined as follows:
	let mut circuit = ArithExpr::Var(8);
	for var in 0..i {
		circuit *= ArithExpr::Var(var)
	}
	circuit
}

/// Returns a circuit that describes the increment operation for an 8-bit addition.
/// We describe the following polynomials
pub fn incr_circuit() -> ArithCircuit<B128> {
	// The circuit is a lookup table for the increment operation, which takes an 8-bit input and
	// returns an 8-bit output and a carry bit. The circuit is defined as follows:
	let mut circuit = ArithExpr::zero();
	for i in 0..8 {
		circuit += ArithExpr::Var(i) * ArithExpr::Const(B128::from(1 << i));

		let carry = carry_in_circuit(i);
		circuit += (ArithExpr::Var(i) + carry.clone()) * ArithExpr::Const(B128::from(1 << (i + 8)));
	}
	circuit += ArithExpr::Var(8) * ArithExpr::Const(B128::from(1 << 16));
	let carry_out = carry_in_circuit(8);
	circuit += carry_out * ArithExpr::Const(B128::from(1 << 17));
	circuit.into()
}

fn merge_incr_cols(
	table: &mut TableBuilder,
	input: Col<B8>,
	carry_in: Col<B1>,
	output: Col<B8>,
	carry_out: Col<B1>,
) -> Col<B32> {
	let beta_1 = ext_basis::<B32, B8>(1);
	let beta_2_0 = ext_basis::<B32, B8>(2);
	let beta_2_1 = beta_2_0 * ext_basis::<B8, B1>(1);
	table.add_computed(
		"merged",
		upcast_col(input)
			+ upcast_col(output) * beta_1
			+ upcast_col(carry_in) * beta_2_0
			+ upcast_col(carry_out) * beta_2_1,
	)
}

fn merge_incr_vals(input: u8, carry_in: bool, output: u8, carry_out: bool) -> u32 {
	((carry_out as u32) << 17) | ((carry_in as u32) << 16) | ((output as u32) << 8) | input as u32
}

#[cfg(test)]
mod tests {
	use std::{cmp::Reverse, iter::repeat_with};

	use binius_core::constraint_system::channel::{Boundary, FlushDirection};
	use binius_field::arch::OptimalUnderlier;
	use bumpalo::Bump;
	use itertools::Itertools;
	use rand::{Rng, SeedableRng, rngs::StdRng};

	use super::*;
	use crate::builder::{
		ConstraintSystem, WitnessIndex, tally,
		test_utils::{ClosureFiller, validate_system_witness},
	};

	/// Unit test for a fixed lookup table, which requires counting lookups during witness
	/// generation of the looker tables.
	#[test]
	fn test_fixed_lookup_producer() {
		let mut cs = ConstraintSystem::new();
		let incr_lookup_chan = cs.add_channel("incr lookup");
		let incr_lookup_perm_chan = cs.add_channel("incr lookup permutation");

		let n_multiplicity_bits = 8;

		let mut incr_table = cs.add_table("increment");
		let incr_lookup = IncrLookup::new(
			&mut incr_table,
			incr_lookup_chan,
			incr_lookup_perm_chan,
			n_multiplicity_bits,
		);

		let mut looker_1 = cs.add_table("looker 1");
		let looker_1_id = looker_1.id();
		let incr_1 = IncrLooker::new(&mut looker_1, incr_lookup_chan);

		let mut looker_2 = cs.add_table("looker 2");
		let looker_2_id = looker_2.id();
		let incr_2 = IncrLooker::new(&mut looker_2, incr_lookup_chan);

		let looker_1_size = 5;
		let looker_2_size = 6;

		let allocator = Bump::new();
		let mut witness = WitnessIndex::new(&cs, &allocator);

		let mut rng = StdRng::seed_from_u64(0);
		let inputs_1 = repeat_with(|| {
			let input = rng.r#gen::<u8>();
			let carry_in_bit = rng.gen_bool(0.5);
			(input, carry_in_bit)
		})
		.take(looker_1_size)
		.collect::<Vec<_>>();

		witness
			.fill_table_sequential(
				&ClosureFiller::new(looker_1_id, |inputs, segment| {
					incr_1.populate(segment, inputs.iter().copied())
				}),
				&inputs_1,
			)
			.unwrap();

		let inputs_2 = repeat_with(|| {
			let input = rng.r#gen::<u8>();
			let carry_in_bit = rng.gen_bool(0.5);
			(input, carry_in_bit)
		})
		.take(looker_2_size)
		.collect::<Vec<_>>();

		witness
			.fill_table_sequential(
				&ClosureFiller::new(looker_2_id, |inputs, segment| {
					incr_2.populate(segment, inputs.iter().copied())
				}),
				&inputs_2,
			)
			.unwrap();

		let boundary_reads = vec![
			merge_incr_vals(111, false, 111, false),
			merge_incr_vals(111, true, 112, false),
			merge_incr_vals(255, false, 255, false),
			merge_incr_vals(255, true, 0, true),
		];
		let boundaries = boundary_reads
			.into_iter()
			.map(|val| Boundary {
				values: vec![B32::new(val).into()],
				direction: FlushDirection::Pull,
				channel_id: incr_lookup_chan,
				multiplicity: 1,
			})
			.collect::<Vec<_>>();

		// Tally the lookup counts from the looker tables
		let counts =
			tally(&cs, &mut witness, &boundaries, incr_lookup_chan, &IncrIndexedLookup).unwrap();

		// Fill the lookup table with the sorted counts
		let sorted_counts = counts
			.into_iter()
			.enumerate()
			.sorted_by_key(|(_, count)| Reverse(*count))
			.collect::<Vec<_>>();

		witness
			.fill_table_sequential(&incr_lookup, &sorted_counts)
			.unwrap();

		validate_system_witness::<OptimalUnderlier>(&cs, witness, boundaries);
	}
}
