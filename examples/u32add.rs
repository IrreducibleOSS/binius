// Copyright 2024 Ulvetanna Inc.

use anyhow::Result;
use binius_core::{
	oracle::{MultilinearOracleSet, ShiftVariant},
	polynomial::{
		composition::{empty_mix_composition, index_composition},
		CompositionPoly, MultilinearComposite,
	},
};
use binius_field::{
	BinaryField128b, BinaryField1b, Field, PackedBinaryField128x1b, PackedField, TowerField,
};
use binius_macros::composition_poly;
use binius_utils::{examples::get_log_trace_size, rayon::adjust_thread_pool};
use bytemuck::{must_cast_slice_mut, Pod};
use rand::{thread_rng, Rng};
use rayon::prelude::*;

// This doesn't actually create any proof, it only generates the trace and validates it against the relevant constraints.

fn main() {
	adjust_thread_pool()
		.as_ref()
		.expect("failed to init thread pool");

	let log_size = get_log_trace_size().unwrap_or(11);
	let oracle = U32AddOracle::new(&mut MultilinearOracleSet::<BinaryField128b>::new(), log_size);
	let witness = U32AddTrace::<PackedBinaryField128x1b>::new(log_size).fill_trace();
	let constraints = MultilinearComposite::from_columns(
		oracle
			.mixed_constraints(<BinaryField128b as PackedField>::random(&mut thread_rng()))
			.unwrap(),
		witness.all_columns(),
	)
	.unwrap();

	// This doesn't validate that c_in is a shifted version of c_out
	for i in 0..(1 << log_size) {
		assert_eq!(constraints.evaluate_on_hypercube(i).unwrap(), BinaryField128b::ZERO);
	}
}

struct U32AddTrace<P: PackedField<Scalar = BinaryField1b>> {
	x_in: Vec<P>,
	y_in: Vec<P>,
	z_out: Vec<P>,
	c_out: Vec<P>,
	c_in: Vec<P>,
}

impl<P: PackedField<Scalar = BinaryField1b> + Pod> U32AddTrace<P> {
	fn new(log_size: usize) -> Self {
		Self {
			x_in: vec![P::default(); 1 << (log_size - P::LOG_WIDTH)],
			y_in: vec![P::default(); 1 << (log_size - P::LOG_WIDTH)],
			z_out: vec![P::default(); 1 << (log_size - P::LOG_WIDTH)],
			c_out: vec![P::default(); 1 << (log_size - P::LOG_WIDTH)],
			c_in: vec![P::default(); 1 << (log_size - P::LOG_WIDTH)],
		}
	}

	fn fill_trace(mut self) -> Self {
		(
			must_cast_slice_mut::<_, u32>(&mut self.x_in),
			must_cast_slice_mut::<_, u32>(&mut self.y_in),
			must_cast_slice_mut::<_, u32>(&mut self.z_out),
			must_cast_slice_mut::<_, u32>(&mut self.c_in),
			must_cast_slice_mut::<_, u32>(&mut self.c_out),
		)
			.into_par_iter()
			.for_each_init(thread_rng, |rng, (x, y, z, cin, cout)| {
				*x = rng.gen();
				*y = rng.gen();
				let carry;
				(*z, carry) = (*x).overflowing_add(*y);
				*cin = (*x) ^ (*y) ^ (*z);
				*cout = *cin >> 1;
				if carry {
					*cout |= 1 << 31;
				}
			});
		self
	}

	fn all_columns(&self) -> impl IntoIterator<Item = &Vec<P>> {
		[&self.x_in, &self.y_in, &self.z_out, &self.c_out, &self.c_in]
	}
}

struct U32AddOracle {
	x_in: usize,
	y_in: usize,
	z_out: usize,
	c_out: usize,
	c_in: usize,
}

impl U32AddOracle {
	pub fn new<F: TowerField>(oracles: &mut MultilinearOracleSet<F>, n_vars: usize) -> Self {
		let mut batch_scope = oracles.build_committed_batch(n_vars, BinaryField1b::TOWER_LEVEL);
		let x_in = batch_scope.add_one();
		let y_in = batch_scope.add_one();
		let z_out = batch_scope.add_one();
		let c_out = batch_scope.add_one();
		let _batch_id = batch_scope.build();

		let c_in = oracles
			.add_shifted(c_out, 1, 5, ShiftVariant::LogicalLeft)
			.unwrap();
		Self {
			x_in,
			y_in,
			z_out,
			c_out,
			c_in,
		}
	}

	pub fn mixed_constraints<F: TowerField>(
		&self,
		challenge: F,
	) -> Result<impl CompositionPoly<F> + Clone> {
		let all_columns = &[self.x_in, self.y_in, self.z_out, self.c_out, self.c_in];
		let mix = empty_mix_composition(all_columns.len(), challenge);
		let mix = mix.include([index_composition(
			all_columns,
			[self.x_in, self.y_in, self.c_in, self.z_out],
			composition_poly!([x, y, cin, z] = x + y + cin - z),
		)?])?;
		let mix = mix.include([index_composition(
			all_columns,
			[self.x_in, self.y_in, self.c_in, self.c_out],
			composition_poly!([x, y, cin, cout] = (x + cin) * (y + cin) + cin - cout),
		)?])?;
		Ok(mix)
	}
}
