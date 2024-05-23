// Copyright 2024 Ulvetanna Inc.

use anyhow::Result;
use binius_core::{
	oracle::{MultilinearOracleSet, ShiftVariant},
	polynomial::{
		composition::{empty_mix_composition, index_composition},
		transparent::step_down::StepDown,
		CompositionPoly, MultilinearComposite,
	},
};
use binius_field::{
	BinaryField128b, BinaryField1b, Field, PackedBinaryField128x1b, PackedField, TowerField,
};
use binius_macros::composition_poly;
use binius_utils::rayon::adjust_thread_pool;
use bytemuck::{must_cast_slice_mut, Pod};
use rand::{thread_rng, Rng};
use std::sync::Arc;

// This doesn't actually create any proof, it only generates the trace and validates it against the relevant constraints.

fn main() {
	adjust_thread_pool()
		.as_ref()
		.expect("failed to init thread pool");

	let log_size = 14;
	let oracle = U32FibOracle::new(&mut MultilinearOracleSet::<BinaryField128b>::new(), log_size);
	let witness = U32FibTrace::<PackedBinaryField128x1b>::new(log_size).fill_trace();
	let constraints = MultilinearComposite::from_columns(
		oracle
			.mixed_constraints(<BinaryField128b as PackedField>::random(&mut thread_rng()))
			.unwrap(),
		witness.all_columns(),
	)
	.unwrap();

	// This doesn't validate that c_in is a shifted version of c_out
	for i in 0..(1 << log_size) {
		assert_eq!(constraints.evaluate_on_hypercube(i).unwrap(), BinaryField128b::ZERO, "i={i}");
	}
}

struct U32FibTrace<P: PackedField<Scalar = BinaryField1b>> {
	x_in: Vec<P>,
	y_in: Vec<P>,
	z_out: Vec<P>,
	c_in: Vec<P>,
	c_out: Vec<P>,

	// We commit this to keep max degree 2 since we need to multiply with `enabled`
	carry_constraint: Vec<P>,

	// Shifting pads with zeros, which will mess with constraints
	// so we need a transparent StepDown to ignore the last rows.
	enabled: Vec<P>,
}

impl<P: PackedField<Scalar = BinaryField1b> + Pod> U32FibTrace<P> {
	fn new(log_size: usize) -> Self {
		Self {
			x_in: vec![P::default(); 1 << (log_size - P::LOG_WIDTH)],
			y_in: vec![P::default(); 1 << (log_size - P::LOG_WIDTH)],
			z_out: vec![P::default(); 1 << (log_size - P::LOG_WIDTH)],
			c_out: vec![P::default(); 1 << (log_size - P::LOG_WIDTH)],
			c_in: vec![P::default(); 1 << (log_size - P::LOG_WIDTH)],

			carry_constraint: vec![P::default(); 1 << (log_size - P::LOG_WIDTH)],
			enabled: vec![P::default(); 1 << (log_size - P::LOG_WIDTH)],
		}
	}

	fn fill_trace(mut self) -> Self {
		let x = must_cast_slice_mut::<_, u32>(&mut self.x_in);
		let y = must_cast_slice_mut::<_, u32>(&mut self.y_in);
		let z = must_cast_slice_mut::<_, u32>(&mut self.z_out);
		let cin = must_cast_slice_mut::<_, u32>(&mut self.c_in);
		let cout = must_cast_slice_mut::<_, u32>(&mut self.c_out);
		let carry_constraint = must_cast_slice_mut::<_, u32>(&mut self.carry_constraint);
		let enabled = must_cast_slice_mut::<_, u32>(&mut self.enabled);
		x[0] = 0;
		y[0] = 1;
		for i in 0..x.len() - 2 {
			let carry;
			(z[i], carry) = (x[i]).overflowing_add(y[i]);
			cin[i] = (x[i]) ^ (y[i]) ^ (z[i]);
			cout[i] = cin[i] >> 1;
			if carry {
				cout[i] |= 1 << 31;
			}
			carry_constraint[i] = ((x[i] ^ cin[i]) & (y[i] ^ cin[i])) ^ cin[i] ^ cout[i];
			enabled[i] = u32::MAX;
			x[i + 1] = y[i];
			y[i + 1] = z[i];
		}

		// We set the last two rows to random data just to prove the relevant constraints are turned off here
		let mut rng = thread_rng();
		for i in x.len() - 2..x.len() {
			x[i] = rng.gen();
			y[i] = rng.gen();
			z[i] = rng.gen();
			cin[i] = rng.gen();
			cout[i] = rng.gen();

			// Carry constraint/enabled are the only things we actually need to set correctly here
			carry_constraint[i] = ((x[i] ^ cin[i]) & (y[i] ^ cin[i])) ^ cin[i] ^ cout[i];
			enabled[i] = 0;
		}

		self
	}

	fn all_columns(&self) -> impl IntoIterator<Item = &Vec<P>> {
		[
			&self.x_in,
			&self.y_in,
			&self.z_out,
			&self.c_out,
			&self.c_in,
			&self.carry_constraint,
			&self.enabled,
		]
	}
}

struct U32FibOracle {
	x_in: usize,
	y_in: usize,
	z_out: usize,
	c_out: usize,
	c_in: usize,
	carry_constraint: usize,
	enabled: usize,
}

impl U32FibOracle {
	pub fn new<F: TowerField>(oracles: &mut MultilinearOracleSet<F>, n_vars: usize) -> Self {
		let mut batch_scope = oracles.build_committed_batch(n_vars, BinaryField1b::TOWER_LEVEL);
		let z_out = batch_scope.add_one();
		let c_out = batch_scope.add_one();
		let carry_constraint = batch_scope.add_one();
		let _batch_id = batch_scope.build();

		let x_in = oracles
			.add_shifted(z_out, 64, n_vars, ShiftVariant::LogicalLeft)
			.unwrap();
		let y_in = oracles
			.add_shifted(z_out, 32, n_vars, ShiftVariant::LogicalLeft)
			.unwrap();
		let c_in = oracles
			.add_shifted(c_out, 1, 5, ShiftVariant::LogicalLeft)
			.unwrap();
		let enabled = oracles
			.add_transparent(Arc::new(StepDown::new(n_vars, (1 << n_vars) - 2).unwrap()))
			.unwrap();
		Self {
			x_in,
			y_in,
			z_out,
			c_out,
			c_in,
			carry_constraint,
			enabled,
		}
	}

	pub fn mixed_constraints<F: TowerField>(
		&self,
		challenge: F,
	) -> Result<impl CompositionPoly<F> + Clone> {
		let all_columns = &[
			self.x_in,
			self.y_in,
			self.z_out,
			self.c_out,
			self.c_in,
			self.carry_constraint,
			self.enabled,
		];
		let mix = empty_mix_composition(all_columns.len(), challenge);
		let mix = mix.include([index_composition(
			all_columns,
			[self.x_in, self.y_in, self.c_in, self.z_out, self.enabled],
			composition_poly!([x, y, cin, z, enabled] = enabled * (x + y + cin - z)),
		)?])?;
		let mix = mix.include([index_composition(
			all_columns,
			[
				self.x_in,
				self.y_in,
				self.c_in,
				self.c_out,
				self.carry_constraint,
			],
			composition_poly!(
				[x, y, cin, cout, carry_constraint] =
					((x + cin) * (y + cin) + cin - cout) - carry_constraint
			),
		)?])?;
		let mix = mix.include([index_composition(
			all_columns,
			[self.carry_constraint, self.enabled],
			composition_poly!([carry_constraint, enabled] = enabled * carry_constraint),
		)?])?;
		Ok(mix)
	}
}
