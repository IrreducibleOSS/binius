// Copyright 2025 Irreducible Inc.

use std::{array, cell::RefMut};

use anyhow::Result;
use binius_field::{
	PackedBinaryField64x1b, PackedExtension, PackedField, PackedFieldIndexable, PackedSubfield,
	TowerField, linear_transformation::PackedTransformationFactory, packed::set_packed_slice,
};

use super::{
	state::{StateMatrix, StateRow},
	trace::{PermutationTrace, RoundTrace},
};
use crate::builder::{B1, B8, B128, Col, Expr, TableBuilder, TableWitnessSegment};

// TODO: move this into the state
pub struct Lane {
	bits: [Col<B1>; 64],
}

impl Lane {
	pub fn new(table: &mut TableBuilder) -> Self {
		Self {
			bits: array::from_fn(|i| table.add_committed(format!("bit[{i}]"))),
		}
	}

	pub fn rot(&self, c: usize) -> Self {
		Self {
			bits: array::from_fn(|i| {
				let j = (i + c) % 64;
				self.bits[j]
			}),
		}
	}
}

fn state_matrix(table: &mut TableBuilder, name: &str) -> StateMatrix<Lane> {
	StateMatrix::from_fn(|(x, y)| {
		let mut table = table.with_namespace(format!("{name}[{x},{y}]"));
		Lane::new(&mut table)
	})
}

pub struct WideRound {
	state_in: StateMatrix<Lane>,
	state_out: StateMatrix<Lane>,
	theta: StateMatrix<Lane>,
	chi: StateMatrix<Lane>,
	rc: Lane,
}

impl WideRound {
	pub fn new(table: &mut TableBuilder) -> Self {
		let state_in = state_matrix(table, "state_in");
		let state_out = state_matrix(table, "state_out");
		let theta = state_matrix(table, "theta");
		let chi = state_matrix(table, "chi");
		let rc = Lane::new(&mut table.with_namespace("rc"));

		// theta
		for x in 0..5 {
			let sum_left: [Expr<B1, 1>; 64] = array::from_fn(|z| {
				sum_expr(&[
					state_in[(x + 4, 0)].bits[z],
					state_in[(x + 4, 1)].bits[z],
					state_in[(x + 4, 2)].bits[z],
					state_in[(x + 4, 3)].bits[z],
					state_in[(x + 4, 4)].bits[z],
				])
			});
			let sum_right: [Expr<B1, 1>; 64] = array::from_fn(|z| {
				sum_expr(&[
					state_in[(x + 1, 0)].bits[(z + 63) % 64],
					state_in[(x + 1, 1)].bits[(z + 63) % 64],
					state_in[(x + 1, 2)].bits[(z + 63) % 64],
					state_in[(x + 1, 3)].bits[(z + 63) % 64],
					state_in[(x + 1, 4)].bits[(z + 63) % 64],
				])
			});
			for y in 0..5 {
				for z in 0..64 {
					table.assert_zero(
						format!("theta[{x}][{y}][{z}]"),
						theta[(x, y)].bits[z]
							+ state_in[(x, y)].bits[z]
							+ sum_left[z].clone() + sum_right[z].clone(),
					);
				}
			}
		}

		// chi

		Self {
			state_in,
			state_out,
			theta,
			chi,
			rc,
		}
	}

	pub fn populate<'a, P>(
		&self,
		index: &mut TableWitnessSegment<P>,
		round_traces: impl Iterator<Item = &'a RoundTrace>,
	) -> Result<()>
	where
		P: PackedFieldIndexable<Scalar = B128> + PackedExtension<B1> + PackedExtension<B8>,
		PackedSubfield<P, B8>: PackedTransformationFactory<PackedSubfield<P, B8>>,
	{
		for (i, round_trace) in round_traces.enumerate() {
			for z in 0..64 {
				for x in 0..5 {
					for y in 0..5 {
						let mut state_in: RefMut<'_, [PackedBinaryField64x1b]> =
							index.get_mut_as(self.state_in[(x, y)].bits[z])?;
						let mut theta: RefMut<'_, [PackedBinaryField64x1b]> =
							index.get_mut_as(self.theta[(x, y)].bits[z])?;

						let state_in_bit = (round_trace.state_in[(x, y)] & (1 << z)) != 0;
						let theta_bit = (round_trace.a_theta[(x, y)] & (1 << z)) != 0;
						set_packed_slice(&mut *state_in, i, state_in_bit.into());
						set_packed_slice(&mut *theta, i, theta_bit.into());
					}
				}
			}
		}
		Ok(())
	}
}

/// Returns an expression representing a sum over the given values.
fn sum_expr<F, const V: usize>(values: &[Col<F, V>]) -> Expr<F, V>
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

#[cfg(test)]
mod tests {
	use binius_field::{arch::OptimalUnderlier128b, as_packed_field::PackedType};
	use bumpalo::Bump;

	use super::*;
	use crate::{
		builder::{ConstraintSystem, Statement, WitnessIndex},
		gadgets::hash::keccak::{test_vector::TEST_VECTOR, trace},
	};

	#[test]
	fn test_round_gadget() {
		// TODO: 2
		const N_ROWS: usize = 24;

		let mut cs = ConstraintSystem::new();
		let mut table = cs.add_table("test");

		let round_gadget = WideRound::new(&mut table);

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
		round_gadget
			.populate(&mut segment, trace.as_ref()[0..N_ROWS].iter())
			.unwrap();

		let ccs = cs.compile(&statement).unwrap();
		let witness = witness.into_multilinear_extension_index();
		binius_core::constraint_system::validate::validate_witness(&ccs, &[], &witness).unwrap();
	}
}
