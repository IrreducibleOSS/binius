// Copyright 2025 Irreducible Inc.

//! Utilities for testing M3 constraint systems and gadgets.

use anyhow::Result;
use binius_core::{constraint_system::channel::Boundary, fiat_shamir::HasherChallenger};
use binius_field::{
	BinaryField128bPolyval, PackedField, PackedFieldIndexable, TowerField,
	as_packed_field::{PackScalar, PackedType},
	linear_transformation::PackedTransformationFactory,
	tower::CanonicalTowerFamily,
	underlier::UnderlierType,
};
use binius_hash::groestl::{Groestl256, Groestl256ByteCompression};
use binius_utils::env::boolean_env_flag_set;

use super::{
	B1, B8, B16, B32, B64,
	constraint_system::ConstraintSystem,
	table::TableId,
	witness::{TableFiller, TableWitnessSegment},
};
use crate::builder::{B128, Statement, WitnessIndex};

/// An easy-to-use implementation of [`TableFiller`] that is constructed with a closure.
///
/// Using this [`TableFiller`] implementation carries some overhead, so it is best to use it only
/// for testing.
#[allow(clippy::type_complexity)]
pub struct ClosureFiller<'a, P, Event>
where
	P: PackedField,
	P::Scalar: TowerField,
{
	table_id: TableId,
	fill: Box<
		dyn for<'b> Fn(&'b [&'b Event], &'b mut TableWitnessSegment<P>) -> Result<()> + Sync + 'a,
	>,
}

impl<'a, P: PackedField<Scalar: TowerField>, Event> ClosureFiller<'a, P, Event> {
	pub fn new(
		table_id: TableId,
		fill: impl for<'b> Fn(&'b [&'b Event], &'b mut TableWitnessSegment<P>) -> Result<()> + Sync + 'a,
	) -> Self {
		Self {
			table_id,
			fill: Box::new(fill),
		}
	}
}

impl<P: PackedField<Scalar: TowerField>, Event: Clone> TableFiller<P>
	for ClosureFiller<'_, P, Event>
{
	type Event = Event;

	fn id(&self) -> TableId {
		self.table_id
	}

	fn fill<'b>(
		&'b self,
		rows: impl Iterator<Item = &'b Self::Event> + Clone,
		witness: &'b mut TableWitnessSegment<P>,
	) -> Result<()> {
		(*self.fill)(&rows.collect::<Vec<_>>(), witness)
	}
}

/// Utility for M3 tests to validate a constraint system and witness.
pub fn validate_system_witness<U>(
	cs: &ConstraintSystem<B128>,
	witness: WitnessIndex<PackedType<U, B128>>,
	boundaries: Vec<Boundary<B128>>,
) where
	U: UnderlierType
		+ PackScalar<B1>
		+ PackScalar<B8>
		+ PackScalar<B16>
		+ PackScalar<B32>
		+ PackScalar<B64>
		+ PackScalar<B128>
		+ PackScalar<BinaryField128bPolyval>,
	PackedType<U, B128>:
		PackedFieldIndexable + PackedTransformationFactory<PackedType<U, BinaryField128bPolyval>>,
	PackedType<U, BinaryField128bPolyval>: PackedTransformationFactory<PackedType<U, B128>>,
{
	const TEST_PROVE_VERIFY_ENV_NAME: &str = "BINIUS_M3_TEST_PROVE_VERIFY";
	validate_system_witness_with_prove_verify::<U>(
		cs,
		witness,
		boundaries,
		boolean_env_flag_set(TEST_PROVE_VERIFY_ENV_NAME),
	)
}

pub fn validate_system_witness_with_prove_verify<U>(
	cs: &ConstraintSystem<B128>,
	witness: WitnessIndex<PackedType<U, B128>>,
	boundaries: Vec<Boundary<B128>>,
	prove_verify: bool,
) where
	U: UnderlierType
		+ PackScalar<B1>
		+ PackScalar<B8>
		+ PackScalar<B16>
		+ PackScalar<B32>
		+ PackScalar<B64>
		+ PackScalar<B128>
		+ PackScalar<BinaryField128bPolyval>,
	PackedType<U, B128>:
		PackedFieldIndexable + PackedTransformationFactory<PackedType<U, BinaryField128bPolyval>>,
	PackedType<U, BinaryField128bPolyval>: PackedTransformationFactory<PackedType<U, B128>>,
{
	let statement = Statement {
		boundaries,
		table_sizes: witness.table_sizes(),
	};
	let ccs = cs.compile(&statement).unwrap();
	let witness = witness.into_multilinear_extension_index();

	binius_core::constraint_system::validate::validate_witness(
		&ccs,
		&statement.boundaries,
		&witness,
	)
	.unwrap();

	if prove_verify {
		const LOG_INV_RATE: usize = 1;
		const SECURITY_BITS: usize = 100;

		let proof = binius_core::constraint_system::prove::<
			U,
			CanonicalTowerFamily,
			Groestl256,
			Groestl256ByteCompression,
			HasherChallenger<Groestl256>,
			_,
		>(
			&ccs,
			LOG_INV_RATE,
			SECURITY_BITS,
			&statement.boundaries,
			witness,
			&binius_hal::make_portable_backend(),
		)
		.unwrap();

		binius_core::constraint_system::verify::<
			U,
			CanonicalTowerFamily,
			Groestl256,
			Groestl256ByteCompression,
			HasherChallenger<Groestl256>,
		>(&ccs, LOG_INV_RATE, SECURITY_BITS, &statement.boundaries, proof)
		.unwrap();
	}
}
