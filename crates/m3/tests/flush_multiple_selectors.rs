// Copyright 2025 Irreducible Inc.

use binius_compute::cpu::alloc::CpuComputeAllocator;
use binius_field::{
	Field, arch::OptimalUnderlier128b, as_packed_field::PackedType, packed::set_packed_slice,
};
use binius_m3::{
	builder::{
		B1, B32, B128, Boundary, ConstraintSystem, FlushDirection, FlushOpts, StructuredDynSize,
		WitnessIndex,
		test_utils::{ClosureFiller, validate_system_witness},
	},
	gadgets::structured::fill_incrementing_b32,
};

#[test]
pub fn test_flush_multiple_selectors() {
	let mut allocator = CpuComputeAllocator::new(1 << 12);
	let allocator = allocator.into_bump_allocator();
	let mut cs = ConstraintSystem::<B128>::new();

	let channel = cs.add_channel("channel");

	let mut table = cs.add_table("multiple_selectors");

	let table_id = table.id();

	table.require_power_of_two_size();

	let structured_col = table.add_structured::<B32>(
		"incrementing",
		StructuredDynSize::Incrementing { max_size_log: 32 },
	);

	let selector1_col = table.add_committed::<B1, 1>("selector1");

	let selector2_col = table.add_committed::<B1, 1>("selector2");

	table.push_with_opts(
		channel,
		[structured_col],
		FlushOpts {
			multiplicity: 1,
			selectors: vec![selector1_col, selector2_col],
		},
	);

	let mut witness = WitnessIndex::<PackedType<OptimalUnderlier128b, B128>>::new(&cs, &allocator);

	witness
		.fill_table_sequential(
			&ClosureFiller::new(table_id, |events, index| {
				{
					let mut selector1_col = index.get_mut(selector1_col)?;
					let mut selector2_col = index.get_mut(selector2_col)?;
					for &i in events {
						set_packed_slice(
							&mut selector1_col,
							i,
							if i > 10 { B1::ZERO } else { B1::ONE },
						);
						set_packed_slice(
							&mut selector2_col,
							i,
							if i >= 8 { B1::ONE } else { B1::ZERO },
						);
					}
				}

				fill_incrementing_b32(index, structured_col)?;
				Ok(())
			}),
			&(0..1 << 10).collect::<Vec<_>>(),
		)
		.unwrap();

	let boundaries = (8..=10)
		.map(|i| Boundary {
			values: vec![B128::new(i)],
			channel_id: channel,
			direction: FlushDirection::Pull,
			multiplicity: 1,
		})
		.collect::<Vec<_>>();

	validate_system_witness::<OptimalUnderlier128b>(&cs, witness, boundaries);
}
