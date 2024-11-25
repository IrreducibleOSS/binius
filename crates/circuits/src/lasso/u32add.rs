// Copyright 2024 Irreducible Inc.

use super::lasso::lasso;
use crate::builder::ConstraintSystemBuilder;
use anyhow::Result;
use binius_core::oracle::{OracleId, ShiftVariant};
use binius_field::{
	as_packed_field::{PackScalar, PackedType},
	packed::set_packed_slice,
	underlier::{UnderlierType, U1},
	BinaryField1b, BinaryField32b, BinaryField8b, ExtensionField, PackedFieldIndexable, TowerField,
};
use bytemuck::Pod;
use itertools::izip;

const ADD_T_LOG_SIZE: usize = 17;

type B1 = BinaryField1b;
type B8 = BinaryField8b;
type B32 = BinaryField32b;

pub fn u32add<U, F, FBase>(
	builder: &mut ConstraintSystemBuilder<U, F, FBase>,
	name: impl ToString + Clone,
	xin_u8: OracleId,
	yin_u8: OracleId,
) -> Result<OracleId, anyhow::Error>
where
	U: UnderlierType
		+ Pod
		+ PackScalar<F>
		+ PackScalar<FBase>
		+ PackScalar<B32>
		+ PackScalar<BinaryField8b>
		+ PackScalar<BinaryField1b>,
	PackedType<U, B32>: PackedFieldIndexable,
	PackedType<U, B8>: PackedFieldIndexable,
	PackedType<U, B32>: PackedFieldIndexable,
	F: TowerField + ExtensionField<FBase> + ExtensionField<B32> + ExtensionField<BinaryField8b>,
	FBase: TowerField,
	B32: TowerField,
{
	builder.push_namespace(name.clone());

	// We use plus 2 because Lasso works with B8 instead of B32.
	let log_rows = builder.log_rows([xin_u8, yin_u8])?;

	let sum = builder.add_committed("sum", log_rows, B8::TOWER_LEVEL);

	let cout = builder.add_committed("cout", log_rows, B1::TOWER_LEVEL);

	let cin = builder.add_shifted("cin", cout, 1, 2, ShiftVariant::LogicalLeft)?;

	let lookup_t = builder.add_committed("lookup_t", ADD_T_LOG_SIZE, B32::TOWER_LEVEL);

	let lookup_u = builder.add_linear_combination(
		"lookup_u",
		log_rows,
		[
			(cin, <F as TowerField>::basis(0, 25)?),
			(cout, <F as TowerField>::basis(0, 24)?),
			(xin_u8, <F as TowerField>::basis(3, 2)?),
			(yin_u8, <F as TowerField>::basis(3, 1)?),
			(sum, <F as TowerField>::basis(3, 0)?),
		],
	)?;

	let channel = builder.add_channel();

	let mut u_to_t_mapping = None;

	if let Some(witness) = builder.witness() {
		let mut sum_witness = witness.new_column::<B8>(sum);
		let mut cin_witness = witness.new_column::<B1>(cin);
		let mut cout_witness = witness.new_column::<B1>(cout);
		let mut lookup_u_witness = witness.new_column::<B32>(lookup_u);
		let mut lookup_t_witness = witness.new_column::<B32>(lookup_t);

		let mut u_to_t_mapping_witness = vec![0; 1 << log_rows];

		let x_ints = witness.get::<B8>(xin_u8)?.as_slice::<u8>();
		let y_ints = witness.get::<B8>(yin_u8)?.as_slice::<u8>();

		let sum_scalars = PackedType::<U, B8>::unpack_scalars_mut(sum_witness.packed());
		let packed_slice_cin = cin_witness.packed();
		let packed_slice_cout = cout_witness.packed();
		let lookup_u_scalars = PackedType::<U, B32>::unpack_scalars_mut(lookup_u_witness.packed());
		let lookup_t_scalars = PackedType::<U, B32>::unpack_scalars_mut(lookup_t_witness.packed());

		let mut temp_cout = 0;

		for (i, (x, y, sum, lookup_u, u_to_t)) in izip!(
			x_ints,
			y_ints,
			sum_scalars.iter_mut(),
			lookup_u_scalars.iter_mut(),
			u_to_t_mapping_witness.iter_mut()
		)
		.enumerate()
		{
			let x = *x as usize;
			let y = *y as usize;

			let cin = if i % 4 == 0 { 0 } else { temp_cout };

			let xy_sum = x + y + cin;

			temp_cout = xy_sum >> 8;

			set_packed_slice(packed_slice_cin, i, BinaryField1b::new(U1::new(cin as u8)));
			set_packed_slice(packed_slice_cout, i, BinaryField1b::new(U1::new(temp_cout as u8)));

			*u_to_t = (x << 8 | y) << 1 | cin;

			let ab_sum = xy_sum & 0xff;

			*sum = BinaryField8b::new(xy_sum as u8);

			let lookup_u_u32 =
				(((((((cin << 1 | temp_cout) << 8) | x) << 8) | y) << 8) | ab_sum) as u32;

			*lookup_u = BinaryField32b::new(lookup_u_u32);
		}

		for (i, lookup_t) in lookup_t_scalars.iter_mut().enumerate() {
			let x = (i >> 9) & 0xff;
			let y = (i >> 1) & 0xff;
			let cin = i & 1;
			let ab_sum = x + y + cin;
			let cout = ab_sum >> 8;
			let ab_sum = ab_sum & 0xff;

			let lookup_t_u32 = (((((((cin << 1 | cout) << 8) | x) << 8) | y) << 8) | ab_sum) as u32;

			*lookup_t = BinaryField32b::new(lookup_t_u32);
		}
		u_to_t_mapping = Some(u_to_t_mapping_witness);
	}

	lasso::<_, _, _, B32, B32>(
		builder,
		format!("{} lasso", name.to_string()),
		u_to_t_mapping,
		lookup_u,
		lookup_t,
		channel,
	)?;

	builder.pop_namespace();
	Ok(sum)
}
