// Copyright 2024 Irreducible Inc.

use std::marker::PhantomData;

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

use super::lasso::lasso;
use crate::{builder::ConstraintSystemBuilder, pack::pack};

const ADD_T_LOG_SIZE: usize = 17;

type B1 = BinaryField1b;
type B8 = BinaryField8b;
type B32 = BinaryField32b;

pub fn u32add<U, F, FInput, FOutput>(
	builder: &mut ConstraintSystemBuilder<U, F>,
	name: impl ToString + Clone,
	xin: OracleId,
	yin: OracleId,
) -> Result<OracleId, anyhow::Error>
where
	U: UnderlierType
		+ Pod
		+ PackScalar<F>
		+ PackScalar<B32>
		+ PackScalar<BinaryField8b>
		+ PackScalar<BinaryField1b>
		+ PackScalar<FInput>
		+ PackScalar<FOutput>,
	PackedType<U, B32>: PackedFieldIndexable,
	PackedType<U, B8>: PackedFieldIndexable,
	PackedType<U, B32>: PackedFieldIndexable,
	B8: ExtensionField<FInput> + ExtensionField<FOutput>,
	F: TowerField
		+ ExtensionField<B32>
		+ ExtensionField<BinaryField8b>
		+ ExtensionField<FInput>
		+ ExtensionField<FOutput>,
	FInput: TowerField,
	FOutput: TowerField,
	B32: TowerField,
{
	let mut several = SeveralU32add::new(builder)?;
	let sum = several.u32add::<FInput, FOutput>(builder, name.clone(), xin, yin)?;
	several.finalize(builder, name)?;
	Ok(sum)
}

pub struct SeveralU32add<U, F> {
	n_lookups: Vec<usize>,
	lookup_t: OracleId,
	lookups_u: Vec<OracleId>,
	u_to_t_mappings: Vec<Vec<usize>>,
	finalized: bool,
	_phantom: PhantomData<(U, F)>,
}

impl<U, F> SeveralU32add<U, F>
where
	U: UnderlierType
		+ Pod
		+ PackScalar<F>
		+ PackScalar<B32>
		+ PackScalar<BinaryField8b>
		+ PackScalar<BinaryField1b>,
	PackedType<U, B32>: PackedFieldIndexable,
	PackedType<U, B8>: PackedFieldIndexable,
	F: TowerField + ExtensionField<B32> + ExtensionField<BinaryField8b>,
{
	pub fn new(builder: &mut ConstraintSystemBuilder<U, F>) -> Result<Self> {
		let lookup_t = builder.add_committed("lookup_t", ADD_T_LOG_SIZE, B32::TOWER_LEVEL);

		if let Some(witness) = builder.witness() {
			let mut lookup_t_witness = witness.new_column::<B32>(lookup_t);

			let lookup_t_scalars =
				PackedType::<U, B32>::unpack_scalars_mut(lookup_t_witness.packed());

			for (i, lookup_t) in lookup_t_scalars.iter_mut().enumerate() {
				let x = (i >> 9) & 0xff;
				let y = (i >> 1) & 0xff;
				let cin = i & 1;
				let ab_sum = x + y + cin;
				let cout = ab_sum >> 8;
				let ab_sum = ab_sum & 0xff;

				let lookup_t_u32 =
					(((((((cin << 1 | cout) << 8) | x) << 8) | y) << 8) | ab_sum) as u32;

				*lookup_t = BinaryField32b::new(lookup_t_u32);
			}
		}
		Ok(Self {
			n_lookups: Vec::new(),
			lookup_t,
			lookups_u: Vec::new(),
			u_to_t_mappings: Vec::new(),
			finalized: false,
			_phantom: PhantomData,
		})
	}

	pub fn u32add<FInput, FOutput>(
		&mut self,
		builder: &mut ConstraintSystemBuilder<U, F>,
		name: impl ToString,
		xin: OracleId,
		yin: OracleId,
	) -> Result<OracleId, anyhow::Error>
	where
		U: PackScalar<FInput> + PackScalar<FOutput>,
		FInput: TowerField,
		FOutput: TowerField,
		F: ExtensionField<FInput> + ExtensionField<FOutput>,
		B8: ExtensionField<FInput> + ExtensionField<FOutput>,
	{
		builder.push_namespace(name);

		let input_log_size = builder.log_rows([xin, yin])?;

		let b8_log_size = input_log_size - B8::TOWER_LEVEL + FInput::TOWER_LEVEL;

		let output_log_size = input_log_size - FOutput::TOWER_LEVEL + FInput::TOWER_LEVEL;

		let sum = builder.add_committed("sum", output_log_size, FOutput::TOWER_LEVEL);

		let sum_packed = if FInput::TOWER_LEVEL == B8::TOWER_LEVEL {
			sum
		} else {
			builder.add_packed("lasso sum packed", sum, B8::TOWER_LEVEL - FInput::TOWER_LEVEL)?
		};

		let cout = builder.add_committed("cout", b8_log_size, B1::TOWER_LEVEL);

		let cin = builder.add_shifted("cin", cout, 1, 2, ShiftVariant::LogicalLeft)?;

		let xin_u8 = pack::<_, _, FInput, B8>(xin, builder, "repacked xin")?;
		let yin_u8 = pack::<_, _, FInput, B8>(yin, builder, "repacked yin")?;

		let lookup_u = builder.add_linear_combination(
			"lookup_u",
			b8_log_size,
			[
				(cin, <F as TowerField>::basis(0, 25)?),
				(cout, <F as TowerField>::basis(0, 24)?),
				(xin_u8, <F as TowerField>::basis(3, 2)?),
				(yin_u8, <F as TowerField>::basis(3, 1)?),
				(sum_packed, <F as TowerField>::basis(3, 0)?),
			],
		)?;

		if let Some(witness) = builder.witness() {
			let mut sum_witness = witness.new_column::<FOutput>(sum);
			let mut cin_witness = witness.new_column::<B1>(cin);
			let mut cout_witness = witness.new_column::<B1>(cout);
			let mut lookup_u_witness = witness.new_column::<B32>(lookup_u);
			let mut u_to_t_mapping_witness = vec![0; 1 << (b8_log_size)];

			let x_ints = witness.get::<B8>(xin_u8)?.as_slice::<u8>();
			let y_ints = witness.get::<B8>(yin_u8)?.as_slice::<u8>();

			let sum_scalars = sum_witness.as_mut_slice::<u8>();
			let packed_slice_cin = cin_witness.packed();
			let packed_slice_cout = cout_witness.packed();
			let lookup_u_scalars =
				PackedType::<U, B32>::unpack_scalars_mut(lookup_u_witness.packed());

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
				set_packed_slice(
					packed_slice_cout,
					i,
					BinaryField1b::new(U1::new(temp_cout as u8)),
				);

				*u_to_t = (x << 8 | y) << 1 | cin;

				let ab_sum = xy_sum & 0xff;

				*sum = xy_sum as u8;

				let lookup_u_u32 =
					(((((((cin << 1 | temp_cout) << 8) | x) << 8) | y) << 8) | ab_sum) as u32;

				*lookup_u = B32::new(lookup_u_u32);
			}

			std::mem::drop(sum_witness);

			let sum_packed_witness = witness.get::<FOutput>(sum)?;

			witness.set::<B8>(sum_packed, sum_packed_witness.repacked::<B8>())?;

			self.u_to_t_mappings.push(u_to_t_mapping_witness)
		}

		self.lookups_u.push(lookup_u);
		self.n_lookups.push(1 << b8_log_size);

		builder.pop_namespace();
		Ok(sum)
	}

	pub fn finalize(
		mut self,
		builder: &mut ConstraintSystemBuilder<U, F>,
		name: impl ToString,
	) -> Result<()> {
		let channel = builder.add_channel();
		self.finalized = true;
		lasso::<_, _, B32, B32>(
			builder,
			name,
			&self.n_lookups,
			&self.u_to_t_mappings,
			&self.lookups_u,
			self.lookup_t,
			channel,
		)
	}
}

impl<U, F> Drop for SeveralU32add<U, F> {
	fn drop(&mut self) {
		assert!(self.finalized)
	}
}
