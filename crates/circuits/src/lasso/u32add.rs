// Copyright 2024-2025 Irreducible Inc.

use std::marker::PhantomData;

use anyhow::Result;
use binius_core::oracle::{OracleId, ShiftVariant};
use binius_field::{
	as_packed_field::{PackScalar, PackedType},
	packed::set_packed_slice,
	underlier::U1,
	BinaryField1b, BinaryField32b, BinaryField8b, ExtensionField, PackedFieldIndexable, TowerField,
};
use itertools::izip;

use super::lasso::lasso;
use crate::{
	builder::{
		types::{F, U},
		ConstraintSystemBuilder,
	},
	pack::pack,
};

const ADD_T_LOG_SIZE: usize = 17;

type B1 = BinaryField1b;
type B8 = BinaryField8b;
type B32 = BinaryField32b;

pub fn u32add<FInput, FOutput>(
	builder: &mut ConstraintSystemBuilder,
	name: impl ToString + Clone,
	xin: OracleId,
	yin: OracleId,
) -> Result<OracleId, anyhow::Error>
where
	FInput: TowerField,
	FOutput: TowerField,
	U: PackScalar<FInput> + PackScalar<FOutput>,
	B8: ExtensionField<FInput> + ExtensionField<FOutput>,
	F: ExtensionField<FInput> + ExtensionField<FOutput>,
{
	let mut several = SeveralU32add::new(builder)?;
	let sum = several.u32add::<FInput, FOutput>(builder, name.clone(), xin, yin)?;
	several.finalize(builder, name)?;
	Ok(sum)
}

pub struct SeveralU32add {
	n_lookups: Vec<usize>,
	lookup_t: OracleId,
	lookups_u: Vec<[OracleId; 1]>,
	u_to_t_mappings: Vec<Vec<usize>>,
	finalized: bool,
	_phantom: PhantomData<(U, F)>,
}

impl SeveralU32add {
	pub fn new(builder: &mut ConstraintSystemBuilder) -> Result<Self> {
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
		builder: &mut ConstraintSystemBuilder,
		name: impl ToString,
		xin: OracleId,
		yin: OracleId,
	) -> Result<OracleId, anyhow::Error>
	where
		FInput: TowerField,
		FOutput: TowerField,
		U: PackScalar<FInput> + PackScalar<FOutput>,
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

		let xin_u8 = pack::<FInput, B8>(xin, builder, "repacked xin")?;
		let yin_u8 = pack::<FInput, B8>(yin, builder, "repacked yin")?;

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

		self.lookups_u.push([lookup_u]);
		self.n_lookups.push(1 << b8_log_size);

		builder.pop_namespace();
		Ok(sum)
	}

	pub fn finalize(
		mut self,
		builder: &mut ConstraintSystemBuilder,
		name: impl ToString,
	) -> Result<()> {
		let channel = builder.add_channel();
		self.finalized = true;
		lasso::<B32>(
			builder,
			name,
			&self.n_lookups,
			&self.u_to_t_mappings,
			&self.lookups_u,
			[self.lookup_t],
			channel,
		)
	}
}

impl Drop for SeveralU32add {
	fn drop(&mut self) {
		assert!(self.finalized)
	}
}

#[cfg(test)]
mod tests {
	use binius_field::{BinaryField1b, BinaryField8b};

	use super::SeveralU32add;
	use crate::{builder::test_utils::test_circuit, unconstrained::unconstrained};

	#[test]
	fn test_several_lasso_u32add() {
		test_circuit(|builder| {
			let mut several_u32_add = SeveralU32add::new(builder).unwrap();
			for log_size in [11, 12, 13] {
				// BinaryField8b is used here because we utilize an 8x8x1→8 table
				let add_a_u8 = unconstrained::<BinaryField8b>(builder, "add_a", log_size).unwrap();
				let add_b_u8 = unconstrained::<BinaryField8b>(builder, "add_b", log_size).unwrap();
				let _sum = several_u32_add
					.u32add::<BinaryField8b, BinaryField8b>(
						builder,
						"lasso_u32add",
						add_a_u8,
						add_b_u8,
					)
					.unwrap();
			}
			several_u32_add.finalize(builder, "lasso_u32add").unwrap();
			Ok(vec![])
		})
		.unwrap();
	}

	#[test]
	fn test_lasso_u32add() {
		test_circuit(|builder| {
			let log_size = 14;
			let add_a = unconstrained::<BinaryField1b>(builder, "add_a", log_size)?;
			let add_b = unconstrained::<BinaryField1b>(builder, "add_b", log_size)?;
			let _sum = super::u32add::<BinaryField1b, BinaryField1b>(
				builder,
				"lasso_u32add",
				add_a,
				add_b,
			)?;
			Ok(vec![])
		})
		.unwrap();
	}
}
