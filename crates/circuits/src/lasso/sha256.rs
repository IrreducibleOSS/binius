// Copyright 2024-2025 Irreducible Inc.

use anyhow::Result;
use binius_core::oracle::OracleId;
use binius_field::{
	as_packed_field::PackedType, BinaryField16b, BinaryField1b, BinaryField32b, BinaryField4b,
	PackedFieldIndexable, TowerField,
};
use itertools::izip;

use super::{lasso::lasso, u32add::SeveralU32add};
use crate::{
	builder::{
		types::{F, U},
		ConstraintSystemBuilder,
	},
	pack::pack,
	sha256::{rotate_and_xor, u32const_repeating, RotateRightType, INIT, ROUND_CONSTS_K},
};

pub const CH_MAJ_T_LOG_SIZE: usize = 12;

type B1 = BinaryField1b;
type B4 = BinaryField4b;
type B16 = BinaryField16b;
type B32 = BinaryField32b;

struct SeveralBitwise {
	n_lookups: Vec<usize>,
	lookup_t: OracleId,
	lookups_u: Vec<[OracleId; 1]>,
	u_to_t_mappings: Vec<Vec<usize>>,
	f: fn(u32, u32, u32) -> u32,
}

impl SeveralBitwise {
	pub fn new(builder: &mut ConstraintSystemBuilder, f: fn(u32, u32, u32) -> u32) -> Result<Self> {
		let lookup_t =
			builder.add_committed("bitwise lookup_t", CH_MAJ_T_LOG_SIZE, B16::TOWER_LEVEL);

		if let Some(witness) = builder.witness() {
			let mut lookup_t_witness = witness.new_column::<B16>(lookup_t);

			let lookup_t_scalars =
				PackedType::<U, B16>::unpack_scalars_mut(lookup_t_witness.packed());

			for (i, lookup_t) in lookup_t_scalars.iter_mut().enumerate() {
				let x = ((i >> 8) & 15) as u16;
				let y = ((i >> 4) & 15) as u16;
				let z = (i & 15) as u16;

				let res = f(x as u32, y as u32, z as u32);

				let lookup_index = (((x << 4) | y) << 4) | z;
				*lookup_t = B16::new((lookup_index << 4) | res as u16);
			}
		}
		Ok(Self {
			n_lookups: Vec::new(),
			lookup_t,
			lookups_u: Vec::new(),
			u_to_t_mappings: Vec::new(),
			f,
		})
	}

	pub fn calculate(
		&mut self,
		builder: &mut ConstraintSystemBuilder,
		name: impl ToString,
		params: [OracleId; 3],
	) -> Result<OracleId> {
		let [xin, yin, zin] = params;

		let log_size = builder.log_rows(params)?;

		let xin_packed = pack::<B1, B4>(xin, builder, "xin_packed")?;
		let yin_packed = pack::<B1, B4>(yin, builder, "yin_packed")?;
		let zin_packed = pack::<B1, B4>(zin, builder, "zin_packed")?;

		let res = builder.add_committed(name, log_size, B1::TOWER_LEVEL);

		let res_packed = builder.add_packed("res_packed", res, B4::TOWER_LEVEL)?;

		let lookup_u = builder.add_linear_combination(
			"ch or maj lookup_u",
			log_size - B4::TOWER_LEVEL,
			[
				(xin_packed, <F as TowerField>::basis(B4::TOWER_LEVEL, 3)?),
				(yin_packed, <F as TowerField>::basis(B4::TOWER_LEVEL, 2)?),
				(zin_packed, <F as TowerField>::basis(B4::TOWER_LEVEL, 1)?),
				(res_packed, <F as TowerField>::basis(B4::TOWER_LEVEL, 0)?),
			],
		)?;

		if let Some(witness) = builder.witness() {
			let mut lookup_u_witness = witness.new_column::<B16>(lookup_u);
			let lookup_u_u16 = PackedType::<U, B16>::unpack_scalars_mut(lookup_u_witness.packed());

			let mut u_to_t_mapping_witness = Vec::with_capacity(1 << (log_size - B4::TOWER_LEVEL));

			let mut res_witness = witness.new_column::<B1>(res);
			let res_u32 = res_witness.as_mut_slice::<u32>();

			let xin_u32 = witness.get::<B1>(xin)?.as_slice::<u32>();

			let yin_u32 = witness.get::<B1>(yin)?.as_slice::<u32>();

			let zin_u32 = witness.get::<B1>(zin)?.as_slice::<u32>();

			for (res, x, y, z, lookup_u) in
				izip!(res_u32.iter_mut(), xin_u32, yin_u32, zin_u32, lookup_u_u16.chunks_mut(8))
			{
				*res = (self.f)(*x, *y, *z);

				#[allow(clippy::needless_range_loop)]
				for i in 0..8 {
					let x = ((*x >> (4 * i)) & 15) as u16;
					let y = ((*y >> (4 * i)) & 15) as u16;
					let z = ((*z >> (4 * i)) & 15) as u16;
					let res = ((*res >> (4 * i)) & 15) as u16;
					let lookup_index = (((x << 4) | y) << 4) | z;
					lookup_u[i] = B16::new((lookup_index << 4) | res);
					u_to_t_mapping_witness.push(lookup_index as usize)
				}
			}

			std::mem::drop(res_witness);

			let res_packed_witness = witness.get::<B1>(res)?;
			witness.set::<B4>(res_packed, res_packed_witness.repacked::<B4>())?;

			self.u_to_t_mappings.push(u_to_t_mapping_witness);
		}

		self.lookups_u.push([lookup_u]);
		self.n_lookups.push(1 << (log_size - B4::TOWER_LEVEL));
		Ok(res)
	}

	pub fn finalize(
		self,
		builder: &mut ConstraintSystemBuilder,
		name: impl ToString,
	) -> Result<()> {
		let channel = builder.add_channel();

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

pub fn sha256(
	builder: &mut ConstraintSystemBuilder,
	input: [OracleId; 16],
	log_size: usize,
) -> Result<[OracleId; 8], anyhow::Error> {
	let n_vars = log_size;

	let mut several_u32_add = SeveralU32add::new(builder)?;

	let mut several_ch = SeveralBitwise::new(builder, |e, f, g| (e & f) ^ ((!e) & g))?;

	let mut several_maj = SeveralBitwise::new(builder, |a, b, c| (a & b) ^ (a & c) ^ (b & c))?;

	let mut w = [OracleId::MAX; 64];

	w[0..16].copy_from_slice(&input);

	for i in 16..64 {
		let s0 = rotate_and_xor(
			n_vars,
			builder,
			&[
				(w[i - 15], 7, RotateRightType::Circular),
				(w[i - 15], 18, RotateRightType::Circular),
				(w[i - 15], 3, RotateRightType::Logical),
			],
		)?;
		let s1 = rotate_and_xor(
			n_vars,
			builder,
			&[
				(w[i - 2], 17, RotateRightType::Circular),
				(w[i - 2], 19, RotateRightType::Circular),
				(w[i - 2], 10, RotateRightType::Logical),
			],
		)?;

		let w_addition =
			several_u32_add.u32add::<B1, B1>(builder, "w_addition", w[i - 16], w[i - 7])?;

		let s_addition = several_u32_add.u32add::<B1, B1>(builder, "s_addition", s0, s1)?;

		w[i] = several_u32_add.u32add::<B1, B1>(
			builder,
			format!("w[{}]", i),
			w_addition,
			s_addition,
		)?;
	}

	let init_oracles = INIT.map(|val| u32const_repeating(n_vars, builder, val, "INIT").unwrap());

	let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = init_oracles;

	let k = ROUND_CONSTS_K
		.map(|val| u32const_repeating(n_vars, builder, val, "ROUND_CONSTS_K").unwrap());

	for i in 0..64 {
		let sigma1 = rotate_and_xor(
			n_vars,
			builder,
			&[
				(e, 6, RotateRightType::Circular),
				(e, 11, RotateRightType::Circular),
				(e, 25, RotateRightType::Circular),
			],
		)?;

		let ch = several_ch.calculate(builder, "ch", [e, f, g])?;

		let h_sigma1 = several_u32_add.u32add::<B1, B1>(builder, "h_sigma1", h, sigma1)?;
		let ch_ki = several_u32_add.u32add::<B1, B1>(builder, "ch_ki", ch, k[i])?;
		let ch_ki_w_i = several_u32_add.u32add::<B1, B1>(builder, "ch_ki_w_i", ch_ki, w[i])?;
		let temp1 = several_u32_add.u32add::<B1, B1>(builder, "temp1", h_sigma1, ch_ki_w_i)?;

		let sigma0 = rotate_and_xor(
			n_vars,
			builder,
			&[
				(a, 2, RotateRightType::Circular),
				(a, 13, RotateRightType::Circular),
				(a, 22, RotateRightType::Circular),
			],
		)?;

		let maj = several_maj.calculate(builder, "maj", [a, b, c])?;

		let temp2 = several_u32_add.u32add::<B1, B1>(builder, "temp2", sigma0, maj)?;

		h = g;
		g = f;
		f = e;
		e = several_u32_add.u32add::<B1, B1>(builder, "ch_ki_w_i", d, temp1)?;
		d = c;
		c = b;
		b = a;
		a = several_u32_add.u32add::<B1, B1>(builder, "ch_ki_w_i", temp1, temp2)?;
	}

	let abcdefgh = [a, b, c, d, e, f, g, h];

	let output = std::array::from_fn(|i| {
		several_u32_add
			.u32add::<B1, B1>(builder, "output", init_oracles[i], abcdefgh[i])
			.unwrap()
	});

	several_u32_add.finalize(builder, "lasso")?;

	several_ch.finalize(builder, "ch")?;
	several_maj.finalize(builder, "maj")?;

	Ok(output)
}

#[cfg(test)]
mod tests {
	use binius_core::{constraint_system::validate::validate_witness, oracle::OracleId};
	use binius_field::{as_packed_field::PackedType, BinaryField1b, BinaryField8b, TowerField};
	use sha2::{compress256, digest::generic_array::GenericArray};

	use crate::{
		builder::{types::U, ConstraintSystemBuilder},
		unconstrained::unconstrained,
	};

	#[test]
	fn test_sha256_lasso() {
		let allocator = bumpalo::Bump::new();
		let mut builder = ConstraintSystemBuilder::new_with_witness(&allocator);
		let log_size = PackedType::<U, BinaryField1b>::LOG_WIDTH + BinaryField8b::TOWER_LEVEL;
		let input: [OracleId; 16] = std::array::from_fn(|i| {
			unconstrained::<BinaryField1b>(&mut builder, i, log_size).unwrap()
		});
		let state_output = super::sha256(&mut builder, input, log_size).unwrap();

		let witness = builder.witness().unwrap();

		let input_witneses: [_; 16] = std::array::from_fn(|i| {
			witness
				.get::<BinaryField1b>(input[i])
				.unwrap()
				.as_slice::<u32>()
		});

		let output_witneses: [_; 8] = std::array::from_fn(|i| {
			witness
				.get::<BinaryField1b>(state_output[i])
				.unwrap()
				.as_slice::<u32>()
		});

		let mut generic_array_input = GenericArray::<u8, _>::default();

		let n_compressions = input_witneses[0].len();

		for j in 0..n_compressions {
			for i in 0..16 {
				for z in 0..4 {
					generic_array_input[i * 4 + z] = input_witneses[i][j].to_be_bytes()[z];
				}
			}

			let mut output = crate::sha256::INIT;
			compress256(&mut output, &[generic_array_input]);

			for i in 0..8 {
				assert_eq!(output[i], output_witneses[i][j]);
			}
		}

		let witness = builder.take_witness().unwrap();
		let constraint_system = builder.build().unwrap();
		let boundaries = vec![];
		validate_witness(&constraint_system, &boundaries, &witness).unwrap();
	}
}
