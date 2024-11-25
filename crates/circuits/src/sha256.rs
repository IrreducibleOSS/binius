// Copyright 2024 Irreducible Inc.

use crate::{builder::ConstraintSystemBuilder, u32add::u32add_committed};
use binius_core::{
	oracle::{OracleId, ShiftVariant},
	transparent::multilinear_extension::MultilinearExtensionTransparent,
};
use binius_field::{
	as_packed_field::{PackScalar, PackedType},
	underlier::{UnderlierType, WithUnderlier},
	BinaryField1b, PackedField, TowerField,
};
use binius_macros::composition_poly;
use binius_utils::checked_arithmetics::checked_log_2;
use bytemuck::{pod_collect_to_vec, Pod};
use itertools::izip;

const LOG_U32_BITS: usize = checked_log_2(32);

type B1 = BinaryField1b;

/// SHA-256 round constants, K
const ROUND_CONSTS_K: [u32; 64] = [
	0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
	0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
	0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
	0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
	0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
	0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
	0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
	0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

const INIT: [u32; 8] = [
	0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

enum RotateRightType {
	Circular,
	Logical,
}

fn rotate_and_xor<F, U>(
	log_size: usize,
	builder: &mut ConstraintSystemBuilder<U, F>,
	r: &[(OracleId, usize, RotateRightType)],
) -> Result<OracleId, anyhow::Error>
where
	F: TowerField,
	U: UnderlierType + Pod + PackScalar<F> + PackScalar<B1>,
{
	let shifted_oracle_ids = r
		.iter()
		.map(|(oracle_id, shift, t)| {
			match t {
				RotateRightType::Circular => builder.add_shifted(
					format!("RotateRightType::Circular shift:{} oracle_id: {}", shift, oracle_id),
					*oracle_id,
					32 - shift,
					LOG_U32_BITS,
					ShiftVariant::CircularLeft,
				),
				RotateRightType::Logical => builder.add_shifted(
					format!("RotateRightType::Logical shift:{} oracle_id: {}", shift, oracle_id),
					*oracle_id,
					*shift,
					LOG_U32_BITS,
					ShiftVariant::LogicalRight,
				),
			}
			.map_err(|e| e.into())
		})
		.collect::<Result<Vec<_>, anyhow::Error>>()?;

	let result_oracle_id = builder.add_linear_combination(
		format!("linear combination of {:?}", shifted_oracle_ids),
		log_size,
		shifted_oracle_ids.iter().map(|s| (*s, F::ONE)),
	)?;

	if let Some(witness) = builder.witness() {
		let mut result_witness = witness.new_column::<B1>(result_oracle_id);
		let result_u32 = result_witness.as_mut_slice::<u32>();

		for ((oracle_id, shift, t), shifted_oracle_id) in r.iter().zip(&shifted_oracle_ids) {
			let values_u32 = witness.get::<B1>(*oracle_id)?.as_slice::<u32>();

			let mut shifted_witness = witness.new_column::<B1>(*shifted_oracle_id);
			let shifted_u32 = shifted_witness.as_mut_slice::<u32>();

			izip!(shifted_u32.iter_mut(), values_u32, result_u32.iter_mut()).for_each(
				|(shifted, val, res)| {
					*shifted = match t {
						RotateRightType::Circular => val.rotate_right(*shift as u32),
						RotateRightType::Logical => val >> shift,
					};
					*res ^= *shifted;
				},
			);
		}
	}

	Ok(result_oracle_id)
}

#[inline]
fn into_packed_vec<P>(src: &[impl Pod]) -> Vec<P>
where
	P: PackedField + WithUnderlier,
	P::Underlier: Pod,
{
	pod_collect_to_vec::<_, P::Underlier>(src)
		.into_iter()
		.map(P::from_underlier)
		.collect()
}

fn u32const_repeating<F, U>(
	log_size: usize,
	builder: &mut ConstraintSystemBuilder<U, F>,
	x: u32,
	name: &str,
) -> Result<OracleId, anyhow::Error>
where
	F: TowerField,
	U: UnderlierType + Pod + PackScalar<F> + PackScalar<B1>,
{
	let brodcasted = vec![x; 1 << (PackedType::<U, B1>::LOG_WIDTH.saturating_sub(LOG_U32_BITS))];

	let transparent_id = builder.add_transparent(
		format!("transparent {}", name),
		MultilinearExtensionTransparent::<_, PackedType<U, F>, _>::from_values(into_packed_vec::<
			PackedType<U, B1>,
		>(&brodcasted))?,
	)?;

	let repeating_id = builder.add_repeating(
		format!("repeating {}", name),
		transparent_id,
		log_size - PackedType::<U, B1>::LOG_WIDTH,
	)?;

	if let Some(witness) = builder.witness() {
		let mut transparent_witness = witness.new_column::<B1>(transparent_id);
		transparent_witness.as_mut_slice::<u32>().fill(x);

		let mut repeating_witness = witness.new_column::<B1>(repeating_id);
		repeating_witness.as_mut_slice::<u32>().fill(x);
	}

	Ok(repeating_id)
}

pub fn sha256<U, F>(
	builder: &mut ConstraintSystemBuilder<U, F>,
	input: [OracleId; 16],
	log_size: usize,
) -> Result<[OracleId; 8], anyhow::Error>
where
	U: UnderlierType + Pod + PackScalar<F> + PackScalar<B1>,
	F: TowerField,
{
	if log_size < <PackedType<U, BinaryField1b>>::LOG_WIDTH {
		Err(anyhow::Error::msg("log_size too small"))?
	}

	let mut w = [OracleId::MAX; 64];

	w[0..16].copy_from_slice(&input);

	for i in 16..64 {
		let s0 = rotate_and_xor(
			log_size,
			builder,
			&[
				(w[i - 15], 7, RotateRightType::Circular),
				(w[i - 15], 18, RotateRightType::Circular),
				(w[i - 15], 3, RotateRightType::Logical),
			],
		)?;
		let s1 = rotate_and_xor(
			log_size,
			builder,
			&[
				(w[i - 2], 17, RotateRightType::Circular),
				(w[i - 2], 19, RotateRightType::Circular),
				(w[i - 2], 10, RotateRightType::Logical),
			],
		)?;
		let w_addition = u32add_committed(builder, "w_addition", w[i - 16], w[i - 7])?;
		let s_addition = u32add_committed(builder, "s_addition", s0, s1)?;

		w[i] = u32add_committed(builder, format!("w[{}]", i), w_addition, s_addition)?;
	}

	let init_oracles = INIT.map(|val| u32const_repeating(log_size, builder, val, "INIT").unwrap());

	let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = init_oracles;

	let k = ROUND_CONSTS_K
		.map(|val| u32const_repeating(log_size, builder, val, "ROUND_CONSTS_K").unwrap());

	let ch: [OracleId; 64] = builder.add_committed_multiple("ch", log_size, B1::TOWER_LEVEL);

	let maj: [OracleId; 64] = builder.add_committed_multiple("maj", log_size, B1::TOWER_LEVEL);

	for i in 0..64 {
		let sigma1 = rotate_and_xor(
			log_size,
			builder,
			&[
				(e, 6, RotateRightType::Circular),
				(e, 11, RotateRightType::Circular),
				(e, 25, RotateRightType::Circular),
			],
		)?;

		if let Some(witness) = builder.witness() {
			let mut ch_witness = witness.new_column::<B1>(ch[i]);
			let ch_u32 = ch_witness.as_mut_slice::<u32>();
			let e_u32 = witness.get::<B1>(e)?.as_slice::<u32>();
			let f_u32 = witness.get::<B1>(f)?.as_slice::<u32>();
			let g_u32 = witness.get::<B1>(g)?.as_slice::<u32>();
			izip!(ch_u32.iter_mut(), e_u32, f_u32, g_u32).for_each(|(ch, e, f, g)| {
				*ch = (e & f) ^ ((!e) & g);
			});
		}

		let h_sigma1 = u32add_committed(builder, "h_sigma1", h, sigma1)?;
		let ch_ki = u32add_committed(builder, "ch_ki", ch[i], k[i])?;
		let ch_ki_w_i = u32add_committed(builder, "ch_ki_w_i", ch_ki, w[i])?;
		let temp1 = u32add_committed(builder, "temp1", h_sigma1, ch_ki_w_i)?;

		let sigma0 = rotate_and_xor(
			log_size,
			builder,
			&[
				(a, 2, RotateRightType::Circular),
				(a, 13, RotateRightType::Circular),
				(a, 22, RotateRightType::Circular),
			],
		)?;

		if let Some(witness) = builder.witness() {
			let mut maj_witness = witness.new_column::<B1>(maj[i]);
			let maj_u32 = maj_witness.as_mut_slice::<u32>();
			let a_u32 = witness.get::<B1>(a)?.as_slice::<u32>();
			let b_u32 = witness.get::<B1>(b)?.as_slice::<u32>();
			let c_u32 = witness.get::<B1>(c)?.as_slice::<u32>();
			izip!(maj_u32.iter_mut(), a_u32, b_u32, c_u32).for_each(|(maj, a, b, c)| {
				*maj = (a & b) ^ (a & c) ^ (b & c);
			});
		}

		let temp2 = u32add_committed(builder, "temp2", sigma0, maj[i])?;

		builder.assert_zero(
			[e, f, g, ch[i]],
			composition_poly!([e, f, g, ch] = (e * f + (1 - e) * g) - ch),
		);

		builder.assert_zero(
			[a, b, c, maj[i]],
			composition_poly!([a, b, c, maj] = maj - a * b + a * c + b * c),
		);

		h = g;
		g = f;
		f = e;
		e = u32add_committed(builder, "e", d, temp1)?;
		d = c;
		c = b;
		b = a;
		a = u32add_committed(builder, "a", temp1, temp2)?;
	}

	let abcdefgh = [a, b, c, d, e, f, g, h];

	let output = std::array::from_fn(|i| {
		u32add_committed(builder, "output", init_oracles[i], abcdefgh[i]).unwrap()
	});

	Ok(output)
}
