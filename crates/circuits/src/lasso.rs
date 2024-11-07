// Copyright 2024 Irreducible Inc.

use anyhow::Result;
use binius_core::{
	constraint_system::channel::ChannelId, oracle::OracleId, transparent,
	witness::MultilinearExtensionIndex,
};
use binius_field::{
	as_packed_field::{PackScalar, PackedType},
	underlier::{UnderlierType, WithUnderlier},
	BinaryField, BinaryField16b, BinaryField32b, BinaryField8b, ExtensionField, Field, PackedField,
	PackedFieldIndexable, TowerField,
};
use binius_macros::composition_poly;
use bytemuck::{must_cast_slice, Pod};
use itertools::izip;

use crate::builder::ConstraintSystemBuilder;

type B8 = BinaryField8b;
type B16 = BinaryField16b;
type B32 = BinaryField32b;

const ALPHA: B32 = B32::MULTIPLICATIVE_GENERATOR;
const T_LOG_SIZE: usize = 16;

pub fn u8mul<U, F>(
	builder: &mut ConstraintSystemBuilder<U, F>,
	name: impl ToString,
	mult_a: OracleId,
	mult_b: OracleId,
	log_size: usize,
) -> Result<OracleId, anyhow::Error>
where
	U: Pod + UnderlierType + PackScalar<B8> + PackScalar<B16> + PackScalar<B32> + PackScalar<F>,
	PackedType<U, B8>: PackedFieldIndexable,
	PackedType<U, B16>: PackedFieldIndexable,
	PackedType<U, B32>: PackedFieldIndexable,
	F: TowerField + BinaryField + ExtensionField<B8> + ExtensionField<B16> + ExtensionField<B32>,
{
	builder.push_namespace(name);
	let trace_oracle = TraceOracle::new(builder, log_size, mult_a, mult_b).unwrap();

	generate_constraints(builder, &trace_oracle).unwrap();

	if let Some(witness) = builder.witness() {
		let u_to_t_mapping = generate_values::<U, F>(log_size, &trace_oracle, witness).unwrap();
		generate_timestamps(log_size, &trace_oracle, &u_to_t_mapping, witness).unwrap();
	}
	builder.pop_namespace();
	Ok(trace_oracle.product)
}

pub struct TraceOracle {
	channel: ChannelId,
	mult_a: OracleId,
	mult_b: OracleId,
	product: OracleId,
	lookup_t: OracleId,
	lookup_u: OracleId,
	// timestamp things below
	lookup_r: OracleId,
	lookup_s: OracleId, // the inverse of r
	lookup_w: OracleId,
	lookup_o: OracleId,
	lookup_f: OracleId,
}

impl TraceOracle {
	pub fn new<F, U>(
		builder: &mut ConstraintSystemBuilder<U, F>,
		n_vars: usize,
		mult_a: OracleId,
		mult_b: OracleId,
	) -> Result<Self>
	where
		F: TowerField + ExtensionField<B32>,
		U: UnderlierType + Pod + PackScalar<F>,
	{
		let product = builder.add_committed("product", n_vars, B16::TOWER_LEVEL);

		let lookup_u = builder.add_linear_combination(
			"lookup_u",
			n_vars,
			[
				(mult_a, <F as TowerField>::basis(3, 3)?),
				(mult_b, <F as TowerField>::basis(3, 2)?),
				(product, <F as TowerField>::basis(3, 0)?),
			],
		)?;
		let lookup_t = builder.add_committed("lookup_t", T_LOG_SIZE, B32::TOWER_LEVEL);
		let lookup_f = builder.add_committed("lookup_f", T_LOG_SIZE, B32::TOWER_LEVEL);
		let lookup_o = builder.add_transparent(
			"lookup_o",
			transparent::constant::Constant {
				n_vars: T_LOG_SIZE,
				value: F::ONE,
			},
		)?;
		let lookup_r = builder.add_committed("lookup_r", n_vars, B32::TOWER_LEVEL);
		let lookup_s = builder.add_committed("lookup_s", n_vars, B32::TOWER_LEVEL);
		let lookup_w =
			builder.add_linear_combination("lookup_w", n_vars, [(lookup_r, F::ONE * ALPHA)])?;

		Ok(TraceOracle {
			channel: builder.add_channel(),
			mult_a,
			mult_b,
			product,
			lookup_t,
			lookup_u,
			lookup_r,
			lookup_s,
			lookup_w,
			lookup_f,
			lookup_o,
		})
	}
}

fn make_underliers<U: UnderlierType + PackScalar<FS>, FS: Field>(log_size: usize) -> Vec<U> {
	let packing_log_width = PackedType::<U, FS>::LOG_WIDTH;
	vec![U::default(); 1 << (log_size - packing_log_width)]
}

fn underliers_unpack_scalars_mut<U: UnderlierType + PackScalar<F>, F: Field>(
	underliers: &mut [U],
) -> &mut [F]
where
	PackedType<U, F>: PackedFieldIndexable,
{
	PackedType::<U, F>::unpack_scalars_mut(PackedType::<U, F>::from_underliers_ref_mut(underliers))
}

pub fn generate_values<U, F>(
	log_size: usize,
	trace_oracle: &TraceOracle,
	witness: &mut MultilinearExtensionIndex<'static, U, F>,
) -> Result<Vec<usize>>
where
	U: UnderlierType + PackScalar<B8> + PackScalar<B16> + PackScalar<B32> + PackScalar<F>,
	PackedType<U, B8>: PackedFieldIndexable,
	PackedType<U, B16>: PackedFieldIndexable,
	PackedType<U, B32>: PackedFieldIndexable,
	F: BinaryField + ExtensionField<B8> + ExtensionField<B16> + ExtensionField<B32>,
{
	let mut product = make_underliers::<_, B16>(log_size);
	let mut lookup_u = make_underliers::<_, B32>(log_size);
	let mut lookup_t = make_underliers::<_, B32>(T_LOG_SIZE);
	let mut u_to_t_mapping = vec![0; 1 << log_size];

	let mult_a_ext = witness.get::<B8>(trace_oracle.mult_a)?;
	let mult_a_ints =
		must_cast_slice::<_, u8>(WithUnderlier::to_underliers_ref(mult_a_ext.evals()));
	let mult_b_ext = witness.get::<B8>(trace_oracle.mult_b)?;
	let mult_b_ints =
		must_cast_slice::<_, u8>(WithUnderlier::to_underliers_ref(mult_b_ext.evals()));

	let product_scalars = underliers_unpack_scalars_mut::<_, B16>(product.as_mut_slice());
	let lookup_u_scalars = underliers_unpack_scalars_mut::<_, B32>(lookup_u.as_mut_slice());
	let lookup_t_scalars = underliers_unpack_scalars_mut::<_, B32>(lookup_t.as_mut_slice());

	for (a, b, lookup_u, product, u_to_t) in izip!(
		mult_a_ints,
		mult_b_ints,
		lookup_u_scalars.iter_mut(),
		product_scalars.iter_mut(),
		u_to_t_mapping.iter_mut()
	) {
		let a_int = *a as usize;
		let b_int = *b as usize;
		let ab_product = a_int * b_int;
		let lookup_index = a_int << 8 | b_int;
		*lookup_u = BinaryField32b::new((lookup_index << 16 | ab_product) as u32);

		*product = BinaryField16b::new(ab_product as u16);
		*u_to_t = lookup_index;
	}

	for (i, lookup_t) in lookup_t_scalars.iter_mut().enumerate() {
		let a_int = (i >> 8) & 0xff;
		let b_int = i & 0xff;
		let ab_product = a_int * b_int;
		let lookup_index = a_int << 8 | b_int;
		assert_eq!(lookup_index, i);
		*lookup_t = BinaryField32b::new((lookup_index << 16 | ab_product) as u32);
	}

	witness.set_owned::<B16, _>([(trace_oracle.product, product)])?;
	witness.set_owned::<B32, _>([
		(trace_oracle.lookup_u, lookup_u),
		(trace_oracle.lookup_t, lookup_t),
	])?;
	Ok(u_to_t_mapping)
}

pub fn generate_timestamps<U, F>(
	log_size: usize,
	trace_oracle: &TraceOracle,
	u_to_t_mapping: &[usize],
	witness: &mut MultilinearExtensionIndex<'static, U, F>,
) -> Result<()>
where
	U: UnderlierType + PackScalar<B8> + PackScalar<B16> + PackScalar<B32> + PackScalar<F>,
	PackedType<U, B8>: PackedFieldIndexable,
	PackedType<U, B16>: PackedFieldIndexable,
	PackedType<U, B32>: PackedFieldIndexable,
	F: BinaryField + ExtensionField<B8> + ExtensionField<B16> + ExtensionField<B32>,
{
	let mut lookup_r = make_underliers::<_, B32>(log_size);
	let mut lookup_s = make_underliers::<_, B32>(log_size);
	let mut lookup_w = make_underliers::<_, B32>(log_size);
	let mut lookup_f = make_underliers::<_, B32>(T_LOG_SIZE);
	let mut lookup_o = make_underliers::<_, B32>(T_LOG_SIZE);

	let lookup_r_scalars = underliers_unpack_scalars_mut::<_, B32>(lookup_r.as_mut_slice());
	let lookup_s_scalars = underliers_unpack_scalars_mut::<_, B32>(lookup_s.as_mut_slice());
	let lookup_w_scalars = underliers_unpack_scalars_mut::<_, B32>(lookup_w.as_mut_slice());
	let lookup_f_scalars = underliers_unpack_scalars_mut::<_, B32>(lookup_f.as_mut_slice());
	let lookup_o_scalars = underliers_unpack_scalars_mut::<_, B32>(lookup_o.as_mut_slice());

	lookup_f_scalars.fill(B32::ONE);
	lookup_o_scalars.fill(B32::ONE);

	for (j, (r, s, w)) in
		izip!(lookup_r_scalars.iter_mut(), lookup_s_scalars.iter_mut(), lookup_w_scalars.iter_mut())
			.enumerate()
	{
		let index = u_to_t_mapping[j];
		let ts = lookup_f_scalars[index];
		*r = ts;
		*s = ts.pow([(1 << 32) - 2]);
		*w = ts * ALPHA;
		lookup_f_scalars[index] *= ALPHA;
	}

	witness.set_owned::<B32, _>([
		(trace_oracle.lookup_r, lookup_r),
		(trace_oracle.lookup_s, lookup_s),
		(trace_oracle.lookup_w, lookup_w),
		(trace_oracle.lookup_f, lookup_f),
		(trace_oracle.lookup_o, lookup_o),
	])?;

	Ok(())
}

pub fn generate_constraints<U, F>(
	builder: &mut ConstraintSystemBuilder<U, F>,
	trace_oracle: &TraceOracle,
) -> Result<()>
where
	U: UnderlierType + Pod + PackScalar<F>,
	F: TowerField,
{
	builder.assert_zero(
		[trace_oracle.lookup_r, trace_oracle.lookup_s],
		composition_poly!([x, y] = x * y - 1),
	);

	// populate table using initial timestamps
	builder.send(trace_oracle.channel, [trace_oracle.lookup_t, trace_oracle.lookup_o]);

	// for every value looked up, pull using current timestamp and push with incremented timestamp
	builder.receive(trace_oracle.channel, [trace_oracle.lookup_u, trace_oracle.lookup_r]);
	builder.send(trace_oracle.channel, [trace_oracle.lookup_u, trace_oracle.lookup_w]);

	// depopulate table using final timestamps
	builder.receive(trace_oracle.channel, [trace_oracle.lookup_t, trace_oracle.lookup_f]);

	Ok(())
}
