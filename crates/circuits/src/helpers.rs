// Copyright 2024 Irreducible Inc.

use binius_field::{
	as_packed_field::{PackScalar, PackedType},
	underlier::{UnderlierType, WithUnderlier},
	Field, PackedField, PackedFieldIndexable,
};

pub fn make_underliers<U: UnderlierType + PackScalar<FS>, FS: Field>(log_size: usize) -> Box<[U]> {
	let packing_log_width = PackedType::<U, FS>::LOG_WIDTH;
	vec![U::default(); 1 << (log_size - packing_log_width)].into_boxed_slice()
}

pub fn underliers_unpack_scalars_mut<U: UnderlierType + PackScalar<F>, F: Field>(
	underliers: &mut [U],
) -> &mut [F]
where
	PackedType<U, F>: PackedFieldIndexable,
{
	PackedType::<U, F>::unpack_scalars_mut(PackedType::<U, F>::from_underliers_ref_mut(underliers))
}
