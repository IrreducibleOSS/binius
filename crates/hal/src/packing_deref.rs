// Copyright 2024 Irreducible Inc.

use binius_field::{
	as_packed_field::{PackScalar, PackedType},
	underlier::{UnderlierType, WithUnderlier},
	Field,
};
use std::{marker::PhantomData, ops::Deref};

/// A wrapper for containers of underlier types that dereferences as packed field slices.
#[derive(Debug, Clone)]
pub struct PackingDeref<U, F, Data>(Data, PhantomData<F>)
where
	Data: Deref<Target = [U]>;

impl<U, F, Data> PackingDeref<U, F, Data>
where
	Data: Deref<Target = [U]>,
{
	pub fn new(data: Data) -> Self {
		Self(data, PhantomData)
	}
}

impl<U, F, Data> Deref for PackingDeref<U, F, Data>
where
	U: UnderlierType + PackScalar<F>,
	F: Field,
	Data: Deref<Target = [U]>,
{
	type Target = [PackedType<U, F>];

	fn deref(&self) -> &Self::Target {
		<PackedType<U, F>>::from_underliers_ref(self.0.deref())
	}
}
