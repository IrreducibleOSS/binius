// Copyright 2025 Irreducible Inc.

use bytes::{Buf, BufMut};
use generic_array::{ArrayLength, GenericArray};

use super::{DeserializeBytes, Error, SerializeBytes};
use crate::TowerField;

/// Serialization where [`TowerField`] elements are written with canonical encoding.
pub trait SerializeCanonical {
	fn serialize_canonical(&self, write_buf: impl BufMut) -> Result<(), Error>;
}

/// Deserialization where [`TowerField`] elements are read with a canonical encoding.
pub trait DeserializeCanonical {
	fn deserialize_canonical(read_buf: impl Buf) -> Result<Self, Error>
	where
		Self: Sized;
}

impl<F: TowerField> SerializeCanonical for F {
	fn serialize_canonical(&self, mut write_buf: impl BufMut) -> Result<(), Error> {
		SerializeBytes::serialize(&F::Canonical::from(*self), &mut write_buf)
	}
}

impl<F: TowerField> DeserializeCanonical for F {
	fn deserialize_canonical(read_buf: impl Buf) -> Result<Self, Error>
	where
		Self: Sized,
	{
		let canonical: F::Canonical = DeserializeBytes::deserialize(read_buf)?;
		Ok(F::from(canonical))
	}
}

impl SerializeCanonical for usize {
	fn serialize_canonical(&self, mut write_buf: impl BufMut) -> Result<(), Error> {
		SerializeCanonical::serialize_canonical(&(*self as u64), &mut write_buf)
	}
}

impl DeserializeCanonical for usize {
	fn deserialize_canonical(mut read_buf: impl Buf) -> Result<Self, Error>
	where
		Self: Sized,
	{
		let value: u64 = DeserializeCanonical::deserialize_canonical(&mut read_buf)?;
		Ok(value as Self)
	}
}

impl SerializeCanonical for u128 {
	fn serialize_canonical(&self, mut write_buf: impl BufMut) -> Result<(), Error> {
		assert_enough_space_for(&write_buf, std::mem::size_of::<Self>())?;
		write_buf.put_u128(*self);
		Ok(())
	}
}

impl DeserializeCanonical for u128 {
	fn deserialize_canonical(mut read_buf: impl Buf) -> Result<Self, Error>
	where
		Self: Sized,
	{
		assert_enough_data_for(&read_buf, std::mem::size_of::<Self>())?;
		Ok(read_buf.get_u128())
	}
}

impl SerializeCanonical for u64 {
	fn serialize_canonical(&self, mut write_buf: impl BufMut) -> Result<(), Error> {
		assert_enough_space_for(&write_buf, std::mem::size_of::<Self>())?;
		write_buf.put_u64(*self);
		Ok(())
	}
}

impl DeserializeCanonical for u64 {
	fn deserialize_canonical(mut read_buf: impl Buf) -> Result<Self, Error>
	where
		Self: Sized,
	{
		assert_enough_data_for(&read_buf, std::mem::size_of::<Self>())?;
		Ok(read_buf.get_u64())
	}
}

impl SerializeCanonical for u32 {
	fn serialize_canonical(&self, mut write_buf: impl BufMut) -> Result<(), Error> {
		assert_enough_space_for(&write_buf, std::mem::size_of::<Self>())?;
		write_buf.put_u32(*self);
		Ok(())
	}
}

impl DeserializeCanonical for u32 {
	fn deserialize_canonical(mut read_buf: impl Buf) -> Result<Self, Error>
	where
		Self: Sized,
	{
		assert_enough_data_for(&read_buf, std::mem::size_of::<Self>())?;
		Ok(read_buf.get_u32())
	}
}

impl SerializeCanonical for u16 {
	fn serialize_canonical(&self, mut write_buf: impl BufMut) -> Result<(), Error> {
		assert_enough_space_for(&write_buf, std::mem::size_of::<Self>())?;
		write_buf.put_u16(*self);
		Ok(())
	}
}

impl DeserializeCanonical for u16 {
	fn deserialize_canonical(mut read_buf: impl Buf) -> Result<Self, Error>
	where
		Self: Sized,
	{
		assert_enough_data_for(&read_buf, std::mem::size_of::<Self>())?;
		Ok(read_buf.get_u16())
	}
}

impl SerializeCanonical for u8 {
	fn serialize_canonical(&self, mut write_buf: impl BufMut) -> Result<(), Error> {
		assert_enough_space_for(&write_buf, std::mem::size_of::<Self>())?;
		write_buf.put_u8(*self);
		Ok(())
	}
}

impl DeserializeCanonical for u8 {
	fn deserialize_canonical(mut read_buf: impl Buf) -> Result<Self, Error>
	where
		Self: Sized,
	{
		assert_enough_data_for(&read_buf, std::mem::size_of::<Self>())?;
		Ok(read_buf.get_u8())
	}
}

impl SerializeCanonical for bool {
	fn serialize_canonical(&self, write_buf: impl BufMut) -> Result<(), Error> {
		u8::serialize_canonical(&(*self as u8), write_buf)
	}
}

impl DeserializeCanonical for bool {
	fn deserialize_canonical(read_buf: impl Buf) -> Result<Self, Error>
	where
		Self: Sized,
	{
		Ok(u8::deserialize_canonical(read_buf)? != 0)
	}
}

impl<T> SerializeCanonical for std::marker::PhantomData<T> {
	fn serialize_canonical(&self, _write_buf: impl BufMut) -> Result<(), Error> {
		Ok(())
	}
}

impl<T> DeserializeCanonical for std::marker::PhantomData<T> {
	fn deserialize_canonical(_read_buf: impl Buf) -> Result<Self, Error>
	where
		Self: Sized,
	{
		Ok(Self)
	}
}

impl SerializeCanonical for &str {
	fn serialize_canonical(&self, mut write_buf: impl BufMut) -> Result<(), Error> {
		let bytes = self.as_bytes();
		SerializeCanonical::serialize_canonical(&bytes.len(), &mut write_buf)?;
		assert_enough_space_for(&write_buf, bytes.len())?;
		write_buf.put_slice(bytes);
		Ok(())
	}
}

impl SerializeCanonical for String {
	fn serialize_canonical(&self, mut write_buf: impl BufMut) -> Result<(), Error> {
		SerializeCanonical::serialize_canonical(&self.as_str(), &mut write_buf)
	}
}

impl DeserializeCanonical for String {
	fn deserialize_canonical(mut read_buf: impl Buf) -> Result<Self, Error>
	where
		Self: Sized,
	{
		let len = DeserializeCanonical::deserialize_canonical(&mut read_buf)?;
		assert_enough_data_for(&read_buf, len)?;
		Ok(Self::from_utf8(read_buf.copy_to_bytes(len).to_vec())?)
	}
}

impl<T: SerializeCanonical> SerializeCanonical for Vec<T> {
	fn serialize_canonical(&self, mut write_buf: impl BufMut) -> Result<(), Error> {
		SerializeCanonical::serialize_canonical(&self.len(), &mut write_buf)?;
		self.iter()
			.try_for_each(|item| SerializeCanonical::serialize_canonical(item, &mut write_buf))
	}
}

impl<T: DeserializeCanonical> DeserializeCanonical for Vec<T> {
	fn deserialize_canonical(mut read_buf: impl Buf) -> Result<Self, Error>
	where
		Self: Sized,
	{
		let len: usize = DeserializeCanonical::deserialize_canonical(&mut read_buf)?;
		(0..len)
			.map(|_| DeserializeCanonical::deserialize_canonical(&mut read_buf))
			.collect()
	}
}

impl<T: SerializeCanonical> SerializeCanonical for Option<T> {
	fn serialize_canonical(&self, mut write_buf: impl BufMut) -> Result<(), Error> {
		match self {
			Some(value) => {
				SerializeCanonical::serialize_canonical(&true, &mut write_buf)?;
				SerializeCanonical::serialize_canonical(value, &mut write_buf)?;
			}
			None => {
				SerializeCanonical::serialize_canonical(&false, write_buf)?;
			}
		}
		Ok(())
	}
}

impl<T: DeserializeCanonical> DeserializeCanonical for Option<T> {
	fn deserialize_canonical(mut read_buf: impl Buf) -> Result<Self, Error>
	where
		Self: Sized,
	{
		Ok(match bool::deserialize_canonical(&mut read_buf)? {
			true => Some(T::deserialize_canonical(&mut read_buf)?),
			false => None,
		})
	}
}

impl<U: SerializeCanonical, V: SerializeCanonical> SerializeCanonical for (U, V) {
	fn serialize_canonical(&self, mut write_buf: impl BufMut) -> Result<(), Error> {
		U::serialize_canonical(&self.0, &mut write_buf)?;
		V::serialize_canonical(&self.1, write_buf)
	}
}

impl<U: DeserializeCanonical, V: DeserializeCanonical> DeserializeCanonical for (U, V) {
	fn deserialize_canonical(mut read_buf: impl Buf) -> Result<Self, Error>
	where
		Self: Sized,
	{
		Ok((U::deserialize_canonical(&mut read_buf)?, V::deserialize_canonical(read_buf)?))
	}
}

impl<N: ArrayLength<u8>> SerializeCanonical for GenericArray<u8, N> {
	fn serialize_canonical(&self, mut write_buf: impl BufMut) -> Result<(), Error> {
		assert_enough_space_for(&write_buf, N::USIZE)?;
		write_buf.put_slice(self);
		Ok(())
	}
}

impl<N: ArrayLength<u8>> DeserializeCanonical for GenericArray<u8, N> {
	fn deserialize_canonical(mut read_buf: impl Buf) -> Result<Self, Error> {
		assert_enough_data_for(&read_buf, N::USIZE)?;
		let mut ret = Self::default();
		read_buf.copy_to_slice(&mut ret);
		Ok(ret)
	}
}

fn assert_enough_space_for(write_buf: &impl BufMut, size: usize) -> Result<(), Error> {
	if write_buf.remaining_mut() < size {
		return Err(Error::WriteBufferFull);
	}
	Ok(())
}

fn assert_enough_data_for(read_buf: &impl Buf, size: usize) -> Result<(), Error> {
	if read_buf.remaining() < size {
		return Err(Error::NotEnoughBytes);
	}
	Ok(())
}
