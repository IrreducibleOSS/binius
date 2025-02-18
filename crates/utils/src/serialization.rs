// Copyright 2024-2025 Irreducible Inc.

use auto_impl::auto_impl;
use bytes::{Buf, BufMut};
use thiserror::Error;

/// Serialize data according to Mode param
#[auto_impl(Box, &)]
pub trait SerializeBytes {
	fn serialize(
		&self,
		write_buf: impl BufMut,
		mode: SerializationMode,
	) -> Result<(), SerializationError>;
}

/// Deserialize data according to Mode param
pub trait DeserializeBytes {
	fn deserialize(read_buf: impl Buf, mode: SerializationMode) -> Result<Self, SerializationError>
	where
		Self: Sized;
}

/// Specifies serialization/deserialization behavior
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SerializationMode {
	/// This mode is faster, and serializes to the underlying bytes
	Native,
	/// Will first convert any tower fields into the Fan-Paar field equivalent
	CanonicalTower,
}

#[derive(Error, Debug, Clone)]
pub enum SerializationError {
	#[error("Write buffer is full")]
	WriteBufferFull,
	#[error("Not enough data in read buffer to deserialize")]
	NotEnoughBytes,
	#[error("Unknown enum variant index {name}::{index}")]
	UnknownEnumVariant { name: &'static str, index: u8 },
	#[error("Serialization has not been implemented")]
	SerializationNotImplemented,
	#[error("Deserializer has not been implemented")]
	DeserializerNotImplented,
	#[error("Multiple deserializers with the same name {name} has been registered")]
	DeserializerNameConflict { name: String },
	#[error("FromUtf8Error: {0}")]
	FromUtf8Error(#[from] std::string::FromUtf8Error),
}

// Copyright 2025 Irreducible Inc.

use generic_array::{ArrayLength, GenericArray};

impl<T: DeserializeBytes> DeserializeBytes for Box<T> {
	fn deserialize(read_buf: impl Buf, mode: SerializationMode) -> Result<Self, SerializationError>
	where
		Self: Sized,
	{
		Ok(Self::new(T::deserialize(read_buf, mode)?))
	}
}

impl SerializeBytes for usize {
	fn serialize(
		&self,
		mut write_buf: impl BufMut,
		mode: SerializationMode,
	) -> Result<(), SerializationError> {
		SerializeBytes::serialize(&(*self as u64), &mut write_buf, mode)
	}
}

impl DeserializeBytes for usize {
	fn deserialize(
		mut read_buf: impl Buf,
		mode: SerializationMode,
	) -> Result<Self, SerializationError>
	where
		Self: Sized,
	{
		let value: u64 = DeserializeBytes::deserialize(&mut read_buf, mode)?;
		Ok(value as Self)
	}
}

impl SerializeBytes for u128 {
	fn serialize(
		&self,
		mut write_buf: impl BufMut,
		_mode: SerializationMode,
	) -> Result<(), SerializationError> {
		assert_enough_space_for(&write_buf, std::mem::size_of::<Self>())?;
		write_buf.put_u128_le(*self);
		Ok(())
	}
}

impl DeserializeBytes for u128 {
	fn deserialize(
		mut read_buf: impl Buf,
		_mode: SerializationMode,
	) -> Result<Self, SerializationError>
	where
		Self: Sized,
	{
		assert_enough_data_for(&read_buf, std::mem::size_of::<Self>())?;
		Ok(read_buf.get_u128_le())
	}
}

impl SerializeBytes for u64 {
	fn serialize(
		&self,
		mut write_buf: impl BufMut,
		_mode: SerializationMode,
	) -> Result<(), SerializationError> {
		assert_enough_space_for(&write_buf, std::mem::size_of::<Self>())?;
		write_buf.put_u64_le(*self);
		Ok(())
	}
}

impl DeserializeBytes for u64 {
	fn deserialize(
		mut read_buf: impl Buf,
		_mode: SerializationMode,
	) -> Result<Self, SerializationError>
	where
		Self: Sized,
	{
		assert_enough_data_for(&read_buf, std::mem::size_of::<Self>())?;
		Ok(read_buf.get_u64_le())
	}
}

impl SerializeBytes for u32 {
	fn serialize(
		&self,
		mut write_buf: impl BufMut,
		_mode: SerializationMode,
	) -> Result<(), SerializationError> {
		assert_enough_space_for(&write_buf, std::mem::size_of::<Self>())?;
		write_buf.put_u32_le(*self);
		Ok(())
	}
}

impl DeserializeBytes for u32 {
	fn deserialize(
		mut read_buf: impl Buf,
		_mode: SerializationMode,
	) -> Result<Self, SerializationError>
	where
		Self: Sized,
	{
		assert_enough_data_for(&read_buf, std::mem::size_of::<Self>())?;
		Ok(read_buf.get_u32_le())
	}
}

impl SerializeBytes for u16 {
	fn serialize(
		&self,
		mut write_buf: impl BufMut,
		_mode: SerializationMode,
	) -> Result<(), SerializationError> {
		assert_enough_space_for(&write_buf, std::mem::size_of::<Self>())?;
		write_buf.put_u16_le(*self);
		Ok(())
	}
}

impl DeserializeBytes for u16 {
	fn deserialize(
		mut read_buf: impl Buf,
		_mode: SerializationMode,
	) -> Result<Self, SerializationError>
	where
		Self: Sized,
	{
		assert_enough_data_for(&read_buf, std::mem::size_of::<Self>())?;
		Ok(read_buf.get_u16_le())
	}
}

impl SerializeBytes for u8 {
	fn serialize(
		&self,
		mut write_buf: impl BufMut,
		_mode: SerializationMode,
	) -> Result<(), SerializationError> {
		assert_enough_space_for(&write_buf, std::mem::size_of::<Self>())?;
		write_buf.put_u8(*self);
		Ok(())
	}
}

impl DeserializeBytes for u8 {
	fn deserialize(
		mut read_buf: impl Buf,
		_mode: SerializationMode,
	) -> Result<Self, SerializationError>
	where
		Self: Sized,
	{
		assert_enough_data_for(&read_buf, std::mem::size_of::<Self>())?;
		Ok(read_buf.get_u8())
	}
}

impl SerializeBytes for bool {
	fn serialize(
		&self,
		write_buf: impl BufMut,
		mode: SerializationMode,
	) -> Result<(), SerializationError> {
		u8::serialize(&(*self as u8), write_buf, mode)
	}
}

impl DeserializeBytes for bool {
	fn deserialize(read_buf: impl Buf, mode: SerializationMode) -> Result<Self, SerializationError>
	where
		Self: Sized,
	{
		Ok(u8::deserialize(read_buf, mode)? != 0)
	}
}

impl<T> SerializeBytes for std::marker::PhantomData<T> {
	fn serialize(
		&self,
		_write_buf: impl BufMut,
		_mode: SerializationMode,
	) -> Result<(), SerializationError> {
		Ok(())
	}
}

impl<T> DeserializeBytes for std::marker::PhantomData<T> {
	fn deserialize(
		_read_buf: impl Buf,
		_mode: SerializationMode,
	) -> Result<Self, SerializationError>
	where
		Self: Sized,
	{
		Ok(Self)
	}
}

impl SerializeBytes for &str {
	fn serialize(
		&self,
		mut write_buf: impl BufMut,
		mode: SerializationMode,
	) -> Result<(), SerializationError> {
		let bytes = self.as_bytes();
		SerializeBytes::serialize(&bytes.len(), &mut write_buf, mode)?;
		assert_enough_space_for(&write_buf, bytes.len())?;
		write_buf.put_slice(bytes);
		Ok(())
	}
}

impl SerializeBytes for String {
	fn serialize(
		&self,
		mut write_buf: impl BufMut,
		mode: SerializationMode,
	) -> Result<(), SerializationError> {
		SerializeBytes::serialize(&self.as_str(), &mut write_buf, mode)
	}
}

impl DeserializeBytes for String {
	fn deserialize(
		mut read_buf: impl Buf,
		mode: SerializationMode,
	) -> Result<Self, SerializationError>
	where
		Self: Sized,
	{
		let len = DeserializeBytes::deserialize(&mut read_buf, mode)?;
		assert_enough_data_for(&read_buf, len)?;
		Ok(Self::from_utf8(read_buf.copy_to_bytes(len).to_vec())?)
	}
}

impl<T: SerializeBytes> SerializeBytes for Vec<T> {
	fn serialize(
		&self,
		mut write_buf: impl BufMut,
		mode: SerializationMode,
	) -> Result<(), SerializationError> {
		SerializeBytes::serialize(&self.len(), &mut write_buf, mode)?;
		self.iter()
			.try_for_each(|item| SerializeBytes::serialize(item, &mut write_buf, mode))
	}
}

impl<T: DeserializeBytes> DeserializeBytes for Vec<T> {
	fn deserialize(
		mut read_buf: impl Buf,
		mode: SerializationMode,
	) -> Result<Self, SerializationError>
	where
		Self: Sized,
	{
		let len: usize = DeserializeBytes::deserialize(&mut read_buf, mode)?;
		(0..len)
			.map(|_| DeserializeBytes::deserialize(&mut read_buf, mode))
			.collect()
	}
}

impl<T: SerializeBytes> SerializeBytes for Option<T> {
	fn serialize(
		&self,
		mut write_buf: impl BufMut,
		mode: SerializationMode,
	) -> Result<(), SerializationError> {
		match self {
			Some(value) => {
				SerializeBytes::serialize(&true, &mut write_buf, mode)?;
				SerializeBytes::serialize(value, &mut write_buf, mode)?;
			}
			None => {
				SerializeBytes::serialize(&false, write_buf, mode)?;
			}
		}
		Ok(())
	}
}

impl<T: DeserializeBytes> DeserializeBytes for Option<T> {
	fn deserialize(
		mut read_buf: impl Buf,
		mode: SerializationMode,
	) -> Result<Self, SerializationError>
	where
		Self: Sized,
	{
		Ok(match bool::deserialize(&mut read_buf, mode)? {
			true => Some(T::deserialize(&mut read_buf, mode)?),
			false => None,
		})
	}
}

impl<U: SerializeBytes, V: SerializeBytes> SerializeBytes for (U, V) {
	fn serialize(
		&self,
		mut write_buf: impl BufMut,
		mode: SerializationMode,
	) -> Result<(), SerializationError> {
		U::serialize(&self.0, &mut write_buf, mode)?;
		V::serialize(&self.1, write_buf, mode)
	}
}

impl<U: DeserializeBytes, V: DeserializeBytes> DeserializeBytes for (U, V) {
	fn deserialize(
		mut read_buf: impl Buf,
		mode: SerializationMode,
	) -> Result<Self, SerializationError>
	where
		Self: Sized,
	{
		Ok((U::deserialize(&mut read_buf, mode)?, V::deserialize(read_buf, mode)?))
	}
}

impl<N: ArrayLength<u8>> SerializeBytes for GenericArray<u8, N> {
	fn serialize(
		&self,
		mut write_buf: impl BufMut,
		_mode: SerializationMode,
	) -> Result<(), SerializationError> {
		assert_enough_space_for(&write_buf, N::USIZE)?;
		write_buf.put_slice(self);
		Ok(())
	}
}

impl<N: ArrayLength<u8>> DeserializeBytes for GenericArray<u8, N> {
	fn deserialize(
		mut read_buf: impl Buf,
		_mode: SerializationMode,
	) -> Result<Self, SerializationError> {
		assert_enough_data_for(&read_buf, N::USIZE)?;
		let mut ret = Self::default();
		read_buf.copy_to_slice(&mut ret);
		Ok(ret)
	}
}

#[inline]
fn assert_enough_space_for(write_buf: &impl BufMut, size: usize) -> Result<(), SerializationError> {
	if write_buf.remaining_mut() < size {
		return Err(SerializationError::WriteBufferFull);
	}
	Ok(())
}

#[inline]
fn assert_enough_data_for(read_buf: &impl Buf, size: usize) -> Result<(), SerializationError> {
	if read_buf.remaining() < size {
		return Err(SerializationError::NotEnoughBytes);
	}
	Ok(())
}

#[cfg(test)]
mod tests {
	use generic_array::typenum::U32;
	use rand::{rngs::StdRng, RngCore, SeedableRng};

	use super::*;

	#[test]
	fn test_generic_array_serialize_deserialize() {
		let mut rng = StdRng::seed_from_u64(0);

		let mut data = GenericArray::<u8, U32>::default();
		rng.fill_bytes(&mut data);

		let mut buf = Vec::new();
		data.serialize(&mut buf, SerializationMode::Native).unwrap();

		let data_deserialized =
			GenericArray::<u8, U32>::deserialize(&mut buf.as_slice(), SerializationMode::Native)
				.unwrap();
		assert_eq!(data_deserialized, data);
	}
}
