// Copyright 2024-2025 Irreducible Inc.

use bytes::{Buf, BufMut};
use generic_array::{ArrayLength, GenericArray};

#[derive(Clone, thiserror::Error, Debug)]
pub enum Error {
	#[error("Write buffer is full")]
	WriteBufferFull,
	#[error("Not enough data in read buffer to deserialize")]
	NotEnoughBytes,
	#[error("Unknown enum variant index {name}::{index}")]
	UnknownEnumVariant { name: &'static str, index: u8 },
	#[error("FromUtf8Error: {0}")]
	FromUtf8Error(#[from] std::string::FromUtf8Error),
}

/// Represents type that can be serialized to a byte buffer.
pub trait SerializeBytes {
	fn serialize(&self, write_buf: impl BufMut) -> Result<(), Error>;
}

/// Represents type that can be deserialized from a byte buffer.
pub trait DeserializeBytes {
	fn deserialize(read_buf: impl Buf) -> Result<Self, Error>
	where
		Self: Sized;
}

impl<N: ArrayLength<u8>> SerializeBytes for GenericArray<u8, N> {
	fn serialize(&self, mut write_buf: impl BufMut) -> Result<(), Error> {
		if write_buf.remaining_mut() < N::USIZE {
			return Err(Error::WriteBufferFull);
		}
		write_buf.put_slice(self);
		Ok(())
	}
}

impl<N: ArrayLength<u8>> DeserializeBytes for GenericArray<u8, N> {
	fn deserialize(mut read_buf: impl Buf) -> Result<Self, Error> {
		if read_buf.remaining() < N::USIZE {
			return Err(Error::NotEnoughBytes);
		}

		let mut ret = Self::default();
		read_buf.copy_to_slice(&mut ret);
		Ok(ret)
	}
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
		data.serialize(&mut buf).unwrap();

		let data_deserialized = GenericArray::<u8, U32>::deserialize(&mut buf.as_slice()).unwrap();
		assert_eq!(data_deserialized, data);
	}
}
