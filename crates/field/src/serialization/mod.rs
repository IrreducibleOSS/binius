// Copyright 2024-2025 Irreducible Inc.

mod error;
mod impls;

use auto_impl::auto_impl;
use bytes::{Buf, BufMut};
pub use error::Error;

/// Serialize data according to Mode param
#[auto_impl(Box, &)]
pub trait SerializeBytes {
	fn serialize(&self, write_buf: impl BufMut, mode: SerializationMode) -> Result<(), Error>;
}

/// Deserialize data according to Mode param
pub trait DeserializeBytes {
	fn deserialize(read_buf: impl Buf, mode: SerializationMode) -> Result<Self, Error>
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
