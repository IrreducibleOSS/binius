// Copyright 2025 Irreducible Inc.

use std::fmt;

use binius_utils::{DeserializeBytes, SerializationError, SerializationMode, SerializeBytes};

/// Identifier for a multilinear oracle in a [`super::MultilinearOracleSet`].
///
/// This is essentially an index.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct OracleId(usize);

impl OracleId {
	/// Create an Oracle ID from the index this oracle stored in the
	/// [`super::MultilinearOracleSet`].
	///
	/// Largely an escape hatch and discouraged to use.
	pub const fn from_index(index: usize) -> Self {
		Self(index)
	}

	/// Returns the index in the associated [`super::MultilinearOracleSet`].
	///
	/// Largely an escape hatch and discouraged to use.
	pub const fn index(&self) -> usize {
		self.0
	}

	/// Returns an invalid OracleId.
	pub const fn invalid() -> Self {
		Self(usize::MAX)
	}
}

impl SerializeBytes for OracleId {
	fn serialize(
		&self,
		write_buf: impl bytes::BufMut,
		mode: SerializationMode,
	) -> Result<(), SerializationError> {
		let index: u32 = self.0 as u32;
		u32::serialize(&index, write_buf, mode)
	}
}
impl DeserializeBytes for OracleId {
	fn deserialize(
		read_buf: impl bytes::Buf,
		mode: SerializationMode,
	) -> Result<Self, SerializationError>
	where
		Self: Sized,
	{
		let index = u32::deserialize(read_buf, mode)?;
		Ok(Self(index as usize))
	}
}

impl fmt::Display for OracleId {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		// Delegate to debug.
		write!(f, "{self:?}")
	}
}

// Technically, there must be no notion of a "default" oracle ID. However, there is some code that
// requires that, so until it's fixed this is going to mean INVALID.
impl Default for OracleId {
	fn default() -> Self {
		Self::invalid()
	}
}
