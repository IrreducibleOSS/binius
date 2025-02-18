// Copyright 2025 Irreducible Inc.

//! The purpose of this module is to enable serialization/deserialization of generic MultivariatePoly implementations
//!
//! The simplest way to do this would be to create an enum with all the possible structs that implement MultivariatePoly
//!
//! This has a few problems, though:
//! - Third party code is not able to define custom transparent polynomials
//! - The enum would inherit, or be forced to enumerate possible type parameters of every struct variant

use std::{collections::HashMap, sync::LazyLock};

use binius_field::{BinaryField128b, TowerField};
use binius_utils::{DeserializeBytes, SerializationError, SerializationMode, SerializeBytes};

use crate::polynomial::MultivariatePoly;

impl<F: TowerField> SerializeBytes for Box<dyn MultivariatePoly<F>> {
	fn serialize(
		&self,
		mut write_buf: impl bytes::BufMut,
		mode: SerializationMode,
	) -> Result<(), SerializationError> {
		self.erased_serialize(&mut write_buf, mode)
	}
}

impl DeserializeBytes for Box<dyn MultivariatePoly<BinaryField128b>> {
	fn deserialize(
		mut read_buf: impl bytes::Buf,
		mode: SerializationMode,
	) -> Result<Self, SerializationError>
	where
		Self: Sized,
	{
		let name = String::deserialize(&mut read_buf, mode)?;
		match REGISTRY.get(name.as_str()) {
			Some(Some(erased_deserialize)) => erased_deserialize(&mut read_buf, mode),
			Some(None) => Err(SerializationError::DeserializerNameConflict { name }),
			None => Err(SerializationError::DeserializerNotImplented),
		}
	}
}

// Using the inventory crate we can collect all deserializers before the main function runs
// This allows third party code to submit their own deserializers as well
inventory::collect!(DeserializerEntry<BinaryField128b>);

static REGISTRY: LazyLock<HashMap<&'static str, Option<ErasedDeserializeBytes<BinaryField128b>>>> =
	LazyLock::new(|| {
		let mut registry = HashMap::new();
		inventory::iter::<DeserializerEntry<BinaryField128b>>
			.into_iter()
			.for_each(|&DeserializerEntry { name, deserializer }| match registry.entry(name) {
				std::collections::hash_map::Entry::Vacant(entry) => {
					entry.insert(Some(deserializer));
				}
				std::collections::hash_map::Entry::Occupied(mut entry) => {
					entry.insert(None);
				}
			});
		registry
	});

impl<F: TowerField> dyn MultivariatePoly<F> {
	pub const fn register_deserializer(
		name: &'static str,
		deserializer: ErasedDeserializeBytes<F>,
	) -> DeserializerEntry<F> {
		DeserializerEntry { name, deserializer }
	}
}

pub struct DeserializerEntry<F: TowerField> {
	name: &'static str,
	deserializer: ErasedDeserializeBytes<F>,
}

type ErasedDeserializeBytes<F> = fn(
	&mut dyn bytes::Buf,
	mode: SerializationMode,
) -> Result<Box<dyn MultivariatePoly<F>>, SerializationError>;
