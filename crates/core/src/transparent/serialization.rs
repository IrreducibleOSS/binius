// Copyright 2025 Irreducible Inc.

//! The purpose of this module is to enable serialization/deserialization of generic MultivariatePoly implementations
//!
//! The simplest way to do this would be to create an enum with all the possible structs that implement MultivariatePoly
//!
//! This has a few problems, though:
//! - Third party code is not able to define custom transparent polynomials
//! - The enum would inherit, or be forced to enumerate possible type parameters of every struct variant

use std::{collections::HashMap, sync::LazyLock};

use binius_field::{
	serialization::Error, BinaryField128b, DeserializeCanonical, SerializeCanonical, TowerField,
};

use crate::polynomial::MultivariatePoly;

impl<F: TowerField> SerializeCanonical for Box<dyn MultivariatePoly<F>> {
	fn serialize_canonical(
		&self,
		mut write_buf: impl bytes::BufMut,
	) -> Result<(), binius_field::serialization::Error> {
		self.erased_serialize_canonical(&mut write_buf)
	}
}

impl DeserializeCanonical for Box<dyn MultivariatePoly<BinaryField128b>> {
	fn deserialize_canonical(mut read_buf: impl bytes::Buf) -> Result<Self, Error>
	where
		Self: Sized,
	{
		let name = String::deserialize_canonical(&mut read_buf)?;
		match REGISTRY.get(name.as_str()) {
			Some(Some(erased_deserialize_canonical)) => erased_deserialize_canonical(&mut read_buf),
			Some(None) => Err(Error::DeserializerNameConflict { name }),
			None => Err(Error::DeserializerNotImplented),
		}
	}
}

// Using the inventory crate we can collect all deserializers before the main function runs
// This allows third party code to submit their own deserializers as well
inventory::collect!(DeserializerEntry<BinaryField128b>);

static REGISTRY: LazyLock<
	HashMap<&'static str, Option<ErasedDeserializeCanonical<BinaryField128b>>>,
> = LazyLock::new(|| {
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
		deserializer: ErasedDeserializeCanonical<F>,
	) -> DeserializerEntry<F> {
		DeserializerEntry { name, deserializer }
	}
}

pub struct DeserializerEntry<F: TowerField> {
	name: &'static str,
	deserializer: ErasedDeserializeCanonical<F>,
}

type ErasedDeserializeCanonical<F> =
	fn(&mut dyn bytes::Buf) -> Result<Box<dyn MultivariatePoly<F>>, Error>;
