// Copyright 2024-2025 Irreducible Inc.

use binius_field::{ExtensionField, TowerField};
use binius_macros::{erased_serialize_canonical, DeserializeCanonical, SerializeCanonical};
use binius_utils::bail;

use crate::polynomial::{Error, MultivariatePoly};

/// A constant polynomial.
#[derive(Debug, Copy, Clone, SerializeCanonical, DeserializeCanonical)]
pub struct Constant<F: TowerField> {
	n_vars: usize,
	value: F,
	tower_level: usize,
}

inventory::submit! {
	<dyn MultivariatePoly<binius_field::BinaryField128b>>::register_deserializer(
		"Constant",
		|buf: &mut dyn bytes::Buf| {
			let deserialized = <Constant<binius_field::BinaryField128b> as binius_field::DeserializeCanonical>::deserialize_canonical(&mut *buf)?;
			Ok(Box::new(deserialized))
		}
	)
}

impl<F: TowerField> Constant<F> {
	pub fn new<FS: TowerField>(n_vars: usize, value: FS) -> Self
	where
		F: ExtensionField<FS>,
	{
		Self {
			value: value.into(),
			tower_level: FS::TOWER_LEVEL,
			n_vars,
		}
	}
}

#[erased_serialize_canonical]
impl<F: TowerField> MultivariatePoly<F> for Constant<F> {
	fn n_vars(&self) -> usize {
		self.n_vars
	}

	fn degree(&self) -> usize {
		0
	}

	fn evaluate(&self, query: &[F]) -> Result<F, Error> {
		if query.len() != self.n_vars {
			bail!(Error::IncorrectQuerySize {
				expected: self.n_vars,
			});
		}
		Ok(self.value)
	}

	fn binary_tower_level(&self) -> usize {
		self.tower_level
	}
}
