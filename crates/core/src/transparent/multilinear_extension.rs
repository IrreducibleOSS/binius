// Copyright 2024-2025 Irreducible Inc.

use std::{fmt::Debug, ops::Deref};

use binius_field::{
	arch::OptimalUnderlier, as_packed_field::PackedType, packed::pack_slice, BinaryField128b,
	BinaryField16b, BinaryField1b, BinaryField2b, BinaryField32b, BinaryField4b, BinaryField64b,
	BinaryField8b, ExtensionField, PackedField, RepackedExtension, TowerField,
};
use binius_hal::{make_portable_backend, ComputationBackendExt};
use binius_macros::erased_serialize_bytes;
use binius_math::{MLEEmbeddingAdapter, MultilinearExtension, MultilinearPoly};
use binius_utils::{DeserializeBytes, SerializationError, SerializationMode, SerializeBytes};

use crate::polynomial::{Error, MultivariatePoly};

/// A transparent multilinear polynomial defined as the multilinear extension over a small
/// hypercube.
///
/// Multilinear polynomials are considered transparent if they can be succinctly evaluated. While
/// evaluation of multilinear extensions is generally exponential in the number of variables, when
/// the number of variables is very small, and thus the evaluation hypercube is small, we can
/// consider such a multilinear extension to be transparent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MultilinearExtensionTransparent<P, PE, Data = Vec<P>>
where
	P: PackedField,
	PE: PackedField,
	PE::Scalar: ExtensionField<P::Scalar>,
	Data: Deref<Target = [P]>,
{
	data: MLEEmbeddingAdapter<P, PE, Data>,
}

impl<P, PE, Data> SerializeBytes for MultilinearExtensionTransparent<P, PE, Data>
where
	P: PackedField,
	PE: RepackedExtension<P>,
	PE::Scalar: TowerField + ExtensionField<P::Scalar>,
	Data: Deref<Target = [P]> + Debug + Send + Sync,
{
	fn serialize(
		&self,
		write_buf: &mut dyn bytes::BufMut,
		mode: SerializationMode,
	) -> Result<(), SerializationError> {
		let elems = PE::iter_slice(
			self.data
				.packed_evals()
				.expect("Evals should always be available here"),
		)
		.collect::<Vec<_>>();
		SerializeBytes::serialize(&elems, write_buf, mode)
	}
}

inventory::submit! {
	<dyn MultivariatePoly<BinaryField128b>>::register_deserializer(
		"MultilinearExtensionTransparent",
		|buf, mode| {
			type U = OptimalUnderlier;
			type F = BinaryField128b;
			type P = PackedType<U, F>;
			let hypercube_evals = Vec::<F>::deserialize(&mut *buf, mode)?;
			let result: Box<dyn MultivariatePoly<F>> = if let Some(packed_evals) = try_pack_slice(&hypercube_evals) {
				Box::new(MultilinearExtensionTransparent::<PackedType<U, BinaryField1b>, P, _>::from_values(packed_evals).unwrap())
			} else if let Some(packed_evals) = try_pack_slice(&hypercube_evals) {
				Box::new(MultilinearExtensionTransparent::<PackedType<U, BinaryField2b>, P, _>::from_values(packed_evals).unwrap())
			} else if let Some(packed_evals) = try_pack_slice(&hypercube_evals) {
				Box::new(MultilinearExtensionTransparent::<PackedType<U, BinaryField4b>, P, _>::from_values(packed_evals).unwrap())
			} else if let Some(packed_evals) = try_pack_slice(&hypercube_evals) {
				Box::new(MultilinearExtensionTransparent::<PackedType<U, BinaryField8b>, P, _>::from_values(packed_evals).unwrap())
			} else if let Some(packed_evals) = try_pack_slice(&hypercube_evals) {
				Box::new(MultilinearExtensionTransparent::<PackedType<U, BinaryField16b>, P, _>::from_values(packed_evals).unwrap())
			} else if let Some(packed_evals) = try_pack_slice(&hypercube_evals) {
				Box::new(MultilinearExtensionTransparent::<PackedType<U, BinaryField32b>, P, _>::from_values(packed_evals).unwrap())
			} else if let Some(packed_evals) = try_pack_slice(&hypercube_evals) {
				Box::new(MultilinearExtensionTransparent::<PackedType<U, BinaryField64b>, P, _>::from_values(packed_evals).unwrap())
			} else {
				Box::new(MultilinearExtensionTransparent::<P, P, _>::from_values(pack_slice(&hypercube_evals)).unwrap())
			};
			Ok(result)
		}
	)
}

fn try_pack_slice<PS, F>(xs: &[F]) -> Option<Vec<PS>>
where
	PS: PackedField,
	F: ExtensionField<PS::Scalar>,
{
	Some(pack_slice(
		&xs.iter()
			.copied()
			.map(TryInto::try_into)
			.collect::<Result<Vec<_>, _>>()
			.ok()?,
	))
}

impl<P, PE, Data> MultilinearExtensionTransparent<P, PE, Data>
where
	P: PackedField,
	PE: PackedField,
	PE::Scalar: ExtensionField<P::Scalar>,
	Data: Deref<Target = [P]>,
{
	pub fn from_values(values: Data) -> Result<Self, Error> {
		let mle = MultilinearExtension::from_values_generic(values)?;
		Ok(Self {
			data: mle.specialize(),
		})
	}

	/// Create a new `MultilinearExtensionTransparent` from a set of values and a possibly smaller number of variables.
	pub fn from_values_and_mu(values: Data, n_vars: usize) -> Result<Self, Error> {
		let mle = MultilinearExtension::new(n_vars, values)?;
		Ok(Self {
			data: mle.specialize(),
		})
	}
}

#[erased_serialize_bytes]
impl<F, P, PE, Data> MultivariatePoly<F> for MultilinearExtensionTransparent<P, PE, Data>
where
	F: TowerField + ExtensionField<P::Scalar>,
	P: PackedField,
	PE: PackedField<Scalar = F> + RepackedExtension<P>,
	Data: Deref<Target = [P]> + Send + Sync + Debug,
{
	fn n_vars(&self) -> usize {
		self.data.n_vars()
	}

	fn degree(&self) -> usize {
		self.data.n_vars()
	}

	fn evaluate(&self, query: &[F]) -> Result<F, Error> {
		// Use the portable CPU backend because the size of the hypercube is small by struct
		// assumption.
		let backend = make_portable_backend();
		let query = backend.multilinear_query(query)?;
		Ok(self.data.evaluate(query.to_ref())?)
	}

	fn binary_tower_level(&self) -> usize {
		F::TOWER_LEVEL - self.data.log_extension_degree()
	}
}
