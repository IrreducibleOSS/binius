// Copyright 2024-2025 Irreducible Inc.

use std::{fmt::Debug, ops::Deref};

use binius_field::{
	arch::OptimalUnderlier, as_packed_field::PackedType, packed::pack_slice, BinaryField128b,
	DeserializeCanonical, ExtensionField, PackedField, RepackedExtension, SerializeCanonical,
	TowerField,
};
use binius_hal::{make_portable_backend, ComputationBackendExt};
use binius_macros::erased_serialize_canonical;
use binius_math::{MLEEmbeddingAdapter, MultilinearExtension, MultilinearPoly};

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

impl<P, PE, Data> SerializeCanonical for MultilinearExtensionTransparent<P, PE, Data>
where
	P: PackedField,
	PE: RepackedExtension<P>,
	PE::Scalar: TowerField + ExtensionField<P::Scalar>,
	Data: Deref<Target = [P]> + Debug + Send + Sync,
{
	fn serialize_canonical(
		&self,
		write_buf: impl bytes::BufMut,
	) -> Result<(), binius_field::serialization::Error> {
		let elems = PE::iter_slice(
			self.data
				.packed_evals()
				.expect("Evals should always be available here"),
		)
		.collect::<Vec<_>>();
		SerializeCanonical::serialize_canonical(&elems, write_buf)
	}
}

inventory::submit! {
	<dyn MultivariatePoly<binius_field::BinaryField128b>>::register_deserializer(
		"MultilinearExtensionTransparent",
		|buf: &mut dyn bytes::Buf| {
			type U = OptimalUnderlier;
			type F = BinaryField128b;
			type P = PackedType<U, F>;
			let hypercube_evals: Vec<F> = DeserializeCanonical::deserialize_canonical(&mut *buf)?;
			let result: Box<dyn MultivariatePoly<F>> = if let Some(packed_evals) = try_pack_slice(&hypercube_evals) {
				Box::new(MultilinearExtensionTransparent::<PackedType<U, binius_field::BinaryField1b>, P, _>::from_values(packed_evals).unwrap())
			} else if let Some(packed_evals) = try_pack_slice(&hypercube_evals) {
				Box::new(MultilinearExtensionTransparent::<PackedType<U, binius_field::BinaryField2b>, P, _>::from_values(packed_evals).unwrap())
			} else if let Some(packed_evals) = try_pack_slice(&hypercube_evals) {
				Box::new(MultilinearExtensionTransparent::<PackedType<U, binius_field::BinaryField4b>, P, _>::from_values(packed_evals).unwrap())
			} else if let Some(packed_evals) = try_pack_slice(&hypercube_evals) {
				Box::new(MultilinearExtensionTransparent::<PackedType<U, binius_field::BinaryField8b>, P, _>::from_values(packed_evals).unwrap())
			} else if let Some(packed_evals) = try_pack_slice(&hypercube_evals) {
				Box::new(MultilinearExtensionTransparent::<PackedType<U, binius_field::BinaryField16b>, P, _>::from_values(packed_evals).unwrap())
			} else if let Some(packed_evals) = try_pack_slice(&hypercube_evals) {
				Box::new(MultilinearExtensionTransparent::<PackedType<U, binius_field::BinaryField32b>, P, _>::from_values(packed_evals).unwrap())
			} else if let Some(packed_evals) = try_pack_slice(&hypercube_evals) {
				Box::new(MultilinearExtensionTransparent::<PackedType<U, binius_field::BinaryField64b>, P, _>::from_values(packed_evals).unwrap())
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

#[erased_serialize_canonical]
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
