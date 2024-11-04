// Copyright 2023-2024 Irreducible Inc.

use super::{Error, MultilinearExtension, MultilinearPoly, MultilinearQueryRef};
use binius_field::{
	packed::{
		get_packed_slice, get_packed_slice_unchecked, set_packed_slice, set_packed_slice_unchecked,
	},
	ExtensionField, Field, PackedField, RepackedExtension,
};
use binius_utils::bail;
use std::{fmt::Debug, marker::PhantomData, ops::Deref, sync::Arc};

/// An adapter for [`MultilinearExtension`] that implements [`MultilinearPoly`] over a packed
/// extension field.
///
/// This struct implements `MultilinearPoly` for an extension field of the base field that the
/// multilinear extension is defined over.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MLEEmbeddingAdapter<P, PE, Data = Vec<P>>(
	MultilinearExtension<P, Data>,
	PhantomData<PE>,
)
where
	P: PackedField,
	PE: PackedField,
	PE::Scalar: ExtensionField<P::Scalar>,
	Data: Deref<Target = [P]>;

impl<'a, P, PE, Data> MLEEmbeddingAdapter<P, PE, Data>
where
	P: PackedField,
	PE: PackedField + RepackedExtension<P>,
	PE::Scalar: ExtensionField<P::Scalar>,
	Data: Deref<Target = [P]> + Send + Sync + Debug + 'a,
{
	pub fn upcast_arc_dyn(self) -> Arc<dyn MultilinearPoly<PE> + Send + Sync + 'a> {
		Arc::new(self)
	}
}

impl<P, PE, Data> From<MultilinearExtension<P, Data>> for MLEEmbeddingAdapter<P, PE, Data>
where
	P: PackedField,
	PE: PackedField,
	PE::Scalar: ExtensionField<P::Scalar>,
	Data: Deref<Target = [P]>,
{
	fn from(inner: MultilinearExtension<P, Data>) -> Self {
		Self(inner, PhantomData)
	}
}

impl<P, PE, Data> AsRef<MultilinearExtension<P, Data>> for MLEEmbeddingAdapter<P, PE, Data>
where
	P: PackedField,
	PE: PackedField,
	PE::Scalar: ExtensionField<P::Scalar>,
	Data: Deref<Target = [P]>,
{
	fn as_ref(&self) -> &MultilinearExtension<P, Data> {
		&self.0
	}
}

impl<P, PE, Data> MultilinearPoly<PE> for MLEEmbeddingAdapter<P, PE, Data>
where
	P: PackedField + Debug,
	PE: PackedField + RepackedExtension<P>,
	PE::Scalar: ExtensionField<P::Scalar>,
	Data: Deref<Target = [P]> + Send + Sync + Debug,
{
	fn n_vars(&self) -> usize {
		self.0.n_vars()
	}

	fn log_extension_degree(&self) -> usize {
		PE::Scalar::LOG_DEGREE
	}

	fn evaluate_on_hypercube(&self, index: usize) -> Result<PE::Scalar, Error> {
		let eval = self.0.evaluate_on_hypercube(index)?;
		Ok(eval.into())
	}

	fn evaluate_on_hypercube_and_scale(
		&self,
		index: usize,
		scalar: PE::Scalar,
	) -> Result<PE::Scalar, Error> {
		let eval = self.0.evaluate_on_hypercube(index)?;
		Ok(scalar * eval)
	}

	fn evaluate(&self, query: MultilinearQueryRef<PE>) -> Result<PE::Scalar, Error> {
		self.0.evaluate(query)
	}

	fn evaluate_partial_low(
		&self,
		query: MultilinearQueryRef<PE>,
	) -> Result<MultilinearExtension<PE>, Error> {
		self.0.evaluate_partial_low(query)
	}

	fn evaluate_partial_high(
		&self,
		query: MultilinearQueryRef<PE>,
	) -> Result<MultilinearExtension<PE>, Error> {
		self.0.evaluate_partial_high(query)
	}

	fn subcube_inner_products(
		&self,
		query: MultilinearQueryRef<PE>,
		subcube_vars: usize,
		subcube_index: usize,
		inner_products: &mut [PE],
	) -> Result<(), Error> {
		let query_n_vars = query.n_vars();
		if query_n_vars + subcube_vars > self.n_vars() {
			bail!(Error::ArgumentRangeError {
				arg: "query.n_vars() + subcube_vars".into(),
				range: 0..self.n_vars(),
			});
		}

		let max_index = 1 << (self.n_vars() - query_n_vars - subcube_vars);
		if subcube_index >= max_index {
			bail!(Error::ArgumentRangeError {
				arg: "subcube_index".into(),
				range: 0..max_index,
			});
		}

		let correct_len = 1 << subcube_vars.saturating_sub(PE::LOG_WIDTH);
		if inner_products.len() != correct_len {
			bail!(Error::ArgumentRangeError {
				arg: "evals.len()".to_string(),
				range: correct_len..correct_len + 1,
			});
		}

		// REVIEW: not spending effort to optimize this as the future of switchover
		//         is somewhat unclear in light of univariate skip
		let subcube_start = subcube_index << (query_n_vars + subcube_vars);
		for scalar_index in 0..1 << subcube_vars {
			let evals_start = subcube_start + (scalar_index << query_n_vars);
			let mut inner_product = PE::Scalar::ZERO;
			for i in 0..1 << query_n_vars {
				inner_product += get_packed_slice(query.expansion(), i)
					* get_packed_slice(self.0.evals(), evals_start + i);
			}

			set_packed_slice(inner_products, scalar_index, inner_product);
		}

		Ok(())
	}

	fn subcube_evals(
		&self,
		subcube_vars: usize,
		subcube_index: usize,
		log_embedding_degree: usize,
		evals: &mut [PE],
	) -> Result<(), Error> {
		let log_extension_degree = PE::Scalar::LOG_DEGREE;

		if subcube_vars > self.n_vars() {
			bail!(Error::ArgumentRangeError {
				arg: "subcube_vars".to_string(),
				range: 0..self.n_vars() + 1,
			});
		}

		// Check that chosen embedding subfield is large enough.
		// We also use a stack allocated array of bases, which imposes
		// a maximum tower height restriction.
		const MAX_TOWER_HEIGHT: usize = 7;
		if log_embedding_degree > log_extension_degree.min(MAX_TOWER_HEIGHT) {
			bail!(Error::LogEmbeddingDegreeTooLarge {
				log_embedding_degree
			});
		}

		let correct_len = 1 << subcube_vars.saturating_sub(log_embedding_degree + PE::LOG_WIDTH);
		if evals.len() != correct_len {
			bail!(Error::ArgumentRangeError {
				arg: "evals.len()".to_string(),
				range: correct_len..correct_len + 1,
			});
		}

		let max_index = 1 << (self.n_vars() - subcube_vars);
		if subcube_index >= max_index {
			bail!(Error::ArgumentRangeError {
				arg: "subcube_index".to_string(),
				range: 0..max_index,
			});
		}

		let subcube_start = subcube_index << subcube_vars;

		if log_embedding_degree == 0 {
			// One-to-one embedding can bypass the extension field construction overhead.
			for i in 0..1 << subcube_vars {
				// Safety: subcube_index < max_index check
				let scalar =
					unsafe { get_packed_slice_unchecked(self.0.evals(), subcube_start + i) };

				let extension_scalar = scalar.into();

				// Safety: i < 1 << min(0, subcube_vars) <= correct_len * PE::WIDTH
				unsafe {
					set_packed_slice_unchecked(evals, i, extension_scalar);
				}
			}
		} else {
			// For many-to-one embedding, use ExtensionField::from_bases_sparse
			let mut bases = [P::Scalar::default(); 1 << MAX_TOWER_HEIGHT];
			let bases = &mut bases[0..1 << log_embedding_degree];

			let bases_count = 1 << log_embedding_degree.min(subcube_vars);
			for i in 0..1 << subcube_vars.saturating_sub(log_embedding_degree) {
				for (j, base) in bases[..bases_count].iter_mut().enumerate() {
					// Safety: i > 0 iff log_embedding_degree < subcube_vars and subcube_index < max_index check
					*base = unsafe {
						get_packed_slice_unchecked(
							self.0.evals(),
							subcube_start + (i << log_embedding_degree) + j,
						)
					};
				}

				let extension_scalar = PE::Scalar::from_bases_sparse(
					bases,
					log_extension_degree - log_embedding_degree,
				)?;

				// Safety: i < 1 << min(0, subcube_vars - log_embedding_degree) <= correct_len * PE::WIDTH
				unsafe {
					set_packed_slice_unchecked(evals, i, extension_scalar);
				}
			}
		}

		Ok(())
	}

	fn packed_evals(&self) -> Option<&[PE]> {
		Some(PE::cast_exts(self.0.evals()))
	}
}

impl<P, Data> MultilinearExtension<P, Data>
where
	P: PackedField,
	Data: Deref<Target = [P]>,
{
	pub fn specialize<PE>(self) -> MLEEmbeddingAdapter<P, PE, Data>
	where
		PE: PackedField,
		PE::Scalar: ExtensionField<P::Scalar>,
	{
		MLEEmbeddingAdapter::from(self)
	}
}

impl<'a, P, Data> MultilinearExtension<P, Data>
where
	P: PackedField,
	Data: Deref<Target = [P]> + Send + Sync + Debug + 'a,
{
	pub fn specialize_arc_dyn<PE>(self) -> Arc<dyn MultilinearPoly<PE> + Send + Sync + 'a>
	where
		PE: PackedField + RepackedExtension<P>,
		PE::Scalar: ExtensionField<P::Scalar>,
	{
		self.specialize().upcast_arc_dyn()
	}
}

/// An adapter for [`MultilinearExtension`] that implements [`MultilinearPoly`] over the same
/// packed field that the [`MultilinearExtension`] stores evaluations in.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MLEDirectAdapter<P, Data = Vec<P>>(MultilinearExtension<P, Data>)
where
	P: PackedField,
	Data: Deref<Target = [P]>;

impl<'a, P, Data> MLEDirectAdapter<P, Data>
where
	P: PackedField,
	Data: Deref<Target = [P]> + Send + Sync + Debug + 'a,
{
	pub fn upcast_arc_dyn(self) -> Arc<dyn MultilinearPoly<P> + Send + Sync + 'a> {
		Arc::new(self)
	}
}

impl<P, Data> From<MultilinearExtension<P, Data>> for MLEDirectAdapter<P, Data>
where
	P: PackedField,
	Data: Deref<Target = [P]>,
{
	fn from(inner: MultilinearExtension<P, Data>) -> Self {
		Self(inner)
	}
}

impl<P, Data> AsRef<MultilinearExtension<P, Data>> for MLEDirectAdapter<P, Data>
where
	P: PackedField,
	Data: Deref<Target = [P]>,
{
	fn as_ref(&self) -> &MultilinearExtension<P, Data> {
		&self.0
	}
}

impl<F, P, Data> MultilinearPoly<P> for MLEDirectAdapter<P, Data>
where
	F: Field,
	P: PackedField<Scalar = F> + Debug,
	Data: Deref<Target = [P]> + Send + Sync + Debug,
{
	#[inline]
	fn n_vars(&self) -> usize {
		self.0.n_vars()
	}

	#[inline]
	fn log_extension_degree(&self) -> usize {
		0
	}

	fn evaluate_on_hypercube(&self, index: usize) -> Result<F, Error> {
		self.0.evaluate_on_hypercube(index)
	}

	fn evaluate_on_hypercube_and_scale(&self, index: usize, scalar: F) -> Result<F, Error> {
		let eval = self.0.evaluate_on_hypercube(index)?;
		Ok(scalar * eval)
	}

	fn evaluate(&self, query: MultilinearQueryRef<P>) -> Result<F, Error> {
		self.0.evaluate(query)
	}

	fn evaluate_partial_low(
		&self,
		query: MultilinearQueryRef<P>,
	) -> Result<MultilinearExtension<P>, Error> {
		self.0.evaluate_partial_low(query)
	}

	fn evaluate_partial_high(
		&self,
		query: MultilinearQueryRef<P>,
	) -> Result<MultilinearExtension<P>, Error> {
		self.0.evaluate_partial_high(query)
	}

	fn subcube_inner_products(
		&self,
		query: MultilinearQueryRef<P>,
		subcube_vars: usize,
		subcube_index: usize,
		inner_products: &mut [P],
	) -> Result<(), Error> {
		let query_n_vars = query.n_vars();
		if query_n_vars + subcube_vars > self.n_vars() {
			bail!(Error::ArgumentRangeError {
				arg: "query.n_vars() + subcube_vars".into(),
				range: 0..self.n_vars(),
			});
		}

		let max_index = 1 << (self.n_vars() - query_n_vars - subcube_vars);
		if subcube_index >= max_index {
			bail!(Error::ArgumentRangeError {
				arg: "subcube_index".into(),
				range: 0..max_index,
			});
		}

		let correct_len = 1 << subcube_vars.saturating_sub(P::LOG_WIDTH);
		if inner_products.len() != correct_len {
			bail!(Error::ArgumentRangeError {
				arg: "evals.len()".to_string(),
				range: correct_len..correct_len + 1,
			});
		}

		// TODO: Maybe optimize me
		let subcube_start = subcube_index << (query_n_vars + subcube_vars);
		for scalar_index in 0..1 << subcube_vars {
			let evals_start = subcube_start + (scalar_index << query_n_vars);
			let mut inner_product = F::ZERO;
			for i in 0..1 << query_n_vars {
				inner_product += get_packed_slice(query.expansion(), i)
					* get_packed_slice(self.0.evals(), evals_start + i);
			}

			set_packed_slice(inner_products, scalar_index, inner_product);
		}

		Ok(())
	}

	fn subcube_evals(
		&self,
		subcube_vars: usize,
		subcube_index: usize,
		log_embedding_degree: usize,
		evals: &mut [P],
	) -> Result<(), Error> {
		let n_vars = self.n_vars();
		if subcube_vars > n_vars {
			bail!(Error::ArgumentRangeError {
				arg: "subcube_vars".to_string(),
				range: 0..n_vars + 1,
			});
		}

		if log_embedding_degree != 0 {
			bail!(Error::LogEmbeddingDegreeTooLarge {
				log_embedding_degree
			});
		}

		let correct_len = 1 << subcube_vars.saturating_sub(P::LOG_WIDTH);
		if evals.len() != correct_len {
			bail!(Error::ArgumentRangeError {
				arg: "evals.len()".to_string(),
				range: correct_len..correct_len + 1,
			});
		}

		let max_index = 1 << (n_vars - subcube_vars);
		if subcube_index >= max_index {
			bail!(Error::ArgumentRangeError {
				arg: "subcube_index".to_string(),
				range: 0..max_index,
			});
		}

		if subcube_vars < P::LOG_WIDTH {
			let subcube_start = subcube_index << subcube_vars;
			for i in 0..1 << subcube_vars {
				// Safety: subcube_index < max_index check
				let scalar =
					unsafe { get_packed_slice_unchecked(self.0.evals(), subcube_start + i) };

				// Safety: i < 1 << min(0, subcube_vars) <= correct_len * P::WIDTH
				unsafe {
					set_packed_slice_unchecked(evals, i, scalar);
				}
			}
		} else {
			let range = subcube_index << (subcube_vars - P::LOG_WIDTH)
				..(subcube_index + 1) << (subcube_vars - P::LOG_WIDTH);
			evals.copy_from_slice(&self.0.evals()[range]);
		}

		Ok(())
	}

	fn packed_evals(&self) -> Option<&[P]> {
		Some(self.0.evals())
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::{tensor_prod_eq_ind, MultilinearQuery};
	use binius_field::{
		arch::OptimalUnderlier256b, as_packed_field::PackedType, packed::iter_packed_slice,
		BinaryField128b, BinaryField16b, BinaryField32b, BinaryField8b, PackedBinaryField16x8b,
		PackedBinaryField1x128b, PackedBinaryField4x32b, PackedBinaryField8x16b, PackedExtension,
		PackedField, PackedFieldIndexable,
	};
	use rand::prelude::*;
	use std::iter::repeat_with;

	type F = BinaryField16b;
	type P = PackedBinaryField8x16b;

	fn multilinear_query<P: PackedField>(p: &[P::Scalar]) -> MultilinearQuery<P, Vec<P>> {
		let mut result = vec![P::default(); 1 << p.len().saturating_sub(P::LOG_WIDTH)];
		result[0] = P::set_single(P::Scalar::ONE);
		tensor_prod_eq_ind(0, &mut result, p).unwrap();
		MultilinearQuery::with_expansion(p.len(), result).unwrap()
	}

	#[test]
	fn test_evaluate_subcube_and_evaluate_partial_low_consistent() {
		let mut rng = StdRng::seed_from_u64(0);
		let poly = MultilinearExtension::from_values(
			repeat_with(|| PackedBinaryField4x32b::random(&mut rng))
				.take(1 << 8)
				.collect(),
		)
		.unwrap()
		.specialize::<PackedBinaryField1x128b>();

		let q = repeat_with(|| <BinaryField128b as PackedField>::random(&mut rng))
			.take(6)
			.collect::<Vec<_>>();
		let query = multilinear_query(&q);

		let partial_low = poly.evaluate_partial_low(query.to_ref()).unwrap();

		let mut inner_products = vec![PackedBinaryField1x128b::zero(); 16];
		poly.subcube_inner_products(query.to_ref(), 4, 0, inner_products.as_mut_slice())
			.unwrap();

		for (idx, inner_product) in iter_packed_slice(&inner_products).enumerate() {
			assert_eq!(inner_product, partial_low.evaluate_on_hypercube(idx).unwrap(),);
		}
	}

	#[test]
	fn test_evaluate_subcube_smaller_than_packed_width() {
		let mut rng = StdRng::seed_from_u64(0);
		let poly = MultilinearExtension::new(
			2,
			vec![PackedBinaryField16x8b::from_scalars(
				[2, 2, 9, 9].map(BinaryField8b::new),
			)],
		)
		.unwrap()
		.specialize::<PackedBinaryField1x128b>();

		let q = repeat_with(|| <BinaryField128b as PackedField>::random(&mut rng))
			.take(1)
			.collect::<Vec<_>>();
		let query = multilinear_query(&q);

		let mut inner_products = vec![PackedBinaryField1x128b::zero(); 2];
		poly.subcube_inner_products(query.to_ref(), 1, 0, inner_products.as_mut_slice())
			.unwrap();

		assert_eq!(get_packed_slice(&inner_products, 0), BinaryField128b::new(2));
		assert_eq!(get_packed_slice(&inner_products, 1), BinaryField128b::new(9));
	}

	#[test]
	fn test_subcube_evals_embeds_correctly() {
		let mut rng = StdRng::seed_from_u64(0);

		type P = PackedBinaryField16x8b;
		type PE = PackedBinaryField1x128b;

		let packed_count = 4;
		let values: Vec<_> = repeat_with(|| P::random(&mut rng))
			.take(1 << packed_count)
			.collect();

		let mle = MultilinearExtension::from_values(values).unwrap();
		let mles = MLEEmbeddingAdapter::<P, PE, _>::from(mle);

		let bytes_values = P::unpack_scalars(mles.0.evals());

		let n_vars = packed_count + P::LOG_WIDTH;
		let mut evals = vec![PE::zero(); 1 << n_vars];
		for subcube_vars in 0..n_vars {
			for subcube_index in 0..1 << (n_vars - subcube_vars) {
				for log_embedding_degree in 0..=4 {
					let evals_subcube = &mut evals
						[0..1 << subcube_vars.saturating_sub(log_embedding_degree + PE::LOG_WIDTH)];

					mles.subcube_evals(
						subcube_vars,
						subcube_index,
						log_embedding_degree,
						evals_subcube,
					)
					.unwrap();

					let bytes_evals = P::unpack_scalars(
						<PE as PackedExtension<BinaryField8b>>::cast_bases(evals_subcube),
					);

					let shift = 4 - log_embedding_degree;
					let skip_mask = (1 << shift) - 1;
					for (i, &b_evals) in bytes_evals.iter().enumerate() {
						let b_values = if i & skip_mask == 0 && i < 1 << (subcube_vars + shift) {
							bytes_values[(subcube_index << subcube_vars) + (i >> shift)]
						} else {
							BinaryField8b::ZERO
						};
						assert_eq!(b_evals, b_values);
					}
				}
			}
		}
	}

	#[test]
	fn test_subcube_inner_products_and_evaluate_partial_low_conform() {
		let mut rng = StdRng::seed_from_u64(0);
		let n_vars = 12;
		let evals = repeat_with(|| P::random(&mut rng))
			.take(1 << (n_vars - P::LOG_WIDTH))
			.collect::<Vec<_>>();
		let mle = MultilinearExtension::from_values(evals).unwrap();
		let mles = MLEDirectAdapter::from(mle);
		let q = repeat_with(|| Field::random(&mut rng))
			.take(6)
			.collect::<Vec<F>>();
		let query = multilinear_query(&q);
		let partial_eval = mles.evaluate_partial_low(query.to_ref()).unwrap();

		let subcube_vars = 4;
		let mut evals = vec![P::default(); 1 << (subcube_vars - P::LOG_WIDTH)];
		for subcube_index in 0..(n_vars - query.n_vars() - subcube_vars) {
			mles.subcube_inner_products(
				query.to_ref(),
				subcube_vars,
				subcube_index,
				evals.as_mut_slice(),
			)
			.unwrap();
			for hypercube_idx in 0..(1 << subcube_vars) {
				assert_eq!(
					get_packed_slice(&evals, hypercube_idx),
					partial_eval
						.evaluate_on_hypercube(hypercube_idx + (subcube_index << subcube_vars))
						.unwrap()
				);
			}
		}
	}

	#[test]
	fn test_packed_evals_against_subcube_evals() {
		type U = OptimalUnderlier256b;
		type P = PackedType<U, BinaryField32b>;
		type PExt = PackedType<U, BinaryField128b>;

		let mut rng = StdRng::seed_from_u64(0);
		let evals = repeat_with(|| P::random(&mut rng))
			.take(2)
			.collect::<Vec<_>>();
		let mle = MultilinearExtension::from_values(evals.clone()).unwrap();
		let poly = MLEEmbeddingAdapter::from(mle);
		assert_eq!(
			<PExt as PackedExtension<BinaryField32b>>::cast_bases(poly.packed_evals().unwrap()),
			&evals
		);

		let mut evals_out = vec![PExt::zero(); 2];
		poly.subcube_evals(poly.n_vars(), 0, poly.log_extension_degree(), evals_out.as_mut_slice())
			.unwrap();
		assert_eq!(evals_out, poly.packed_evals().unwrap());
	}
}
