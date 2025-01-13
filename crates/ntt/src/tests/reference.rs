// Copyright 2024-2025 Irreducible Inc.

use binius_field::{
	packed::{get_packed_slice, set_packed_slice},
	BinaryField, ExtensionField, PackedField,
};
use binius_math::BinarySubspace;

use crate::{twiddle::TwiddleAccess, AdditiveNTT, Error, SingleThreadedNTT};

/// This trait allows passing packed batches to the simple NTT implementation.
trait DataAccess<T> {
	fn get(&self, index: usize) -> T;
	fn set(&mut self, index: usize, value: T);

	fn len(&self) -> usize;
}

impl<T: Clone> DataAccess<T> for [T] {
	fn len(&self) -> usize {
		self.len()
	}

	fn get(&self, index: usize) -> T {
		self[index].clone()
	}

	fn set(&mut self, index: usize, value: T) {
		self[index] = value;
	}
}

/// A slice of packed field elements with an access to a batch with the given index:
/// [batch_0_element_0, batch_1_element_0, ..., batch_0_element_1, batch_0_element_1, ...]
struct BatchedPackedFieldSlice<'a, P> {
	data: &'a mut [P],
	log_batch_count: usize,
	batch_index: usize,
}

impl<'a, P> BatchedPackedFieldSlice<'a, P> {
	fn new(data: &'a mut [P], log_batch_count: usize, batch_index: usize) -> Self {
		Self {
			data,
			log_batch_count,
			batch_index,
		}
	}
}

impl<P> DataAccess<P::Scalar> for BatchedPackedFieldSlice<'_, P>
where
	P: PackedField,
{
	fn get(&self, index: usize) -> P::Scalar {
		get_packed_slice(self.data, self.batch_index + (index << self.log_batch_count))
	}

	fn set(&mut self, index: usize, value: P::Scalar) {
		set_packed_slice(self.data, self.batch_index + (index << self.log_batch_count), value)
	}

	fn len(&self) -> usize {
		(self.data.len() * P::WIDTH) >> self.log_batch_count
	}
}

/// Reference implementation of the forward NTT to compare against in tests.
fn forward_transform_simple<F, FF>(
	log_domain_size: usize,
	s_evals: &[impl TwiddleAccess<F>],
	data: &mut impl DataAccess<FF>,
	coset: u32,
) -> Result<(), Error>
where
	F: BinaryField,
	FF: ExtensionField<F>,
{
	let n = data.len();
	assert!(n.is_power_of_two());

	let log_n = n.trailing_zeros() as usize;
	let coset_bits = 32 - coset.leading_zeros() as usize;
	if log_n + coset_bits > log_domain_size {
		return Err(Error::DomainTooSmall {
			log_required_domain_size: log_n + coset_bits,
		});
	}

	for i in (0..log_n).rev() {
		let s_evals_i = &s_evals[i];
		for j in 0..1 << (log_n - 1 - i) {
			let twiddle = s_evals_i.get((coset as usize) << (log_n - 1 - i) | j);
			for k in 0..1 << i {
				let idx0 = j << (i + 1) | k;
				let idx1 = idx0 | 1 << i;

				let (mut u, mut v) = (data.get(idx0), data.get(idx1));

				u += v * twiddle;
				v += u;

				data.set(idx0, u);
				data.set(idx1, v);
			}
		}
	}

	Ok(())
}

/// Reference implementation of the inverse NTT to compare against in tests.
fn inverse_transform_simple<F, FF>(
	log_domain_size: usize,
	s_evals: &[impl TwiddleAccess<F>],
	data: &mut impl DataAccess<FF>,
	coset: u32,
) -> Result<(), Error>
where
	F: BinaryField,
	FF: ExtensionField<F>,
{
	let n = data.len();
	assert!(n.is_power_of_two());

	let log_n = n.trailing_zeros() as usize;
	let coset_bits = 32 - coset.leading_zeros() as usize;
	if log_n + coset_bits > log_domain_size {
		return Err(Error::DomainTooSmall {
			log_required_domain_size: log_n + coset_bits,
		});
	}

	#[allow(clippy::needless_range_loop)]
	for i in 0..log_n {
		let s_evals_i = &s_evals[i];
		for j in 0..1 << (log_n - 1 - i) {
			let twiddle = s_evals_i.get((coset as usize) << (log_n - 1 - i) | j);
			for k in 0..1 << i {
				let idx0 = j << (i + 1) | k;
				let idx1 = idx0 | 1 << i;

				let (mut u, mut v) = (data.get(idx0), data.get(idx1));

				v += u;
				u += v * twiddle;

				data.set(idx0, u);
				data.set(idx1, v);
			}
		}
	}

	Ok(())
}

/// Simple NTT implementation that uses the reference implementation for the forward and inverse NTT.
pub struct SimpleAdditiveNTT<F: BinaryField, TA: TwiddleAccess<F>> {
	subspace: BinarySubspace<F>,
	s_evals: Vec<TA>,
}

impl<F, TA: TwiddleAccess<F>> AdditiveNTT<F> for SimpleAdditiveNTT<F, TA>
where
	F: BinaryField,
	TA: TwiddleAccess<F>,
{
	fn subspace(&self) -> &BinarySubspace<F> {
		&self.subspace
	}

	fn get_subspace_eval(&self, _i: usize, _j: usize) -> F {
		unimplemented!()
	}

	fn forward_transform<P>(
		&self,
		data: &mut [P],
		coset: u32,
		log_batch_size: usize,
	) -> Result<(), Error>
	where
		P: PackedField<Scalar = F>,
	{
		for batch_index in 0..1 << log_batch_size {
			let mut batch = BatchedPackedFieldSlice::new(data, log_batch_size, batch_index);
			forward_transform_simple(self.log_domain_size(), &self.s_evals, &mut batch, coset)?;
		}

		Ok(())
	}

	fn inverse_transform<P>(
		&self,
		data: &mut [P],
		coset: u32,
		log_batch_size: usize,
	) -> Result<(), Error>
	where
		P: PackedField<Scalar = F>,
	{
		for batch_index in 0..1 << log_batch_size {
			let mut batch = BatchedPackedFieldSlice::new(data, log_batch_size, batch_index);
			inverse_transform_simple(self.log_domain_size(), &self.s_evals, &mut batch, coset)?;
		}

		Ok(())
	}
}

impl<F, TA> SingleThreadedNTT<F, TA>
where
	F: BinaryField,
	TA: TwiddleAccess<F>,
{
	pub(super) fn into_simple_ntt(self) -> SimpleAdditiveNTT<F, TA> {
		SimpleAdditiveNTT {
			subspace: self.subspace().clone(),
			s_evals: self.s_evals,
		}
	}
}
