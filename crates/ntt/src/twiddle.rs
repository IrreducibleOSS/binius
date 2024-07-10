// Copyright 2024 Ulvetanna Inc.

use crate::Error;
use binius_field::{BinaryField, Field};
use std::{marker::PhantomData, ops::Deref};

/// A trait for accessing twiddle factors in a single NTT round.
///
/// Twiddle factors in the additive NTT are subspace polynomial evaluations over linear subspaces.
/// This structure accesses the subspace polynomial evaluations for a normalized subspace
/// polynomial $\hat{W}_i(X)$, where $i$ indexes the NTT round. These polynomials, by definition,
/// vanish on domains of size $2^i$ that are linear subspaces of a binary field. The evaluations of
/// the vanishing polynomial over all elements in any coset of the subspace are equal. We write
/// $\{j\}$ for the $j$th coset of the subspace. The twiddle factor $t_{i,j}$ is then
/// $\hat{W}_i(\{j\})$.
///
/// These polynomials are linear, which allows for flexibility in how they are computed. Namely,
/// for an evaluation domain of size $2^i$, there is a strategy for computing polynomial
/// evaluations "on-the-fly" with $O(\ell)$ field
/// additions using $O(\ell)$ stored elements or precomputing the $2^\ell$ evaluations and
/// looking them up in constant time (see the [`OnTheFlyTwiddleAccess`] and
/// [`PrecomputedTwiddleAccess`] implementations, respectively).
///
/// See [LCH14] and [DP24] Section 2.3 for more details.
///
/// [LCH14]: <https://arxiv.org/abs/1404.3458>
/// [DP24]: <https://eprint.iacr.org/2024/504>
pub trait TwiddleAccess<F: BinaryField> {
	/// Base-2 logarithm of the number of twiddle factors in this round.
	fn log_n(&self) -> usize;

	/// Get the twiddle factor at the given index.
	///
	/// Panics if `index` is not in the range 0 to `1 << self.log_n()`.
	fn get(&self, index: usize) -> F;

	/// Get the pair of twiddle factors at the indices `index` and `(1 << index_bits) | index`.
	///
	/// Panics if `index_bits` is not in the range 0 to `self.log_n()` or `index` is not in the
	/// range 0 to `1 << index_bits`.
	fn get_pair(&self, index_bits: usize, index: usize) -> (F, F);

	/// Returns a scoped twiddle access for the coset that fixes the upper `coset_bits` of the
	/// index to `coset`.
	///
	/// Panics if `coset_bits` is not in the range 0 to `self.log_n()` or `coset` is not in the
	/// range 0 to `1 << coset_bits`.
	fn coset(&self, coset_bits: usize, coset: usize) -> impl TwiddleAccess<F>;
}

/// Twiddle access method that does on-the-fly computation to reduce its memory footprint.
///
/// This implementation uses a small amount of precomputed constants from which the twiddle factors
/// are derived on the fly (OTF). The number of constants is ~1/2 k^2 field elements for a domain
/// of size 2^k.
#[derive(Debug)]
pub struct OnTheFlyTwiddleAccess<F, SEvals = Vec<F>> {
	log_n: usize,
	offset: F,
	s_evals: SEvals,
}

impl<F: BinaryField> OnTheFlyTwiddleAccess<F> {
	pub fn generate<DomainField: BinaryField + Into<F>>(
		log_domain_size: usize,
	) -> Result<Vec<Self>, Error> {
		let s_evals = precompute_subspace_evals::<F, DomainField>(log_domain_size)?
			.into_iter()
			.enumerate()
			.map(|(i, s_evals_i)| OnTheFlyTwiddleAccess {
				log_n: log_domain_size - 1 - i,
				offset: F::ZERO,
				s_evals: s_evals_i,
			})
			.collect();
		Ok(s_evals)
	}
}

impl<F, SEvals> TwiddleAccess<F> for OnTheFlyTwiddleAccess<F, SEvals>
where
	F: BinaryField,
	SEvals: Deref<Target = [F]>,
{
	#[inline]
	fn log_n(&self) -> usize {
		self.log_n
	}

	#[inline]
	fn get(&self, i: usize) -> F {
		self.offset + subset_sum(&self.s_evals, self.log_n, i)
	}

	#[inline]
	fn get_pair(&self, index_bits: usize, i: usize) -> (F, F) {
		let t0 = self.offset + subset_sum(&self.s_evals, index_bits, i);
		(t0, t0 + self.s_evals[index_bits])
	}

	#[inline]
	fn coset(&self, coset_bits: usize, coset: usize) -> impl TwiddleAccess<F> {
		let log_n = self.log_n - coset_bits;
		let offset = subset_sum(&self.s_evals[log_n..], coset_bits, coset);
		OnTheFlyTwiddleAccess {
			log_n,
			offset: self.offset + offset,
			s_evals: &self.s_evals[..log_n],
		}
	}
}

fn subset_sum<F: Field>(values: &[F], n_bits: usize, index: usize) -> F {
	(0..n_bits)
		.filter(|b| (index >> b) & 1 != 0)
		.map(|b| values[b])
		.sum()
}

/// Twiddle access method using a larger table of precomputed constants.
///
/// This implementation precomputes all 2^k twiddle factors for a domain of size 2^k.
#[derive(Debug)]
pub struct PrecomputedTwiddleAccess<F, SEvals = Vec<F>> {
	log_n: usize,
	s_evals: SEvals,
	_marker: PhantomData<F>,
}

impl<F: BinaryField> PrecomputedTwiddleAccess<F> {
	pub fn generate<DomainField: BinaryField + Into<F>>(
		log_domain_size: usize,
	) -> Result<Vec<Self>, Error> {
		let on_the_fly = OnTheFlyTwiddleAccess::<F, _>::generate::<DomainField>(log_domain_size)?;
		Ok(expand_subspace_evals(&on_the_fly))
	}
}

impl<F, SEvals> TwiddleAccess<F> for PrecomputedTwiddleAccess<F, SEvals>
where
	F: BinaryField,
	SEvals: Deref<Target = [F]>,
{
	#[inline]
	fn log_n(&self) -> usize {
		self.log_n
	}

	#[inline]
	fn get(&self, i: usize) -> F {
		self.s_evals[i]
	}

	#[inline]
	fn get_pair(&self, index_bits: usize, i: usize) -> (F, F) {
		(self.s_evals[i], self.s_evals[1 << index_bits | i])
	}

	#[inline]
	fn coset(&self, coset_bits: usize, coset: usize) -> impl TwiddleAccess<F> {
		let log_n = self.log_n - coset_bits;
		PrecomputedTwiddleAccess {
			log_n,
			s_evals: &self.s_evals[coset << log_n..(coset + 1) << log_n],
			_marker: PhantomData,
		}
	}
}

fn precompute_subspace_evals<F: BinaryField, DomainField: BinaryField + Into<F>>(
	log_domain_size: usize,
) -> Result<Vec<Vec<F>>, Error> {
	if DomainField::N_BITS < log_domain_size {
		return Err(Error::FieldTooSmall { log_domain_size });
	}

	let mut s_evals = Vec::with_capacity(log_domain_size);

	// normalization_consts[i] = W_i(2^i)
	let mut normalization_consts = Vec::with_capacity(log_domain_size);
	normalization_consts.push(F::ONE);

	let s0_evals = (1..log_domain_size)
		.map(|i| {
			DomainField::basis(i)
				.expect("basis vector must exist because of FieldTooSmall check above")
				.into()
		})
		.collect::<Vec<F>>();

	s_evals.push(s0_evals);

	for _ in 1..log_domain_size {
		let (norm_const_i, s_i_evals) = {
			let norm_prev = *normalization_consts
				.last()
				.expect("normalization_consts is not empty");
			let s_prev_evals = s_evals.last().expect("s_evals is not empty");

			let norm_const_i = subspace_map(s_prev_evals[0], norm_prev);
			let s_i_evals = s_prev_evals
				.iter()
				.skip(1)
				.map(|&s_ij_prev| subspace_map(s_ij_prev, norm_prev))
				.collect::<Vec<_>>();

			(norm_const_i, s_i_evals)
		};

		normalization_consts.push(norm_const_i);
		s_evals.push(s_i_evals);
	}

	for (norm_const_i, s_evals_i) in normalization_consts.iter().zip(s_evals.iter_mut()) {
		let inv_norm_const = norm_const_i
			.invert()
			.expect("normalization constants are nonzero");
		for s_ij in s_evals_i.iter_mut() {
			*s_ij *= inv_norm_const;
		}
	}

	Ok(s_evals)
}

fn subspace_map<F: Field>(elem: F, constant: F) -> F {
	elem.square() + constant * elem
}

pub fn expand_subspace_evals<F, SEvals>(
	on_the_fly: &[OnTheFlyTwiddleAccess<F, SEvals>],
) -> Vec<PrecomputedTwiddleAccess<F>>
where
	F: BinaryField,
	SEvals: Deref<Target = [F]>,
{
	let log_domain_size = on_the_fly.len();
	on_the_fly
		.iter()
		.enumerate()
		.map(|(i, on_the_fly_i)| {
			let s_evals_i = &on_the_fly_i.s_evals;

			let mut expanded = Vec::with_capacity(1 << s_evals_i.len());
			expanded.push(F::ZERO);
			for &eval in s_evals_i.iter() {
				for i in 0..expanded.len() {
					expanded.push(expanded[i] + eval);
				}
			}

			PrecomputedTwiddleAccess {
				log_n: log_domain_size - 1 - i,
				s_evals: expanded,
				_marker: PhantomData,
			}
		})
		.collect()
}
