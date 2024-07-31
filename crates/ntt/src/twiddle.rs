// Copyright 2024 Ulvetanna Inc.

use crate::Error;
use binius_field::{BinaryField, Field};
use std::{marker::PhantomData, ops::Deref};

/// A trait for accessing twiddle factors in a single NTT round.
///
/// Twiddle factors in the additive NTT are subspace polynomial evaluations over linear subspaces,
/// with an implicit NTT round $i$.
/// Setup: let $K/\mathbb{F}\_2$ be a finite extension of degree $d$, and let $\beta_0,\ldots ,\beta_{d-1}$ be a
/// linear basis. Let $U_i$ be the $\mathbb{F}\_2$-linear span of $\beta_0,\ldots ,\beta_{i-1}$. Let $\hat{W}_i(X)$
/// be the normalized subspace polynomial of degree $2^i$ that vanishes on $U_i$ and is $1$ on $\beta_i$.
/// Evaluating $\hat{W}_i(X)$ turns out to yield an $\mathbb{F}\_2$-linear function $K\rightarrow K$.
///
/// This trait accesses the subspace polynomial evaluations for $\hat{W}\_i(X)$.
/// The evaluations of the vanishing polynomial over all elements in any coset of the subspace
/// are equal. Equivalently, the evaluations of $\hat{W}\_i(X)$ are well-defined on
/// the $d-i$-dimensional vector space $K/U_{i-1}$. Note that $K/U_{i-1}$ has a natural induced basis.
/// Write $\{j\}$ for the $j$th coset of the subspace, where $j$ is in $[0,2^{d-i})$, with respect
/// to this natural basis. This means: write $j$ in binary: $j = j_0 + \ldots +j_{d-i-1}2^{d-i-1}$
/// and consider the following element of $K$: $j_0\beta_i + \ldots  + j_{d-i-1}\beta_{d-1}$.
/// This element of course specifies an element of $K/U_i$.
/// The twiddle factor $t_{i,j}$ is then $\hat{W}\_i(\{j\})$, i.e., $\hat{W}\_j$ evaluated at the aforementioned element of
/// the quotient $K/U_i$.
///
/// As explained, the evaluations of these polynomial yield linear functions, which allows for flexibility in how they are computed.
/// Namely, for an evaluation domain of size $2^{i}$, there is a strategy for computing polynomial
/// evaluations "on-the-fly" with $O(\ell)$ field additions using $O(\ell)$ stored elements or precomputing the $2^\ell$ evaluations and
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
	/// Evaluate $\hat{W}\_i(X)$ at the element `index`: write `index` in binary,
	/// and evaluate at the element $index_0\beta_i \ldots + index_{d-i-1}\beta_{d-1}$.
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
	/// Recall that a `TwiddleAccess` has an implicit NTT round $i$. Let $j=d-coset_{bits}$.
	/// Then`coset` returns a `TwiddleAccess` object (of NTT round i) for the following affine  
	/// subspace of $K/U_{i-1}$: the set of all elements of $K/U_{i-1}$
	/// whose coordinates in the basis $\beta_i,\ldots ,\beta_{d-1}$ is:
	/// $(*, \cdots, *, coset_{0}, \ldots , coset_{bits-1})$, where the first $j$ coordinates are arbitrary.
	/// Here $coset = coset_0 + \ldots  + coset_{bits-1}2^{bits-1}$. In sum, this amounts to *evaluations* of $\hat{W}\_i$
	/// at all such elements.
	///
	/// Therefore, the `self.log_n` of the new `TwiddleAccess` object is computed as `self.log_n() - coset_bits`.
	///
	/// Panics if `coset_bits` is not in the range 0 to `self.log_n()` or `coset` is not in the
	/// range 0 to `1 << coset_bits`.
	fn coset(&self, coset_bits: usize, coset: usize) -> impl TwiddleAccess<F>;
}

/// Twiddle access method that does on-the-fly computation to reduce its memory footprint.
///
/// This implementation uses a small amount of precomputed constants from which the twiddle factors
/// are derived on the fly (OTF). The number of constants is ~$1/2 d^2$ field elements for a domain
/// of size $2^d$.
#[derive(Debug)]
pub struct OnTheFlyTwiddleAccess<F, SEvals = Vec<F>> {
	log_n: usize,
	/// `offset` is a constant that is added to all twiddle factors.
	offset: F,
	/// `s_evals` is $<\hat{W}\_i(\beta_{i+1}),\ldots ,\hat{W}\_i(\beta_{d-1})>$ for the implicit round $i$.
	s_evals: SEvals,
}

impl<F: BinaryField> OnTheFlyTwiddleAccess<F> {
	/// Generate a vector of OnTheFlyTwiddleAccess objects, one for each NTT round.
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
		// The `s_evals` for the $i$th round contains the evaluations of $\hat{W}\_i$ on
		// $\beta_{i+1},\ldots ,\beta_{d-1}$.
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
/// This implementation precomputes all $2^k$ twiddle factors for a domain of size $2^k$.
#[derive(Debug)]
pub struct PrecomputedTwiddleAccess<F, SEvals = Vec<F>> {
	log_n: usize,
	/// If we are implicitly in NTT round i, then `s_evals` contains the evaluations of $\hat{W}\_i$
	/// on the entire space $K/U_{i-1}$, where the order is the usual "binary counting order"
	/// in the basis vectors $\beta_i,\ldots ,\beta_{d-1}$.
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

/// Precompute the evaluations of the normalized subspace polynomials $\hat{W}_i$ on a basis.
///
/// Let $K/\mathbb{F}_2$ be a finite extension of degree $d$, and let $\beta_0,\ldots ,\beta_{d-1}$ be a linear basis,
/// with $\beta_0$ = 1. Let $U_i$ be the $\mathbb{F}_2$-linear span of $\beta_0,\ldots ,\beta_{i-1}$, so $U_0$ is the zero subspace.
/// Let $\hat{W}\_i(X)$ be the normalized subspace polynomial of degree $2^i$ that vanishes on $U_i$
/// and is $1$ on $\beta_i$.
/// Return a vector whose $i$th entry is a vector of evaluations of $\hat{W}\_i$ at $\beta_{i+1},\ldots ,\beta_{d-1}$.
fn precompute_subspace_evals<F: BinaryField, DomainField: BinaryField + Into<F>>(
	log_domain_size: usize,
) -> Result<Vec<Vec<F>>, Error> {
	if DomainField::N_BITS < log_domain_size {
		return Err(Error::FieldTooSmall { log_domain_size });
	}

	let mut s_evals = Vec::with_capacity(log_domain_size);

	// `normalization_consts[i]` = $W\_i(2^i):=  W\_i(\beta_i)$
	let mut normalization_consts = Vec::with_capacity(log_domain_size);
	// $\beta_0 = 1$ and $W\_0(X) = X$, so $W\_0(\beta_0) = \beta_0 = 1$
	normalization_consts.push(F::ONE);
	//`s0_evals` = $(\beta_1,\ldots ,\beta_{d-1}) = (W\_0(\beta_1), \ldots , W\_0(\beta_{d-1}))$
	let s0_evals = (1..log_domain_size)
		.map(|i| {
			DomainField::basis(i)
				.expect("basis vector must exist because of FieldTooSmall check above")
				.into()
		})
		.collect::<Vec<F>>();

	s_evals.push(s0_evals);
	// let $W\_i(X)$ be the *unnormalized* subspace polynomial, i.e., $\prod_{u\in U_{i-1}}(X-u)$.
	// Then $W\_{i+1}(X) = W\_i(X)(W\_i(X)+W\_i(\beta_i))$. This crucially uses the "linearity" of
	// $W\_i(X)$. This fundamental relation allows us to iteratively compute `s_evals` layer by layer.
	for _ in 1..log_domain_size {
		let (norm_const_i, s_i_evals) = {
			let norm_prev = *normalization_consts
				.last()
				.expect("normalization_consts is not empty");
			let s_prev_evals = s_evals.last().expect("s_evals is not empty");
			// `norm_prev` = $W\_{i-1}(\beta_{i-1})$
			// s_prev_evals = $W\_{i-1}(\beta_i),\ldots ,W\_{i-1}(\beta_{d-1})$
			let norm_const_i = subspace_map(s_prev_evals[0], norm_prev);
			let s_i_evals = s_prev_evals
				.iter()
				.skip(1)
				.map(|&s_ij_prev| subspace_map(s_ij_prev, norm_prev))
				.collect::<Vec<_>>();
			// the two calls to the function subspace_map yield the following:
			// `norm_const_i` = $W\_{i}(\beta_i)$; and
			// `s_i_evals` = $W\_{i}(\beta_{i+1}),\ldots ,W\_{i}(\beta_{d-1})$.
			(norm_const_i, s_i_evals)
		};

		normalization_consts.push(norm_const_i);
		s_evals.push(s_i_evals);
	}

	for (norm_const_i, s_evals_i) in normalization_consts.iter().zip(s_evals.iter_mut()) {
		let inv_norm_const = norm_const_i
			.invert()
			.expect("normalization constants are nonzero");
		// replace all terms $W\_{i}(\beta_j)$ with $W\_{i}(\beta_j)/W\_{i}(\beta_i)$
		// to obtain the evaluations of the *normalized* subspace polynomials.
		for s_ij in s_evals_i.iter_mut() {
			*s_ij *= inv_norm_const;
		}
	}

	Ok(s_evals)
}
/// Computes the function $(e,c)\mapsto e^2+ce$.
///
/// This is primarily used to compute $W\_{i+1}(X)$ from $W\_i(X)$ in the binary field setting.
fn subspace_map<F: Field>(elem: F, constant: F) -> F {
	elem.square() + constant * elem
}
/// Given `OnTheFlyTwiddleAccess` instances for each NTT round, returns a vector of `PrecomputedTwiddleAccess` objects,
/// one for each NTT round.
///
/// For each round $i$, the input contains the value of $\hat{W}\_i$ on the basis $\beta_i,\ldots ,\beta_{d-1}$.
/// The ith element of the output contains the evaluations of $\hat{W}\_i$ on the entire space $K/U_{i-1}$,
/// where the order is the usual "binary counting order" in the above basis.
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
