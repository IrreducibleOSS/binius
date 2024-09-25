// Copyright 2024 Ulvetanna Inc.

use crate::{
	challenger::{CanObserve, CanSample, CanSampleBits},
	poly_commit::PolyCommitScheme,
	polynomial::{Error as PolynomialError, MultilinearExtension, MultilinearQuery},
};

use binius_field::{
	packed::{get_packed_slice, iter_packed_slice},
	Field, PackedField,
};
use binius_hal::ComputationBackend;
use binius_utils::bail;
// use rayon::prelude::*;
use std::{iter, marker::PhantomData, mem, ops::Deref};

#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error("inner PCS error: {0}")]
	InnerPCS(#[source] Box<dyn std::error::Error + Send + Sync>),
	#[error("polynomial error: {0}")]
	Polynomial(#[from] PolynomialError),
	#[error("verification failure: {0}")]
	Verification(#[from] VerificationError),
}

#[derive(Debug, thiserror::Error)]
pub enum VerificationError {
	#[error("evaluation value is inconsistent with the tensor evaluation")]
	IncorrectEvaluation,
}

/// Creates a new multilinear from a batch of multilinears:
///
/// If we have a collection of $2^m$ multilinear polynomials $t_u$, each of which is $n$-variate,
/// indexed over $u\in \{0,1\}^m$, construct the function $T$ on $\{0,1\}^{m+n}$ such that
/// $T(u||v):=t_u(v)$ for all $u\in \{0,1\}^m$ and $v\in \{0,1\}^n$. By abuse of notation
/// we consider $T$ an $n+m$-variate multilinear polynomial.
fn merge_polynomials<F, P, Data>(
	polys: &[MultilinearExtension<P, Data>],
	n_vars: usize,
	// the number of polynomials is $2^m$
	m_polys: usize,
) -> Result<MultilinearExtension<P>, Error>
where
	F: Field,
	P: PackedField<Scalar = F>,
	Data: Deref<Target = [P]> + Send + Sync,
{
	for poly_i in polys {
		if poly_i.n_vars() != n_vars {
			// wrong error, just trying to write some baby code.
			bail!(PolynomialError::IncorrectOutputPolynomialSize { expected: n_vars });
		}
	}
	if polys.len() != 1 << m_polys {
		// wrong error, just trying to write some baby code.
		bail!(PolynomialError::IncorrectOutputPolynomialSize { expected: m_polys });
	}

	// $T(u||v):=t_{u}(v)$, where  $u||v = 2^m * v + u$
	// The below code is my attempt at implementing the above, keeping in mind packing.
	// when packing is not an issue, the internal part of the loop just consists of:
	// let current_index = v * (1 << m_polys) + u;
	// let vec_of_values[current_index] = polys[u].evals()[v];
	let mut vec_of_values = vec![P::zero(); 1 << (n_vars + m_polys - P::LOG_WIDTH)];

	for u in 0..1 << m_polys {
		for v in 0..1 << n_vars {
			let current_index = v * (1 << m_polys) + u;
			let external_index = current_index / P::WIDTH;
			let internal_index = current_index % P::WIDTH;
			vec_of_values[external_index]
				.set_checked(internal_index, get_packed_slice(polys[u].evals(), v))
				.unwrap();
		}
	}

	let merged_poly = MultilinearExtension::from_values(vec_of_values.clone()).unwrap();
	Ok(merged_poly)
}

/// A block-box batching scheme for multilinear commitments.
///
/// Suppose we have a collection of $2^m$ multilinear polynomials $t_u$,
/// each of which is $n$-variate, and we want to prove the evaluations at a point $\vec{r}=(r_0,...,r_{n-1})$
/// are $(s_u)$.
/// Build the multilinear, n+m-variate polynomial T, whose values on $B_{m+n}$ are given as:
/// $T(u||v) = t_u(v)$, for all u in $\{0,1\}^m$ and v in $\{0,1\}^n$.
/// Sample random challenges $\vec{r'}:=(r'_0,...,r'_{m-1})$. Finally, pass off the evaluation of
/// $T$ at $(r'_0,...,r'_{m-1},r_0,...,r_{n-1})$ to the inner polynomial commitment scheme.
///
/// If the prover is honest, $T(\vec{r'}||\vec{r})$ is the dot product of the tensor expansion of $\vec{r'}$ with
/// $(s_u)$. Therefore, given the claimed evaluations $(s_u)$, the verifier can compute the desired mixing
/// herself.
///
/// ## Type parameters
///
/// * `F` - the coefficient subfield
/// * `P` - a packed field of `F`
/// * `Inner` - the inner polynomial commitment scheme over the extension field
///
/// [DP24]: <https://eprint.iacr.org/2024/504>
#[derive(Debug)]
pub struct BatchPCS<F, P, Inner> {
	inner: Inner,
	vars: usize, // number of variables
	m: usize,    // log number of multilinears
	_marker: PhantomData<(F, P)>,
}

impl<F, P, Inner> BatchPCS<F, P, Inner>
where
	F: Field,
	P: PackedField<Scalar = F>,
	Inner: PolyCommitScheme<P, F>,
{
	pub fn new(inner: Inner, vars: usize, m: usize) -> Result<Self, Error> {
		// check that the inner PCS has the correct number of variables.
		assert_eq!(inner.n_vars(), vars + m);
		Ok(Self {
			inner,
			vars, // the number of variables in the polynomials
			m,    // there are 2^{m_polys} multilinears
			_marker: PhantomData,
		})
	}
}

impl<F, P, Inner> PolyCommitScheme<P, F> for BatchPCS<F, P, Inner>
where
	F: Field,
	P: PackedField<Scalar = F>,
	Inner: PolyCommitScheme<P, F>,
{
	type Commitment = Inner::Commitment;
	type Committed = Inner::Committed;
	type Proof = Proof<F, P, Inner::Proof>;
	type Error = Error;

	fn n_vars(&self) -> usize {
		// the number of variables in the merged polynomial is vars + m, and
		// the merged polynomial is what inner has access to.
		assert_eq!(self.inner.n_vars(), self.vars + self.m);
		self.vars
	}

	fn commit<Data>(
		&self,
		polys: &[MultilinearExtension<P, Data>],
	) -> Result<(Self::Commitment, Self::Committed), Self::Error>
	where
		Data: Deref<Target = [P]> + Send + Sync,
	{
		// TODO: can I peform the following two lines in one line?
		let dummy_binding = merge_polynomials(polys, self.vars, self.m).unwrap();
		let merged_poly = [dummy_binding.to_ref()];
		self.inner
			.commit(&merged_poly)
			.map_err(|err| Error::InnerPCS(Box::new(err)))
	}

	fn prove_evaluation<Data, CH, Backend>(
		&self,
		challenger: &mut CH,
		committed: &Self::Committed,
		polys: &[MultilinearExtension<P, Data>],
		query: &[F],
		backend: Backend,
	) -> Result<Self::Proof, Self::Error>
	where
		Data: Deref<Target = [P]> + Send + Sync,
		CH: CanObserve<F> + CanObserve<Self::Commitment> + CanSample<F> + CanSampleBits<usize>,
		Backend: ComputationBackend,
	{
		if query.len() != self.vars {
			return Err(PolynomialError::IncorrectQuerySize {
				expected: self.vars,
			}
			.into());
		}
		// r'_0,...,r'_{m-1}
		let challenges = challenger.sample_vec(self.m);

		// new_query := challenges || query.
		// TODO: Aleksei suggests this can be done more simply via something like [a,b].concat();
		let new_query: Vec<F> = challenges
			.clone()
			.into_iter()
			.chain(query.iter().cloned())
			.collect();

		let merged_poly = merge_polynomials(polys, self.vars, self.m).unwrap();
		let merged_polys = [merged_poly.to_ref()];

		let inner_pcs_proof = self
			.inner
			.prove_evaluation(challenger, committed, &merged_polys, &new_query, backend)
			.unwrap();

		Ok(Proof {
			inner_pcs_proof,
			_marker: PhantomData,
		})
	}

	fn verify_evaluation<CH, Backend>(
		&self,
		challenger: &mut CH,
		commitment: &Self::Commitment,
		query: &[F],
		proof: Self::Proof,
		values: &[F],
		backend: Backend,
	) -> Result<(), Self::Error>
	where
		CH: CanObserve<F> + CanObserve<Self::Commitment> + CanSample<F> + CanSampleBits<usize>,
		Backend: ComputationBackend,
	{
		let mixing_challenges = challenger.sample_vec(self.m);
		// interpolate_from_evaluations is the multilinear polynomial
		// whose values on u\in B_{m} is s_u. Then the mixed evaluation, i.e.,
		// (tensor expansion of r')\cdot (s_u), is just given by *evaluating*
		// interpolate_from_evaluations on the mixing challenge.
		let interpolate_from_evaluations =
			MultilinearExtension::from_values(values.to_vec()).unwrap();
		let mixed_evaluation = interpolate_from_evaluations
			.evaluate(&MultilinearQuery::<P>::with_full_query(&mixing_challenges, backend.clone())?)
			.unwrap();
		let mixed_value = &[mixed_evaluation];

		// check that the inner PCS proof verifies with the value mixed_evaluation
		// TODO: is this too verbose to concatenate the challenges and the query?
		let new_query: Vec<F> = mixing_challenges
			.into_iter()
			.chain(query.iter().cloned())
			.collect();
		self.inner
			.verify_evaluation(
				challenger,
				commitment,
				&new_query,
				proof.inner_pcs_proof,
				mixed_value,
				backend,
			)
			.map_err(|err| Error::InnerPCS(Box::new(err)))?;
		Ok(())
	}

	fn proof_size(&self, _n_polys: usize) -> usize {
		todo!()
	}
}

/// A [`BatchPCS`] proof.
#[derive(Debug, Clone)]
pub struct Proof<F, P, Inner>
where
	F: Field,
	P: PackedField<Scalar = F>,
{
	inner_pcs_proof: Inner,
	_marker: PhantomData<(F, P)>,
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::{
		challenger::new_hasher_challenger,
		poly_commit::{tensor_pcs::find_proof_size_optimal_pcs, BasicTensorPCS, TensorPCS},
		reed_solomon::reed_solomon::ReedSolomonCode,
	};
	use binius_field::{
		arch::OptimalUnderlier128b,
		as_packed_field::{PackScalar, PackedType},
		underlier::{Divisible, UnderlierType},
		BinaryField, BinaryField128b, BinaryField1b, BinaryField32b, BinaryField8b,
		PackedBinaryField16x8b, PackedBinaryField1x128b, PackedBinaryField4x32b,
	};
	use binius_hal::make_backend;
	use binius_hash::GroestlHasher;
	use rand::{prelude::StdRng, SeedableRng};
	use std::iter::repeat_with;

	// TODO: write tests to take U, F, and Packed as generics.
	// I ran into issues because of some trait bounds with the TensorPCS so I gave up and just wrote
	// mutliple verbose tests.
	#[test]
	fn test_commit_prove_verify_success_128b() {
		type U = OptimalUnderlier128b;
		type F = BinaryField128b;
		type Packed = PackedBinaryField1x128b;
		let mut rng = StdRng::seed_from_u64(0);
		// set the variables: n_vars is the number of variables in the polynomials and 2ˆm is the number of polynomials.
		let n_vars = 7;
		let m = 3;
		let total_new_vars = n_vars + m;

		let multilins = (0..1 << m)
			.map(|_| {
				MultilinearExtension::from_values(
					repeat_with(|| <PackedType<U, F>>::random(&mut rng))
						.take(1 << (n_vars))
						.collect(),
				)
				.unwrap()
			})
			.collect::<Vec<_>>();

		let eval_point = repeat_with(|| <F as Field>::random(&mut rng))
			.take(n_vars)
			.collect::<Vec<_>>();

		let backend = make_backend();
		let eval_query =
			MultilinearQuery::<F>::with_full_query(&eval_point, backend.clone()).unwrap();
		let values = multilins
			.iter()
			.map(|x| x.evaluate(&eval_query).unwrap())
			.collect::<Vec<_>>();
		// for the TensorPCS, we need to have the following equality:
		// log_rows + log_dimension = inner_pcs.n_vars() = n + m
		// we force this; for simplicity, we take log_dimension_rs_code = 5, but this can also be changed.
		let log_dimension_rs_code = 5;
		let rs_code = ReedSolomonCode::new(log_dimension_rs_code, 2, Default::default()).unwrap();
		let n_test_queries = 10;

		let inner_pcs = <BasicTensorPCS<
			OptimalUnderlier128b,
			F,
			F,
			F,
			_,
			_,
			_,
		>>::new_using_groestl_merkle_tree(total_new_vars - log_dimension_rs_code, rs_code, n_test_queries)
		.unwrap();

		let backend = make_backend();
		let pcs: BatchPCS<F, Packed, TensorPCS<_, _, _, _, _, _, _, _>> =
			BatchPCS::new(inner_pcs, n_vars, m).unwrap();

		let polys = multilins.iter().map(|x| x.to_ref()).collect::<Vec<_>>();

		let (commitment, committed) = pcs.commit(&polys).unwrap();
		let mut challenger = new_hasher_challenger::<_, GroestlHasher<_>>();
		challenger.observe(commitment.clone());

		let mut prover_challenger = challenger.clone();
		let proof = pcs
			.prove_evaluation(
				&mut prover_challenger,
				&committed,
				&polys,
				&eval_point,
				backend.clone(),
			)
			.unwrap();

		let mut verifier_challenger = challenger.clone();
		pcs.verify_evaluation(
			&mut verifier_challenger,
			&commitment,
			&eval_point,
			proof,
			&values,
			backend.clone(),
		)
		.unwrap();
	}
	#[test]
	fn test_commit_prove_verify_success_32b() {
		type U = OptimalUnderlier128b;
		type F = BinaryField32b;
		type Packed = PackedBinaryField4x32b;
		let mut rng = StdRng::seed_from_u64(0);
		// set the variables: n_vars is the number of variables in the polynomials and 2ˆm is the number of polynomials.
		let n_vars = 3;
		let m = 3;
		let total_new_vars = n_vars + m;

		let multilins = (0..1 << m)
			.map(|_| {
				MultilinearExtension::from_values(
					repeat_with(|| <PackedType<U, F>>::random(&mut rng))
						.take(1 << (n_vars - Packed::LOG_WIDTH))
						.collect(),
				)
				.unwrap()
			})
			.collect::<Vec<_>>();

		let eval_point = repeat_with(|| <F as Field>::random(&mut rng))
			.take(n_vars)
			.collect::<Vec<_>>();

		let backend = make_backend();
		let eval_query =
			MultilinearQuery::<F>::with_full_query(&eval_point, backend.clone()).unwrap();
		let values = multilins
			.iter()
			.map(|x| x.evaluate(&eval_query).unwrap())
			.collect::<Vec<_>>();

		// log_rows + log_dimensions will be inner_pcs.n_vars(), which we need to be equal to n + m.
		// not sure how to modulate this.
		let log_dimension_rs_code = 3;
		let rs_code = ReedSolomonCode::new(log_dimension_rs_code, 2, Default::default()).unwrap();
		let n_test_queries = 10;

		let inner_pcs = <BasicTensorPCS<
			OptimalUnderlier128b,
			F,
			F,
			F,
			_,
			_,
			_,
		>>::new_using_groestl_merkle_tree(total_new_vars - log_dimension_rs_code, rs_code, n_test_queries)
		.unwrap();

		// no luck with using find_proof_size_optimal_pcs, keep getting None.
		// let inner_pcs =
		// 	find_proof_size_optimal_pcs::<U, F, F, F, _>(30, total_new_vars, 1, 2, false).unwrap();

		let backend = make_backend();
		let pcs: BatchPCS<F, Packed, TensorPCS<_, _, _, _, _, _, _, _>> =
			BatchPCS::new(inner_pcs, n_vars, m).unwrap();

		let polys = multilins.iter().map(|x| x.to_ref()).collect::<Vec<_>>();

		let (commitment, committed) = pcs.commit(&polys).unwrap();
		let mut challenger = new_hasher_challenger::<_, GroestlHasher<_>>();
		challenger.observe(commitment.clone());

		let mut prover_challenger = challenger.clone();
		let proof = pcs
			.prove_evaluation(
				&mut prover_challenger,
				&committed,
				&polys,
				&eval_point,
				backend.clone(),
			)
			.unwrap();

		let mut verifier_challenger = challenger.clone();
		pcs.verify_evaluation(
			&mut verifier_challenger,
			&commitment,
			&eval_point,
			proof,
			&values,
			backend.clone(),
		)
		.unwrap();
	}
}
