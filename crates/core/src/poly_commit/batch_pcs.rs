// Copyright 2024 Irreducible Inc.

use crate::{
	challenger::{CanObserve, CanSample, CanSampleBits},
	poly_commit::PolyCommitScheme,
	polynomial::Error as PolynomialError,
};

use crate::transcript::{CanRead, CanWrite};
use binius_field::{util::inner_product_unchecked, ExtensionField, Field, PackedField, TowerField};
use binius_hal::ComputationBackend;
use binius_math::{MultilinearExtension, MultilinearQuery};
use binius_utils::bail;
use bytemuck::zeroed_vec;
use std::{marker::PhantomData, ops::Deref};

#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error("number of polynomials must be less than {max}")]
	NumPolys { max: usize },
	#[error("expected all polynomials to have {expected} variables")]
	NumVars { expected: usize },
	#[error("number of variables in the inner PCS, {n_inner} is not what is expected, {n_vars} + {log_num_polys}")]
	NumVarsInnerOuter {
		n_inner: usize,
		n_vars: usize,
		log_num_polys: usize,
	},
	#[error("inner PCS error: {0}")]
	InnerPCS(#[source] Box<dyn std::error::Error + Send + Sync>),
	#[error("polynomial error: {0}")]
	Polynomial(#[from] PolynomialError),
	#[error("HAL error: {0}")]
	HalError(#[from] binius_hal::Error),
	#[error("Math error: {0}")]
	MathError(#[from] binius_math::Error),
}

/// Creates a new multilinear from a batch of multilinears via \emph{merging}.
///
/// If we have a collection of $2^m$ multilinear polynomials $t_u$, each of which is $n$-variate,
/// indexed over $u\in \{0,1\}^m$, construct the function $T$ on $\{0,1\}^{m+n}$ such that
/// $T(v||u):=t_u(v)$ for all $u\in \{0,1\}^m$ and $v\in \{0,1\}^n$. By abuse of notation
/// we consider $T$ an $m+n$-variate multilinear polynomial.
///
/// If the number of polynomials is not a power of 2, we pad the list up to the next power of two
/// with constant zero polynomials.
///
/// In [Example 4.10, DP23], there is a *different* definition of `merge`: they take: $T(u||v): = t_u(v).$
/// We choose our convention to make the actual process of merging slightly more efficient: indeed, it amounts
/// to simply concatenating the evaluations of the individual multilinears (as opposed to a mildly
/// more expensive interleaving process). This is all downstream of the fact that the underlying
/// list of evaluations of a multilinear is in Little Endian order.
fn merge_polynomials<P, Data>(
	n_vars: usize,
	log_n_polys: usize,
	polys: &[MultilinearExtension<P, Data>],
) -> Result<MultilinearExtension<P>, Error>
where
	P: PackedField,
	Data: Deref<Target = [P]> + Send + Sync,
{
	if polys.len() > 1 << log_n_polys {
		bail!(Error::NumPolys {
			max: 1 << log_n_polys
		});
	}
	if polys.iter().any(|poly| poly.n_vars() != n_vars) {
		bail!(Error::NumVars { expected: n_vars });
	}

	// $T(v||u):=t_{u}(v)$. Note that $v||u = 2^n * u + v$ as we are working with the little Endian binary expansion.
	let poly_packed_size = 1 << (n_vars - P::LOG_WIDTH);
	let mut packed_merged = zeroed_vec(poly_packed_size << log_n_polys);

	for (u, poly) in polys.iter().enumerate() {
		packed_merged[u * poly_packed_size..(u + 1) * poly_packed_size]
			.copy_from_slice(poly.evals())
	}

	Ok(MultilinearExtension::from_values(packed_merged)?)
}

/// A block-box batching scheme for multilinear commitments, as explained in [Section 5.3, DP24].
///
/// In particular, the scheme allows us to open the evaluations of a collection of multilinear
/// polynomials at a point $\vec{r}$.
///
/// Given a collection of $2^m$ multilinear $n$-variate polynomials $t_u$, jointly commit to them with the following
/// functionality: jointly open the evaluations of the polynomials at a point $\vec{r}=(r_0,\ldots,r_{n-1})$.
///
/// Suppose we have a collection of $2^m$ multilinear polynomials $t_u$,
/// each of which is $n$-variate, and we want to prove the evaluations at a point $\vec{r}=(r_0,\ldots ,r_{n-1})$
/// are $(s_u)$.
/// Build the multilinear, $n+m$-variate polynomial T, whose values on $B_{m+n}$ are given as:
/// $T(v||u) = t_u(v)$, for all u in $\{0,1\}^m$ and v in $\{0,1\}^n$.
/// Sample random challenges $\vec{r'}:=(r'_0,\ldots,r'_{m-1})$. Finally, pass off the evaluation of
/// $T$ at $(r_0,\ldots,r_{n-1}, r'_0,\ldots,r'_{m-1})$ to the inner polynomial commitment scheme.
///
/// If the prover is honest, $T(\vec{r}||\vec{r'})$ is the dot product of the tensor expansion of $\vec{r'}$ with
/// $(s_u)$. Equivalently, it is the *evaluation* of the multilinear polynomial defined via MLE on the hypercube:
/// $B_m\rightarrow \mathbb F$ by $u\mapsto s_u$ at the point $(r'_0,\ldots, r'_{m-1})$. Therefore, given the
/// claimed evaluations $(s_u)$, the verifier can compute the desired mixing herself.
///
/// ## Type parameters
///
/// * `P` - the packed coefficient subfield
/// * `FE` - an extension field of `P::Scalar` (used for the inner PCS)
/// * `Inner` - the inner polynomial commitment scheme over the extension field
///
/// [DP24]: <https://eprint.iacr.org/2024/504>
#[derive(Debug)]
pub struct BatchPCS<P, FE, InnerPCS>
where
	P: PackedField,
	FE: ExtensionField<P::Scalar> + TowerField,
	InnerPCS: PolyCommitScheme<P, FE>,
{
	inner: InnerPCS,
	n_vars: usize,        // number of variables
	log_num_polys: usize, // log_2 number of multilinears
	_marker: PhantomData<(P, FE)>,
}

impl<F, FE, P, Inner> BatchPCS<P, FE, Inner>
where
	F: Field,
	P: PackedField<Scalar = F>,
	FE: ExtensionField<F> + TowerField,
	Inner: PolyCommitScheme<P, FE>,
{
	pub fn new(inner: Inner, n_vars: usize, log_num_polys: usize) -> Result<Self, Error> {
		// check that the inner PCS has the correct number of variables.
		if inner.n_vars() != n_vars + log_num_polys {
			bail!(Error::NumVarsInnerOuter {
				n_inner: inner.n_vars(),
				n_vars,
				log_num_polys,
			});
		}
		Ok(Self {
			inner,
			n_vars,        // the number of variables in the polynomials
			log_num_polys, // there are 2^{log_num_polys} multilinears
			_marker: PhantomData,
		})
	}
}

impl<F, FE, P, Inner> PolyCommitScheme<P, FE> for BatchPCS<P, FE, Inner>
where
	F: Field,
	P: PackedField<Scalar = F>,
	FE: ExtensionField<F> + TowerField,
	Inner: PolyCommitScheme<P, FE>,
{
	type Commitment = Inner::Commitment;
	type Committed = Inner::Committed;
	type Proof = Proof<Inner::Proof>;
	type Error = Error;

	fn n_vars(&self) -> usize {
		self.n_vars
	}

	fn commit<Data>(
		&self,
		polys: &[MultilinearExtension<P, Data>],
	) -> Result<(Self::Commitment, Self::Committed), Self::Error>
	where
		Data: Deref<Target = [P]> + Send + Sync,
	{
		let merged_poly = merge_polynomials(self.n_vars, self.log_num_polys, polys)?;
		self.inner
			.commit(&[merged_poly])
			.map_err(|err| Error::InnerPCS(Box::new(err)))
	}

	fn prove_evaluation<Data, Transcript, Backend>(
		&self,
		transcript: &mut Transcript,
		committed: &Self::Committed,
		polys: &[MultilinearExtension<P, Data>],
		query: &[FE],
		backend: &Backend,
	) -> Result<Self::Proof, Self::Error>
	where
		Data: Deref<Target = [P]> + Send + Sync,
		Transcript: CanObserve<FE>
			+ CanObserve<Self::Commitment>
			+ CanSample<FE>
			+ CanSampleBits<usize>
			+ CanWrite,
		Backend: ComputationBackend,
	{
		if query.len() != self.n_vars {
			bail!(PolynomialError::IncorrectQuerySize {
				expected: self.n_vars
			});
		}
		// r'_0,...,r'_{m-1} are drawn from FE.
		let challenges = transcript.sample_vec(self.log_num_polys);

		// new_query := query || challenges.
		let new_query = query
			.iter()
			.copied()
			.chain(challenges.iter().copied())
			.collect::<Vec<_>>();

		let merged_poly = merge_polynomials(self.n_vars, self.log_num_polys, polys)?;

		let inner_pcs_proof = self
			.inner
			.prove_evaluation(transcript, committed, &[merged_poly], &new_query, backend)
			.map_err(|err| Error::InnerPCS(Box::new(err)))?;
		Ok(Proof(inner_pcs_proof))
	}

	fn verify_evaluation<Transcript, Backend>(
		&self,
		transcript: &mut Transcript,
		commitment: &Self::Commitment,
		query: &[FE],
		proof: Self::Proof,
		values: &[FE],
		backend: &Backend,
	) -> Result<(), Self::Error>
	where
		Transcript: CanObserve<FE>
			+ CanObserve<Self::Commitment>
			+ CanSample<FE>
			+ CanSampleBits<usize>
			+ CanRead,
		Backend: ComputationBackend,
	{
		if values.len() > 1 << self.log_num_polys {
			bail!(Error::NumPolys {
				max: 1 << self.log_num_polys
			});
		}

		let mixing_challenges = transcript.sample_vec(self.log_num_polys);
		let mixed_evaluation = inner_product_unchecked(
			MultilinearQuery::expand(&mixing_challenges).into_expansion(),
			values.iter().copied(),
		);
		let mixed_value = &[mixed_evaluation];

		// new_query := query || mixing_challenges.
		let new_query = query
			.iter()
			.copied()
			.chain(mixing_challenges.iter().copied())
			.collect::<Vec<_>>();

		// check that the inner PCS proof verifies with the value mixed_evaluation
		self.inner
			.verify_evaluation(transcript, commitment, &new_query, proof.0, mixed_value, backend)
			.map_err(|err| Error::InnerPCS(Box::new(err)))?;
		Ok(())
	}

	fn proof_size(&self, _n_polys: usize) -> usize {
		// The proof size is the size of the inner PCS for a single polynomial.
		self.inner.proof_size(1)
	}
}

/// A [`BatchPCS`] proof.
#[derive(Debug, Clone)]
pub struct Proof<Inner>(Inner);

#[cfg(test)]
mod tests {
	use super::*;
	use crate::{
		fiat_shamir::HasherChallenger,
		poly_commit::{tensor_pcs::find_proof_size_optimal_pcs, TensorPCS},
		reed_solomon::reed_solomon::ReedSolomonCode,
		transcript::{AdviceWriter, TranscriptWriter},
	};
	use binius_field::{
		arch::OptimalUnderlier128b, as_packed_field::PackedType, BinaryField128b, BinaryField32b,
	};
	use binius_hal::{make_portable_backend, ComputationBackendExt};
	use binius_ntt::NTTOptions;
	use groestl_crypto::Groestl256;
	use p3_util::log2_ceil_usize;
	use rand::{prelude::StdRng, SeedableRng};
	use std::iter::repeat_with;

	#[test]
	fn test_commit_prove_verify_success_128b() {
		type U = OptimalUnderlier128b;
		type F = BinaryField128b;
		let mut rng = StdRng::seed_from_u64(0);
		// set the variables: n_vars is the number of variables in the polynomials and 2ˆm is the number of polynomials.
		let n_vars = 7;
		let n_polys = 6;
		let m = log2_ceil_usize(n_polys);
		let total_new_vars = n_vars + m;

		let multilins = repeat_with(|| {
			MultilinearExtension::from_values(
				repeat_with(|| <PackedType<U, F>>::random(&mut rng))
					.take(1 << (n_vars))
					.collect(),
			)
			.unwrap()
		})
		.take(n_polys)
		.collect::<Vec<_>>();

		let eval_point = repeat_with(|| <F as Field>::random(&mut rng))
			.take(n_vars)
			.collect::<Vec<_>>();

		let backend = make_portable_backend();
		let eval_query = backend.multilinear_query::<F>(&eval_point).unwrap();
		let values = multilins
			.iter()
			.map(|x| x.evaluate(&eval_query).unwrap())
			.collect::<Vec<_>>();

		let inner_pcs =
			find_proof_size_optimal_pcs::<U, F, F, F, _>(100, total_new_vars, 1, 1, false).unwrap();

		let backend = make_portable_backend();
		let pcs = BatchPCS::new(inner_pcs, n_vars, m).unwrap();

		let polys = multilins.iter().map(|x| x.to_ref()).collect::<Vec<_>>();

		let (commitment, committed) = pcs.commit(&polys).unwrap();
		let mut prover_proof = crate::transcript::Proof {
			transcript: TranscriptWriter::<HasherChallenger<Groestl256>>::default(),
			advice: AdviceWriter::new(),
		};
		prover_proof.transcript.observe(commitment.clone());

		let proof = pcs
			.prove_evaluation(
				&mut prover_proof.transcript,
				&committed,
				&polys,
				&eval_point,
				&backend,
			)
			.unwrap();

		let mut verifier_proof = prover_proof.into_verifier();
		verifier_proof.transcript.observe(commitment.clone());
		pcs.verify_evaluation(
			&mut verifier_proof.transcript,
			&commitment,
			&eval_point,
			proof,
			&values,
			&backend,
		)
		.unwrap();
	}
	#[test]
	fn test_commit_prove_verify_success_32b() {
		type U = OptimalUnderlier128b;
		type F = BinaryField32b;
		type FE = BinaryField128b;
		type Packed = PackedType<U, F>;
		let mut rng = StdRng::seed_from_u64(0);
		// set the variables: n_vars is the number of variables in the polynomials and 2ˆm is the number of polynomials.
		let n_vars = 6;
		let n_polys = 6;
		let m = log2_ceil_usize(n_polys);

		let multilins = repeat_with(|| {
			MultilinearExtension::from_values(
				repeat_with(|| <PackedType<U, F>>::random(&mut rng))
					.take(1 << (n_vars - Packed::LOG_WIDTH))
					.collect(),
			)
			.unwrap()
		})
		.take(n_polys)
		.collect::<Vec<_>>();

		let eval_point = repeat_with(|| <FE as Field>::random(&mut rng))
			.take(n_vars)
			.collect::<Vec<_>>();

		let backend = make_portable_backend();
		let eval_query = backend.multilinear_query::<FE>(&eval_point).unwrap();
		let values = multilins
			.iter()
			.map(|x| x.evaluate(&eval_query).unwrap())
			.collect::<Vec<_>>();

		let rs_code = ReedSolomonCode::<Packed>::new(4, 1, NTTOptions::default()).unwrap();
		let inner_pcs =
			TensorPCS::<U, F, F, F, FE, _, _, _>::new_using_groestl_merkle_tree(5, rs_code, 10)
				.unwrap();

		let backend = make_portable_backend();
		let pcs = BatchPCS::new(inner_pcs, n_vars, m).unwrap();

		let polys = multilins.iter().map(|x| x.to_ref()).collect::<Vec<_>>();

		let (commitment, committed) = pcs.commit(&polys).unwrap();

		let mut prover_proof = crate::transcript::Proof {
			transcript: TranscriptWriter::<HasherChallenger<Groestl256>>::default(),
			advice: AdviceWriter::default(),
		};
		prover_proof.transcript.observe(commitment.clone());
		let proof = pcs
			.prove_evaluation(
				&mut prover_proof.transcript,
				&committed,
				&polys,
				&eval_point,
				&backend,
			)
			.unwrap();

		let mut verifier_proof = prover_proof.into_verifier();
		verifier_proof.transcript.observe(commitment.clone());
		pcs.verify_evaluation(
			&mut verifier_proof.transcript,
			&commitment,
			&eval_point,
			proof,
			&values,
			&backend,
		)
		.unwrap();
	}
}
