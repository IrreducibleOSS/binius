// Copyright 2024 Irreducible Inc.

use crate::{
	composition::BivariateProduct,
	fiat_shamir::{CanSample, CanSampleBits},
	poly_commit::PolyCommitScheme,
	polynomial::{multivariate::MultivariatePoly, Error as PolynomialError},
	protocols::sumcheck::{
		self, immediate_switchover_heuristic, prove::RegularSumcheckProver, BatchSumcheckOutput,
		CompositeSumClaim, SumcheckClaim,
	},
	tensor_algebra::TensorAlgebra,
	transcript::{read_u64, write_u64, AdviceReader, AdviceWriter, CanRead, CanWrite},
	transparent::ring_switch::RingSwitchEqInd,
};
use binius_field::{
	packed::iter_packed_slice, util::inner_product_unchecked, ExtensionField, Field,
	PackedExtension, PackedField, PackedFieldIndexable, TowerField,
};
use binius_hal::{ComputationBackend, ComputationBackendExt};
use binius_math::{
	EvaluationDomainFactory, MLEDirectAdapter, MLEEmbeddingAdapter, MultilinearExtension,
};
use std::{marker::PhantomData, mem, ops::Deref};
use tracing::instrument;

/// A polynomial commitment scheme constructed as a reduction to an inner PCS over a field
/// extension.
///
/// This implements the Construction 4.1 in [DP24]. The polynomial is committed by committing its
/// corresponding packed polynomial with the inner PCS. Evaluation proofs consist of a sumcheck
/// reduction followed by a PCS evaluation proof for the packed polynomial using the inner PCS.
///
/// ## Type parameters
///
/// * `F` - the coefficient subfield
/// * `FDomain` - the field containing the sumcheck evaluation domains
/// * `PE` - a packed extension field of `F` that is cryptographically big
/// * `DomainFactory` - a domain factory for the sumcheck reduction
/// * `Inner` - the inner polynomial commitment scheme over the extension field
///
/// [DP24]: <https://eprint.iacr.org/2024/504>
#[derive(Debug)]
pub struct RingSwitchPCS<F, FDomain, PE, DomainFactory, Inner> {
	inner: Inner,
	domain_factory: DomainFactory,
	_marker: PhantomData<(F, FDomain, PE)>,
}

impl<F, FDomain, PE, DomainFactory, Inner> RingSwitchPCS<F, FDomain, PE, DomainFactory, Inner>
where
	F: Field,
	PE: PackedField,
	PE::Scalar: ExtensionField<F> + TowerField,
	Inner: PolyCommitScheme<PE, PE::Scalar>,
{
	pub fn new(inner: Inner, domain_factory: DomainFactory) -> Result<Self, Error> {
		Ok(Self {
			inner,
			domain_factory,
			_marker: PhantomData,
		})
	}

	/// Returns $\kappa$, the base-2 logarithm of the extension degree.
	pub const fn kappa() -> usize {
		<TensorAlgebra<F, PE::Scalar>>::kappa()
	}
}

impl<F, FDomain, FE, P, PE, DomainFactory, Inner> PolyCommitScheme<P, FE>
	for RingSwitchPCS<F, FDomain, PE, DomainFactory, Inner>
where
	F: TowerField,
	FDomain: Field,
	FE: ExtensionField<F>
		+ ExtensionField<FDomain>
		+ PackedField<Scalar = FE>
		+ PackedExtension<F>
		+ TowerField,
	P: PackedField<Scalar = F>,
	PE: PackedFieldIndexable<Scalar = FE>
		+ PackedExtension<F, PackedSubfield = P>
		+ PackedExtension<FDomain>,
	DomainFactory: EvaluationDomainFactory<FDomain>,
	Inner: PolyCommitScheme<PE, FE>,
{
	type Commitment = Inner::Commitment;
	type Committed = Inner::Committed;
	type Error = Error;

	fn n_vars(&self) -> usize {
		self.inner.n_vars() + Self::kappa()
	}

	fn commit<Data>(
		&self,
		polys: &[MultilinearExtension<P, Data>],
	) -> Result<(Self::Commitment, Self::Committed), Self::Error>
	where
		Data: Deref<Target = [P]> + Send + Sync,
	{
		if polys.len() != 1 {
			todo!("handle batches of size greater than 1");
		}

		let packed_polys = polys
			.iter()
			.map(|poly| {
				let packed_evals = <PE as PackedExtension<F>>::cast_exts(poly.evals());
				MultilinearExtension::from_values_slice(packed_evals)
			})
			.collect::<Result<Vec<_>, _>>()?;
		self.inner
			.commit(&packed_polys)
			.map_err(|err| Error::InnerPCS(Box::new(err)))
	}

	// Clippy allow is due to bug: https://github.com/rust-lang/rust-clippy/pull/12892
	#[allow(clippy::needless_borrows_for_generic_args)]
	#[instrument(skip_all, name = "ring_switch::prove_evaluation")]
	fn prove_evaluation<Data, Transcript, Backend>(
		&self,
		advice: &mut AdviceWriter,
		mut transcript: &mut Transcript,
		committed: &Self::Committed,
		polys: &[MultilinearExtension<P, Data>],
		query: &[PE::Scalar],
		backend: &Backend,
	) -> Result<(), Self::Error>
	where
		Data: Deref<Target = [P]> + Send + Sync,
		Transcript: CanWrite + CanSample<PE::Scalar> + CanSampleBits<usize>,
		Backend: ComputationBackend,
	{
		if query.len() != self.n_vars() {
			return Err(PolynomialError::IncorrectQuerySize {
				expected: self.n_vars(),
			}
			.into());
		}
		if polys.len() != 1 {
			todo!("handle batches of size greater than 1");
		}

		let poly = &polys[0];
		let packed_poly = MultilinearExtension::from_values_slice(
			<PE as PackedExtension<F>>::cast_exts(poly.evals()),
		)?;

		let (_, query_from_kappa) = query.split_at(Self::kappa());

		let expanded_query = backend.multilinear_query::<PE>(query_from_kappa)?;
		let poly = MLEEmbeddingAdapter::from(poly.to_ref());
		let partial_eval = backend.evaluate_partial_high(&poly, expanded_query.to_ref())?;
		let sumcheck_eval =
			TensorAlgebra::<F, _>::new(iter_packed_slice(partial_eval.evals()).collect());

		let sumcheck_eval_elems = sumcheck_eval.vertical_elems();
		write_u64(advice, sumcheck_eval_elems.len() as u64);
		transcript.write_scalar_slice(sumcheck_eval.vertical_elems());

		// The challenges used to mix the rows of the tensor algebra coefficients.
		let tensor_mixing_challenges = transcript.sample_vec(Self::kappa());

		let sumcheck_claim = reduce_tensor_claim(
			self.n_vars(),
			sumcheck_eval.clone(),
			&tensor_mixing_challenges,
			backend,
		);
		let rs_eq = RingSwitchEqInd::<F, _>::new(
			query_from_kappa.to_vec(),
			tensor_mixing_challenges.to_vec(),
		)
		.map_err(|_| Error::RingSwitchConstructionFailed)?;

		let transparent = rs_eq.multilinear_extension::<PE, _>(backend)?;

		let sumcheck_prover = RegularSumcheckProver::<_, PE, _, _, _>::new(
			[packed_poly.to_ref(), transparent.to_ref()]
				.map(MLEDirectAdapter::from)
				.into(),
			sumcheck_claim.composite_sums().iter().cloned(),
			&self.domain_factory,
			immediate_switchover_heuristic,
			backend,
		)?;
		let sumcheck_output = sumcheck::batch_prove(vec![sumcheck_prover], &mut transcript)?;
		let (_, eval_point) =
			verify_sumcheck_output(sumcheck_output, query_from_kappa, &tensor_mixing_challenges)?;

		self.inner
			.prove_evaluation(advice, transcript, committed, &[packed_poly], &eval_point, backend)
			.map_err(|err| Error::InnerPCS(Box::new(err)))?;

		Ok(())
	}

	fn verify_evaluation<Transcript, Backend>(
		&self,
		advice: &mut AdviceReader,
		mut transcript: &mut Transcript,
		commitment: &Self::Commitment,
		query: &[FE],
		values: &[FE],
		backend: &Backend,
	) -> Result<(), Self::Error>
	where
		Transcript: CanSample<FE> + CanSampleBits<usize> + CanRead,
		Backend: ComputationBackend,
	{
		if query.len() != self.n_vars() {
			return Err(PolynomialError::IncorrectQuerySize {
				expected: self.n_vars(),
			}
			.into());
		}
		if values.len() != 1 {
			todo!("handle batches of size greater than 1");
		}

		let (query_to_kappa, query_from_kappa) = query.split_at(Self::kappa());

		let sumcheck_eval_elems_len = read_u64(advice).map_err(Error::TranscriptError)? as usize;
		let sumcheck_eval = transcript
			.read_scalar_slice(sumcheck_eval_elems_len)
			.map_err(Error::TranscriptError)?;
		// This is s₀ in Protocol 4.1
		let sumcheck_eval = TensorAlgebra::new(sumcheck_eval);

		// Check that the claimed sum is consistent with the tensor algebra element received.
		let expanded_query = backend.multilinear_query::<FE>(query_to_kappa)?;
		let computed_eval =
			MultilinearExtension::from_values_slice(sumcheck_eval.vertical_elems())?
				.evaluate(&expanded_query)?;
		if values[0] != computed_eval {
			return Err(VerificationError::IncorrectEvaluation.into());
		}

		// The challenges used to mix the rows of the tensor algebra coefficients.
		let tensor_mixing_challenges = transcript.sample_vec(Self::kappa());

		let sumcheck_claim =
			reduce_tensor_claim(self.n_vars(), sumcheck_eval, &tensor_mixing_challenges, &backend);
		let output = sumcheck::batch_verify(&[sumcheck_claim], &mut transcript)?;

		let (eval, eval_point) =
			verify_sumcheck_output(output, query_from_kappa, &tensor_mixing_challenges)?;

		self.inner
			.verify_evaluation(advice, transcript, commitment, &eval_point, &[eval], backend)
			.map_err(|err| Error::InnerPCS(Box::new(err)))
	}

	fn proof_size(&self, n_polys: usize) -> usize {
		if n_polys != 1 {
			todo!("handle batches of size greater than 1");
		}
		let sumcheck_eval_size = <TensorAlgebra<F, FE>>::byte_size();
		// We have a product of two multilinear polynomials. Each round of the sumcheck
		// (of which there are self.inner.n_vars()) has 2 FE elements, due to an optimization.
		// The final evaluations yield 2 FE elements.
		let sumcheck_proof_size = mem::size_of::<FE>() * (2 * self.inner.n_vars() + 2);
		sumcheck_eval_size + sumcheck_proof_size + self.inner.proof_size(n_polys)
	}
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error("inner PCS error: {0}")]
	InnerPCS(#[source] Box<dyn std::error::Error + Send + Sync>),
	#[error("sumcheck error: {0}")]
	Sumcheck(#[from] sumcheck::Error),
	#[error("polynomial error: {0}")]
	Polynomial(#[from] PolynomialError),
	#[error("failed to construct the eq indicator evaluation")]
	RingSwitchConstructionFailed,
	#[error("failed to compute the eq indicator evaluation")]
	RingSwitchComputationFailed,
	#[error("verification failure: {0}")]
	Verification(#[from] VerificationError),
	#[error("HAL error: {0}")]
	HalError(#[from] binius_hal::Error),
	#[error("Math error: {0}")]
	MathError(#[from] binius_math::Error),
	#[error("Transcript error: {0}")]
	TranscriptError(#[from] crate::transcript::Error),
}

#[derive(Debug, thiserror::Error)]
pub enum VerificationError {
	#[error("evaluation value is inconsistent with the tensor evaluation")]
	IncorrectEvaluation,
	#[error("ring switch eq indicator evaluation is incorrect")]
	IncorrectRingSwitchIndEvaluation,
}

pub(super) fn reduce_tensor_claim<F, FE, Backend>(
	n_vars: usize,
	tensor_sum: TensorAlgebra<F, FE>,
	tensor_mixing_challenges: &[FE],
	backend: &Backend,
) -> SumcheckClaim<FE, BivariateProduct>
where
	F: Field,
	FE: ExtensionField<F> + PackedField<Scalar = FE> + PackedExtension<F>,
	Backend: ComputationBackend,
{
	// Precondition
	let kappa = <TensorAlgebra<F, FE>>::kappa();
	assert_eq!(tensor_mixing_challenges.len(), kappa);

	let expanded_mixing_coeffs = backend
		.tensor_product_full_query(tensor_mixing_challenges)
		.expect("FE extension degree is less than 2^31");
	let mixed_sum = inner_product_unchecked::<FE, _>(
		tensor_sum.transpose().vertical_elems().iter().copied(),
		expanded_mixing_coeffs.iter().copied(),
	);

	SumcheckClaim::new(
		n_vars - kappa, // Number of variables in the packed polynomial
		// First polynomial is the packed committed polynomial and the second is the ring-switching
		// eq indicator
		BivariateProduct {}.degree(),
		vec![CompositeSumClaim {
			composition: BivariateProduct {},
			sum: mixed_sum,
		}],
	)
	.expect("composition degree matches number of multilinears")
}

/// Reduce the output of the ring-switching sumcheck to an inner PCS claim.
///
/// ## Arguments
///
/// * `output` - the sumcheck output
/// * `eval_point` - the evaluation point of the packed polynomial
/// * `tensor_mixing_challenges` - the mixing challenges used to mix the tensor algebra rows
fn verify_sumcheck_output<F, FE>(
	output: BatchSumcheckOutput<FE>,
	eval_point: &[FE],
	tensor_mixing_challenges: &[FE],
) -> Result<(FE, Vec<FE>), Error>
where
	F: TowerField,
	FE: ExtensionField<F> + PackedField<Scalar = FE> + PackedExtension<F> + TowerField,
{
	// Precondition
	let kappa = <TensorAlgebra<F, FE>>::kappa();
	assert_eq!(tensor_mixing_challenges.len(), kappa);

	let BatchSumcheckOutput {
		challenges: sumcheck_challenges,
		mut multilinear_evals,
	} = output;

	// Assertions are preconditions
	assert_eq!(eval_point.len(), sumcheck_challenges.len());
	assert_eq!(multilinear_evals.len(), 1);
	let multilinear_evals = multilinear_evals
		.pop()
		.expect("multilinear_evals has exactly one element");
	assert_eq!(multilinear_evals.len(), 2);

	let rs_eq =
		RingSwitchEqInd::<F, _>::new(eval_point.to_vec(), tensor_mixing_challenges.to_vec())
			.map_err(|_| Error::RingSwitchConstructionFailed)?;
	let ring_switch_eq_ind_eval = rs_eq
		.evaluate(&sumcheck_challenges)
		.map_err(|_| Error::RingSwitchComputationFailed)?;

	if multilinear_evals[1] != ring_switch_eq_ind_eval {
		return Err(VerificationError::IncorrectRingSwitchIndEvaluation.into());
	}

	Ok((multilinear_evals[0], sumcheck_challenges))
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::{
		fiat_shamir::HasherChallenger,
		merkle_tree_vcs::BinaryMerkleTreeProver,
		poly_commit::FRIPCS,
		transcript::{AdviceWriter, TranscriptWriter},
	};
	use binius_field::{
		arch::OptimalUnderlier128b,
		as_packed_field::{PackScalar, PackedType},
		underlier::{Divisible, UnderlierType},
		BinaryField128b, BinaryField1b, BinaryField32b, BinaryField8b,
	};
	use binius_hal::make_portable_backend;
	use binius_hash::{GroestlDigestCompression, GroestlHasher};
	use binius_math::IsomorphicEvaluationDomainFactory;
	use binius_ntt::NTTOptions;
	use binius_utils::checked_arithmetics::checked_log_2;
	use groestl_crypto::Groestl256;
	use rand::{prelude::StdRng, SeedableRng};
	use std::iter::repeat_with;

	fn test_commit_prove_verify_success<U, F, FE>()
	where
		U: UnderlierType
			+ PackScalar<F>
			+ PackScalar<FE>
			+ PackScalar<BinaryField8b>
			+ Divisible<u8>,
		F: TowerField,
		FE: TowerField
			+ ExtensionField<F>
			+ ExtensionField<BinaryField8b>
			+ PackedField<Scalar = FE>
			+ PackedExtension<F>
			+ PackedExtension<BinaryField8b, PackedSubfield: PackedFieldIndexable>,
		PackedType<U, FE>: PackedFieldIndexable,
	{
		let mut rng = StdRng::seed_from_u64(0);
		let n_vars = 8 + checked_log_2(<FE as ExtensionField<F>>::DEGREE);

		let multilin = MultilinearExtension::from_values(
			repeat_with(|| <PackedType<U, F>>::random(&mut rng))
				.take(1 << (n_vars - <PackedType<U, F>>::LOG_WIDTH))
				.collect(),
		)
		.unwrap();
		assert_eq!(multilin.n_vars(), n_vars);

		let eval_point = repeat_with(|| <FE as Field>::random(&mut rng))
			.take(n_vars)
			.collect::<Vec<_>>();

		let backend = make_portable_backend();
		let eval_query = backend.multilinear_query::<FE>(&eval_point).unwrap();
		let eval = multilin.evaluate(&eval_query).unwrap();

		// let rs_code = ReedSolomonCode::new(5, 2, Default::default()).unwrap();
		// let n_test_queries = 10;
		let domain_factory = IsomorphicEvaluationDomainFactory::<BinaryField8b>::default();
		let merkle_prover = BinaryMerkleTreeProver::<_, GroestlHasher<_>, _>::new(
			GroestlDigestCompression::default(),
		);
		let inner_pcs = FRIPCS::<FE, FE, FE, PackedType<U, FE>, _, _, _>::new(
			8,
			2,
			vec![2, 2, 2],
			32,
			merkle_prover,
			domain_factory,
			NTTOptions::default(),
		)
		.unwrap();

		let domain_factory = IsomorphicEvaluationDomainFactory::<BinaryField8b>::default();
		let backend = make_portable_backend();
		let pcs =
			RingSwitchPCS::<F, BinaryField8b, _, _, _>::new(inner_pcs, domain_factory).unwrap();

		let (mut commitment, committed) = pcs.commit(&[multilin.to_ref()]).unwrap();

		let mut prover_challenger = crate::transcript::Proof {
			transcript: TranscriptWriter::<HasherChallenger<Groestl256>>::default(),
			advice: AdviceWriter::default(),
		};
		prover_challenger.transcript.write_packed(commitment);
		pcs.prove_evaluation(
			&mut prover_challenger.advice,
			&mut prover_challenger.transcript,
			&committed,
			&[multilin],
			&eval_point,
			&backend,
		)
		.unwrap();

		let mut verifier_challenger = prover_challenger.into_verifier();
		commitment = verifier_challenger.transcript.read_packed().unwrap();
		pcs.verify_evaluation(
			&mut verifier_challenger.advice,
			&mut verifier_challenger.transcript,
			&commitment,
			&eval_point,
			&[eval],
			&backend,
		)
		.unwrap();

		verifier_challenger.finalize().unwrap()
	}

	#[test]
	fn test_commit_prove_verify_success_1b_128b() {
		test_commit_prove_verify_success::<OptimalUnderlier128b, BinaryField1b, BinaryField128b>();
	}

	#[test]
	fn test_commit_prove_verify_success_32b_128b() {
		test_commit_prove_verify_success::<OptimalUnderlier128b, BinaryField32b, BinaryField128b>();
	}
}
