// Copyright 2024 Ulvetanna Inc.

use crate::{
	challenger::{CanObserve, CanSample, CanSampleBits},
	composition::BivariateProduct,
	poly_commit::PolyCommitScheme,
	polynomial::{Error as PolynomialError, MultilinearExtension, MultilinearQuery},
	protocols::{
		abstract_sumcheck::ReducedClaim,
		sumcheck_v2::{
			self, prove::RegularSumcheckProver, BatchSumcheckOutput, CompositeSumClaim,
			SumcheckClaim,
		},
	},
	tensor_algebra::TensorAlgebra,
};
use binius_field::{
	packed::iter_packed_slice, util::inner_product_unchecked, ExtensionField, Field,
	PackedExtension, PackedField, PackedFieldIndexable,
};
use binius_hal::{ComputationBackend, VecOrImmutableSlice};
use binius_math::EvaluationDomainFactory;
use rayon::prelude::*;
use std::{iter, marker::PhantomData, mem, ops::Deref};

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
	PE::Scalar: ExtensionField<F>,
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
	F: Field,
	FDomain: Field,
	FE: ExtensionField<F> + ExtensionField<FDomain> + PackedField<Scalar = FE> + PackedExtension<F>,
	P: PackedField<Scalar = F>,
	PE: PackedFieldIndexable<Scalar = FE>
		+ PackedExtension<F, PackedSubfield = P>
		+ PackedExtension<FDomain>,
	DomainFactory: EvaluationDomainFactory<FDomain>,
	Inner: PolyCommitScheme<PE, FE>,
{
	type Commitment = Inner::Commitment;
	type Committed = Inner::Committed;
	type Proof = Proof<F, FE, Inner::Proof>;
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
	fn prove_evaluation<Data, CH, Backend>(
		&self,
		mut challenger: &mut CH,
		committed: &Self::Committed,
		polys: &[MultilinearExtension<P, Data>],
		query: &[PE::Scalar],
		backend: Backend,
	) -> Result<Self::Proof, Self::Error>
	where
		Data: Deref<Target = [P]> + Send + Sync,
		CH: CanObserve<PE::Scalar>
			+ CanObserve<Self::Commitment>
			+ CanSample<PE::Scalar>
			+ CanSampleBits<usize>,
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

		let expanded_query =
			MultilinearQuery::<PE>::with_full_query(query_from_kappa, backend.clone())?;
		let partial_eval = poly.evaluate_partial_high(&expanded_query)?;
		let sumcheck_eval =
			TensorAlgebra::<F, _>::new(iter_packed_slice(partial_eval.evals()).collect());

		challenger.observe_slice(sumcheck_eval.vertical_elems());

		// The challenges used to mix the rows of the tensor algebra coefficients.
		let tensor_mixing_challenges = challenger.sample_vec(Self::kappa());

		let sumcheck_claim = reduce_tensor_claim(
			self.n_vars(),
			sumcheck_eval.clone(),
			&tensor_mixing_challenges,
			backend.clone(),
		);
		let transparent =
			MultilinearExtension::from_values_generic(ring_switch_eq_ind_partial_eval(
				query_from_kappa,
				&tensor_mixing_challenges,
				backend.clone(),
			)?)?;
		let sumcheck_prover = RegularSumcheckProver::<_, PE, _, _, _>::new(
			vec![
				MultilinearExtension::<PE, _>::specialize(packed_poly.to_ref()),
				MultilinearExtension::<PE, _>::specialize(transparent.to_ref()),
			],
			sumcheck_claim.composite_sums().iter().cloned(),
			&self.domain_factory,
			|_| 1,
			backend.clone(),
		)?;
		let (sumcheck_output, sumcheck_proof) =
			sumcheck_v2::batch_prove(vec![sumcheck_prover], &mut challenger)?;
		let ReducedClaim {
			eval: _,
			eval_point,
		} = verify_sumcheck_output(
			sumcheck_output,
			query_from_kappa,
			&tensor_mixing_challenges,
			backend.clone(),
		)?;

		let inner_pcs_proof = self
			.inner
			.prove_evaluation(challenger, committed, &[packed_poly], &eval_point, backend)
			.map_err(|err| Error::InnerPCS(Box::new(err)))?;

		Ok(Proof {
			sumcheck_eval,
			sumcheck_proof,
			inner_pcs_proof,
		})
	}

	fn verify_evaluation<CH, Backend>(
		&self,
		mut challenger: &mut CH,
		commitment: &Self::Commitment,
		query: &[FE],
		proof: Self::Proof,
		values: &[FE],
		backend: Backend,
	) -> Result<(), Self::Error>
	where
		CH: CanObserve<FE> + CanObserve<Self::Commitment> + CanSample<FE> + CanSampleBits<usize>,
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

		let Proof {
			// This is s₀ in Protocol 4.1
			sumcheck_eval,
			sumcheck_proof,
			inner_pcs_proof,
		} = proof;

		let (query_to_kappa, query_from_kappa) = query.split_at(Self::kappa());

		challenger.observe_slice(sumcheck_eval.vertical_elems());

		// Check that the claimed sum is consistent with the tensor algebra element received.
		let expanded_query =
			MultilinearQuery::<FE>::with_full_query(query_to_kappa, backend.clone())?;
		let computed_eval =
			MultilinearExtension::from_values_slice(sumcheck_eval.vertical_elems())?
				.evaluate(&expanded_query)?;
		if values[0] != computed_eval {
			return Err(VerificationError::IncorrectEvaluation.into());
		}

		// The challenges used to mix the rows of the tensor algebra coefficients.
		let tensor_mixing_challenges = challenger.sample_vec(Self::kappa());

		let sumcheck_claim = reduce_tensor_claim(
			self.n_vars(),
			sumcheck_eval,
			&tensor_mixing_challenges,
			backend.clone(),
		);
		let output = sumcheck_v2::batch_verify(&[sumcheck_claim], sumcheck_proof, &mut challenger)?;

		let ReducedClaim { eval, eval_point } = verify_sumcheck_output(
			output,
			query_from_kappa,
			&tensor_mixing_challenges,
			backend.clone(),
		)?;

		self.inner
			.verify_evaluation(
				challenger,
				commitment,
				&eval_point,
				inner_pcs_proof,
				&[eval],
				backend,
			)
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

/// A [`RingSwitchPCS`] proof.
#[derive(Debug, Clone)]
pub struct Proof<F, FE, Inner>
where
	F: Field,
	FE: ExtensionField<F>,
{
	sumcheck_eval: TensorAlgebra<F, FE>,
	sumcheck_proof: sumcheck_v2::Proof<FE>,
	inner_pcs_proof: Inner,
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error("inner PCS error: {0}")]
	InnerPCS(#[source] Box<dyn std::error::Error + Send + Sync>),
	#[error("sumcheck error: {0}")]
	Sumcheck(#[from] sumcheck_v2::Error),
	#[error("polynomial error: {0}")]
	Polynomial(#[from] PolynomialError),
	#[error("verification failure: {0}")]
	Verification(#[from] VerificationError),
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
	backend: Backend,
) -> SumcheckClaim<FE, BivariateProduct>
where
	F: Field,
	FE: ExtensionField<F> + PackedField<Scalar = FE> + PackedExtension<F>,
	Backend: ComputationBackend,
{
	// Precondition
	let kappa = <TensorAlgebra<F, FE>>::kappa();
	assert_eq!(tensor_mixing_challenges.len(), kappa);

	let expanded_mixing_coeffs =
		MultilinearQuery::<FE>::with_full_query(tensor_mixing_challenges, backend)
			.expect("FE extension degree is less than 2^31")
			.into_expansion();
	let mixed_sum = inner_product_unchecked::<FE, _>(
		tensor_sum.transpose().vertical_elems().iter().copied(),
		// TODO: Avoid copying.
		expanded_mixing_coeffs.iter().copied(),
	);

	SumcheckClaim::new(
		n_vars - kappa, // Number of variables in the packed polynomial
		// First polynomial is the packed committed polynomial and the second is the ring-switching
		// eq indicator
		BivariateProduct.degree(),
		vec![CompositeSumClaim {
			composition: BivariateProduct,
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
fn verify_sumcheck_output<F, FE, Backend>(
	output: BatchSumcheckOutput<FE>,
	eval_point: &[FE],
	tensor_mixing_challenges: &[FE],
	backend: Backend,
) -> Result<ReducedClaim<FE>, VerificationError>
where
	F: Field,
	FE: ExtensionField<F> + PackedField<Scalar = FE> + PackedExtension<F>,
	Backend: ComputationBackend,
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

	let ring_switch_eq_ind_eval = evaluate_ring_switch_eq_ind::<F, _, _>(
		eval_point,
		&sumcheck_challenges,
		tensor_mixing_challenges,
		backend,
	);
	if multilinear_evals[1] != ring_switch_eq_ind_eval {
		return Err(VerificationError::IncorrectRingSwitchIndEvaluation);
	}

	Ok(ReducedClaim {
		eval: multilinear_evals[0],
		eval_point: sumcheck_challenges,
	})
}

pub fn evaluate_ring_switch_eq_ind<FS, F, Backend>(
	eval_point: &[F],
	sumcheck_challenges: &[F],
	mixing_challenges: &[F],
	backend: Backend,
) -> F
where
	FS: Field,
	F: ExtensionField<FS> + PackedField<Scalar = F> + PackedExtension<FS>,
	Backend: ComputationBackend,
{
	assert_eq!(mixing_challenges.len(), <TensorAlgebra<FS, F>>::kappa());

	let tensor_eval = iter::zip(eval_point.iter().copied(), sumcheck_challenges.iter().copied())
		.fold(TensorAlgebra::one(), |eval, (vert_i, hztl_i)| {
			// This formula is specific to characteristic 2 fields
			// Here we know that $h v + (1 - h) (1 - v) = 1 + h + v$.
			let vert_scaled = eval.clone().scale_vertical(vert_i);
			let hztl_scaled = eval.clone().scale_horizontal(hztl_i);
			eval + &vert_scaled + &hztl_scaled
		});

	let expanded_mixing_coeffs = MultilinearQuery::<F>::with_full_query(mixing_challenges, backend)
		.expect("F extension degree is less than 2^31")
		.into_expansion();
	let folded_eval = inner_product_unchecked::<F, _>(
		tensor_eval.transpose().vertical_elems().iter().copied(),
		// TODO: Avoid copying.
		expanded_mixing_coeffs.iter().copied(),
	);
	folded_eval
}

/// The evaluations of the partially evaluated ring-switch eq indicator over the boolean hypercube.
///
/// See [DP24] Section 4.
///
/// [DP24]: <https://eprint.iacr.org/2024/504>
pub fn ring_switch_eq_ind_partial_eval<FS, F, P, Backend>(
	eval_point: &[F],
	mixing_challenges: &[F],
	backend: Backend,
) -> Result<VecOrImmutableSlice<P>, PolynomialError>
where
	FS: Field,
	F: ExtensionField<FS> + PackedField<Scalar = F> + PackedExtension<FS>,
	P: PackedFieldIndexable<Scalar = F>,
	Backend: ComputationBackend,
{
	assert_eq!(mixing_challenges.len(), <TensorAlgebra<FS, F>>::kappa());
	let expanded_mixing_coeffs =
		MultilinearQuery::<F>::with_full_query(mixing_challenges, backend.clone())?;
	let mut evals = MultilinearQuery::<P>::with_full_query(eval_point, backend)?.into_expansion();
	P::unpack_scalars_mut(&mut evals)
		.par_iter_mut()
		.for_each(|val| {
			let vert = *val;
			*val = inner_product_unchecked(
				expanded_mixing_coeffs.expansion().iter().copied(),
				ExtensionField::<FS>::iter_bases(&vert),
			);
		});
	Ok(evals)
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::{
		challenger::new_hasher_challenger, poly_commit::BasicTensorPCS,
		reed_solomon::reed_solomon::ReedSolomonCode,
	};
	use binius_backend_provider::make_best_backend;
	use binius_field::{
		arch::OptimalUnderlier128b,
		as_packed_field::{PackScalar, PackedType},
		underlier::{Divisible, UnderlierType},
		BinaryField, BinaryField128b, BinaryField1b, BinaryField32b, BinaryField8b,
	};
	use binius_hash::GroestlHasher;
	use binius_math::IsomorphicEvaluationDomainFactory;
	use binius_utils::checked_arithmetics::checked_log_2;
	use rand::{prelude::StdRng, SeedableRng};
	use std::iter::repeat_with;

	fn test_commit_prove_verify_success<U, F, FE>()
	where
		U: UnderlierType
			+ PackScalar<F>
			+ PackScalar<FE>
			+ PackScalar<BinaryField8b>
			+ Divisible<u8>,
		F: Field,
		FE: BinaryField
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

		let backend = make_best_backend();
		let eval_query =
			MultilinearQuery::<FE>::with_full_query(&eval_point, backend.clone()).unwrap();
		let eval = multilin.evaluate(&eval_query).unwrap();

		let rs_code = ReedSolomonCode::new(5, 2, Default::default()).unwrap();
		let n_test_queries = 10;
		let inner_pcs = BasicTensorPCS::<U, FE, FE, FE, _, _, _>::new_using_groestl_merkle_tree(
			3,
			rs_code,
			n_test_queries,
		)
		.unwrap();

		let domain_factory = IsomorphicEvaluationDomainFactory::<BinaryField8b>::default();
		let backend = make_best_backend();
		let pcs =
			RingSwitchPCS::<F, BinaryField8b, _, _, _>::new(inner_pcs, domain_factory).unwrap();

		let (commitment, committed) = pcs.commit(&[multilin.to_ref()]).unwrap();

		let mut challenger = new_hasher_challenger::<_, GroestlHasher<_>>();
		challenger.observe(commitment.clone());

		let mut prover_challenger = challenger.clone();
		let proof = pcs
			.prove_evaluation(
				&mut prover_challenger,
				&committed,
				&[multilin],
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
			&[eval],
			backend.clone(),
		)
		.unwrap();
	}

	#[test]
	fn test_commit_prove_verify_success_1b_128b() {
		test_commit_prove_verify_success::<OptimalUnderlier128b, BinaryField1b, BinaryField128b>();
	}

	#[test]
	fn test_commit_prove_verify_success_32b_128b() {
		test_commit_prove_verify_success::<OptimalUnderlier128b, BinaryField32b, BinaryField128b>();
	}

	#[test]
	fn test_ring_switch_eq_ind() {
		type F = BinaryField8b;
		type FE = BinaryField128b;

		let mut rng = StdRng::seed_from_u64(0);

		let n_vars = 10;
		let eval_point = repeat_with(|| <FE as Field>::random(&mut rng))
			.take(n_vars)
			.collect::<Vec<_>>();

		let mixing_challenges = repeat_with(|| <FE as Field>::random(&mut rng))
			.take(4)
			.collect::<Vec<_>>();
		let backend = make_best_backend();

		let sumcheck_challenges = repeat_with(|| <FE as Field>::random(&mut rng))
			.take(n_vars)
			.collect::<Vec<_>>();

		let val1 = evaluate_ring_switch_eq_ind::<F, _, _>(
			&eval_point,
			&sumcheck_challenges,
			&mixing_challenges,
			backend.clone(),
		);

		let partial_evals = ring_switch_eq_ind_partial_eval::<F, _, FE, _>(
			&eval_point,
			&mixing_challenges,
			backend.clone(),
		)
		.unwrap();
		let val2 = MultilinearExtension::from_values_generic(partial_evals).unwrap();
		let val2 = val2
			.evaluate(
				&MultilinearQuery::<FE>::with_full_query(&sumcheck_challenges, backend).unwrap(),
			)
			.unwrap();
		assert_eq!(val1, val2);
	}
}
