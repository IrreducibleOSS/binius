// Copyright 2024 Irreducible Inc.

use crate::{
	composition::{BivariateProduct, IndexComposition},
	protocols::sumcheck::{
		common::{
			equal_n_vars_check, immediate_switchover_heuristic, small_field_embedding_degree_check,
		},
		prove::{common::fold_partial_eq_ind, RegularSumcheckProver},
		univariate::{
			lagrange_evals_multilinear_extension, univariatizing_reduction_composite_sum_claims,
		},
		univariate_zerocheck::{domain_size, extrapolated_scalars_count},
		Error, VerificationError,
	},
};
use binius_field::{
	util::{eq, inner_product_unchecked},
	BinaryField, ExtensionField, Field, PackedExtension, PackedField, PackedFieldIndexable,
	RepackedExtension,
};
use binius_hal::{ComputationBackend, ComputationBackendExt};
use binius_math::{
	make_ntt_domain_points, CompositionPoly, Error as MathError, EvaluationDomain,
	EvaluationDomainFactory, MLEDirectAdapter, MultilinearExtension, MultilinearPoly,
};
use binius_ntt::{AdditiveNTT, OddInterpolate, SingleThreadedNTT};
use binius_utils::bail;
use bytemuck::zeroed_vec;
use itertools::izip;
use p3_util::log2_ceil_usize;
use rayon::prelude::*;
use stackalloc::stackalloc_with_iter;
use std::{collections::HashMap, iter::repeat_n};
use tracing::instrument;
use transpose::transpose;

/// Helper method to reduce zerocheck witness to skipped variables only via a partial high projection.
///
/// Resulting set of reduced multilinears has an additional column of the projected zerocheck equality
/// indicator at the end, constructed in an efficient manner.
#[instrument(skip_all, level = "debug")]
pub fn reduce_to_skipped_zerocheck_projection<F, P, M, Backend>(
	multilinears: Vec<M>,
	sumcheck_challenges: &[F],
	zerocheck_challenges: &[F],
	backend: &'_ Backend,
) -> Result<Vec<MLEDirectAdapter<P>>, Error>
where
	F: Field,
	P: PackedFieldIndexable<Scalar = F>,
	M: MultilinearPoly<P> + Send + Sync,
	Backend: ComputationBackend,
{
	if sumcheck_challenges.len() > zerocheck_challenges.len() {
		bail!(Error::IncorrectZerocheckChallengesLength);
	}

	// Reduce the witness proper
	let mut reduced_multilinears =
		reduce_to_skipped_projection(multilinears, sumcheck_challenges, backend)?;
	let skip_rounds = equal_n_vars_check(&reduced_multilinears)?;

	// Efficiently construct projection of the zerocheck equality indicator to skipped variables
	let mut partial_eq_ind_eval = backend
		.tensor_product_full_query(&zerocheck_challenges[..skip_rounds])?
		.to_vec();

	let eq_ind_factor = zerocheck_challenges[skip_rounds..]
		.iter()
		.zip(sumcheck_challenges)
		.fold(F::ONE, |accum, (&zerocheck_challenge, &sumcheck_challenge)| {
			accum * eq(sumcheck_challenge, zerocheck_challenge)
		});

	partial_eq_ind_eval
		.iter_mut()
		.for_each(|packed| *packed *= eq_ind_factor);

	reduced_multilinears.push(MultilinearExtension::new(skip_rounds, partial_eq_ind_eval)?.into());
	Ok(reduced_multilinears)
}

/// Helper method to reduce the witness to skipped variables via a partial high projection.
#[instrument(skip_all, level = "debug")]
pub fn reduce_to_skipped_projection<F, P, M, Backend>(
	multilinears: Vec<M>,
	sumcheck_challenges: &[F],
	backend: &'_ Backend,
) -> Result<Vec<MLEDirectAdapter<P>>, Error>
where
	F: Field,
	P: PackedFieldIndexable<Scalar = F>,
	M: MultilinearPoly<P> + Send + Sync,
	Backend: ComputationBackend,
{
	let n_vars = equal_n_vars_check(&multilinears)?;

	if sumcheck_challenges.len() > n_vars {
		bail!(Error::IncorrectNumberOfChallenges);
	}

	let query = backend.multilinear_query(sumcheck_challenges)?;

	let reduced_multilinears = multilinears
		.into_iter()
		.map(|multilinear| {
			multilinear
				.evaluate_partial_high(query.to_ref())
				.expect("0 <= sumcheck_challenges.len() < n_vars")
				.into()
		})
		.collect();

	Ok(reduced_multilinears)
}

pub type Prover<'a, FDomain, P, Backend> = RegularSumcheckProver<
	'a,
	FDomain,
	P,
	IndexComposition<BivariateProduct, 2>,
	MLEDirectAdapter<P>,
	Backend,
>;

/// Create the sumcheck prover for the univariatizing reduction of multilinears
/// (see [verifier side](crate::protocols::sumcheck::univariate::univariatizing_reduction_claim))
///
/// This method takes multilinears projected to first `skip_rounds` variables, constructs a multilinear
/// extension of Lagrange evaluations at `univariate_challenge`, and creates a regular sumcheck prover,
/// placing Lagrange evaluation in the last witness column.
///
/// Note that `univariatized_multilinear_evals` come from a previous sumcheck with a univariate first round.
pub fn univariatizing_reduction_prover<'a, F, FDomain, P, Backend>(
	mut reduced_multilinears: Vec<MLEDirectAdapter<P>>,
	univariatized_multilinear_evals: &[F],
	univariate_challenge: F,
	evaluation_domain_factory: impl EvaluationDomainFactory<FDomain>,
	backend: &'a Backend,
) -> Result<Prover<'a, FDomain, P, Backend>, Error>
where
	F: Field + ExtensionField<FDomain>,
	FDomain: BinaryField,
	P: PackedFieldIndexable<Scalar = F> + PackedExtension<FDomain>,
	Backend: ComputationBackend,
{
	let skip_rounds = equal_n_vars_check(&reduced_multilinears)?;

	if univariatized_multilinear_evals.len() != reduced_multilinears.len() {
		bail!(VerificationError::NumberOfFinalEvaluations);
	}

	let evaluation_domain =
		EvaluationDomain::from_points(make_ntt_domain_points(1 << skip_rounds)?)?;

	reduced_multilinears.push(
		lagrange_evals_multilinear_extension(&evaluation_domain, univariate_challenge)?.into(),
	);

	let composite_sum_claims =
		univariatizing_reduction_composite_sum_claims(univariatized_multilinear_evals);

	let prover = RegularSumcheckProver::new(
		reduced_multilinears,
		composite_sum_claims,
		evaluation_domain_factory,
		immediate_switchover_heuristic,
		backend,
	)?;

	Ok(prover)
}

#[derive(Debug)]
struct ParFoldStates<PBase: PackedField, P: PackedField> {
	/// Evaluations of a multilinear subcube, embedded into P (see MultilinearPoly::subcube_evals). Scratch space.
	evals: Vec<P>,
	/// `evals` cast to base field and transposed to 2^skip_rounds * 2^log_batch row-major form. Scratch space.
	interleaved_evals: Vec<PBase>,
	/// `interleaved_evals` extrapolated beyond first 2^skip_rounds domain points, per multilinear.
	extrapolated_evals: Vec<Vec<PBase>>,
	/// Evals of a single composition over extrapolated multilinears. Scratch space.
	composition_evals: Vec<PBase>,
	/// Round evals accumulators, per multilinear.
	round_evals: Vec<Vec<P::Scalar>>,
}

impl<PBase: PackedField, P: PackedField> ParFoldStates<PBase, P> {
	fn new(
		n_multilinears: usize,
		skip_rounds: usize,
		log_batch: usize,
		log_embedding_degree: usize,
		composition_degrees: impl Iterator<Item = usize> + Clone,
	) -> Self {
		let subcube_vars = skip_rounds + log_batch;
		let composition_max_degree = composition_degrees.clone().max().unwrap_or(0);
		let extrapolated_packed_pbase_len =
			extrapolated_evals_packed_len::<PBase>(composition_max_degree, skip_rounds, log_batch);

		let evals =
			zeroed_vec(1 << subcube_vars.saturating_sub(P::LOG_WIDTH + log_embedding_degree));
		let interleaved_evals = zeroed_vec(1 << subcube_vars.saturating_sub(PBase::LOG_WIDTH));

		let extrapolated_evals = (0..n_multilinears)
			.map(|_| zeroed_vec(extrapolated_packed_pbase_len))
			.collect();

		let composition_evals = zeroed_vec(extrapolated_packed_pbase_len);

		let round_evals = composition_degrees
			.map(|composition_degree| {
				zeroed_vec(extrapolated_scalars_count(composition_degree, skip_rounds))
			})
			.collect();

		Self {
			evals,
			interleaved_evals,
			extrapolated_evals,
			composition_evals,
			round_evals,
		}
	}
}

#[derive(Debug)]
pub struct ZerocheckUnivariateEvalsOutput<F, P, Backend>
where
	F: Field,
	P: PackedField<Scalar = F>,
	Backend: ComputationBackend,
{
	pub round_evals: Vec<Vec<F>>,
	skip_rounds: usize,
	remaining_rounds: usize,
	max_domain_size: usize,
	partial_eq_ind_evals: Backend::Vec<P>,
	partial_eq_ind_evals_skipped: Vec<P>,
}

pub struct ZerocheckUnivariateFoldResult<F, P, Backend>
where
	F: Field,
	P: PackedField<Scalar = F>,
	Backend: ComputationBackend,
{
	pub skip_rounds: usize,
	pub eq_ind_eval: F,
	pub subcube_lagrange_coeffs: Vec<F>,
	pub claimed_sums: Vec<F>,
	pub partial_eq_ind_evals: Backend::Vec<P>,
}

impl<F, P, Backend> ZerocheckUnivariateEvalsOutput<F, P, Backend>
where
	F: Field,
	P: PackedFieldIndexable<Scalar = F>,
	Backend: ComputationBackend,
{
	// Univariate round can be folded once the challenge has been sampled.
	#[instrument(
		skip_all,
		name = "ZerocheckUnivariateEvalsOutput::fold",
		level = "debug"
	)]
	pub fn fold<FDomain>(
		self,
		challenge: F,
	) -> Result<ZerocheckUnivariateFoldResult<F, P, Backend>, Error>
	where
		FDomain: BinaryField,
		F: ExtensionField<FDomain>,
	{
		let Self {
			round_evals,
			skip_rounds,
			remaining_rounds,
			max_domain_size,
			mut partial_eq_ind_evals,
			partial_eq_ind_evals_skipped,
		} = self;

		// Zerocheck equality indicator over skipped variables
		let partial_eq_ind_evals_skipped =
			&P::unpack_scalars(&partial_eq_ind_evals_skipped)[..1 << skip_rounds];

		let domain_points = make_ntt_domain_points::<FDomain>(max_domain_size)?;

		// Lagrange extrapolation over skipped subcube
		let subcube_evaluation_domain =
			EvaluationDomain::from_points(domain_points[..1 << skip_rounds].to_vec())?;

		let eq_ind_eval =
			subcube_evaluation_domain.extrapolate(partial_eq_ind_evals_skipped, challenge)?;
		let subcube_lagrange_coeffs = subcube_evaluation_domain.lagrange_evals(challenge);

		// Zerocheck tensor expansion for the reduced zerocheck should be one variable less
		fold_partial_eq_ind::<P, Backend>(remaining_rounds, &mut partial_eq_ind_evals);

		// Lagrange extrapolation for the entire univariate domain
		let round_evals_lagrange_coeffs =
			EvaluationDomain::from_points(domain_points)?.lagrange_evals(challenge);

		let claimed_sums = round_evals
			.into_iter()
			.map(|evals| {
				inner_product_unchecked::<F, F>(
					evals,
					round_evals_lagrange_coeffs[1 << skip_rounds..]
						.iter()
						.copied(),
				)
			})
			.collect();

		Ok(ZerocheckUnivariateFoldResult {
			skip_rounds,
			eq_ind_eval,
			subcube_lagrange_coeffs,
			claimed_sums,
			partial_eq_ind_evals,
		})
	}
}

/// Compute univariate skip round evaluations for zerocheck.
///
/// When all witness multilinear hypercube evaluations can be embedded into a small field
/// `PBase::Scalar` that is significantly smaller than `F`, we naturally want to refrain from
/// folding for `skip_rounds` (denoted below as $k$) to reap the benefits of faster small field
/// multiplications. Naive extensions to sumcheck protocol which compute multivariate round
/// polynomials do not work though, given that for a composition of degree $d$ one would need
/// $(d+1)^k-2^k$ evaluations (assuming [Gruen24] section 3.2 optimizations), which usually grows faster
/// than $2^k$ and thus will typically require more work than large field sumcheck. We adopt a
/// univariatizing approach instead, where we define "oblong" multivariates:
/// $$\hat{M}(\hat{u}_1,x_1,\ldots,x_n) = \sum M(u_1,\ldots, u_k, x_1, \ldots, x_n) \cdot L_u(\hat{u}_1)$$
/// with $\mathbb{M}: \hat{u}_1 \rightarrow (u_1, \ldots, u_k)$ being some map from the univariate domain to
/// the $\mathcal{B}_k$ hypercube and $L_u(\hat{u})$ being Lagrange polynomials.
///
/// The main idea of the univariatizing approach is that $\hat{M}$ are of degree $2^k-1$ in
/// $\hat{u}_1$ and multilinear in other variables, thus evaluating a composition of degree $d$ over
/// $\hat{M}$ yields a total degree of $d(2^k-1)$ in the first round (again, assuming [Gruen24]
/// section 3.2 trick to avoid multiplication by the equality indicator), which is comparable to
/// what a regular non-skipping zerocheck prover would do. The only issue is that we normally don't
/// have an oracle for $\hat{M}$, which necessitates an extra sumcheck reduction to multilinear claims
/// (see [univariatizing_reduction_claim](`super::super::univariate::univariatizing_reduction_claim`)).
///
/// One special trick of the univariate round is that round polynomial is represented in Lagrange form:
///  1. Honest prover evaluates to zero on $2^k$ domain points mapping to $\mathcal{B}_k$, reducing proof size
///  2. Avoiding monomial conversion saves prover time by skipping $O(N^3)$ inverse Vandermonde precomp
///  3. Evaluation in the verifier can be made linear time when barycentric weights are precomputed
///
/// The returned round polynomial is multiplied by zerocheck equality indicator and thus has a degree
/// of $(d+1)(2^k - 1)$; this multiplication happens after summation and is not in the hot path.
///
/// This implementation defines $\mathbb{M}$ to be the basis-induced mapping of the binary field `FDomain`;
/// the main reason for that is to be able to use additive NTT from [LCH14] for extrapolation. The choice
/// of domain field impacts performance, thus generally the smallest field with cardinality not less than
/// the degree of the round polynomial should be used.
///
/// [LCH14]: <https://arxiv.org/abs/1404.3458>
/// [Gruen24]: <https://eprint.iacr.org/2024/108>
#[instrument(skip_all, level = "debug")]
pub fn zerocheck_univariate_evals<F, FDomain, PBase, P, Composition, M, Backend>(
	multilinears: &[M],
	compositions: &[Composition],
	zerocheck_challenges: &[F],
	skip_rounds: usize,
	max_domain_size: usize,
	backend: &Backend,
) -> Result<ZerocheckUnivariateEvalsOutput<F, P, Backend>, Error>
where
	FDomain: BinaryField,
	F: BinaryField + ExtensionField<PBase::Scalar> + ExtensionField<FDomain>,
	PBase: PackedFieldIndexable<Scalar: ExtensionField<FDomain>>
		+ PackedExtension<FDomain, PackedSubfield: PackedFieldIndexable>,
	P: PackedFieldIndexable<Scalar = F> + RepackedExtension<PBase>,
	Composition: CompositionPoly<PBase>,
	M: MultilinearPoly<P> + Send + Sync,
	Backend: ComputationBackend,
{
	let n_vars = equal_n_vars_check(multilinears)?;
	let n_multilinears = multilinears.len();

	if skip_rounds > n_vars {
		bail!(Error::TooManySkippedRounds);
	}

	if zerocheck_challenges.len() != n_vars {
		bail!(Error::IncorrectZerocheckChallengesLength);
	}

	small_field_embedding_degree_check::<PBase, P, _>(multilinears)?;

	let log_embedding_degree = <P::Scalar as ExtensionField<PBase::Scalar>>::LOG_DEGREE;
	let composition_degrees = compositions.iter().map(|composition| composition.degree());
	let composition_max_degree = composition_degrees.clone().max().unwrap_or(0);

	// Consider equality indicator a factor on the composition, which increases
	// the composition degree by one.
	if max_domain_size < domain_size(composition_max_degree + 1, skip_rounds) {
		bail!(Error::LagrangeDomainTooSmall);
	}

	let min_domain_bits = log2_ceil_usize(max_domain_size).max(1);
	if min_domain_bits > FDomain::N_BITS {
		bail!(MathError::DomainSizeTooLarge);
	}

	// Batching factors for strided NTTs.
	let log_extension_degree_base_domain = <PBase::Scalar as ExtensionField<FDomain>>::LOG_DEGREE;

	// Only a domain size NTT is needed.
	let fdomain_ntt = SingleThreadedNTT::<FDomain>::new(min_domain_bits)
		.expect("FDomain cardinality checked before")
		.precompute_twiddles();

	// Smaller subcubes are batched together to reduce interpolation/evaluation overhead.
	const MIN_SUBCUBE_VARS: usize = 8;
	let log_batch = MIN_SUBCUBE_VARS.min(n_vars).saturating_sub(skip_rounds);

	// Expand the multilinear query in all but the first `skip_rounds` variables,
	// where each tensor expansion element serves as a constant factor of the whole
	// univariatized subcube.
	// NB: expansion of the first `skip_rounds` variables is applied to the round evals sum
	let partial_eq_ind_evals =
		backend.tensor_product_full_query(&zerocheck_challenges[skip_rounds..])?;
	let partial_eq_ind_evals_scalars = P::unpack_scalars(&partial_eq_ind_evals[..]);

	// Evaluate each composition on a minimal packed prefix corresponding to the degree
	let pbase_prefix_lens = composition_degrees
		.clone()
		.map(|composition_degree| {
			extrapolated_evals_packed_len::<PBase>(composition_degree, skip_rounds, log_batch)
		})
		.collect::<Vec<_>>();

	let subcube_vars = log_batch + skip_rounds;
	let log_subcube_count = n_vars - subcube_vars;

	// NB: we avoid evaluation on the first 2^skip_rounds points because honest
	// prover would always evaluate to zero there; we also factor out first
	// skip_rounds terms of the equality indicator and apply them pointwise to
	// the final round evaluations, which equates to lowering the composition_degree
	// by one (this is an extension of Gruen section 3.2 trick)
	let staggered_round_evals = (0..1 << log_subcube_count)
		.into_par_iter()
		.try_fold(
			|| {
				ParFoldStates::<PBase, P>::new(
					n_multilinears,
					skip_rounds,
					log_batch,
					log_embedding_degree,
					composition_degrees.clone(),
				)
			},
			|mut par_fold_states, subcube_index| -> Result<_, Error> {
				let ParFoldStates {
					evals,
					interleaved_evals,
					extrapolated_evals,
					composition_evals,
					round_evals,
					..
				} = &mut par_fold_states;

				// Interpolate multilinear evals for each multilinear
				for (multilinear, extrapolated_evals) in
					izip!(multilinears, extrapolated_evals.iter_mut())
				{
					// Sample evals subcube from a multilinear poly
					multilinear.subcube_evals(
						subcube_vars,
						subcube_index,
						log_embedding_degree,
						evals.as_mut_slice(),
					)?;

					// Use the PackedExtension bound to cast evals to base field.
					let evals_base = P::cast_bases_mut(evals.as_mut_slice());

					// The evals subcube can be seen as 2^log_batch subcubes of skip_rounds
					// variables, laid out sequentially in memory; this can be seen as
					// row major 2^log_batch * 2^skip_rounds scalar matrix. We need to transpose
					// it to 2^skip_rounds * 2_log_batch shape in order for the strided NTT to work.
					let interleaved_evals_ref = if log_batch == 0 {
						// In case of no batching, pass the slice as is.
						evals_base
					} else {
						let evals_base_scalars =
							&PBase::unpack_scalars(evals_base)[..1 << subcube_vars];
						let interleaved_evals_scalars =
							&mut PBase::unpack_scalars_mut(interleaved_evals.as_mut_slice())
								[..1 << subcube_vars];

						transpose(
							evals_base_scalars,
							interleaved_evals_scalars,
							1 << skip_rounds,
							1 << log_batch,
						);

						interleaved_evals.as_mut_slice()
					};

					// Extrapolate evals using a conservative upper bound of the composition
					// degree. When evals are correctly strided, we can use additive NTT to
					// extrapolate them beyond the first 2^skip_rounds. We use the fact that an NTT
					// over the extension is just a strided NTT over the base field.
					let interleaved_evals_bases = PBase::cast_bases_mut(interleaved_evals_ref);
					let extrapolated_evals_bases = PBase::cast_bases_mut(extrapolated_evals);

					ntt_extrapolate(
						&fdomain_ntt,
						skip_rounds,
						log_batch + log_extension_degree_base_domain,
						composition_max_degree,
						interleaved_evals_bases,
						extrapolated_evals_bases,
					)?
				}

				// Obtain 1 << log_batch partial equality indicator constant factors for each
				// of the subcubes of size 1 << skip_rounds.
				let partial_eq_ind_evals_scalars_subslice =
					&partial_eq_ind_evals_scalars[subcube_index << log_batch..][..1 << log_batch];

				// Evaluate the compositions and accumulate round results
				for (composition, round_evals, &pbase_prefix_len) in
					izip!(compositions, round_evals, &pbase_prefix_lens)
				{
					let extrapolated_evals_iter = extrapolated_evals
						.iter()
						.map(|evals| &evals[..pbase_prefix_len]);

					stackalloc_with_iter(n_multilinears, extrapolated_evals_iter, |batch_query| {
						// Evaluate the small field composition
						composition
							.batch_evaluate(batch_query, &mut composition_evals[..pbase_prefix_len])
					})?;

					// Accumulate round evals and multiply by the constant part of the
					// zerocheck equality indicator
					let composition_evals_scalars =
						PBase::unpack_scalars_mut(composition_evals.as_mut_slice());

					for (round_eval, composition_evals) in izip!(
						round_evals.iter_mut(),
						composition_evals_scalars.chunks_exact(1 << log_batch),
					) {
						// Inner product is with the high n_vars - skip_rounds projection
						// of the zerocheck equality indicator (one factor per subcube).
						*round_eval += inner_product_unchecked(
							partial_eq_ind_evals_scalars_subslice.iter().copied(),
							composition_evals.iter().copied(),
						);
					}
				}

				Ok(par_fold_states)
			},
		)
		.map(|states| -> Result<_, Error> { Ok(states?.round_evals) })
		.try_reduce(
			|| {
				composition_degrees
					.clone()
					.map(|composition_degree| {
						zeroed_vec(extrapolated_scalars_count(composition_degree, skip_rounds))
					})
					.collect()
			},
			|lhs, rhs| -> Result<_, Error> {
				let round_evals_sum = izip!(lhs, rhs)
					.map(|(mut lhs_vals, rhs_vals)| {
						for (lhs_val, rhs_val) in izip!(&mut lhs_vals, rhs_vals) {
							*lhs_val += rhs_val;
						}
						lhs_vals
					})
					.collect();

				Ok(round_evals_sum)
			},
		)?;

	// So far evals of each composition are "staggered" in a sense that they are evaluated on the smallest
	// domain which guarantees uniqueness of the round polynomial. We extrapolate them to max_domain_size to
	// aid in Gruen section 3.2 optimization below and batch mixing.
	let max_domain_size_round_evals =
		extrapolate_round_evals::<FDomain, _>(staggered_round_evals, skip_rounds, max_domain_size)?;

	// We also extrapolate the tensor expansion of the first skip_rounds zerocheck challenges to the same
	// max domain size and then pointwise multiply it by each of the round evals vectors.
	let (partial_eq_ind_evals_skipped, partial_eq_ind_evals_skipped_extrapolated) =
		extrapolate_partial_eq_ind::<F, FDomain, PBase, P, _, _>(
			&fdomain_ntt,
			skip_rounds,
			max_domain_size,
			&zerocheck_challenges[..skip_rounds],
			backend,
		)?;

	let round_evals = multiply_round_evals(
		max_domain_size_round_evals,
		&P::unpack_scalars(partial_eq_ind_evals_skipped_extrapolated.as_slice())
			[..max_domain_size.saturating_sub(1 << skip_rounds)],
	);

	let remaining_rounds = n_vars - skip_rounds;

	Ok(ZerocheckUnivariateEvalsOutput {
		round_evals,
		skip_rounds,
		remaining_rounds,
		max_domain_size,
		partial_eq_ind_evals,
		partial_eq_ind_evals_skipped,
	})
}

// Extrapolate round evaluations to the full domain.
// NB: this method relies on the fact that `round_evals` have specific lengths
// (namely `d * 2^n`, where `n` is not less than the number of skipped rounds and thus d
// is not larger than the composition degree), which enables additive-NTT based subquadratic
// techniques.
#[instrument(skip_all, level = "debug")]
fn extrapolate_round_evals<FDomain, F>(
	mut round_evals: Vec<Vec<F>>,
	skip_rounds: usize,
	max_domain_size: usize,
) -> Result<Vec<Vec<F>>, Error>
where
	FDomain: BinaryField,
	F: BinaryField + ExtensionField<FDomain>,
{
	// Instantiate a large enough NTT over F to be able to forward transform to full domain size.
	// REVIEW: should be possible to use an existing FDomain NTT with striding.
	let ntt = SingleThreadedNTT::new(log2_ceil_usize(max_domain_size))?;

	// Cache OddInterpolate instances, which, albeit small in practice, take cubic time to create.
	let mut odd_interpolates = HashMap::new();

	for round_evals in &mut round_evals {
		// Re-add zero evaluations at the beginning.
		round_evals.splice(0..0, repeat_n(F::ZERO, 1 << skip_rounds));

		let n = round_evals.len();

		// Get OddInterpolate instance of required size.
		let odd_interpolate = odd_interpolates.entry(n).or_insert_with(|| {
			let ell = n.trailing_zeros() as usize;
			assert!(ell >= skip_rounds);

			OddInterpolate::new(n >> ell, ell, ntt.twiddles())
				.expect("domain large enough by construction")
		});

		// Obtain novel polynomial basis representation of round evaluations.
		odd_interpolate.inverse_transform(&ntt, round_evals)?;

		// Use forward NTT to extrapolate novel representation to the max domain size.
		let next_power_of_two = 1 << log2_ceil_usize(max_domain_size);
		round_evals.resize(next_power_of_two, F::ZERO);

		ntt.forward_transform(round_evals, 0, 0)?;

		// Sanity check: first 1 << skip_rounds evals are still zeros.
		debug_assert!(round_evals[..1 << skip_rounds]
			.iter()
			.all(|&coeff| coeff == F::ZERO));

		// Trim the result.
		round_evals.resize(max_domain_size, F::ZERO);
		round_evals.drain(..1 << skip_rounds);
	}

	Ok(round_evals)
}

#[instrument(skip_all, level = "debug")]
fn multiply_round_evals<F: Field>(
	mut round_evals: Vec<Vec<F>>,
	pointwise_multiplier: &[F],
) -> Vec<Vec<F>> {
	for round_evals in &mut round_evals {
		debug_assert_eq!(round_evals.len(), pointwise_multiplier.len());
		for (round_eval, &multiplier) in izip!(round_evals, pointwise_multiplier) {
			*round_eval *= multiplier;
		}
	}

	round_evals
}

fn extrapolate_partial_eq_ind<F, FDomain, PBase, P, NTT, Backend>(
	ntt: &NTT,
	skip_rounds: usize,
	max_domain_size: usize,
	zerocheck_challenges: &[F],
	backend: &Backend,
) -> Result<(Vec<P>, Vec<P>), Error>
where
	FDomain: Field,
	F: Field + ExtensionField<PBase::Scalar> + ExtensionField<FDomain>,
	P: PackedField<Scalar = F> + RepackedExtension<PBase>,
	PBase: PackedField<Scalar: ExtensionField<FDomain>>
		+ PackedExtension<FDomain, PackedSubfield: PackedFieldIndexable>,
	NTT: AdditiveNTT<PBase::PackedSubfield> + AdditiveNTT<FDomain>,
	Backend: ComputationBackend,
{
	debug_assert_eq!(zerocheck_challenges.len(), skip_rounds);

	// Expand the first skip_rounds variables of the zerocheck equality indicator
	// and extrapolate it to the full domain using additive NTT (Gruen 3.2 adaptation)
	let partial_eq_ind_evals_skipped = backend
		.tensor_product_full_query(zerocheck_challenges)?
		.to_vec();

	// We extensively validate the sizes of slices within NTT interpolation routines,
	// thus let's find the smallest composition_max_degree that covers the max_domain_size.
	let composition_max_degree = (0..)
		.find(|&composition_degree| domain_size(composition_degree, skip_rounds) >= max_domain_size)
		.expect("usize overflow");

	let mut partial_eq_ind_evals_skipped_extrapolated =
		zeroed_vec::<P>(extrapolated_evals_packed_len::<P>(composition_max_degree, skip_rounds, 0));

	// Strided NTT as a way to extrapolate over the large field P::Scalar.
	let mut partial_eq_ind_evals_skipped_scratch = partial_eq_ind_evals_skipped.clone();

	let partial_eq_ind_evals_skipped_scratch_bases = PBase::cast_bases_mut(P::cast_bases_mut(
		partial_eq_ind_evals_skipped_scratch.as_mut_slice(),
	));

	let partial_eq_ind_evals_skipped_extrapolated_bases = PBase::cast_bases_mut(P::cast_bases_mut(
		partial_eq_ind_evals_skipped_extrapolated.as_mut_slice(),
	));

	let log_extension_degree_p_domain = <P::Scalar as ExtensionField<FDomain>>::LOG_DEGREE;

	ntt_extrapolate(
		ntt,
		skip_rounds,
		log_extension_degree_p_domain,
		composition_max_degree,
		partial_eq_ind_evals_skipped_scratch_bases,
		partial_eq_ind_evals_skipped_extrapolated_bases,
	)?;

	Ok((partial_eq_ind_evals_skipped, partial_eq_ind_evals_skipped_extrapolated))
}

fn ntt_extrapolate<NTT, P>(
	ntt: &NTT,
	skip_rounds: usize,
	log_batch: usize,
	composition_max_degree: usize,
	interleaved_evals: &mut [P],
	extrapolated_evals: &mut [P],
) -> Result<(), Error>
where
	P: PackedFieldIndexable,
	NTT: AdditiveNTT<P> + AdditiveNTT<P::Scalar>,
{
	let subcube_vars = skip_rounds + log_batch;
	debug_assert_eq!(
		1 << (skip_rounds + log_batch).saturating_sub(P::LOG_WIDTH),
		interleaved_evals.len()
	);
	debug_assert_eq!(
		extrapolated_evals_packed_len::<P>(composition_max_degree, skip_rounds, log_batch),
		extrapolated_evals.len()
	);
	debug_assert!(
		<NTT as AdditiveNTT<P>>::log_domain_size(ntt)
			>= log2_ceil_usize(domain_size(composition_max_degree, skip_rounds))
	);

	if subcube_vars >= P::LOG_WIDTH {
		// Things are simple when each NTT spans whole packed fields
		ntt_extrapolate_chunks_exact(ntt, log_batch, interleaved_evals, extrapolated_evals)
	} else {
		// In the rare case where subcubes are smaller we downcast to scalars
		let extrapolated_scalars_count =
			extrapolated_scalars_count(composition_max_degree, skip_rounds)
				.div_ceil(1 << skip_rounds)
				<< subcube_vars;

		ntt_extrapolate_chunks_exact(
			ntt,
			log_batch,
			&mut P::unpack_scalars_mut(interleaved_evals)[..1 << subcube_vars],
			&mut P::unpack_scalars_mut(extrapolated_evals)[..extrapolated_scalars_count],
		)
	}
}

fn ntt_extrapolate_chunks_exact<NTT, P>(
	ntt: &NTT,
	log_batch: usize,
	interleaved_evals: &mut [P],
	extrapolated_evals: &mut [P],
) -> Result<(), Error>
where
	P: PackedField,
	NTT: AdditiveNTT<P>,
{
	debug_assert!(interleaved_evals.len().is_power_of_two());
	debug_assert!(extrapolated_evals.len() % interleaved_evals.len() == 0);

	// Inverse NTT: convert evals to novel basis representation
	ntt.inverse_transform(interleaved_evals, 0, log_batch)?;

	// Forward NTT: evaluate novel basis representation at consecutive cosets
	for (i, extrapolated_chunk) in extrapolated_evals
		.chunks_exact_mut(interleaved_evals.len())
		.enumerate()
	{
		extrapolated_chunk.copy_from_slice(interleaved_evals);
		ntt.forward_transform(extrapolated_chunk, (i + 1) as u32, log_batch)?;
	}

	Ok(())
}

fn extrapolated_evals_packed_len<P: PackedField>(
	composition_degree: usize,
	skip_rounds: usize,
	log_batch: usize,
) -> usize {
	composition_degree.saturating_sub(1) << (skip_rounds + log_batch).saturating_sub(P::LOG_WIDTH)
}

#[cfg(test)]
mod tests {
	use crate::{
		composition::{IndexComposition, ProductComposition},
		polynomial::CompositionScalarAdapter,
		protocols::{
			sumcheck::prove::univariate::{domain_size, zerocheck_univariate_evals},
			test_utils::generate_zero_product_multilinears,
		},
		transparent::eq_ind::EqIndPartialEval,
	};
	use binius_field::{
		BinaryField128b, BinaryField16b, BinaryField8b, ExtensionField, Field,
		PackedBinaryField128x1b, PackedBinaryField1x128b, PackedBinaryField4x32b,
		PackedBinaryField8x16b, PackedExtension, PackedField, PackedFieldIndexable,
	};
	use binius_hal::make_portable_backend;
	use binius_math::{
		make_ntt_domain_points, CompositionPoly, DefaultEvaluationDomainFactory, EvaluationDomain,
		EvaluationDomainFactory, MultilinearPoly,
	};
	use binius_ntt::SingleThreadedNTT;
	use rand::{prelude::StdRng, SeedableRng};
	use std::sync::Arc;

	#[test]
	fn ntt_extrapolate_correctness() {
		type P = PackedBinaryField4x32b;
		type FDomain = BinaryField16b;
		let log_extension_degree_p_domain = 1;

		let mut rng = StdRng::seed_from_u64(0);
		let ntt = SingleThreadedNTT::<FDomain>::new(10).unwrap();
		let domain_points = make_ntt_domain_points::<FDomain>(1 << 10).unwrap();

		for skip_rounds in 0..5usize {
			let domain =
				EvaluationDomain::from_points(domain_points[..1 << skip_rounds].to_vec()).unwrap();
			for log_batch in 0..3usize {
				for composition_degree in 0..5usize {
					let subcube_vars = skip_rounds + log_batch;
					let interleaved_len = 1 << subcube_vars.saturating_sub(P::LOG_WIDTH);
					let interleaved_evals = (0..interleaved_len)
						.map(|_| P::random(&mut rng))
						.collect::<Vec<_>>();

					let extrapolated_scalars_cnt =
						composition_degree.saturating_sub(1) << skip_rounds;
					let extrapolated_ntts = composition_degree.saturating_sub(1);
					let extrapolated_len = extrapolated_ntts * interleaved_len;
					let mut extrapolated_evals = vec![P::zero(); extrapolated_len];

					let mut interleaved_evals_scratch = interleaved_evals.clone();

					let interleaved_evals_domain =
						P::cast_bases_mut(interleaved_evals_scratch.as_mut_slice());
					let extrapolated_evals_domain =
						P::cast_bases_mut(extrapolated_evals.as_mut_slice());

					super::ntt_extrapolate(
						&ntt,
						skip_rounds,
						log_batch + log_extension_degree_p_domain,
						composition_degree,
						interleaved_evals_domain,
						extrapolated_evals_domain,
					)
					.unwrap();

					let interleaved_scalars =
						&P::unpack_scalars(interleaved_evals.as_slice())[..1 << subcube_vars];
					let extrapolated_scalars = &P::unpack_scalars(extrapolated_evals.as_slice())
						[..extrapolated_scalars_cnt << log_batch];

					for batch_idx in 0..1 << log_batch {
						let values = (0..1 << skip_rounds)
							.map(|i| interleaved_scalars[(i << log_batch) + batch_idx])
							.collect::<Vec<_>>();

						for (i, &point) in domain_points[1 << skip_rounds..]
							[..extrapolated_scalars_cnt]
							.iter()
							.enumerate()
						{
							let extrapolated =
								domain.extrapolate(values.as_slice(), point.into()).unwrap();
							let expected = extrapolated_scalars[(i << log_batch) + batch_idx];
							assert_eq!(extrapolated, expected);
						}
					}
				}
			}
		}
	}

	#[test]
	fn zerocheck_univariate_evals_invariants() {
		type F = BinaryField128b;
		type FDomain = BinaryField8b;
		type FBase = BinaryField16b;
		type P = PackedBinaryField128x1b;
		type PE = PackedBinaryField1x128b;
		type PBase = PackedBinaryField8x16b;

		let mut rng = StdRng::seed_from_u64(0);

		let n_vars = 7;
		let log_embedding_degree = <F as ExtensionField<FBase>>::LOG_DEGREE;

		let mut multilinears = generate_zero_product_multilinears::<P, PE>(&mut rng, n_vars, 2);
		multilinears.extend(generate_zero_product_multilinears(&mut rng, n_vars, 3));
		multilinears.extend(generate_zero_product_multilinears(&mut rng, n_vars, 4));

		let compositions = [
			Arc::new(IndexComposition::new(9, [0, 1], ProductComposition::<2> {}).unwrap())
				as Arc<dyn CompositionPoly<PBase>>,
			Arc::new(IndexComposition::new(9, [2, 3, 4], ProductComposition::<3> {}).unwrap())
				as Arc<dyn CompositionPoly<PBase>>,
			Arc::new(IndexComposition::new(9, [5, 6, 7, 8], ProductComposition::<4> {}).unwrap())
				as Arc<dyn CompositionPoly<PBase>>,
		];

		let backend = make_portable_backend();
		let zerocheck_challenges = (0..n_vars)
			.map(|_| <F as Field>::random(&mut rng))
			.collect::<Vec<_>>();
		let zerocheck_eq_ind_mle = EqIndPartialEval::new(n_vars, zerocheck_challenges.clone())
			.unwrap()
			.multilinear_extension::<F, _>(&backend)
			.unwrap();

		for skip_rounds in 0usize..=5 {
			let max_domain_size = domain_size(5, skip_rounds);
			let output = zerocheck_univariate_evals::<F, FDomain, PBase, PE, _, _, _>(
				multilinears.as_slice(),
				compositions.as_slice(),
				zerocheck_challenges.as_slice(),
				skip_rounds,
				max_domain_size,
				&backend,
			)
			.unwrap();

			// naive computation of the univariate skip output
			let round_evals_len = 4usize << skip_rounds;
			assert!(output
				.round_evals
				.iter()
				.all(|round_evals| round_evals.len() == round_evals_len));

			let compositions = compositions
				.iter()
				.cloned()
				.map(CompositionScalarAdapter::new)
				.collect::<Vec<_>>();

			let mut query = [FBase::ZERO; 9];
			let mut evals = vec![PE::zero(); 1 << skip_rounds.saturating_sub(log_embedding_degree)];
			let domain = DefaultEvaluationDomainFactory::<FDomain>::default()
				.create(1 << skip_rounds)
				.unwrap();
			for round_evals_index in 0..round_evals_len {
				let x = FDomain::new(((1 << skip_rounds) + round_evals_index) as u8);
				let mut composition_sums = vec![F::ZERO; compositions.len()];
				for subcube_index in 0..1 << (n_vars - skip_rounds) {
					for (query, multilinear) in query.iter_mut().zip(&multilinears) {
						multilinear
							.subcube_evals(
								skip_rounds,
								subcube_index,
								log_embedding_degree,
								evals.as_mut_slice(),
							)
							.unwrap();
						let evals_scalars =
							&PBase::unpack_scalars(PE::cast_bases(evals.as_slice()))
								[..1 << skip_rounds];
						let extrapolated = domain.extrapolate(evals_scalars, x.into()).unwrap();
						*query = extrapolated;
					}
					let eq_ind_factor = domain
						.extrapolate(
							&zerocheck_eq_ind_mle.evals()[subcube_index << skip_rounds..]
								[..1 << skip_rounds],
							x.into(),
						)
						.unwrap();
					for (composition, sum) in compositions.iter().zip(composition_sums.iter_mut()) {
						*sum += eq_ind_factor * composition.evaluate(query.as_slice()).unwrap();
					}
				}

				let univariate_skip_composition_sums = output
					.round_evals
					.iter()
					.map(|round_evals| round_evals[round_evals_index])
					.collect::<Vec<_>>();
				assert_eq!(univariate_skip_composition_sums, composition_sums);
			}
		}
	}
}
