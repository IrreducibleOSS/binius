// Copyright 2024-2025 Irreducible Inc.

use std::{collections::HashMap, iter::repeat_n};

use binius_field::{
	BinaryField, ExtensionField, Field, PackedExtension, PackedField, PackedSubfield, TowerField,
	packed::{get_packed_slice, get_packed_slice_checked},
	recast_packed_mut,
	util::inner_product_unchecked,
};
use binius_hal::ComputationBackend;
use binius_math::{
	BinarySubspace, CompositionPoly, Error as MathError, EvaluationDomain, MLEDirectAdapter,
	MultilinearPoly, RowsBatchRef,
};
use binius_maybe_rayon::prelude::*;
use binius_ntt::{
	AdditiveNTT, NTTShape, OddInterpolate, SingleThreadedNTT, twiddle::TwiddleAccess,
};
use binius_utils::{bail, checked_arithmetics::log2_ceil_usize};
use bytemuck::zeroed_vec;
use itertools::izip;
use stackalloc::stackalloc_with_iter;
use tracing::instrument;

use crate::{
	composition::{BivariateProduct, IndexComposition},
	protocols::sumcheck::{
		Error,
		common::{equal_n_vars_check, small_field_embedding_degree_check},
		prove::{
			RegularSumcheckProver,
			logging::{ExpandQueryData, UnivariateSkipCalculateCoeffsData},
		},
		zerocheck::{domain_size, extrapolated_scalars_count},
	},
};

pub type Prover<'a, FDomain, P, Backend> = RegularSumcheckProver<
	'a,
	FDomain,
	P,
	IndexComposition<BivariateProduct, 2>,
	MLEDirectAdapter<P>,
	Backend,
>;

#[derive(Debug)]
struct ParFoldStates<FBase: Field, P: PackedExtension<FBase>> {
	/// Evaluations of a multilinear subcube, embedded into P (see MultilinearPoly::subcube_evals).
	/// Scratch space.
	evals: Vec<P>,
	/// `evals` extrapolated beyond first 2^skip_rounds domain points, per multilinear.
	extrapolated_evals: Vec<Vec<PackedSubfield<P, FBase>>>,
	/// Evals of a single composition over extrapolated multilinears. Scratch space.
	composition_evals: Vec<PackedSubfield<P, FBase>>,
	/// Packed round evals accumulators, per multilinear.
	packed_round_evals: Vec<Vec<P>>,
}

impl<FBase: Field, P: PackedExtension<FBase>> ParFoldStates<FBase, P> {
	fn new(
		n_multilinears: usize,
		skip_rounds: usize,
		log_batch: usize,
		log_embedding_degree: usize,
		composition_degrees: impl Iterator<Item = usize> + Clone,
	) -> Self {
		let subcube_vars = skip_rounds + log_batch;
		let composition_max_degree = composition_degrees.clone().max().unwrap_or(0);
		let extrapolated_packed_pbase_len = extrapolated_evals_packed_len::<PackedSubfield<P, FBase>>(
			composition_max_degree,
			skip_rounds,
			log_batch,
		);

		let evals =
			zeroed_vec(1 << subcube_vars.saturating_sub(P::LOG_WIDTH + log_embedding_degree));

		let extrapolated_evals = (0..n_multilinears)
			.map(|_| zeroed_vec(extrapolated_packed_pbase_len))
			.collect();

		let composition_evals = zeroed_vec(extrapolated_packed_pbase_len);

		let packed_round_evals = composition_degrees
			.map(|composition_degree| {
				zeroed_vec(extrapolated_evals_packed_len::<P>(composition_degree, skip_rounds, 0))
			})
			.collect();

		Self {
			evals,
			extrapolated_evals,
			composition_evals,
			packed_round_evals,
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
}

pub struct ZerocheckUnivariateFoldResult<F, P, Backend>
where
	F: Field,
	P: PackedField<Scalar = F>,
	Backend: ComputationBackend,
{
	pub remaining_rounds: usize,
	pub subcube_lagrange_coeffs: Vec<F>,
	pub claimed_sums: Vec<F>,
	pub partial_eq_ind_evals: Backend::Vec<P>,
}

impl<F, P, Backend> ZerocheckUnivariateEvalsOutput<F, P, Backend>
where
	F: Field,
	P: PackedField<Scalar = F>,
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
		FDomain: TowerField,
		F: ExtensionField<FDomain>,
	{
		let Self {
			round_evals,
			skip_rounds,
			remaining_rounds,
			max_domain_size,
			partial_eq_ind_evals,
		} = self;

		// REVIEW: consider using novel basis for the univariate round representation
		//         (instead of Lagrange)
		let max_dim = log2_ceil_usize(max_domain_size);
		let subspace =
			BinarySubspace::<FDomain::Canonical>::with_dim(max_dim)?.isomorphic::<FDomain>();
		let max_domain = EvaluationDomain::from_points(
			subspace.iter().take(max_domain_size).collect::<Vec<_>>(),
			false,
		)?;

		// Lagrange extrapolation over skipped subcube
		let subcube_lagrange_coeffs = EvaluationDomain::from_points(
			subspace.reduce_dim(skip_rounds)?.iter().collect::<Vec<_>>(),
			false,
		)?
		.lagrange_evals(challenge);

		// Lagrange extrapolation for the entire univariate domain
		let round_evals_lagrange_coeffs = max_domain.lagrange_evals(challenge);

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
			remaining_rounds,
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
/// $(d+1)^k-2^k$ evaluations (assuming [Gruen24] section 3.2 optimizations), which usually grows
/// faster than $2^k$ and thus will typically require more work than large field sumcheck. We adopt
/// a univariatizing approach instead, where we define "oblong" multivariates:
/// $$\hat{M}(\hat{u}_1,x_1,\ldots,x_n) = \sum M(u_1,\ldots, u_k, x_1, \ldots, x_n) \cdot
/// L_u(\hat{u}_1)$$ with $\mathbb{M}: \hat{u}_1 \rightarrow (u_1, \ldots, u_k)$ being some map from
/// the univariate domain to the $\mathcal{B}_k$ hypercube and $L_u(\hat{u})$ being Lagrange
/// polynomials.
///
/// The main idea of the univariatizing approach is that $\hat{M}$ are of degree $2^k-1$ in
/// $\hat{u}_1$ and multilinear in other variables, thus evaluating a composition of degree $d$ over
/// $\hat{M}$ yields a total degree of $d(2^k-1)$ in the first round (again, assuming [Gruen24]
/// section 3.2 trick to avoid multiplication by the equality indicator), which is comparable to
/// what a regular non-skipping zerocheck prover would do. The only issue is that we normally don't
/// have an oracle for $\hat{M}$, which necessitates an extra sumcheck reduction to multilinear
/// claims
/// (see [univariatizing_reduction_claim](`super::super::zerocheck::univariatizing_reduction_claim`)).
///
/// One special trick of the univariate round is that round polynomial is represented in Lagrange
/// form:
///  1. Honest prover evaluates to zero on $2^k$ domain points mapping to $\mathcal{B}_k$, reducing
///     proof size
///  2. Avoiding monomial conversion saves prover time by skipping $O(N^3)$ inverse Vandermonde
///     precomp
///  3. Evaluation in the verifier can be made linear time when barycentric weights are precomputed
///
/// This implementation defines $\mathbb{M}$ to be the basis-induced mapping of the binary field
/// `FDomain`; the main reason for that is to be able to use additive NTT from [LCH14] for
/// extrapolation. The choice of domain field impacts performance, thus generally the smallest field
/// with cardinality not less than the degree of the round polynomial should be used.
///
/// [LCH14]: <https://arxiv.org/abs/1404.3458>
/// [Gruen24]: <https://eprint.iacr.org/2024/108>
pub fn zerocheck_univariate_evals<F, FDomain, FBase, P, Composition, M, Backend>(
	multilinears: &[M],
	compositions: &[Composition],
	zerocheck_challenges: &[F],
	skip_rounds: usize,
	max_domain_size: usize,
	backend: &Backend,
) -> Result<ZerocheckUnivariateEvalsOutput<F, P, Backend>, Error>
where
	FDomain: TowerField,
	FBase: ExtensionField<FDomain>,
	F: TowerField,
	P: PackedField<Scalar = F> + PackedExtension<FBase> + PackedExtension<FDomain>,
	Composition: CompositionPoly<PackedSubfield<P, FBase>>,
	M: MultilinearPoly<P> + Send + Sync,
	Backend: ComputationBackend,
{
	let n_vars = equal_n_vars_check(multilinears)?;
	let n_multilinears = multilinears.len();

	if skip_rounds > n_vars {
		bail!(Error::TooManySkippedRounds);
	}

	let remaining_rounds = n_vars - skip_rounds;
	if zerocheck_challenges.len() != remaining_rounds {
		bail!(Error::IncorrectZerocheckChallengesLength);
	}

	small_field_embedding_degree_check::<_, FBase, P, _>(multilinears)?;

	let log_embedding_degree = <F as ExtensionField<FBase>>::LOG_DEGREE;
	let composition_degrees = compositions.iter().map(|composition| composition.degree());
	let composition_max_degree = composition_degrees.clone().max().unwrap_or(0);

	if max_domain_size < domain_size(composition_max_degree, skip_rounds) {
		bail!(Error::LagrangeDomainTooSmall);
	}

	// Batching factors for strided NTTs.
	let log_extension_degree_base_domain = <FBase as ExtensionField<FDomain>>::LOG_DEGREE;

	// Check that domain field contains the required NTT cosets.
	let min_domain_bits = log2_ceil_usize(max_domain_size);
	if min_domain_bits > FDomain::N_BITS {
		bail!(MathError::DomainSizeTooLarge);
	}

	// Only a domain size NTT is needed.
	let fdomain_ntt = SingleThreadedNTT::<FDomain>::with_canonical_field(min_domain_bits)
		.expect("FDomain cardinality checked before")
		.precompute_twiddles();

	// Smaller subcubes are batched together to reduce interpolation/evaluation overhead.
	// REVIEW: make this a heuristic dependent on base field size and/or number of multilinears
	//         to guarantee L1 cache (or accelerator scratchpad) non-eviction.
	const MAX_SUBCUBE_VARS: usize = 12;
	let log_batch = MAX_SUBCUBE_VARS.min(n_vars).saturating_sub(skip_rounds);

	// Expand the multilinear query in all but the first `skip_rounds` variables,
	// where each tensor expansion element serves as a constant factor of the whole
	// univariatized subcube.
	// NB: expansion of the first `skip_rounds` variables is applied to the round evals sum
	let dimensions_data = ExpandQueryData::new(zerocheck_challenges);
	let expand_span = tracing::debug_span!(
		"[task] Expand Query",
		phase = "zerocheck",
		perfetto_category = "task.main",
		?dimensions_data,
	)
	.entered();
	let partial_eq_ind_evals: <Backend as ComputationBackend>::Vec<P> =
		backend.tensor_product_full_query(zerocheck_challenges)?;
	drop(expand_span);

	// Evaluate each composition on a minimal packed prefix corresponding to the degree
	let pbase_prefix_lens = composition_degrees
		.clone()
		.map(|composition_degree| {
			extrapolated_evals_packed_len::<PackedSubfield<P, FBase>>(
				composition_degree,
				skip_rounds,
				log_batch,
			)
		})
		.collect::<Vec<_>>();
	let dimensions_data =
		UnivariateSkipCalculateCoeffsData::new(n_vars, skip_rounds, n_multilinears, log_batch);
	let coeffs_span = tracing::debug_span!(
		"[task] Univariate Skip Calculate coeffs",
		phase = "zerocheck",
		perfetto_category = "task.main",
		?dimensions_data,
	)
	.entered();

	let subcube_vars = log_batch + skip_rounds;
	let log_subcube_count = n_vars - subcube_vars;

	let p_coset_round_evals_len = 1 << skip_rounds.saturating_sub(P::LOG_WIDTH);
	let pbase_coset_composition_evals_len =
		1 << subcube_vars.saturating_sub(P::LOG_WIDTH + log_embedding_degree);

	// NB: we avoid evaluation on the first 2^skip_rounds points because honest
	// prover would always evaluate to zero there; we also factor out first
	// skip_rounds terms of the equality indicator and apply them pointwise to
	// the final round evaluations, which equates to lowering the composition_degree
	// by one (this is an extension of Gruen section 3.2 trick)
	let staggered_round_evals = (0..1 << log_subcube_count)
		.into_par_iter()
		.try_fold(
			|| {
				ParFoldStates::<FBase, P>::new(
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
					extrapolated_evals,
					composition_evals,
					packed_round_evals,
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
						evals,
					)?;

					// Extrapolate evals using a conservative upper bound of the composition
					// degree. We use Additive NTT to extrapolate evals beyond the first
					// 2^skip_rounds, exploiting the fact that extension field NTT is a strided
					// base field NTT.
					let evals_base = <P as PackedExtension<FBase>>::cast_bases_mut(evals);
					let evals_domain = recast_packed_mut::<P, FBase, FDomain>(evals_base);
					let extrapolated_evals_domain =
						recast_packed_mut::<P, FBase, FDomain>(extrapolated_evals);

					ntt_extrapolate(
						&fdomain_ntt,
						skip_rounds,
						log_extension_degree_base_domain,
						log_batch,
						evals_domain,
						extrapolated_evals_domain,
					)?
				}

				// Evaluate the compositions and accumulate round results
				for (composition, packed_round_evals, &pbase_prefix_len) in
					izip!(compositions, packed_round_evals, &pbase_prefix_lens)
				{
					let extrapolated_evals_iter = extrapolated_evals
						.iter()
						.map(|evals| &evals[..pbase_prefix_len]);

					stackalloc_with_iter(n_multilinears, extrapolated_evals_iter, |batch_query| {
						let batch_query = RowsBatchRef::new(batch_query, pbase_prefix_len);

						// Evaluate the small field composition
						composition.batch_evaluate(
							&batch_query,
							&mut composition_evals[..pbase_prefix_len],
						)
					})?;

					// Accumulate round evals and multiply by the constant part of the
					// zerocheck equality indicator
					for (packed_round_evals_coset, composition_evals_coset) in izip!(
						packed_round_evals.chunks_exact_mut(p_coset_round_evals_len,),
						composition_evals.chunks_exact(pbase_coset_composition_evals_len)
					) {
						// At this point, the composition evals are laid out as a 3D array,
						// with dimensions being (ordered by increasing stride):
						//  1) 2^skip_rounds           - a low indexed subcube being "skipped"
						//  2) 2^log_batch             - batch of adjacent subcubes
						//  3) composition_degree - 1  - cosets of the subcube evaluation domain
						// NB: each complete span of dim 1 gets multiplied by a constant from
						// the equality indicator expansion, and dims 1+2 are padded up to the
						// nearest packed field due to ntt_extrapolate implementation details
						// (not performing sub-packed-field NTTs). This helper method handles
						// multiplication of each dim 1 + 2 submatrix by the corresponding
						// equality indicator subslice.
						spread_product::<_, FBase>(
							packed_round_evals_coset,
							composition_evals_coset,
							&partial_eq_ind_evals,
							subcube_index,
							skip_rounds,
							log_batch,
						);
					}
				}

				Ok(par_fold_states)
			},
		)
		.map(|states| -> Result<_, Error> {
			let scalar_round_evals = izip!(composition_degrees.clone(), states?.packed_round_evals)
				.map(|(composition_degree, packed_round_evals)| {
					let mut composition_round_evals = Vec::with_capacity(
						extrapolated_scalars_count(composition_degree, skip_rounds),
					);

					for packed_round_evals_coset in
						packed_round_evals.chunks_exact(p_coset_round_evals_len)
					{
						let coset_scalars = packed_round_evals_coset
							.iter()
							.flat_map(|packed| packed.iter())
							.take(1 << skip_rounds);

						composition_round_evals.extend(coset_scalars);
					}

					composition_round_evals
				})
				.collect::<Vec<_>>();

			Ok(scalar_round_evals)
		})
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
						debug_assert_eq!(lhs_vals.len(), rhs_vals.len());
						for (lhs_val, rhs_val) in izip!(&mut lhs_vals, rhs_vals) {
							*lhs_val += rhs_val;
						}
						lhs_vals
					})
					.collect();

				Ok(round_evals_sum)
			},
		)?;

	// So far evals of each composition are "staggered" in a sense that they are evaluated on the
	// smallest domain which guarantees uniqueness of the round polynomial. We extrapolate them to
	// max_domain_size to aid in Gruen section 3.2 optimization below and batch mixing.
	let round_evals =
		extrapolate_round_evals(&fdomain_ntt, staggered_round_evals, skip_rounds, max_domain_size)?;
	drop(coeffs_span);

	Ok(ZerocheckUnivariateEvalsOutput {
		round_evals,
		skip_rounds,
		remaining_rounds,
		max_domain_size,
		partial_eq_ind_evals,
	})
}

// A helper to perform spread multiplication of small field composition evals by appropriate
// equality indicator scalars. See `zerocheck_univariate_evals` impl for intuition.
fn spread_product<P, FBase>(
	accum: &mut [P],
	small: &[PackedSubfield<P, FBase>],
	large: &[P],
	subcube_index: usize,
	log_n: usize,
	log_batch: usize,
) where
	P: PackedExtension<FBase>,
	FBase: Field,
{
	let log_embedding_degree = <P::Scalar as ExtensionField<FBase>>::LOG_DEGREE;
	let pbase_log_width = P::LOG_WIDTH + log_embedding_degree;

	debug_assert_eq!(accum.len(), 1 << log_n.saturating_sub(P::LOG_WIDTH));
	debug_assert_eq!(small.len(), 1 << (log_n + log_batch).saturating_sub(pbase_log_width));

	if log_n >= P::LOG_WIDTH {
		// Use spread multiplication on fast path.
		let mask = (1 << log_embedding_degree) - 1;
		for batch_idx in 0..1 << log_batch {
			let mult = get_packed_slice(large, subcube_index << log_batch | batch_idx);
			let spread_large = P::cast_base(P::broadcast(mult));

			for (block_idx, dest) in accum.iter_mut().enumerate() {
				let block_offset = block_idx | batch_idx << (log_n - P::LOG_WIDTH);
				let spread_small = small[block_offset >> log_embedding_degree]
					.spread(P::LOG_WIDTH, block_offset & mask);
				*dest += P::cast_ext(spread_large * spread_small);
			}
		}
	} else {
		// Multiple skipped subcube evaluations do fit into a single packed field
		// This never occurs with large traces under frontloaded univariate skip batching,
		// making this a non-critical slow path.
		for (outer_idx, dest) in accum.iter_mut().enumerate() {
			*dest = P::from_fn(|inner_idx| {
				if inner_idx >= 1 << log_n {
					return P::Scalar::ZERO;
				}
				(0..1 << log_batch)
					.map(|batch_idx| {
						let large = get_packed_slice(large, subcube_index << log_batch | batch_idx);
						let small = get_packed_slice_checked(
							small,
							batch_idx << log_n | outer_idx << P::LOG_WIDTH | inner_idx,
						)
						.unwrap_or_default();
						large * small
					})
					.sum()
			})
		}
	}
}

// Extrapolate round evaluations to the full domain.
// NB: this method relies on the fact that `round_evals` have specific lengths
// (namely `d * 2^n`, where `n` is not less than the number of skipped rounds and thus d
// is not larger than the composition degree), which enables additive-NTT based subquadratic
// techniques.
#[instrument(skip_all, level = "debug")]
fn extrapolate_round_evals<F, FDomain, TA>(
	ntt: &SingleThreadedNTT<FDomain, TA>,
	mut round_evals: Vec<Vec<F>>,
	skip_rounds: usize,
	max_domain_size: usize,
) -> Result<Vec<Vec<F>>, Error>
where
	F: BinaryField + ExtensionField<FDomain>,
	FDomain: BinaryField,
	TA: TwiddleAccess<FDomain>,
{
	// Instantiate a large enough NTT over F to be able to forward transform to full domain size.
	// TODO: We can't currently use the `ntt` directly because it is over FDomain. It'd be nice to
	// have helpers that apply a subfield NTT to an extension field vector, without the
	// PackedExtension relation that `forward_transform_ext` and `inverse_transform_ext` require.
	let subspace_upcast = BinarySubspace::new_unchecked(
		ntt.subspace(0)
			.basis()
			.iter()
			.copied()
			.map(F::from)
			.collect(),
	);
	let ntt = SingleThreadedNTT::with_subspace(&subspace_upcast)
		.expect("ntt provided is valid; subspace is equivalent but upcast to F");

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
		let next_log_n = ntt.log_domain_size();
		round_evals.resize(1 << next_log_n, F::ZERO);

		let shape = NTTShape {
			log_y: next_log_n,
			..Default::default()
		};
		ntt.forward_transform(round_evals, shape, 0, 0, 0)?;

		// Sanity check: first 1 << skip_rounds evals are still zeros.
		debug_assert!(
			round_evals[..1 << skip_rounds]
				.iter()
				.all(|&coeff| coeff == F::ZERO)
		);

		// Trim the result.
		round_evals.resize(max_domain_size, F::ZERO);
		round_evals.drain(..1 << skip_rounds);
	}

	Ok(round_evals)
}

fn ntt_extrapolate<NTT, P>(
	ntt: &NTT,
	skip_rounds: usize,
	log_stride_batch: usize,
	log_batch: usize,
	evals: &mut [P],
	extrapolated_evals: &mut [P],
) -> Result<(), Error>
where
	P: PackedField<Scalar: BinaryField>,
	NTT: AdditiveNTT<P::Scalar>,
{
	let shape = NTTShape {
		log_x: log_stride_batch,
		log_y: skip_rounds,
		log_z: log_batch,
	};

	let n_chunks = extrapolated_evals.len() / evals.len();
	let coset_bits = min_bits(n_chunks);

	// Inverse NTT: convert evals to novel basis representation
	ntt.inverse_transform(evals, shape, 0, coset_bits, 0)?;

	// Forward NTT: evaluate novel basis representation at consecutive cosets
	for (coset, extrapolated_chunk) in izip!(1.., extrapolated_evals.chunks_exact_mut(evals.len()))
	{
		// REVIEW: can avoid that copy (and extrapolated_evals scratchpad) when
		// composition_max_degree == 2
		extrapolated_chunk.copy_from_slice(evals);
		ntt.forward_transform(extrapolated_chunk, shape, coset, coset_bits, 0)?;
	}

	Ok(())
}

const fn extrapolated_evals_packed_len<P: PackedField>(
	composition_degree: usize,
	skip_rounds: usize,
	log_batch: usize,
) -> usize {
	composition_degree.saturating_sub(1) << (skip_rounds + log_batch).saturating_sub(P::LOG_WIDTH)
}

#[cfg(test)]
mod tests {
	use std::sync::Arc;

	use binius_field::{
		BinaryField1b, BinaryField8b, BinaryField16b, BinaryField128b, ExtensionField, Field,
		PackedBinaryField4x32b, PackedExtension, PackedField, PackedFieldIndexable, TowerField,
		arch::{OptimalUnderlier128b, OptimalUnderlier512b},
		as_packed_field::{PackScalar, PackedType},
		underlier::UnderlierType,
	};
	use binius_hal::make_portable_backend;
	use binius_math::{BinarySubspace, CompositionPoly, EvaluationDomain, MultilinearPoly};
	use binius_ntt::SingleThreadedNTT;
	use rand::{SeedableRng, prelude::StdRng};

	use crate::{
		composition::{IndexComposition, ProductComposition},
		polynomial::CompositionScalarAdapter,
		protocols::{
			sumcheck::prove::univariate::{domain_size, zerocheck_univariate_evals},
			test_utils::generate_zero_product_multilinears,
		},
		transparent::eq_ind::EqIndPartialEval,
	};

	#[test]
	fn ntt_extrapolate_correctness() {
		type P = PackedBinaryField4x32b;
		type FDomain = BinaryField16b;
		let log_extension_degree_p_domain = 1;

		let mut rng = StdRng::seed_from_u64(0);
		let ntt = SingleThreadedNTT::<FDomain>::new(10).unwrap();
		let subspace = BinarySubspace::<FDomain>::with_dim(10).unwrap();
		let max_domain =
			EvaluationDomain::from_points(subspace.iter().collect::<Vec<_>>(), false).unwrap();

		for skip_rounds in 0..5usize {
			let subsubspace = subspace.reduce_dim(skip_rounds).unwrap();
			let domain =
				EvaluationDomain::from_points(subsubspace.iter().collect::<Vec<_>>(), false)
					.unwrap();
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
						P::cast_bases_mut(&mut interleaved_evals_scratch);
					let extrapolated_evals_domain = P::cast_bases_mut(&mut extrapolated_evals);

					super::ntt_extrapolate(
						&ntt,
						skip_rounds,
						log_extension_degree_p_domain,
						log_batch,
						interleaved_evals_domain,
						extrapolated_evals_domain,
					)
					.unwrap();

					let interleaved_scalars =
						&P::unpack_scalars(&interleaved_evals)[..1 << subcube_vars];
					let extrapolated_scalars = &P::unpack_scalars(&extrapolated_evals)
						[..extrapolated_scalars_cnt << log_batch];

					for batch_idx in 0..1 << log_batch {
						let values =
							&interleaved_scalars[batch_idx << skip_rounds..][..1 << skip_rounds];

						for (i, &point) in max_domain.finite_points()[1 << skip_rounds..]
							[..extrapolated_scalars_cnt]
							.iter()
							.take(1 << skip_rounds)
							.enumerate()
						{
							let extrapolated = domain.extrapolate(values, point.into()).unwrap();
							let expected = extrapolated_scalars[batch_idx << skip_rounds | i];
							assert_eq!(extrapolated, expected);
						}
					}
				}
			}
		}
	}

	#[test]
	fn zerocheck_univariate_evals_invariants_basic() {
		zerocheck_univariate_evals_invariants_helper::<
			OptimalUnderlier128b,
			BinaryField128b,
			BinaryField8b,
			BinaryField16b,
		>()
	}

	#[test]
	fn zerocheck_univariate_evals_with_nontrivial_packing() {
		// Using a 512-bit underlier with a 128-bit extension field means the packed field will have
		// a non-trivial packing width of 4.
		zerocheck_univariate_evals_invariants_helper::<
			OptimalUnderlier512b,
			BinaryField128b,
			BinaryField8b,
			BinaryField16b,
		>()
	}

	fn zerocheck_univariate_evals_invariants_helper<U, F, FDomain, FBase>()
	where
		U: UnderlierType
			+ PackScalar<F>
			+ PackScalar<FBase>
			+ PackScalar<FDomain>
			+ PackScalar<BinaryField1b>,
		F: TowerField + ExtensionField<FDomain> + ExtensionField<FBase>,
		FBase: TowerField + ExtensionField<FDomain>,
		FDomain: TowerField + From<u8>,
		PackedType<U, FBase>: PackedFieldIndexable,
		PackedType<U, FDomain>: PackedFieldIndexable,
		PackedType<U, F>: PackedFieldIndexable,
	{
		let mut rng = StdRng::seed_from_u64(0);

		let n_vars = 7;
		let log_embedding_degree = <F as ExtensionField<FBase>>::LOG_DEGREE;

		let mut multilinears = generate_zero_product_multilinears::<
			PackedType<U, BinaryField1b>,
			PackedType<U, F>,
		>(&mut rng, n_vars, 2);
		multilinears.extend(generate_zero_product_multilinears(&mut rng, n_vars, 3));
		multilinears.extend(generate_zero_product_multilinears(&mut rng, n_vars, 4));

		let compositions = [
			Arc::new(IndexComposition::new(9, [0, 1], ProductComposition::<2> {}).unwrap())
				as Arc<dyn CompositionPoly<PackedType<U, FBase>>>,
			Arc::new(IndexComposition::new(9, [2, 3, 4], ProductComposition::<3> {}).unwrap())
				as Arc<dyn CompositionPoly<PackedType<U, FBase>>>,
			Arc::new(IndexComposition::new(9, [5, 6, 7, 8], ProductComposition::<4> {}).unwrap())
				as Arc<dyn CompositionPoly<PackedType<U, FBase>>>,
		];

		let backend = make_portable_backend();
		let zerocheck_challenges = (0..n_vars)
			.map(|_| <F as Field>::random(&mut rng))
			.collect::<Vec<_>>();

		for skip_rounds in 0usize..=5 {
			let max_domain_size = domain_size(5, skip_rounds);
			let output =
				zerocheck_univariate_evals::<F, FDomain, FBase, PackedType<U, F>, _, _, _>(
					&multilinears,
					&compositions,
					&zerocheck_challenges[skip_rounds..],
					skip_rounds,
					max_domain_size,
					&backend,
				)
				.unwrap();

			let zerocheck_eq_ind = EqIndPartialEval::new(&zerocheck_challenges[skip_rounds..])
				.multilinear_extension::<F, _>(&backend)
				.unwrap();

			// naive computation of the univariate skip output
			let round_evals_len = 4usize << skip_rounds;
			assert!(
				output
					.round_evals
					.iter()
					.all(|round_evals| round_evals.len() == round_evals_len)
			);

			let compositions = compositions
				.iter()
				.cloned()
				.map(CompositionScalarAdapter::new)
				.collect::<Vec<_>>();

			let mut query = [FBase::ZERO; 9];
			let mut evals = vec![
				PackedType::<U, F>::zero();
				1 << skip_rounds.saturating_sub(
					log_embedding_degree + PackedType::<U, F>::LOG_WIDTH
				)
			];
			let subspace = BinarySubspace::<FDomain>::with_dim(skip_rounds).unwrap();
			let domain =
				EvaluationDomain::from_points(subspace.iter().collect::<Vec<_>>(), false).unwrap();
			for round_evals_index in 0..round_evals_len {
				let x = FDomain::from(((1 << skip_rounds) + round_evals_index) as u8);
				let mut composition_sums = vec![F::ZERO; compositions.len()];
				for subcube_index in 0..1 << (n_vars - skip_rounds) {
					for (query, multilinear) in query.iter_mut().zip(&multilinears) {
						multilinear
							.subcube_evals(
								skip_rounds,
								subcube_index,
								log_embedding_degree,
								&mut evals,
							)
							.unwrap();
						let evals_scalars = &PackedType::<U, FBase>::unpack_scalars(
							PackedExtension::<FBase>::cast_bases(&evals),
						)[..1 << skip_rounds];
						let extrapolated = domain.extrapolate(evals_scalars, x.into()).unwrap();
						*query = extrapolated;
					}

					let eq_ind_factor = zerocheck_eq_ind
						.evaluate_on_hypercube(subcube_index)
						.unwrap();
					for (composition, sum) in compositions.iter().zip(composition_sums.iter_mut()) {
						*sum += eq_ind_factor * composition.evaluate(&query).unwrap();
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
