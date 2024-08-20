// Copyright 2024 Ulvetanna Inc.

use super::Error;
use crate::polynomial::{
	Error as PolynomialError, MultilinearExtensionSpecialized, MultilinearPoly, MultilinearQuery,
};
use binius_field::PackedField;
use binius_utils::{array_2d::Array2D, bail};
use rayon::prelude::*;
use std::{cmp::max, collections::HashMap, hash::Hash, ops::Range};

/// An individual multilinear polynomial in a multivariate composite.
#[derive(Debug, Clone)]
enum SumcheckMultilinear<P, M>
where
	P: PackedField,
	M: MultilinearPoly<P> + Send + Sync,
{
	/// Small field polynomial - to be folded into large field at `switchover` round
	Transparent {
		multilinear: M,
		introduction_round: usize,
		switchover_round: usize,
	},
	/// Large field polynomial - halved in size each round
	Folded {
		large_field_folded_multilinear: MultilinearExtensionSpecialized<P, P>,
	},
}

/// Parallel fold state, consisting of scratch area and result accumulator.
struct ParFoldStates<P: PackedField> {
	// Evaluations at 0, 1 and domain points, per MLE. Scratch space.
	evals_0: Array2D<P>,
	evals_1: Array2D<P>,
	evals_z: Array2D<P>,

	// Accumulated sums of evaluations over univariate domain.
	round_evals: Array2D<P>,
}

impl<P: PackedField> ParFoldStates<P> {
	fn new(n_multilinears: usize, n_round_evals: usize, n_states: usize) -> Self {
		Self {
			evals_0: Array2D::zeroes(n_states, n_multilinears),
			evals_1: Array2D::zeroes(n_states, n_multilinears),
			evals_z: Array2D::zeroes(n_states, n_multilinears),
			round_evals: Array2D::zeroes(n_states, n_round_evals),
		}
	}
}

/// Represents an object that can evaluate the composition function of a generalized sumcheck.
///
/// Generalizes handling of regular sumcheck and zerocheck protocols.
pub trait AbstractSumcheckEvaluator<P: PackedField>: Sync {
	type VertexState;

	/// The number of points to evaluate at.
	fn n_round_evals(&self) -> usize;

	/// Process and update the round evaluations with the evaluations at a hypercube vertex.
	///
	/// ## Arguments
	///
	/// * `i`: index of the hypercube vertex under processing
	/// * `vertex_state`: state of the hypercube vertex under processing
	/// * `evals_0`: the n multilinear polynomial evaluations at 0
	/// * `evals_1`: the n multilinear polynomial evaluations at 1
	/// * `evals_z`: a scratch buffer of size n for storing multilinear polynomial evaluations at a
	///              point z
	/// * `round_evals`: the accumulated evaluations for the round
	fn process_vertex(
		&self,
		i: usize,
		vertex_state: Self::VertexState,
		evals_0: &[P],
		evals_1: &[P],
		evals_z: &mut [P],
		round_evals: &mut [P],
	);

	/// Given evaluations of the round polynomial, interpolate and return monomial coefficients
	///
	/// ## Arguments
	///
	/// * `current_round_sum`: the claimed sum for the current round
	/// * `round_evals`: the computed evaluations of the round polynomial
	fn round_evals_to_coeffs(
		&self,
		current_round_sum: P::Scalar,
		round_evals: Vec<P::Scalar>,
	) -> Result<Vec<P::Scalar>, PolynomialError>;
}

/// A common provers state for a generalized batched sumcheck protocol.
///
/// The family of generalized sumcheck protocols includes regular sumcheck, zerocheck and others. The
/// zerocheck case permits many important optimizations, enumerated in [Gruen24]. These algorithms
/// are used to prove the interactive multivariate sumcheck protocol in the specific case that the
/// polynomial is a composite of multilinears. This prover state is responsible for updating and
/// evaluating the composed multilinears.
///
/// Once initialized, the expected caller behavior is, for a total of `n_rounds`:
///   1. At the beginning of each step, call [`Self::extend`] with multilinears introduced in this round
///   2. Then call [`Self::pre_execute_rounds`] to perform query expansion, folding, and other bookkeeping.
///   3. Call [`Self::calculate_round_coeffs`] on each multilinear subset with an appropriate evaluator.
///
/// We associate with each multilinear a `switchover` round number, which controls small field
/// optimization and corresponding time/memory tradeoff. In rounds $0, \ldots, switchover-1$ the
/// partial evaluation of a specific multilinear is obtained by doing $2^{n\\_vars - round}$ inner
/// products, with total time complexity proportional to the number of polynomial coefficients.
///
/// After switchover the inner products are stored in a new MLE in large field, which is halved on
/// each round. There are two tradeoffs at play:
///   1) Pre-switchover rounds perform Small * Large field multiplications, but do $2^{round}$ as many of them.
///   2) Pre-switchover rounds require no additional memory, but initial folding allocates a new MLE in a
///      large field that is $2^{switchover}$ times smaller - for example for 1-bit polynomial and 128-bit large
///      field a switchover of 7 would require additional memory identical to the polynomial size.
///
/// NB. Note that `switchover=0` does not make sense, as first round is never folded. Also note that
///     switchover rounds are numbered relative introduction round.
///
/// [Gruen24]: https://eprint.iacr.org/2024/108
pub struct CommonProversState<MultilinearId, PW, M>
where
	MultilinearId: Hash + Eq + Sync,
	PW: PackedField,
	M: MultilinearPoly<PW> + Send + Sync,
{
	n_vars: usize,
	switchover_fn: Box<dyn Fn(usize) -> usize>,
	next_round: usize,
	multilinears: HashMap<MultilinearId, SumcheckMultilinear<PW, M>>,
	max_query_vars: Option<usize>,
	queries: Vec<Option<MultilinearQuery<PW>>>,
}

impl<MultilinearId, PW, M> CommonProversState<MultilinearId, PW, M>
where
	MultilinearId: Clone + Hash + Eq + Sync,
	PW: PackedField,
	M: MultilinearPoly<PW> + Sync + Send,
{
	pub fn new(n_vars: usize, switchover_fn: impl Fn(usize) -> usize + 'static) -> Self {
		Self {
			n_vars,
			switchover_fn: Box::new(switchover_fn),
			next_round: 0,
			multilinears: HashMap::new(),
			max_query_vars: None,
			queries: Vec::new(),
		}
	}

	pub fn extend(
		&mut self,
		multilinears: impl IntoIterator<Item = (MultilinearId, M)>,
	) -> Result<(), Error> {
		let introduction_round = self.next_round;

		for (multilinear_id, multilinear) in multilinears {
			let switchover_round = max(1, (self.switchover_fn)(multilinear.extension_degree()));
			self.max_query_vars = Some(max(self.max_query_vars.unwrap_or(1), switchover_round));

			if introduction_round + multilinear.n_vars() != self.n_vars {
				bail!(PolynomialError::IncorrectNumberOfVariables {
					expected: self.n_vars - introduction_round,
					actual: multilinear.n_vars(),
				});
			}

			// TODO Consider bailing out on non-idempotent rewrites?
			self.multilinears.insert(
				multilinear_id.clone(),
				SumcheckMultilinear::Transparent {
					multilinear,
					introduction_round,
					switchover_round,
				},
			);
		}

		Ok(())
	}

	pub fn pre_execute_rounds(
		&mut self,
		prev_rd_challenge: Option<PW::Scalar>,
	) -> Result<(), PolynomialError> {
		assert_eq!(self.next_round == 0, prev_rd_challenge.is_none());

		// Update queries (have to be done before switchover)
		if let Some(prev_rd_challenge) = prev_rd_challenge {
			for query in self.queries.iter_mut() {
				if let Some(prev_query) = query.take() {
					let expanded_query = prev_query.update(&[prev_rd_challenge])?;
					query.replace(expanded_query);
				}
			}
		}

		let new_query = self
			.max_query_vars
			.take()
			.map(|max_query_vars| MultilinearQuery::new(max_query_vars))
			.transpose()?;

		self.queries.push(new_query);
		self.next_round += 1;
		debug_assert_eq!(self.next_round, self.queries.len());

		// Fold multilinears
		if let Some(prev_rd_challenge) = prev_rd_challenge {
			self.fold(prev_rd_challenge)?;
		}

		Ok(())
	}

	/// Fold all stored multilinears with the verifier challenge received in the previous round.
	///
	/// This manages whether to partially evaluate the multilinear at an extension point
	/// (post-switchover) or to store the extended tensor product of the received queries
	/// (pre-switchover).
	///
	/// See struct documentation for more details on the generalized sumcheck proving algorithm.
	fn fold(&mut self, prev_rd_challenge: PW::Scalar) -> Result<(), PolynomialError> {
		let &mut Self {
			ref mut multilinears,
			ref mut queries,
			next_round,
			..
		} = self;

		// Partial query for folding
		let single_variable_partial_query =
			MultilinearQuery::with_full_query(&[prev_rd_challenge])?;

		// Perform switchover and/or folding
		let any_transparent_left = multilinears
			.par_iter_mut()
			.map(|(_, sc_multilinear)| -> Result<Option<usize>, PolynomialError> {
				match sc_multilinear {
					&mut SumcheckMultilinear::Transparent {
						ref mut multilinear,
						introduction_round,
						switchover_round,
					} => {
						// NB. next_round is already incremented there, hence "less than" comparison.
						if switchover_round + introduction_round < next_round {
							let query_ref = queries[introduction_round].as_ref().expect(
								"query is guaranteed to be Some while there are transparent \
								     multilinears remaining",
							);
							// At switchover, perform inner products in large field and save them
							// in a newly created MLE.
							let large_field_folded_multilinear =
								multilinear.evaluate_partial_low(query_ref)?;

							*sc_multilinear = SumcheckMultilinear::Folded {
								large_field_folded_multilinear,
							};

							Ok(None)
						} else {
							Ok(Some(introduction_round))
						}
					}

					SumcheckMultilinear::Folded {
						ref mut large_field_folded_multilinear,
					} => {
						// Post-switchover, simply halve large field MLE.
						*large_field_folded_multilinear = large_field_folded_multilinear
							.evaluate_partial_low(&single_variable_partial_query)?;

						Ok(None)
					}
				}
			})
			.try_fold(
				|| vec![false; queries.len()],
				|mut any_transparent_left,
				 opt_round: Result<Option<usize>, PolynomialError>|
				 -> Result<Vec<bool>, PolynomialError> {
					if let Some(round) = opt_round? {
						any_transparent_left[round] = true;
					}
					Ok(any_transparent_left)
				},
			)
			.try_reduce(
				|| vec![false; queries.len()],
				|mut any_transparent_lhs, any_transparent_rhs| {
					any_transparent_lhs
						.iter_mut()
						.zip(any_transparent_rhs)
						.for_each(|(lhs, rhs)| *lhs |= rhs);

					Ok(any_transparent_lhs)
				},
			)?;

		// All folded large field - tensor is no more needed.
		for (query, keep) in queries.iter_mut().zip(any_transparent_left) {
			if !keep {
				*query = None;
			}
		}

		Ok(())
	}

	/// Compute the sum of the partial polynomial evaluations over the hypercube.
	pub fn calculate_round_coeffs<VS>(
		&self,
		multilinear_ids: &[MultilinearId],
		evaluator: impl AbstractSumcheckEvaluator<PW, VertexState = VS>,
		current_round_sum: PW::Scalar,
		vertex_state_iterator: impl IndexedParallelIterator<Item = VS>,
	) -> Result<Vec<PW::Scalar>, Error> {
		assert!(
			self.max_query_vars.is_none(),
			"extend() called after pre_execute_rounds() but before calculate_round_coeffs()"
		);

		// Extract multilinears & round
		let &Self {
			ref multilinears,
			next_round,
			..
		} = self;

		// Handling different cases separately for more inlining opportunities
		// (especially in early rounds)
		let mut any_transparent = false;
		let mut any_folded = false;

		for multilinear_id in multilinear_ids {
			let multilinear = multilinears
				.get(multilinear_id)
				.ok_or(Error::WitnessNotFound)?;

			match multilinear {
				SumcheckMultilinear::Transparent { .. } => {
					any_transparent = true;
				}

				SumcheckMultilinear::Folded { .. } => {
					any_folded = true;
				}
			}
		}

		let opt_query = self.get_subset_query(multilinear_ids);

		match (any_transparent, any_folded, opt_query) {
			(true, false, Some((introduction_round, _)))
				if introduction_round + 1 == next_round =>
			{
				// All transparent, first round - direct sampling
				self.calculate_round_coeffs_helper(
					multilinear_ids,
					Self::only_transparent,
					Self::direct_sample,
					evaluator,
					vertex_state_iterator,
					current_round_sum,
				)
			}

			(true, false, Some((introduction_round, query)))
				if introduction_round + 1 < next_round =>
			{
				self.calculate_round_coeffs_helper(
					multilinear_ids,
					Self::only_transparent,
					|multilin, indices, evals0, evals1, col| {
						Self::subcube_inner_product(query, multilin, indices, evals0, evals1, col)
					},
					evaluator,
					vertex_state_iterator,
					current_round_sum,
				)
			}

			// All folded - direct sampling
			(false, true, _) => self.calculate_round_coeffs_helper(
				multilinear_ids,
				Self::only_folded,
				Self::direct_sample,
				evaluator,
				vertex_state_iterator,
				current_round_sum,
			),

			// Heterogeneous case
			(_, _, Some((_, query))) => self.calculate_round_coeffs_helper(
				multilinear_ids,
				|x| x,
				|sc_multilin, indices, evals0, evals1, col| match sc_multilin {
					SumcheckMultilinear::Transparent { multilinear, .. } => {
						Self::subcube_inner_product(
							query,
							multilinear,
							indices,
							evals0,
							evals1,
							col,
						)
					}

					SumcheckMultilinear::Folded {
						large_field_folded_multilinear,
					} => Self::direct_sample(
						large_field_folded_multilinear,
						indices,
						evals0,
						evals1,
						col,
					),
				},
				evaluator,
				vertex_state_iterator,
				current_round_sum,
			),

			_ => panic!("tensor not present during sumcheck, or some other invalid case"),
		}
	}

	// The gist of sumcheck - summing over evaluations of the multivariate composite on evaluation domain
	// for the remaining variables: there are `round-1` already assigned variables with values from large
	// field, and `rd_vars = n_vars - round` remaining variables that are being summed over. `eval01` closure
	// computes 0 & 1 evaluations at some index - either by performing inner product over assigned variables
	// pre-switchover or directly sampling MLE representation during first round or post-switchover.
	fn calculate_round_coeffs_helper<'b, T, VS>(
		&'b self,
		multilinear_ids: &[MultilinearId],
		precomp: impl Fn(&'b SumcheckMultilinear<PW, M>) -> T,
		eval01: impl Fn(T, Range<usize>, &mut Array2D<PW>, &mut Array2D<PW>, usize) + Sync,
		evaluator: impl AbstractSumcheckEvaluator<PW, VertexState = VS>,
		vertex_state_iterator: impl IndexedParallelIterator<Item = VS>,
		current_round_sum: PW::Scalar,
	) -> Result<Vec<PW::Scalar>, Error>
	where
		T: Copy + Sync + 'b,
		M: 'b,
	{
		// When possible to pre-process unpacking sumcheck multilinears, we do so.
		// For performance, it's ideal to hoist this out of the tight loop.
		let precomps = multilinear_ids
			.iter()
			.map(|multilinear_id| -> Result<T, Error> {
				let sc_multilinear = self
					.multilinears
					.get(multilinear_id)
					.ok_or(Error::WitnessNotFound)?;
				Ok(precomp(sc_multilinear))
			})
			.collect::<Result<Vec<_>, _>>()?;

		let n_multilinears = precomps.len();
		let n_round_evals = evaluator.n_round_evals();

		/// Process batches of vertices in parallel, accumulating the round evaluations.
		const BATCH_SIZE: usize = 64;

		let evals = vertex_state_iterator
			.chunks(BATCH_SIZE)
			.enumerate()
			.fold(
				|| ParFoldStates::new(n_multilinears, n_round_evals, BATCH_SIZE),
				|mut par_fold_states, (vertex, vertex_states)| {
					let begin = vertex * BATCH_SIZE;
					let end = begin + vertex_states.len();
					for (j, precomp) in precomps.iter().enumerate() {
						eval01(
							*precomp,
							begin..end,
							&mut par_fold_states.evals_0,
							&mut par_fold_states.evals_1,
							j,
						);
					}

					for (k, vertex_state) in vertex_states.into_iter().enumerate() {
						evaluator.process_vertex(
							begin + k,
							vertex_state,
							par_fold_states.evals_0.get_row(k),
							par_fold_states.evals_1.get_row(k),
							par_fold_states.evals_z.get_row_mut(k),
							par_fold_states.round_evals.get_row_mut(k),
						);
					}

					par_fold_states
				},
			)
			.map(|states| states.round_evals.sum_rows())
			// Simply sum up the fold partitions.
			.reduce(
				|| vec![PW::zero(); n_round_evals],
				|mut overall_round_evals, partial_round_evals| {
					overall_round_evals
						.iter_mut()
						.zip(partial_round_evals.iter())
						.for_each(|(f, s)| *f += *s);
					overall_round_evals
				},
			)
			.iter()
			.map(|x| x.iter().sum())
			.collect();

		Ok(evaluator.round_evals_to_coeffs(current_round_sum, evals)?)
	}

	// Obtain the appropriate switchover multilinear query, relative introduction round.
	fn get_subset_query(
		&self,
		oracle_ids: &[MultilinearId],
	) -> Option<(usize, &MultilinearQuery<PW>)> {
		let introduction_rounds = oracle_ids
			.iter()
			.flat_map(|oracle_id| match self.multilinears.get(oracle_id)? {
				SumcheckMultilinear::Transparent {
					introduction_round, ..
				} => Some(*introduction_round),
				_ => None,
			})
			.collect::<Vec<_>>();

		let first_introduction_round = *introduction_rounds.first()?;
		if introduction_rounds
			.iter()
			.any(|&round| round != first_introduction_round)
		{
			return None;
		}

		self.queries
			.get(first_introduction_round)?
			.as_ref()
			.map(|query| (first_introduction_round, query))
	}

	// Note the generic parameter - this method samples small field in first round and
	// large field post-switchover.
	#[inline]
	fn direct_sample<MD>(
		multilin: MD,
		indices: Range<usize>,
		evals_0: &mut Array2D<PW>,
		evals_1: &mut Array2D<PW>,
		col_index: usize,
	) where
		MD: MultilinearPoly<PW>,
	{
		for (k, i) in indices.enumerate() {
			evals_0[(k, col_index)] = PW::from_fn(|j| {
				multilin
					.evaluate_on_hypercube((i * PW::WIDTH + j) << 1)
					.unwrap_or_default()
			});
			evals_1[(k, col_index)] = PW::from_fn(|j| {
				multilin
					.evaluate_on_hypercube(((i * PW::WIDTH + j) << 1) + 1)
					.unwrap_or_default()
			});
		}
	}

	#[inline]
	fn subcube_inner_product(
		query: &MultilinearQuery<PW>,
		multilin: &M,
		indices: Range<usize>,
		evals_0: &mut Array2D<PW>,
		evals_1: &mut Array2D<PW>,
		col_index: usize,
	) {
		multilin
			.evaluate_subcube(indices, query, evals_0, evals_1, col_index)
			.expect("indices within range");
	}

	fn only_transparent(sc_multilin: &SumcheckMultilinear<PW, M>) -> &M {
		match sc_multilin {
			SumcheckMultilinear::Transparent { multilinear, .. } => multilinear,
			_ => panic!("all transparent by invariant"),
		}
	}

	fn only_folded(
		sc_multilin: &SumcheckMultilinear<PW, M>,
	) -> &MultilinearExtensionSpecialized<PW, PW> {
		match sc_multilin {
			SumcheckMultilinear::Folded {
				large_field_folded_multilinear,
			} => large_field_folded_multilinear,
			_ => panic!("all folded by invariant"),
		}
	}
}
