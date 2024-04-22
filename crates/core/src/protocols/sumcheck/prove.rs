// Copyright 2023 Ulvetanna Inc.

use super::{
	error::Error,
	sumcheck::{
		reduce_sumcheck_claim_final, reduce_sumcheck_claim_round, reduce_zerocheck_claim_round,
		SumcheckClaim, SumcheckRound, SumcheckRoundClaim, SumcheckWitness,
	},
};
use crate::{
	oracle::CompositePolyOracle,
	polynomial::{
		extrapolate_line, multilinear_query::MultilinearQuery,
		transparent::eq_ind::EqIndPartialEval, CompositionPoly, EvaluationDomain,
		MultilinearExtension, MultilinearExtensionSpecialized, MultilinearPoly,
	},
	protocols::evalcheck::EvalcheckClaim,
};
use binius_field::{Field, PackedField};
use getset::Getters;
use rayon::prelude::*;
use std::{borrow::Borrow, cmp::max, fmt::Debug, marker::PhantomData};
use tracing::instrument;

/// An individual multilinear polynomial in a multivariate composite
#[derive(Debug)]
enum SumcheckMultilinear<P: PackedField, M> {
	/// Small field polynomial - to be folded into large field at `switchover` round
	Transparent {
		switchover: usize,
		small_field_multilin: M,
	},
	/// Large field polynomial - halved in size each round
	Folded {
		large_field_folded_multilin: MultilinearExtensionSpecialized<'static, P, P>,
	},
}

/// Parallel fold state, consisting of scratch area and result accumulator.
struct ParFoldState<F: Field> {
	// Evaluations at 0, 1 and domain points, per MLE. Scratch space.
	evals_0: Vec<F>,
	evals_1: Vec<F>,
	evals_z: Vec<F>,

	// Accumulated sums of evaluations over univariate domain.
	round_evals: Vec<F>,
}

impl<F: Field> ParFoldState<F> {
	fn new(n_multilinears: usize, n_round_evals: usize) -> Self {
		Self {
			evals_0: vec![F::ZERO; n_multilinears],
			evals_1: vec![F::ZERO; n_multilinears],
			evals_z: vec![F::ZERO; n_multilinears],
			round_evals: vec![F::ZERO; n_round_evals],
		}
	}
}

#[derive(Debug)]
struct ZerocheckAuxiliaryState<F, PW>
where
	F: Field,
	PW: PackedField,
	PW::Scalar: From<F>,
{
	challenges: Vec<F>,
	round_eq_ind: MultilinearExtension<'static, PW::Scalar>,
}

impl<F, PW> ZerocheckAuxiliaryState<F, PW>
where
	F: Field,
	PW: PackedField,
	PW::Scalar: From<F>,
{
	// Update the round_eq_ind for the next sumcheck round
	//
	// Let
	//  * $n$ be the number of variables in the sumcheck claim
	//  * $eq_k(X, Y)$ denote the equality indicator polynomial on $2 * k$ variables.
	//  * $\alpha_1, \ldots, \alpha_{n-1}$ be the $n-1$ zerocheck challenges
	// In round $i$, before computing the round polynomial, we seek the invariant that
	// * round_eq_ind is MLE of $eq_{n-i-1}(X, Y)$ partially evaluated at $Y = (\alpha_{i+1}, \ldots, \alpha_{n-1})$.
	//
	// To update the round_eq_ind, from $eq_{n-i}(X, \alpha_i, \ldots, \alpha_{n-1})$
	// to $eq_{n-i-1}(X, \alpha_{i+1}, \ldots, \alpha_{n-1})$, we sum consecutive hypercube evaluations.
	//
	// For example consider the hypercube evaluations of $eq_2(X, $\alpha_1, \alpha_2)$
	// * [$(1-\alpha_1) * (1-\alpha_2)$, $\alpha_1 * (1-\alpha_2)$, $(1-\alpha_1) * \alpha_2$, $\alpha_1 * \alpha_2$]
	// and consider the hypercube evaluations of $eq_1(X, \alpha_2)$
	// * [$(1-\alpha_2)$, $\alpha_2$]
	// We obtain the ith hypercube evaluation of $eq_1(X, \alpha_2)$ by summing the $(2*i)$ and $(2*i+1)$
	// hypercube evaluations of $eq_2(X, \alpha_1, \alpha_2)$.
	fn update_round_eq_ind(&mut self) -> Result<(), Error> {
		let current_evals = self.round_eq_ind.evals();
		let mut new_evals = vec![PW::Scalar::default(); current_evals.len() >> 1];
		new_evals.par_iter_mut().enumerate().for_each(|(i, e)| {
			*e = current_evals[i << 1] + current_evals[(i << 1) + 1];
		});
		let new_multilin = MultilinearExtension::from_values(new_evals)?;
		self.round_eq_ind = new_multilin;

		Ok(())
	}
}

/// A mutable prover state. To prove a sumcheck claim, supply a multivariate composite witness. In
/// some cases it makes sense to do so in an different yet isomorphic field PW (witness packed
/// field) which may preferable due to superior performance. One example of such operating field
/// would be BinaryField128bPolyval, which tends to be much faster than 128-bit tower field on x86
/// CPUs. The only constraint is that constituent MLEs should have MultilinearPoly impls for PW -
/// something which is trivially satisfied for MLEs with tower field scalars for claims in tower
/// field as well.
///
/// Prover state is instantiated via `new` method, followed by exactly n_vars `execute_round` invocations.
/// Each of those takes in an optional challenge (None on first round and Some on following rounds) and
/// evaluation domain. Proof and Evalcheck claim are obtained via `finalize` call at the end.
///
/// Each MLE in the multivariate composite is parameterized by `switchover` round number, which
/// controls small field optimization and corresponding time/memory tradeoff. In rounds
/// 0..(switchover-1) the partial evaluation of a specific MLE is obtained by doing
/// 2**(n_vars-round) inner products, with total time complexity proportional to the MLE size.
/// After switchover the inner products are stored in a new MLE in large field, which is halved on each
/// round. There are two tradeoffs at play:
///   1) Pre-switchover rounds perform Small * Large field multiplications, but do 2^round as many of them.
///   2) Pre-switchover rounds require no additional memory, but initial folding allocates a new MLE in a
///      large field that is 2^switchover times smaller - for example for 1-bit polynomial and 128-bit large
///      field a switchover of 7 would require additional memory identical to the polynomial size.
///
/// NB. Note that switchover=0 does not make sense, as first round is never folded.
#[derive(Debug, Getters)]
pub struct SumcheckProverState<'a, F, PW, CW, M>
where
	F: Field + From<PW::Scalar>,
	PW: PackedField,
	PW::Scalar: From<F>,
	CW: CompositionPoly<PW>,
	M: MultilinearPoly<PW> + Sync,
{
	oracle: CompositePolyOracle<F>,
	composition: CW,
	multilinears: Vec<SumcheckMultilinear<PW, M>>,

	domain: &'a EvaluationDomain<F>,
	query: Option<MultilinearQuery<PW>>,

	#[getset(get = "pub")]
	round_claim: SumcheckRoundClaim<F>,

	round: usize,
	last_round_proof: Option<SumcheckRound<F>>,
	zerocheck_aux_state: Option<ZerocheckAuxiliaryState<F, PW>>,
	_m_marker: PhantomData<M>,
}

impl<'a, F, PW, CW, M> SumcheckProverState<'a, F, PW, CW, M>
where
	F: Field + From<PW::Scalar>,
	PW: PackedField,
	PW::Scalar: From<F>,
	CW: CompositionPoly<PW>,
	M: MultilinearPoly<PW> + Sync,
{
	/// Start a new sumcheck instance with claim in field `F`. Witness may be given in
	/// a different (but isomorphic) packed field PW. `switchovers` slice contains rounds
	/// given multilinear index - expectation is that the caller would have enough context to
	/// introspect on field & polynomial sizes. This is a stopgap measure until a proper type
	/// reflection mechanism is introduced.
	pub fn new(
		domain: &'a EvaluationDomain<F>,
		sumcheck_claim: SumcheckClaim<F>,
		sumcheck_witness: SumcheckWitness<PW, CW, M>,
		switchovers: &[usize],
	) -> Result<Self, Error> {
		let n_vars = sumcheck_claim.n_vars();

		if sumcheck_claim.poly.max_individual_degree() == 0 {
			return Err(Error::PolynomialDegreeIsZero);
		}

		if sumcheck_witness.n_vars() != n_vars {
			let err_str = format!(
				"Claim and Witness n_vars mismatch in sumcheck. Claim: {}, Witness: {}",
				sumcheck_claim.n_vars(),
				sumcheck_witness.n_vars(),
			);

			return Err(Error::ProverClaimWitnessMismatch(err_str));
		}

		if switchovers.len() != sumcheck_witness.n_multilinears() {
			let err_str = format!(
				"Witness contains {} multilinears but {} switchovers are given.",
				sumcheck_witness.n_multilinears(),
				switchovers.len()
			);

			return Err(Error::ImproperInput(err_str));
		}

		check_evaluation_domain(sumcheck_claim.poly.max_individual_degree(), domain)?;

		let mut max_query_vars = 1;
		let mut multilinears = Vec::new();

		for (small_field_multilin, &switchover) in
			sumcheck_witness.multilinears.into_iter().zip(switchovers)
		{
			max_query_vars = max(max_query_vars, switchover);
			multilinears.push(SumcheckMultilinear::Transparent {
				switchover,
				small_field_multilin,
			});
		}

		let composition = sumcheck_witness.composition;

		let query = Some(MultilinearQuery::new(max_query_vars)?);

		let round_claim = SumcheckRoundClaim {
			partial_point: Vec::new(),
			current_round_sum: sumcheck_claim.sum,
		};

		let zerocheck_aux_state =
			if let Some(zc_challenges) = sumcheck_claim.zerocheck_challenges.clone() {
				let pw_challenges = zc_challenges
					.iter()
					.map(|&f| f.into())
					.collect::<Vec<PW::Scalar>>();

				let round_eq_multilin =
					EqIndPartialEval::new(n_vars - 1, pw_challenges)?.multilinear_extension()?;
				Some(ZerocheckAuxiliaryState {
					challenges: zc_challenges,
					round_eq_ind: round_eq_multilin,
				})
			} else {
				None
			};

		let prover_state = SumcheckProverState {
			oracle: sumcheck_claim.poly,
			composition,
			multilinears,
			domain,
			query,
			round_claim,
			round: 0,
			last_round_proof: None,
			zerocheck_aux_state,
			_m_marker: PhantomData,
		};

		Ok(prover_state)
	}

	pub fn n_vars(&self) -> usize {
		self.oracle.n_vars()
	}

	/// Generic parameters allow to pass a different witness type to the inner Evalcheck claim.
	#[instrument(skip_all, name = "sumcheck::SumcheckProverState::finalize")]
	pub fn finalize(mut self, prev_rd_challenge: Option<F>) -> Result<EvalcheckClaim<F>, Error> {
		// First round has no challenge, other rounds should have it
		self.validate_rd_challenge(prev_rd_challenge)?;

		if self.round != self.n_vars() {
			return Err(Error::ImproperInput(format!(
				"finalize() called on round {} while n_vars={}",
				self.round,
				self.n_vars()
			)));
		}

		// Last reduction to obtain eval value at eval_point
		if let Some(prev_rd_challenge) = prev_rd_challenge {
			self.reduce_claim(prev_rd_challenge)?;
		}

		reduce_sumcheck_claim_final(&self.oracle, self.round_claim)
	}

	fn is_zerocheck(&self) -> bool {
		self.zerocheck_aux_state.is_some()
	}

	fn zerocheck_challenges(&self) -> Option<&Vec<F>> {
		self.zerocheck_aux_state.as_ref().map(|aux| &aux.challenges)
	}

	fn zerocheck_eq_ind(&self) -> Option<&MultilinearExtension<'static, PW::Scalar>> {
		self.zerocheck_aux_state
			.as_ref()
			.map(|aux| &aux.round_eq_ind)
	}

	fn get_round_polynomial_degree(&self) -> usize {
		self.composition.degree()
	}

	fn evals_to_coeffs(
		&self,
		mut evals: Vec<F>,
		domain: &EvaluationDomain<F>,
	) -> Result<Vec<F>, Error> {
		// We have partial information about the degree $d$ univariate round polynomial $r(X)$,
		// but in each of the following cases, we can complete the picture and attain evaluations
		// at $r(0), \ldots, r(d+1)$.
		if let Some(zc_challenges) = self.zerocheck_challenges() {
			if self.round == 0 {
				// This is the case where we are processing the first round of a sumcheck that came from zerocheck.
				// We are given $r(2), \ldots, r(d+1)$.
				// From context, we infer that $r(0) = r(1) = 0$.
				evals.insert(0, F::ZERO);
				evals.insert(0, F::ZERO);
			} else {
				// This is a subsequent round of a sumcheck that came from zerocheck, given $r(1), \ldots, r(d+1)$
				// Letting $s$ be the current round's claimed sum, and $\alpha_i$ the ith zerocheck challenge
				// we have the identity $r(0) = \frac{1}{1 - \alpha_i} * (s - \alpha_i * r(1))$
				// which allows us to compute the value of $r(0)$
				let alpha = zc_challenges[self.round - 1];
				let alpha_bar = F::ONE - alpha;
				let one_evaluation = evals[0];
				let zero_evaluation_numerator =
					self.round_claim().current_round_sum - one_evaluation * alpha;
				let zero_evaluation_denominator_inv = alpha_bar.invert().unwrap();
				let zero_evaluation = zero_evaluation_numerator * zero_evaluation_denominator_inv;

				evals.insert(0, zero_evaluation);
			}
		} else {
			// In the case where this is a sumcheck that did not come from zerocheck,
			// Given $r(1), \ldots, r(d+1)$, letting $s$ be the current round's claimed sum,
			// we can compute $r(0)$ using the identity $r(0) = s - r(1)$
			evals.insert(0, self.round_claim().current_round_sum - evals[0]);
		}

		assert_eq!(evals.len(), domain.size());
		let coeffs = domain.interpolate(&evals)?;
		Ok(coeffs)
	}

	// NB: Can omit some coefficients to reduce proof size because verifier can compute
	// these missing coefficients.
	// This is documented in detail in the common prover-verifier logic for
	// reducing a sumcheck round claim.
	fn trim_coeffs<T: Clone>(&self, coeffs: Vec<T>) -> Vec<T> {
		if self.is_zerocheck() && self.round == 0 {
			coeffs[2..].to_vec()
		} else if self.is_zerocheck() {
			coeffs[1..].to_vec()
		} else {
			coeffs[..coeffs.len() - 1].to_vec()
		}
	}

	#[instrument(skip_all, name = "sumcheck::SumcheckProverState::execute_round")]
	pub fn execute_round(
		&mut self,
		prev_rd_challenge: Option<F>,
	) -> Result<SumcheckRound<F>, Error> {
		// First round has no challenge, other rounds should have it
		self.validate_rd_challenge(prev_rd_challenge)?;

		if self.round >= self.n_vars() {
			return Err(Error::ImproperInput("too many execute_round calls".to_string()));
		}

		// Rounds 1..n_vars-1 - Some(..) challenge is given
		if let Some(prev_rd_challenge) = prev_rd_challenge {
			// If zerocheck, update round_eq_ind
			if self.is_zerocheck() {
				self.zerocheck_aux_state
					.as_mut()
					.unwrap()
					.update_round_eq_ind()?;
			}

			// Process switchovers of small field multilinears and folding of large field ones
			self.handle_switchover_and_fold(prev_rd_challenge.into())?;

			// Reduce Evalcheck claim
			self.reduce_claim(prev_rd_challenge)?;
		}

		// Extract multilinears & round
		let &mut Self {
			round,
			ref multilinears,
			..
		} = self;

		// Transform evaluation domain points to operating field
		let wf_domain = self
			.domain
			.points()
			.iter()
			.copied()
			.map(Into::into)
			.collect::<Vec<_>>();

		// Handling different cases separately for more inlining opportunities
		// (especially in early rounds)
		let any_transparent = multilinears
			.iter()
			.any(|ml| matches!(ml, SumcheckMultilinear::Transparent { .. }));
		let any_folded = multilinears
			.iter()
			.any(|ml| matches!(ml, SumcheckMultilinear::Folded { .. }));

		let evals = match (round, any_transparent, any_folded) {
			// All transparent, first round - direct sampling
			(0, true, false) => {
				self.sum_to_round_evals(&wf_domain, Self::only_transparent, Self::direct_sample)
			}

			// All transparent, rounds 1..n_vars - small field inner product
			(_, true, false) => {
				self.sum_to_round_evals(&wf_domain, Self::only_transparent, |multilin, i| {
					self.subcube_inner_product(multilin, i)
				})
			}

			// All folded - direct sampling
			(_, false, true) => {
				self.sum_to_round_evals(&wf_domain, Self::only_folded, Self::direct_sample)
			}

			// Heterogeneous case
			_ => self.sum_to_round_evals(
				&wf_domain,
				|x| x,
				|sc_multilin, i| match sc_multilin {
					SumcheckMultilinear::Transparent {
						small_field_multilin,
						..
					} => self.subcube_inner_product(small_field_multilin.borrow(), i),

					SumcheckMultilinear::Folded {
						large_field_folded_multilin,
					} => Self::direct_sample(large_field_folded_multilin, i),
				},
			),
		}?;

		let untrimmed_coeffs = self.evals_to_coeffs(evals, self.domain)?;
		let trimmed_coeffs = self.trim_coeffs(untrimmed_coeffs);
		let proof_round = SumcheckRound {
			coeffs: trimmed_coeffs,
		};
		self.last_round_proof = Some(proof_round.clone());

		self.round += 1;

		Ok(proof_round)
	}

	// Note the generic parameter - this method samples small field in first round and
	// large field post-switchover.
	#[inline]
	fn direct_sample<MD>(multilin: MD, i: usize) -> (PW::Scalar, PW::Scalar)
	where
		MD: MultilinearPoly<PW>,
	{
		let eval0 = multilin
			.evaluate_on_hypercube(i << 1)
			.expect("eval 0 within range");
		let eval1 = multilin
			.evaluate_on_hypercube((i << 1) + 1)
			.expect("eval 1 within range");

		(eval0, eval1)
	}

	#[inline]
	fn subcube_inner_product(&self, multilin: &M, i: usize) -> (PW::Scalar, PW::Scalar) where {
		let query = self.query.as_ref().expect("tensor present by invariant");

		let eval0 = multilin
			.evaluate_subcube(i << 1, query)
			.expect("eval 0 within range");
		let eval1 = multilin
			.evaluate_subcube((i << 1) + 1, query)
			.expect("eval 1 within range");

		(eval0, eval1)
	}

	fn only_transparent(sc_multilin: &SumcheckMultilinear<PW, M>) -> &M {
		match sc_multilin {
			SumcheckMultilinear::Transparent {
				small_field_multilin,
				..
			} => small_field_multilin.borrow(),
			_ => panic!("all transparent by invariant"),
		}
	}

	fn only_folded(
		sc_multilin: &SumcheckMultilinear<PW, M>,
	) -> &MultilinearExtensionSpecialized<'static, PW, PW> {
		match sc_multilin {
			SumcheckMultilinear::Folded {
				large_field_folded_multilin,
			} => large_field_folded_multilin,
			_ => panic!("all folded by invariant"),
		}
	}

	fn calculate_fold_result_sumcheck<'b, T>(
		&self,
		rd_vars: usize,
		n_multilinears: usize,
		n_round_evals: usize,
		eval_points: &[PW::Scalar],
		eval01: impl Fn(T, usize) -> (PW::Scalar, PW::Scalar) + Sync,
		precomps: Vec<T>,
	) -> Vec<F>
	where
		T: Copy + Sync + 'b,
		M: 'b,
	{
		let fold_result = (0..1 << (rd_vars - 1)).into_par_iter().fold(
			|| ParFoldState::new(n_multilinears, n_round_evals),
			|mut state, i| {
				for (j, precomp) in precomps.iter().enumerate() {
					let (eval0, eval1) = eval01(*precomp, i);
					state.evals_0[j] = eval0;
					state.evals_1[j] = eval1;
				}

				process_round_evals_sumcheck::<PW, CW>(
					&self.composition,
					&state.evals_0,
					&state.evals_1,
					&mut state.evals_z,
					&mut state.round_evals,
					eval_points,
				);

				state
			},
		);

		// Simply sum up the fold partitions.
		let wf_round_evals = fold_result.map(|state| state.round_evals).reduce(
			|| vec![PW::Scalar::ZERO; n_round_evals],
			|mut overall_round_evals, partial_round_evals| {
				overall_round_evals
					.iter_mut()
					.zip(partial_round_evals.iter())
					.for_each(|(f, s)| *f += s);
				overall_round_evals
			},
		);

		wf_round_evals.into_iter().map(Into::into).collect()
	}

	fn calculate_fold_result_zerocheck<'b, T>(
		&self,
		rd_vars: usize,
		n_multilinears: usize,
		n_round_evals: usize,
		eval_points: &[PW::Scalar],
		eval01: impl Fn(T, usize) -> (PW::Scalar, PW::Scalar) + Sync,
		precomps: Vec<T>,
	) -> Vec<F>
	where
		T: Copy + Sync + 'b,
		M: 'b,
	{
		let zerocheck_eq_ind = self
			.zerocheck_eq_ind()
			.expect("zerocheck eq ind available by invariant");

		let fold_result = (0..1 << (rd_vars - 1)).into_par_iter().fold(
			|| ParFoldState::new(n_multilinears, n_round_evals),
			|mut state, i| {
				for (j, precomp) in precomps.iter().enumerate() {
					let (eval0, eval1) = eval01(*precomp, i);
					state.evals_0[j] = eval0;
					state.evals_1[j] = eval1;
				}

				let zerocheck_eq_factor = zerocheck_eq_ind
					.evaluate_on_hypercube(i)
					.expect("zerocheck eq ind hypercube eval within range");

				process_round_evals_zerocheck::<PW, CW>(
					zerocheck_eq_factor,
					&self.composition,
					&state.evals_0,
					&state.evals_1,
					&mut state.evals_z,
					&mut state.round_evals,
					eval_points,
				);

				state
			},
		);

		// Simply sum up the fold partitions.
		let wf_round_evals = fold_result.map(|state| state.round_evals).reduce(
			|| vec![PW::Scalar::ZERO; n_round_evals],
			|mut overall_round_evals, partial_round_evals| {
				overall_round_evals
					.iter_mut()
					.zip(partial_round_evals.iter())
					.for_each(|(f, s)| *f += s);
				overall_round_evals
			},
		);

		wf_round_evals.into_iter().map(Into::into).collect()
	}

	// The gist of sumcheck - summing over evaluations of the multivariate composite on evaluation domain
	// for the remaining variables: there are `round-1` already assigned variables with values from large
	// field, and `rd_vars = n_vars - round` remaining variables that are being summed over. `eval01` closure
	// computes 0 & 1 evaluations at some index - either by performing inner product over assigned variables
	// pre-switchover or directly sampling MLE representation during first round or post-switchover.
	fn sum_to_round_evals<'b, T>(
		&'b self,
		eval_points: &[PW::Scalar],
		precomp: impl Fn(&'b SumcheckMultilinear<PW, M>) -> T,
		eval01: impl Fn(T, usize) -> (PW::Scalar, PW::Scalar) + Sync,
	) -> Result<Vec<F>, Error>
	where
		T: Copy + Sync + 'b,
		M: 'b,
	{
		let rd_vars = self.n_vars() - self.round;
		let n_multilinears = self.multilinears.len();
		let n_round_evals = {
			if self.round == 0 && self.is_zerocheck() {
				// In the very first round of a sumcheck that comes from zerocheck
				// we can uniquely determine the degree d univariate round polynomial r
				// with evaluations at X = 2, ..., d+1 because we know r(0) = r(1) = 0
				self.get_round_polynomial_degree() - 1
			} else {
				// Generally, we can uniquely derive the degree d univariate round polynomial r
				// from evaluations at X = 1, ..., d+1 because we have an identity that
				// relates r(0), r(1), and the current round's claimed sum
				self.get_round_polynomial_degree()
			}
		};
		debug_assert_eq!(eval_points.len(), self.get_round_polynomial_degree() + 1);

		// When possible to pre-process unpacking sumcheck multilinears, we do so.
		// For performance, it's ideal to hoist this out of the tight loop.
		let precomps = self.multilinears.iter().map(precomp).collect::<Vec<_>>();

		// Note: to avoid extra logic inside the tight loop, we have two separate implementations at the cost
		// of slight code duplication. This is a tradeoff for better performance.
		if self.is_zerocheck() {
			Ok(self.calculate_fold_result_zerocheck(
				rd_vars,
				n_multilinears,
				n_round_evals,
				eval_points,
				eval01,
				precomps,
			))
		} else {
			Ok(self.calculate_fold_result_sumcheck(
				rd_vars,
				n_multilinears,
				n_round_evals,
				eval_points,
				eval01,
				precomps,
			))
		}
	}

	fn validate_rd_challenge(&self, prev_rd_challenge: Option<F>) -> Result<(), Error> {
		if prev_rd_challenge.is_none() != (self.round == 0) {
			return Err(Error::ImproperInput(format!(
				"incorrect optional challenge: is_some()={:?} at round {}",
				prev_rd_challenge.is_some(),
				self.round
			)));
		}

		Ok(())
	}

	fn reduce_claim(&mut self, prev_rd_challenge: F) -> Result<(), Error> {
		let round_claim = self.round_claim.clone();
		let round_proof = self
			.last_round_proof
			.as_ref()
			.expect("round is at least 1 by invariant")
			.clone();

		let new_round_claim = if let Some(zc_challenges) = self.zerocheck_challenges() {
			let alpha = if self.round == 1 {
				None
			} else {
				Some(zc_challenges[self.round - 2])
			};
			reduce_zerocheck_claim_round(round_claim, prev_rd_challenge, round_proof, alpha)
		} else {
			reduce_sumcheck_claim_round(round_claim, prev_rd_challenge, round_proof)
		}?;

		self.round_claim = new_round_claim;

		Ok(())
	}

	fn handle_switchover_and_fold(&mut self, prev_rd_challenge: PW::Scalar) -> Result<(), Error> {
		let &mut Self {
			round,
			ref mut multilinears,
			ref mut query,
			..
		} = self;

		// Update query (has to be done before switchover)
		if let Some(prev_query) = query.take() {
			let expanded_query = prev_query.update(&[prev_rd_challenge])?;
			query.replace(expanded_query);
		}

		// Partial query (for folding)
		let partial_query = MultilinearQuery::with_full_query(&[prev_rd_challenge])?;

		// Perform switchover and/or folding
		let mut any_transparent_left = false;

		for multilin in multilinears.iter_mut() {
			match *multilin {
				SumcheckMultilinear::Transparent {
					switchover,
					ref small_field_multilin,
				} => {
					if switchover <= round {
						// At switchover, perform inner products in large field and save them
						// in a newly created MLE.
						let query_ref = query.as_ref().expect("tensor available by invariant");
						let large_field_folded_multilin = small_field_multilin
							.borrow()
							.evaluate_partial_low(query_ref)?;

						*multilin = SumcheckMultilinear::Folded {
							large_field_folded_multilin,
						};
					} else {
						any_transparent_left = true;
					}
				}

				SumcheckMultilinear::Folded {
					ref mut large_field_folded_multilin,
				} => {
					// Post-switchover, simply halve large field MLE.
					*large_field_folded_multilin =
						large_field_folded_multilin.evaluate_partial_low(&partial_query)?;
				}
			}
		}

		// All folded large field - tensor is no more needed.
		if !any_transparent_left {
			*query = None;
		}

		Ok(())
	}
}

// Sumcheck evaluation at a specific point - given an array of 0 & 1 evaluations at some index,
// uses them to linearly interpolate each MLE value at domain point, and then evaluate multivariate
// composite over those.
fn process_round_evals_sumcheck<P: PackedField, C: CompositionPoly<P> + ?Sized>(
	composition: &C,
	evals_0: &[P::Scalar],
	evals_1: &[P::Scalar],
	evals_z: &mut [P::Scalar],
	round_evals: &mut [P::Scalar],
	domain: &[P::Scalar],
) {
	let degree = domain.len() - 1;
	// NB: We skip evaluation of $r(X)$ at $X = 0$ as it is derivable from the current_round_sum - $r(1)$.
	assert!(domain.len() - round_evals.len() == 1);

	round_evals[0] += composition
		.evaluate(evals_1)
		.expect("evals_1 is initialized with a length of poly.composition.n_vars()");

	// The rest require interpolation.
	for d in 2..degree + 1 {
		evals_0
			.iter()
			.zip(evals_1.iter())
			.zip(evals_z.iter_mut())
			.for_each(|((&evals_0_j, &evals_1_j), evals_z_j)| {
				*evals_z_j = extrapolate_line(evals_0_j, evals_1_j, domain[d]);
			});

		round_evals[d - 1] += composition
			.evaluate(evals_z)
			.expect("evals_z is initialized with a length of poly.composition.n_vars()");
	}
}

// Zerocheck evaluation at a specific point - given an array of 0 & 1 evaluations at some index,
// uses them to linearly interpolate each MLE value at domain point, and then evaluate multivariate
// composite over those.
// This is a special version of the function used when the sumcheck claim came directly from a zerocheck reduction.
fn process_round_evals_zerocheck<P: PackedField, C: CompositionPoly<P> + ?Sized>(
	zerocheck_eq_ind_factor: P::Scalar,
	composition: &C,
	evals_0: &[P::Scalar],
	evals_1: &[P::Scalar],
	evals_z: &mut [P::Scalar],
	round_evals: &mut [P::Scalar],
	domain: &[P::Scalar],
) {
	let degree = domain.len() - 1;
	// NB: We should skip evaluating $r(X)$ at $X = 0$, and in some contexts, also skip at $X = 1$.
	// Generally, we can skip evaluation at 0;
	// * It can be dervied from current_round_sum, zerocheck_eq_ind_factor, and $r(1)$.
	// However, in the special case where this is the first round of zerocheck, we know
	// * $r(0) = r(1) = 0$.
	// We can therefore skip explicit evaluations at $X = 0$ and $X = 1$.
	let n_skipped_evaluations = domain.len() - round_evals.len();
	assert!((1..=2).contains(&n_skipped_evaluations));

	if n_skipped_evaluations == 1 {
		round_evals[0] += zerocheck_eq_ind_factor
			* composition
				.evaluate(evals_1)
				.expect("evals_1 is initialized with a length of poly.composition.n_vars()");
	}

	// The rest require interpolation.
	for d in 2..degree + 1 {
		evals_0
			.iter()
			.zip(evals_1.iter())
			.zip(evals_z.iter_mut())
			.for_each(|((&evals_0_j, &evals_1_j), evals_z_j)| {
				*evals_z_j = extrapolate_line(evals_0_j, evals_1_j, domain[d]);
			});

		round_evals[d - n_skipped_evaluations] += zerocheck_eq_ind_factor
			* composition
				.evaluate(evals_z)
				.expect("evals_z is initialized with a length of poly.composition.n_vars()");
	}
}

/// Validate that evaluation domain starts with 0 & 1 and the size is exactly one greater than the
/// maximum individual degree of the polynomial.
fn check_evaluation_domain<F: Field>(
	max_individual_degree: usize,
	domain: &EvaluationDomain<F>,
) -> Result<(), Error> {
	if max_individual_degree == 0
		|| domain.size() != max_individual_degree + 1
		|| domain.points()[0] != F::ZERO
		|| domain.points()[1] != F::ONE
	{
		return Err(Error::EvaluationDomainMismatch);
	}
	Ok(())
}
