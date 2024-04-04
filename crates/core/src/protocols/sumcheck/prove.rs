// Copyright 2023 Ulvetanna Inc.

use super::{
	error::Error,
	sumcheck::{
		reduce_sumcheck_claim_final, reduce_sumcheck_claim_round, SumcheckClaim, SumcheckRound,
		SumcheckRoundClaim, SumcheckWitness,
	},
};
use crate::{
	oracle::CompositePolyOracle,
	polynomial::{
		extrapolate_line, multilinear_query::MultilinearQuery, CompositionPoly, EvaluationDomain,
		MultilinearExtensionSpecialized, MultilinearPoly,
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
	fn new(n_multilinears: usize, degree: usize) -> Self {
		Self {
			evals_0: vec![F::ZERO; n_multilinears],
			evals_1: vec![F::ZERO; n_multilinears],
			evals_z: vec![F::ZERO; n_multilinears],
			round_evals: vec![F::ZERO; degree],
		}
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
pub struct SumcheckProverState<'a, F, PW, C, CW, M>
where
	F: Field + From<PW::Scalar>,
	PW: PackedField,
	PW::Scalar: From<F>,
	CW: CompositionPoly<PW>,
	M: MultilinearPoly<PW> + Sync,
{
	oracle: CompositePolyOracle<F, C>,
	composition: CW,
	multilinears: Vec<SumcheckMultilinear<PW, M>>,

	domain: &'a EvaluationDomain<F>,
	query: Option<MultilinearQuery<PW>>,

	#[getset(get = "pub")]
	round_claim: SumcheckRoundClaim<F>,

	round: usize,
	last_round_proof: Option<SumcheckRound<F>>,
	_m_marker: PhantomData<M>,
}

impl<'a, F, PW, C, CW, M> SumcheckProverState<'a, F, PW, C, CW, M>
where
	F: Field + From<PW::Scalar>,
	PW: PackedField,
	PW::Scalar: From<F>,
	C: CompositionPoly<F>,
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
		sumcheck_claim: SumcheckClaim<F, C>,
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

		let prover_state = SumcheckProverState {
			oracle: sumcheck_claim.poly,
			composition,
			multilinears,
			domain,
			query,
			round_claim,
			round: 0,
			last_round_proof: None,
			_m_marker: PhantomData,
		};

		Ok(prover_state)
	}

	pub fn n_vars(&self) -> usize {
		self.oracle.n_vars()
	}

	/// Generic parameters allow to pass a different witness type to the inner Evalcheck claim.
	#[instrument(skip_all, name = "sumcheck::SumcheckProverState::finalize")]
	pub fn finalize(mut self, prev_rd_challenge: Option<F>) -> Result<EvalcheckClaim<F, C>, Error> {
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

		let mut evals = match (round, any_transparent, any_folded) {
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
		};

		self.round += 1;

		evals.insert(0, self.round_claim.current_round_sum - evals[0]);
		let mut coeffs = self.domain.interpolate(&evals)?;
		// Omit the last coefficient because the verifier can compute it themself
		coeffs.truncate(coeffs.len() - 1);

		let proof_round = SumcheckRound { coeffs };
		self.last_round_proof = Some(proof_round.clone());
		Ok(proof_round)
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
	) -> Vec<F>
	where
		T: Copy + Sync + 'b,
		M: 'b,
	{
		let rd_vars = self.n_vars() - self.round;

		let n_multilinears = self.multilinears.len();
		let degree = self.composition.degree();

		debug_assert_eq!(eval_points.len(), degree + 1);

		// When there is some unpacking to do on each multilinear, hoist it out of the tight loop.
		let precomps = self.multilinears.iter().map(precomp).collect::<Vec<_>>();

		let fold_result = (0..1 << (rd_vars - 1)).into_par_iter().fold(
			|| ParFoldState::new(n_multilinears, degree),
			|mut state, i| {
				for (j, precomp) in precomps.iter().enumerate() {
					let (eval0, eval1) = eval01(*precomp, i);
					state.evals_0[j] = eval0;
					state.evals_1[j] = eval1;
				}

				process_round_evals::<PW, CW>(
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
			|| vec![PW::Scalar::ZERO; degree],
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
		self.round_claim = reduce_sumcheck_claim_round(
			self.round_claim.clone(),
			prev_rd_challenge,
			self.last_round_proof
				.as_ref()
				.expect("not first round by invariant")
				.clone(),
		)?;
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
// NB: Evaluation at 0 is not computed, due to it being derivable from the claim (sum = eval_0 + eval_1).
fn process_round_evals<P: PackedField, C: CompositionPoly<P> + ?Sized>(
	composition: &C,
	evals_0: &[P::Scalar],
	evals_1: &[P::Scalar],
	evals_z: &mut [P::Scalar],
	round_evals: &mut [P::Scalar],
	domain: &[P::Scalar],
) {
	let degree = domain.len() - 1;

	// Having evaluation at 1, can directly compute multivariate there.
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
