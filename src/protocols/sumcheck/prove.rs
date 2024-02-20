// Copyright 2023 Ulvetanna Inc.

use std::{borrow::Borrow, cmp::max, convert::Into, fmt::Debug, marker::PhantomData, sync::Arc};

use rayon::prelude::*;

use tracing::instrument;

use super::{
	error::Error,
	sumcheck::{
		check_evaluation_domain, reduce_sumcheck_claim_final, reduce_sumcheck_claim_round,
		SumcheckClaim, SumcheckProof, SumcheckProveOutput, SumcheckRound, SumcheckRoundClaim,
		SumcheckWitness,
	},
};

use crate::{
	field::{Field, PackedField},
	oracle::MultivariatePolyOracle,
	polynomial::{
		extrapolate_line, multilinear_query::MultilinearQuery, CompositionPoly, EvaluationDomain,
		MultilinearExtension, MultilinearPoly,
	},
	protocols::evalcheck::EvalcheckWitness,
};

/// An individual multilinear polynomial in a multivariate composite
enum SumcheckMultilinear<P: PackedField, BM> {
	/// Small field polynomial - to be folded into large field at `switchover` round
	Transparent {
		switchover: usize,
		small_field_multilin: BM,
	},
	/// Large field polynomial - halved in size each round
	Folded {
		large_field_folded_multilin: MultilinearExtension<'static, P>,
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
/// some cases it makes sense to do so in an different yet isomorphic field OPF (operating packed
/// field) which may preferable due to superior performance. One example of such operating field
/// would be BinaryField128bPolyval, which tends to be much faster than 128-bit tower field on x86
/// CPUs. The only constraint is that constituent MLEs should have MultilinearPoly impls for OPF -
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
pub struct SumcheckProverState<F, OPF, M, BM>
where
	F: Field + From<OPF::Scalar> + Into<OPF::Scalar>,
	OPF: PackedField + Debug,
	M: MultilinearPoly<OPF> + Sync + ?Sized,
	BM: Borrow<M> + Sync,
{
	n_vars: usize,
	max_individual_degree: usize,

	composition: Arc<dyn CompositionPoly<OPF>>,
	multilinears: Vec<SumcheckMultilinear<OPF, BM>>,

	query: Option<MultilinearQuery<OPF::Scalar>>,

	proof: SumcheckProof<F>,
	round_claim: Option<SumcheckRoundClaim<F>>,

	round: usize,
	_m_marker: PhantomData<M>,
}

impl<F, OPF, M, BM> SumcheckProverState<F, OPF, M, BM>
where
	F: Field + From<OPF::Scalar> + Into<OPF::Scalar>,
	OPF: PackedField + Debug,
	M: MultilinearPoly<OPF> + Sync + ?Sized,
	BM: Borrow<M> + Sync,
{
	/// Start a new sumcheck instance with claim in field `F`. Witness may be given in
	/// a different (but isomorphic) packed field OPF. `switchovers` slice contains rounds
	/// given multilinear index - expectation is that the caller would have enough context to
	/// introspect on field & polynomial sizes. This is a stopgap measure until a proper type
	/// reflection mechanism is introduced.
	pub fn new(
		sumcheck_claim: &SumcheckClaim<F>,
		sumcheck_witness: SumcheckWitness<OPF, M, BM>,
		switchovers: &[usize],
	) -> Result<Self, Error> {
		let n_vars = sumcheck_witness.n_vars();
		let max_individual_degree = sumcheck_claim.poly.max_individual_degree();

		if max_individual_degree == 0 {
			return Err(Error::PolynomialDegreeIsZero);
		}

		if sumcheck_claim.poly.n_vars() != n_vars {
			let err_str = format!(
				"Claim and Witness n_vars mismatch in sumcheck. Claim: {}, Witness: {}",
				sumcheck_claim.poly.n_vars(),
				n_vars
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

		let proof = SumcheckProof { rounds: Vec::new() };

		let round_claim = Some(SumcheckRoundClaim {
			partial_point: Vec::new(),
			current_round_sum: sumcheck_claim.sum,
		});

		let prover_state = SumcheckProverState {
			n_vars,
			max_individual_degree,

			composition,
			multilinears,

			query,

			proof,
			round_claim,

			round: 0,
			_m_marker: PhantomData,
		};

		Ok(prover_state)
	}

	/// Generic parameters allow to pass a different witness type to the inner Evalcheck claim.
	#[instrument(skip_all, name = "sumcheck::SumcheckProverState::finalize")]
	pub fn finalize<WPF, WM, WBM>(
		&mut self,
		poly_oracle: &MultivariatePolyOracle<F>,
		sumcheck_witness: SumcheckWitness<WPF, WM, WBM>,
		domain: &EvaluationDomain<F>,
		prev_rd_challenge: Option<F>,
	) -> Result<SumcheckProveOutput<F, WPF, WM, WBM>, Error>
	where
		WPF: PackedField,
		WM: MultilinearPoly<WPF> + ?Sized,
		WBM: Borrow<WM>,
	{
		// First round has no challenge, other rounds should have it
		self.validate_rd_challenge(prev_rd_challenge)?;

		if self.round != self.n_vars {
			return Err(Error::ImproperInput(format!(
				"finalize() called on round {} while n_vars={}",
				self.round, self.n_vars
			)));
		}

		// Last reduction to obtain eval value at eval_point
		if let Some(prev_rd_challenge) = prev_rd_challenge {
			self.reduce_claim(domain, prev_rd_challenge)?;
		}

		let evalcheck_claim = reduce_sumcheck_claim_final(
			poly_oracle,
			self.round_claim
				.take()
				.expect("round_claim always present by invariant"),
		)?;

		self.round += 1;

		let sumcheck_proof = self.proof.clone();
		let evalcheck_witness = EvalcheckWitness::composite(sumcheck_witness.multilinears);

		Ok(SumcheckProveOutput {
			sumcheck_proof,
			evalcheck_claim,
			evalcheck_witness,
		})
	}

	#[instrument(skip_all, name = "sumcheck::SumcheckProverState::execute_round")]
	pub fn execute_round(
		&mut self,
		domain: &EvaluationDomain<F>,
		prev_rd_challenge: Option<F>,
	) -> Result<SumcheckRound<F>, Error> {
		// First round has no challenge, other rounds should have it
		self.validate_rd_challenge(prev_rd_challenge)?;

		if self.round >= self.n_vars {
			return Err(Error::ImproperInput("too many execute_round calls".to_string()));
		}

		// Validate that evaluation domain starts with 0 & 1 and is large enough to
		// interpolate a univariate of a maximum individual degree
		check_evaluation_domain(self.max_individual_degree, domain)?;

		// Rounds 1..n_vars-1 - Some(..) challenge is given
		if let Some(prev_rd_challenge) = prev_rd_challenge {
			// Process switchovers of small field multilinears and folding of large field ones
			self.handle_switchover_and_fold(prev_rd_challenge.into())?;

			// Reduce Evalcheck claim
			self.reduce_claim(domain, prev_rd_challenge)?;
		}

		// Extract multilinears & round
		let &mut Self {
			round,
			ref multilinears,
			..
		} = self;

		// Transform evaluation domain points to operating field
		let opf_domain = domain
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

		let coeffs = match (round, any_transparent, any_folded) {
			// All transparent, first round - direct sampling
			(0, true, false) => {
				self.sum_to_round_evals(&opf_domain, Self::only_transparent, Self::direct_sample)
			}

			// All transparent, rounds 1..n_vars - small field inner product
			(_, true, false) => {
				self.sum_to_round_evals(&opf_domain, Self::only_transparent, |multilin, i| {
					self.subcube_inner_product(multilin, i)
				})
			}

			// All folded - direct sampling
			(_, false, true) => {
				self.sum_to_round_evals(&opf_domain, Self::only_folded, Self::direct_sample)
			}

			// Heterogeneous case
			_ => self.sum_to_round_evals(
				&opf_domain,
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

		let proof_round = SumcheckRound { coeffs };
		self.proof.rounds.push(proof_round.clone());

		Ok(proof_round)
	}

	fn only_transparent(sc_multilin: &SumcheckMultilinear<OPF, BM>) -> &M {
		match sc_multilin {
			SumcheckMultilinear::Transparent {
				small_field_multilin,
				..
			} => small_field_multilin.borrow(),
			_ => panic!("all transparent by invariant"),
		}
	}

	fn only_folded(
		sc_multilin: &SumcheckMultilinear<OPF, BM>,
	) -> &MultilinearExtension<'static, OPF> {
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
	fn direct_sample<MD>(multilin: &MD, i: usize) -> (OPF::Scalar, OPF::Scalar)
	where
		MD: MultilinearPoly<OPF> + ?Sized,
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
	fn subcube_inner_product(&self, multilin: &M, i: usize) -> (OPF::Scalar, OPF::Scalar) where {
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
	fn sum_to_round_evals<'a, T>(
		&'a self,
		eval_points: &[OPF::Scalar],
		precomp: impl Fn(&'a SumcheckMultilinear<OPF, BM>) -> T,
		eval01: impl Fn(T, usize) -> (OPF::Scalar, OPF::Scalar) + Sync,
	) -> Vec<F>
	where
		T: Copy + Sync + 'a,
		BM: 'a,
	{
		let rd_vars = self.n_vars - self.round;

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

				process_round_evals(
					self.composition.as_ref(),
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
		let opf_round_evals = fold_result.map(|state| state.round_evals).reduce(
			|| vec![OPF::Scalar::ZERO; degree],
			|mut overall_round_evals, partial_round_evals| {
				overall_round_evals
					.iter_mut()
					.zip(partial_round_evals.iter())
					.for_each(|(f, s)| *f += s);
				overall_round_evals
			},
		);

		opf_round_evals.into_iter().map(Into::into).collect()
	}

	fn validate_rd_challenge(&self, prev_rd_challenge: Option<F>) -> Result<(), Error> {
		if prev_rd_challenge.map_or(self.round > 0, |_| self.round == 0) {
			return Err(Error::ImproperInput(format!(
				"incorrect optional challenge: is_some()={:?} at round {}",
				prev_rd_challenge.is_some(),
				self.round
			)));
		}

		Ok(())
	}

	pub fn get_claim(&self) -> SumcheckRoundClaim<F> {
		self.round_claim
			.clone()
			.expect("round_claim always present by invariant")
	}

	fn reduce_claim(
		&mut self,
		domain: &EvaluationDomain<F>,
		prev_rd_challenge: F,
	) -> Result<(), Error> {
		let new_round_claim = reduce_sumcheck_claim_round(
			self.max_individual_degree,
			domain,
			self.proof
				.rounds
				.last()
				.expect("not first round by invariant")
				.clone(),
			self.round_claim
				.take()
				.expect("round_claim always present by invariant"),
			prev_rd_challenge,
		)?;

		self.round_claim.replace(new_round_claim);

		Ok(())
	}

	fn handle_switchover_and_fold(&mut self, prev_rd_challenge: OPF::Scalar) -> Result<(), Error> {
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

#[cfg(test)]
mod tests {
	use super::*;
	use crate::{
		challenger::HashChallenger,
		field::{BinaryField128b, BinaryField128bPolyval, BinaryField32b, TowerField},
		hash::GroestlHasher,
		oracle::{CommittedId, CompositePolyOracle, MultilinearPolyOracle, MultivariatePolyOracle},
		polynomial::{CompositionPoly, MultilinearComposite, MultilinearExtension},
		protocols::{
			sumcheck::SumcheckClaim,
			test_utils::{
				full_prove_with_operating_field, full_prove_with_switchover, full_verify,
				transform_poly, TestProductComposition,
			},
		},
	};
	use p3_util::log2_ceil_usize;
	use rand::{rngs::StdRng, SeedableRng};
	use rayon::current_num_threads;
	use std::{iter::repeat_with, sync::Arc};

	fn test_prove_verify_interaction_helper(
		n_vars: usize,
		n_multilinears: usize,
		switchover_rd: usize,
	) {
		type F = BinaryField32b;
		type FE = BinaryField128b;

		let mut rng = StdRng::seed_from_u64(0);

		// Setup Witness
		let composition: Arc<dyn CompositionPoly<FE>> =
			Arc::new(TestProductComposition::new(n_multilinears));
		let multilinears = repeat_with(|| {
			let values = repeat_with(|| Field::random(&mut rng))
				.take(1 << n_vars)
				.collect::<Vec<F>>();
			MultilinearExtension::from_values(values).unwrap()
		})
		.take(composition.n_vars())
		.collect::<Vec<_>>();
		let poly = MultilinearComposite::new(n_vars, composition, multilinears.clone()).unwrap();

		// Get the sum
		let sum = (0..1 << n_vars)
			.map(|i| {
				let mut prod = F::ONE;
				(0..n_multilinears).for_each(|j| {
					prod *= multilinears[j].packed_evaluate_on_hypercube(i).unwrap();
				});
				prod
			})
			.sum::<F>();

		let sumcheck_witness = poly.clone();

		// Setup Claim
		let h = (0..n_multilinears)
			.map(|i| MultilinearPolyOracle::Committed {
				id: CommittedId {
					batch_id: 0,
					index: i,
				},
				n_vars,
				tower_level: F::TOWER_LEVEL,
			})
			.collect();
		let composite_poly = CompositePolyOracle::new(
			n_vars,
			h,
			Arc::new(TestProductComposition::new(n_multilinears)),
		)
		.unwrap();
		let poly_oracle = MultivariatePolyOracle::Composite(composite_poly);
		let sumcheck_claim = SumcheckClaim {
			sum: sum.into(),
			poly: poly_oracle,
		};

		// Setup evaluation domain
		let domain = EvaluationDomain::new(n_multilinears + 1).unwrap();

		let challenger = <HashChallenger<_, GroestlHasher<_>>>::new();

		let (prover_rd_claims, final_prove_output) = full_prove_with_switchover(
			&sumcheck_claim,
			sumcheck_witness,
			&domain,
			challenger.clone(),
			switchover_rd,
		);

		let (verifier_rd_claims, final_verify_output) = full_verify(
			&sumcheck_claim,
			final_prove_output.sumcheck_proof,
			&domain,
			challenger.clone(),
		);

		assert_eq!(prover_rd_claims, verifier_rd_claims);
		assert_eq!(final_prove_output.evalcheck_claim.eval, final_verify_output.eval);
		assert_eq!(final_prove_output.evalcheck_claim.eval_point, final_verify_output.eval_point);
		assert_eq!(final_prove_output.evalcheck_claim.poly.n_vars(), n_vars);
		assert!(final_prove_output.evalcheck_claim.is_random_point);
		assert_eq!(final_verify_output.poly.n_vars(), n_vars);

		// Verify that the evalcheck claim is correct
		let eval_point = &final_verify_output.eval_point;
		let multilin_query = MultilinearQuery::with_full_query(eval_point).unwrap();
		let actual = poly.evaluate(&multilin_query).unwrap();
		assert_eq!(actual, final_verify_output.eval);
	}

	fn test_prove_verify_interaction_with_monomial_basis_conversion_helper(
		n_vars: usize,
		n_multilinears: usize,
	) {
		type F = BinaryField128b;
		type OF = BinaryField128bPolyval;

		let mut rng = StdRng::seed_from_u64(0);

		let composition = Arc::new(TestProductComposition::new(n_multilinears));
		let prover_composition = composition.clone();
		let composition_nvars = n_multilinears;

		let multilinears = repeat_with(|| {
			let values = repeat_with(|| Field::random(&mut rng))
				.take(1 << n_vars)
				.collect::<Vec<F>>();
			MultilinearExtension::from_values(values).unwrap()
		})
		.take(composition_nvars)
		.collect::<Vec<_>>();
		let poly = MultilinearComposite::new(n_vars, composition, multilinears.clone()).unwrap();
		let prover_poly = transform_poly::<_, OF>(&poly, prover_composition).unwrap();

		let sum = (0..1 << n_vars)
			.map(|i| {
				let mut prod = F::ONE;
				(0..n_multilinears).for_each(|j| {
					prod *= multilinears[j].packed_evaluate_on_hypercube(i).unwrap();
				});
				prod
			})
			.sum();

		let operating_witness = prover_poly;
		let witness = poly.clone();

		// CLAIM
		let h = (0..n_multilinears)
			.map(|i| MultilinearPolyOracle::Committed {
				id: CommittedId {
					batch_id: 0,
					index: i,
				},
				n_vars,
				tower_level: F::TOWER_LEVEL,
			})
			.collect();
		let composite_poly = CompositePolyOracle::new(
			n_vars,
			h,
			Arc::new(TestProductComposition::new(n_multilinears)),
		)
		.unwrap();
		let poly_oracle = MultivariatePolyOracle::Composite(composite_poly);
		let sumcheck_claim = SumcheckClaim {
			sum,
			poly: poly_oracle,
		};

		// Setup evaluation domain
		let domain = EvaluationDomain::new(n_multilinears + 1).unwrap();

		let challenger = <HashChallenger<_, GroestlHasher<_>>>::new();
		let (prover_rd_claims, final_prove_output) = full_prove_with_operating_field(
			&sumcheck_claim,
			witness,
			operating_witness,
			&domain,
			challenger.clone(),
		);

		let (verifier_rd_claims, final_verify_output) = full_verify(
			&sumcheck_claim,
			final_prove_output.sumcheck_proof,
			&domain,
			challenger.clone(),
		);

		assert_eq!(prover_rd_claims, verifier_rd_claims);
		assert_eq!(final_prove_output.evalcheck_claim.eval, final_verify_output.eval);
		assert_eq!(final_prove_output.evalcheck_claim.eval_point, final_verify_output.eval_point);
		assert_eq!(final_prove_output.evalcheck_claim.poly.n_vars(), n_vars);
		assert!(final_prove_output.evalcheck_claim.is_random_point);
		assert_eq!(final_verify_output.poly.n_vars(), n_vars);

		// Verify that the evalcheck claim is correct
		let eval_point = &final_verify_output.eval_point;
		let multilin_query = MultilinearQuery::with_full_query(eval_point).unwrap();
		let actual = poly.evaluate(&multilin_query).unwrap();
		assert_eq!(actual, final_verify_output.eval);
	}

	#[test]
	fn test_prove_verify_interaction_basic() {
		crate::util::init_tracing();

		for n_vars in 2..8 {
			for n_multilinears in 1..4 {
				for switchover_rd in 1..=n_vars / 2 {
					test_prove_verify_interaction_helper(n_vars, n_multilinears, switchover_rd);
				}
			}
		}
	}

	#[test]
	fn test_prove_verify_interaction_pigeonhole_cores() {
		let n_threads = current_num_threads();
		let n_vars = log2_ceil_usize(n_threads) + 1;
		for n_multilinears in 1..4 {
			for switchover_rd in 1..=n_vars / 2 {
				test_prove_verify_interaction_helper(n_vars, n_multilinears, switchover_rd);
			}
		}
	}

	#[test]
	fn test_prove_verify_interaction_with_monomial_basis_conversion_basic() {
		for n_vars in 2..8 {
			for n_multilinears in 1..4 {
				test_prove_verify_interaction_with_monomial_basis_conversion_helper(
					n_vars,
					n_multilinears,
				);
			}
		}
	}

	#[test]
	fn test_prove_verify_interaction_with_monomial_basis_conversion_pigeonhole_cores() {
		let n_threads = current_num_threads();
		let n_vars = log2_ceil_usize(n_threads) + 1;
		for n_multilinears in 1..6 {
			test_prove_verify_interaction_with_monomial_basis_conversion_helper(
				n_vars,
				n_multilinears,
			);
		}
	}
}
