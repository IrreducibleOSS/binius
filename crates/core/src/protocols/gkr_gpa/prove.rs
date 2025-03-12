// Copyright 2024-2025 Irreducible Inc.

use binius_field::{Field, PackedExtension, PackedField, TowerField};
use binius_hal::ComputationBackend;
use binius_math::{
	extrapolate_line_scalar, EvaluationDomainFactory, EvaluationOrder, MLEDirectAdapter,
	MultilinearExtension, MultilinearPoly,
};
use binius_utils::{
	bail,
	sorting::{stable_sort, unsort},
};
use tracing::instrument;

use super::{
	gkr_gpa::{GrandProductBatchProveOutput, LayerClaim},
	packed_field_storage::PackedFieldStorage,
	Error, GrandProductClaim, GrandProductWitness,
};
use crate::{
	composition::{BivariateProduct, IndexComposition},
	fiat_shamir::{CanSample, Challenger},
	protocols::sumcheck::{
		self, immediate_switchover_heuristic, prove::eq_ind::EqIndSumcheckProver, CompositeSumClaim,
	},
	transcript::ProverTranscript,
};

/// Proves batch reduction turning each GrandProductClaim into an EvalcheckMultilinearClaim
///
/// REQUIRES:
/// * witnesses and claims are of the same length
/// * The ith witness corresponds to the ith claim
#[instrument(skip_all, name = "gkr_gpa::batch_prove", level = "debug")]
pub fn batch_prove<F, P, FDomain, Challenger_, Backend>(
	evaluation_order: EvaluationOrder,
	witnesses: impl IntoIterator<Item = GrandProductWitness<P>>,
	claims: &[GrandProductClaim<F>],
	evaluation_domain_factory: impl EvaluationDomainFactory<FDomain>,
	transcript: &mut ProverTranscript<Challenger_>,
	backend: &Backend,
) -> Result<GrandProductBatchProveOutput<F>, Error>
where
	F: TowerField,
	P: PackedField<Scalar = F> + PackedExtension<FDomain>,
	FDomain: Field,
	Challenger_: Challenger,
	Backend: ComputationBackend,
{
	//  Ensure witnesses and claims are of the same length, zip them together
	// 	For each witness-claim pair, create GrandProductProver
	let witness_vec = witnesses.into_iter().collect::<Vec<_>>();

	let n_claims = claims.len();
	if n_claims == 0 {
		return Ok(GrandProductBatchProveOutput::default());
	}
	if witness_vec.len() != n_claims {
		bail!(Error::MismatchedWitnessClaimLength);
	}

	// Create a vector of GrandProductProverStates
	let provers_vec = witness_vec
		.iter()
		.zip(claims)
		.map(|(witness, claim)| GrandProductProverState::new(claim, witness, backend))
		.collect::<Result<Vec<_>, _>>()?;

	let (original_indices, mut sorted_provers) =
		stable_sort(provers_vec, |prover| prover.input_vars(), true);

	let max_n_vars = sorted_provers
		.first()
		.expect("sorted_provers is not empty by invariant")
		.input_vars();

	let mut reverse_sorted_final_layer_claims = Vec::with_capacity(n_claims);

	for layer_no in 0..max_n_vars {
		// Step 1: Process finished provers
		process_finished_provers(
			layer_no,
			&mut sorted_provers,
			&mut reverse_sorted_final_layer_claims,
		)?;

		// Now we must create the batch layer proof for the kth to k+1th layer reduction

		// Step 2: Create sumcheck batch proof
		let batch_sumcheck_output = {
			let gpa_sumcheck_prover = GrandProductProverState::stage_gpa_sumcheck_provers(
				evaluation_order,
				&sorted_provers,
				evaluation_domain_factory.clone(),
			)?;

			sumcheck::batch_prove(vec![gpa_sumcheck_prover], transcript)?
		};

		// Step 3: Sample a challenge for the next layer
		let gpa_challenge = transcript.sample();

		// Step 4: Finalize each prover to update its internal current_layer_claim
		for (i, prover) in sorted_provers.iter_mut().enumerate() {
			prover.finalize_batch_layer_proof(
				batch_sumcheck_output.multilinear_evals[0][2 * i],
				batch_sumcheck_output.multilinear_evals[0][2 * i + 1],
				batch_sumcheck_output.challenges.clone(),
				gpa_challenge,
			)?;
		}
	}
	process_finished_provers(
		max_n_vars,
		&mut sorted_provers,
		&mut reverse_sorted_final_layer_claims,
	)?;

	debug_assert!(sorted_provers.is_empty());
	debug_assert_eq!(reverse_sorted_final_layer_claims.len(), n_claims);

	reverse_sorted_final_layer_claims.reverse();
	let sorted_final_layer_claim = reverse_sorted_final_layer_claims;

	let final_layer_claims = unsort(original_indices, sorted_final_layer_claim);

	Ok(GrandProductBatchProveOutput { final_layer_claims })
}

fn process_finished_provers<F, P, Backend>(
	layer_no: usize,
	sorted_provers: &mut Vec<GrandProductProverState<'_, F, P, Backend>>,
	reverse_sorted_final_layer_claims: &mut Vec<LayerClaim<F>>,
) -> Result<(), Error>
where
	F: TowerField,
	P: PackedField<Scalar = F>,
	Backend: ComputationBackend,
{
	while let Some(prover) = sorted_provers.last() {
		if prover.input_vars() != layer_no {
			break;
		}
		debug_assert!(layer_no > 0);
		let finished_prover = sorted_provers.pop().expect("not empty");
		let final_layer_claim = finished_prover.finalize()?;
		reverse_sorted_final_layer_claims.push(final_layer_claim);
	}

	Ok(())
}

/// GPA protocol prover state
///
/// Coordinates the proving of a grand product claim before and after
/// the sumcheck-based layer reductions.
#[derive(Debug)]
struct GrandProductProverState<'a, F, P, Backend>
where
	F: Field + From<P::Scalar>,
	P: PackedField,
	P::Scalar: Field + From<F>,
	Backend: ComputationBackend,
{
	n_vars: usize,
	// Layers of the product circuit as multilinear polynomials
	// The ith element is the ith layer of the product circuit
	layers: Vec<MLEDirectAdapter<P, PackedFieldStorage<'a, P>>>,
	// The ith element consists of a tuple of the
	// first and second halves of the (i+1)th layer of the product circuit
	next_layer_halves: Vec<[MLEDirectAdapter<P, PackedFieldStorage<'a, P>>; 2]>,
	// The current claim about a layer multilinear of the product circuit
	current_layer_claim: LayerClaim<F>,

	backend: Backend,
}

impl<'a, F, P, Backend> GrandProductProverState<'a, F, P, Backend>
where
	F: TowerField + From<P::Scalar>,
	P: PackedField<Scalar = F>,
	Backend: ComputationBackend,
{
	/// Create a new GrandProductProverState
	fn new(
		claim: &GrandProductClaim<F>,
		witness: &'a GrandProductWitness<P>,
		backend: Backend,
	) -> Result<Self, Error> {
		let n_vars = claim.n_vars;
		if n_vars != witness.n_vars() || witness.grand_product_evaluation() != claim.product {
			bail!(Error::ProverClaimWitnessMismatch);
		}

		// Build multilinear polynomials from circuit evaluations
		let n_layers = n_vars + 1;
		let next_layer_halves = (1..n_layers)
			.map(|i| {
				let (left_evals, right_evals) = witness.ith_layer_eval_halves(i)?;
				let left = MultilinearExtension::try_from(left_evals)?;
				let right = MultilinearExtension::try_from(right_evals)?;
				Ok([left, right].map(MLEDirectAdapter::from))
			})
			.collect::<Result<Vec<_>, Error>>()?;

		let layers = (0..n_layers)
			.map(|i| {
				let ith_layer_evals = witness.ith_layer_evals(i)?;
				let ith_layer_evals = if P::LOG_WIDTH < i {
					PackedFieldStorage::from(ith_layer_evals)
				} else {
					debug_assert_eq!(ith_layer_evals.len(), 1);
					PackedFieldStorage::new_inline(ith_layer_evals[0].iter().take(1 << i))
						.expect("length is a power of 2")
				};

				let mle = MultilinearExtension::try_from(ith_layer_evals)?;
				Ok(mle.into())
			})
			.collect::<Result<Vec<_>, Error>>()?;

		debug_assert_eq!(next_layer_halves.len(), n_vars);
		debug_assert_eq!(layers.len(), n_vars + 1);

		// Initialize Layer Claim
		let layer_claim = LayerClaim {
			eval_point: vec![],
			eval: claim.product,
		};

		// Return new GrandProductProver and the common product
		Ok(Self {
			n_vars,
			next_layer_halves,
			layers,
			current_layer_claim: layer_claim,
			backend,
		})
	}

	const fn input_vars(&self) -> usize {
		self.n_vars
	}

	fn current_layer_no(&self) -> usize {
		self.current_layer_claim.eval_point.len()
	}

	#[allow(clippy::type_complexity)]
	#[instrument(skip_all, level = "debug")]
	fn stage_gpa_sumcheck_provers<FDomain>(
		evaluation_order: EvaluationOrder,
		provers: &[Self],
		evaluation_domain_factory: impl EvaluationDomainFactory<FDomain>,
	) -> Result<
		EqIndSumcheckProver<
			FDomain,
			P,
			IndexComposition<BivariateProduct, 2>,
			impl MultilinearPoly<P> + Send + Sync + 'a,
			Backend,
		>,
		Error,
	>
	where
		FDomain: Field,
		P: PackedExtension<FDomain>,
	{
		// test same layer
		let Some(first_prover) = provers.first() else {
			unreachable!();
		};

		// construct witness
		let n_claims = provers.len();
		let n_multilinears = provers.len() * 2;
		let current_layer_no = first_prover.current_layer_no();

		let mut composite_claims = Vec::with_capacity(n_claims);
		let mut multilinears = Vec::with_capacity(n_multilinears);

		for (i, prover) in provers.iter().enumerate() {
			let indices = [2 * i, 2 * i + 1];

			let composite_claim = CompositeSumClaim {
				sum: prover.current_layer_claim.eval,
				composition: IndexComposition::new(n_multilinears, indices, BivariateProduct {})?,
			};

			composite_claims.push(composite_claim);
			multilinears.extend(prover.next_layer_halves[current_layer_no].clone());
		}

		let first_layer_mle_advice = provers
			.iter()
			.map(|prover| prover.layers[current_layer_no].clone())
			.collect::<Vec<_>>();

		Ok(EqIndSumcheckProver::new(
			evaluation_order,
			multilinears,
			// Some(first_layer_mle_advice),
			&first_prover.current_layer_claim.eval_point,
			composite_claims,
			evaluation_domain_factory,
			// We use GPA protocol only for big fields, which is why switchover is trivial
			immediate_switchover_heuristic,
			&first_prover.backend,
		)?)
	}

	fn finalize_batch_layer_proof(
		&mut self,
		zero_eval: F,
		one_eval: F,
		sumcheck_challenge: Vec<F>,
		gpa_challenge: F,
	) -> Result<(), Error> {
		if self.current_layer_no() >= self.input_vars() {
			bail!(Error::TooManyRounds);
		}
		let new_eval = extrapolate_line_scalar::<F, F>(zero_eval, one_eval, gpa_challenge);
		let mut layer_challenge = sumcheck_challenge;
		layer_challenge.push(gpa_challenge);

		self.current_layer_claim = LayerClaim {
			eval_point: layer_challenge,
			eval: new_eval,
		};

		Ok(())
	}

	fn finalize(self) -> Result<LayerClaim<F>, Error> {
		if self.current_layer_no() != self.input_vars() {
			bail!(Error::PrematureFinalize);
		}

		let final_layer_claim = LayerClaim {
			eval_point: self.current_layer_claim.eval_point,
			eval: self.current_layer_claim.eval,
		};
		Ok(final_layer_claim)
	}
}
