// Copyright 2024-2025 Irreducible Inc.

use binius_field::{Field, PackedExtension, PackedField, TowerField};
use binius_hal::ComputationBackend;
use binius_math::{EvaluationDomainFactory, EvaluationOrder, extrapolate_line_scalar};
use binius_utils::{
	bail,
	sorting::{stable_sort, unsort},
};
use itertools::izip;
use tracing::instrument;

use super::{
	Error, GrandProductClaim, GrandProductWitness,
	gkr_gpa::{GrandProductBatchProveOutput, LayerClaim},
};
use crate::{
	composition::{BivariateProduct, IndexComposition},
	fiat_shamir::{CanSample, Challenger},
	protocols::sumcheck::{
		BatchSumcheckOutput, CompositeSumClaim,
		prove::{SumcheckProver, eq_ind::EqIndSumcheckProverBuilder, front_loaded},
	},
	transcript::ProverTranscript,
};

/// Proves batch reduction turning each GrandProductClaim into LayerClaim on original multilinear.
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
	// 	For each witness-claim pair, create GrandProductProverState
	let witnesses = witnesses.into_iter().collect::<Vec<_>>();

	if witnesses.len() != claims.len() {
		bail!(Error::MismatchedWitnessClaimLength);
	}

	// Create a vector of GrandProductProverStates
	let prover_states = izip!(witnesses, claims)
		.map(|(witness, claim)| GrandProductProverState::new(claim, witness))
		.collect::<Result<Vec<_>, _>>()?;

	let (original_indices, mut sorted_prover_states) =
		stable_sort(prover_states, |state| state.remaining_layers.len(), true);

	let mut reverse_sorted_final_layer_claims = Vec::with_capacity(claims.len());
	let mut eval_point = Vec::new();

	loop {
		// Step 1: Process finished provers
		process_finished_provers(
			&mut sorted_prover_states,
			&mut reverse_sorted_final_layer_claims,
			&eval_point,
		)?;

		if sorted_prover_states.is_empty() {
			break;
		}

		// Now we must create the batch layer proof for the kth to k+1th layer reduction

		// Step 2: Create sumcheck batch proof
		let BatchSumcheckOutput {
			challenges,
			multilinear_evals,
		} = {
			let eq_ind_sumcheck_prover = GrandProductProverState::stage_sumcheck_provers(
				evaluation_order,
				&mut sorted_prover_states,
				evaluation_domain_factory.clone(),
				&eval_point,
				backend,
			)?;

			let batch_sumcheck_prover =
				front_loaded::BatchProver::new(vec![eq_ind_sumcheck_prover], transcript)?;

			let mut batch_output = batch_sumcheck_prover.run(transcript)?;

			if evaluation_order == EvaluationOrder::HighToLow {
				batch_output.challenges.reverse();
			}

			batch_output
		};

		// Step 3: Sample a challenge for the next layer
		let gpa_challenge = transcript.sample();

		eval_point.copy_from_slice(&challenges);
		eval_point.push(gpa_challenge);

		// Step 4: Finalize each prover to update its internal current_layer_claim
		debug_assert_eq!(multilinear_evals.len(), 1);
		let multilinear_evals = multilinear_evals
			.first()
			.expect("exactly one prover in a batch");
		for (state, evals) in izip!(&mut sorted_prover_states, multilinear_evals.chunks_exact(2)) {
			state.update_layer_eval(evals[0], evals[1], gpa_challenge);
		}
	}
	process_finished_provers(
		&mut sorted_prover_states,
		&mut reverse_sorted_final_layer_claims,
		&eval_point,
	)?;

	debug_assert!(sorted_prover_states.is_empty());
	debug_assert_eq!(reverse_sorted_final_layer_claims.len(), claims.len());

	reverse_sorted_final_layer_claims.reverse();
	let sorted_final_layer_claims = reverse_sorted_final_layer_claims;

	let final_layer_claims = unsort(original_indices, sorted_final_layer_claims);
	Ok(GrandProductBatchProveOutput { final_layer_claims })
}

fn process_finished_provers<F, P>(
	sorted_prover_states: &mut Vec<GrandProductProverState<P>>,
	reverse_sorted_final_layer_claims: &mut Vec<LayerClaim<F>>,
	eval_point: &[F],
) -> Result<(), Error>
where
	F: TowerField,
	P: PackedField<Scalar = F>,
{
	let first_finished =
		sorted_prover_states.partition_point(|state| !state.remaining_layers.is_empty());

	for state in sorted_prover_states.drain(first_finished..).rev() {
		reverse_sorted_final_layer_claims.push(state.finalize(eval_point)?);
	}

	Ok(())
}

/// GPA protocol state for a single witness
///
/// Coordinates the proving of a grand product claim before and after
/// the sumcheck-based layer reductions.
#[derive(Debug)]
struct GrandProductProverState<P>
where
	P: PackedField<Scalar: TowerField>,
{
	// Remaining layers of the product circuit, ordered from largest to smallest.
	// Each step removes the last layer.
	remaining_layers: Vec<Vec<P>>,
	// The current eval claim (on a shared eval point).
	layer_eval: P::Scalar,
}

impl<F, P> GrandProductProverState<P>
where
	F: TowerField,
	P: PackedField<Scalar = F>,
{
	/// Create a new GrandProductProverState
	fn new(claim: &GrandProductClaim<F>, witness: GrandProductWitness<P>) -> Result<Self, Error> {
		if claim.n_vars != witness.n_vars() || witness.grand_product_evaluation() != claim.product {
			bail!(Error::ProverClaimWitnessMismatch);
		}

		let mut remaining_layers = witness.into_circuit_layers();
		debug_assert_eq!(remaining_layers.len(), claim.n_vars + 1);
		let _ = remaining_layers
			.pop()
			.expect("remaining_layers cannot be empty");

		// Initialize Layer Claim
		let layer_eval = claim.product;

		// Return new GrandProductProver and the common product
		Ok(Self {
			remaining_layers,
			layer_eval,
		})
	}

	#[allow(clippy::type_complexity)]
	#[instrument(skip_all, level = "debug")]
	fn stage_sumcheck_provers<'a, FDomain, Backend>(
		evaluation_order: EvaluationOrder,
		states: &mut [Self],
		evaluation_domain_factory: impl EvaluationDomainFactory<FDomain>,
		eq_ind_challenges: &[P::Scalar],
		backend: &'a Backend,
	) -> Result<impl SumcheckProver<P::Scalar> + 'a, Error>
	where
		FDomain: Field,
		P: PackedExtension<FDomain>,
		Backend: ComputationBackend,
	{
		let n_vars = eq_ind_challenges.len();
		let n_claims = states.len();
		let n_multilinears = n_claims * 2;

		let mut composite_claims = Vec::with_capacity(n_claims);
		let mut multilinears = Vec::with_capacity(n_multilinears);
		let mut const_suffixes = Vec::with_capacity(n_multilinears);

		for (i, state) in states.iter_mut().enumerate() {
			let indices = [2 * i, 2 * i + 1];

			let composite_claim = CompositeSumClaim {
				sum: state.layer_eval,
				composition: IndexComposition::new(n_multilinears, indices, BivariateProduct {})?,
			};

			composite_claims.push(composite_claim);

			let layer = state
				.remaining_layers
				.pop()
				.expect("not staging more than n_vars times");

			let multilinear_pair =
				if n_vars >= P::LOG_WIDTH && layer.len() < 1 << (n_vars - P::LOG_WIDTH) {
					[layer, vec![]]
				} else if n_vars >= P::LOG_WIDTH {
					let mut evals_0 = layer;
					let evals_1 = evals_0.split_off(1 << (n_vars - P::LOG_WIDTH));
					[evals_0, evals_1]
				} else {
					let mut evals_0 = P::zero();
					let mut evals_1 = P::zero();
					let only_packed = layer.first().copied().unwrap_or_else(P::one);

					for i in 0..1 << n_vars {
						evals_0.set(i, only_packed.get(i));
						evals_1.set(i, only_packed.get(i | 1 << n_vars));
					}

					[vec![evals_0], vec![evals_1]]
				};

			for multilinear in multilinear_pair {
				let suffix_len = (1usize << n_vars).saturating_sub(multilinear.len() * P::WIDTH);
				const_suffixes.push((F::ONE, suffix_len));
				multilinears.push(multilinear);
			}
		}

		let prover = EqIndSumcheckProverBuilder::without_switchover(n_vars, multilinears, backend)
			.with_const_suffixes(&const_suffixes)?
			.build(
				evaluation_order,
				eq_ind_challenges,
				composite_claims,
				evaluation_domain_factory,
			)?;

		Ok(prover)
	}

	fn update_layer_eval(&mut self, zero_eval: F, one_eval: F, gpa_challenge: F) {
		self.layer_eval = extrapolate_line_scalar::<F, F>(zero_eval, one_eval, gpa_challenge);
	}

	fn finalize(self, eval_point: &[F]) -> Result<LayerClaim<F>, Error> {
		if !self.remaining_layers.is_empty() {
			bail!(Error::PrematureFinalize);
		}

		Ok(LayerClaim {
			eval_point: eval_point.to_vec(),
			eval: self.layer_eval,
		})
	}
}
