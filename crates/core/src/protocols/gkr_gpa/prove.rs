// Copyright 2024 Irreducible Inc.

use super::{
	gkr_gpa::{GrandProductBatchProveOutput, LayerClaim},
	gpa_sumcheck::prove::GPAProver,
	Error, GrandProductBatchProof, GrandProductClaim, GrandProductWitness,
};
use crate::{protocols::sumcheck, transcript::CanWrite};
use binius_field::{
	ExtensionField, Field, PackedExtension, PackedField, PackedFieldIndexable, TowerField,
};
use binius_hal::ComputationBackend;
use binius_math::{
	extrapolate_line_scalar, EvaluationDomainFactory, MLEDirectAdapter, MultilinearExtension,
	MultilinearPoly,
};
use binius_utils::{
	bail,
	sorting::{stable_sort, unsort},
};
use p3_challenger::{CanObserve, CanSample};
use tracing::instrument;

/// Proves batch reduction turning each GrandProductClaim into an EvalcheckMultilinearClaim
///
/// REQUIRES:
/// * witnesses and claims are of the same length
/// * The ith witness corresponds to the ith claim
#[instrument(skip_all, name = "gkr_gpa::batch_prove", level = "debug")]
pub fn batch_prove<'a, F, P, FDomain, Transcript, Backend>(
	witnesses: impl IntoIterator<Item = GrandProductWitness<'a, P>>,
	claims: &[GrandProductClaim<F>],
	evaluation_domain_factory: impl EvaluationDomainFactory<FDomain>,
	mut transcript: Transcript,
	backend: &Backend,
) -> Result<GrandProductBatchProveOutput<F>, Error>
where
	F: TowerField,
	P: PackedFieldIndexable<Scalar = F> + PackedExtension<FDomain>,
	FDomain: Field,
	P::Scalar: Field + ExtensionField<FDomain>,
	Transcript: CanSample<F> + CanObserve<F> + CanWrite,
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

	let mut batch_layer_proofs = Vec::with_capacity(max_n_vars);
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
		let (gpa_sumcheck_batch_proof, sumcheck_challenge) = {
			let stage_gpa_sumcheck_provers = sorted_provers
				.iter_mut()
				.map(|p| p.stage_gpa_sumcheck_prover(evaluation_domain_factory.clone()))
				.collect::<Result<Vec<_>, _>>()?;
			let (batch_sumcheck_output, proof) =
				sumcheck::batch_prove(stage_gpa_sumcheck_provers, &mut transcript)?;
			let sumcheck_challenge = batch_sumcheck_output.challenges;

			(proof, sumcheck_challenge)
		};

		// Step 3: Sample a challenge for the next layer
		let gpa_challenge = transcript.sample();

		// Step 4: Finalize each prover to update its internal current_layer_claim
		for (i, prover) in sorted_provers.iter_mut().enumerate() {
			prover.finalize_batch_layer_proof(
				gpa_sumcheck_batch_proof.multilinear_evals[i][0],
				gpa_sumcheck_batch_proof.multilinear_evals[i][1],
				sumcheck_challenge.clone(),
				gpa_challenge,
			)?;
		}

		batch_layer_proofs.push(gpa_sumcheck_batch_proof);
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

	Ok(GrandProductBatchProveOutput {
		final_layer_claims,
		proof: GrandProductBatchProof { batch_layer_proofs },
	})
}

fn process_finished_provers<F, P, Backend>(
	layer_no: usize,
	sorted_provers: &mut Vec<GrandProductProverState<'_, F, P, Backend>>,
	reverse_sorted_final_layer_claims: &mut Vec<LayerClaim<F>>,
) -> Result<(), Error>
where
	P: PackedFieldIndexable<Scalar = F>,
	F: Field + From<P::Scalar>,
	P::Scalar: Field + From<F>,
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
	layers: Vec<MLEDirectAdapter<P, &'a [P]>>,
	// The ith element consists of a tuple of the
	// first and second halves of the (i+1)th layer of the product circuit
	next_layer_halves: Vec<[MLEDirectAdapter<P, &'a [P]>; 2]>,
	// The current claim about a layer multilinear of the product circuit
	current_layer_claim: LayerClaim<F>,

	backend: Backend,
}

impl<'a, F, P, Backend> GrandProductProverState<'a, F, P, Backend>
where
	F: Field + From<P::Scalar>,
	P: PackedFieldIndexable<Scalar = F>,
	P::Scalar: Field + From<F>,
	Backend: ComputationBackend,
{
	/// Create a new GrandProductProverState
	fn new(
		claim: &GrandProductClaim<F>,
		witness: &'a GrandProductWitness<'a, P>,
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
				let left = MultilinearExtension::from_values_slice(left_evals)?;
				let right = MultilinearExtension::from_values_slice(right_evals)?;
				Ok([left, right].map(MLEDirectAdapter::from))
			})
			.collect::<Result<Vec<_>, Error>>()?;

		let layers = (0..n_layers)
			.map(|i| {
				let ith_layer_evals = witness.ith_layer_evals(i)?;
				let mle = MultilinearExtension::from_values_slice(ith_layer_evals)?;
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

	fn input_vars(&self) -> usize {
		self.n_vars
	}

	fn current_layer_no(&self) -> usize {
		self.current_layer_claim.eval_point.len()
	}

	// Create GPA sumcheck prover
	#[allow(clippy::type_complexity)]
	fn stage_gpa_sumcheck_prover<FDomain>(
		&self,
		evaluation_domain_factory: impl EvaluationDomainFactory<FDomain>,
	) -> Result<GPAProver<FDomain, P, impl MultilinearPoly<P> + Send + Sync + 'a, Backend>, Error>
	where
		FDomain: Field,
		P: PackedExtension<FDomain>,
		F: ExtensionField<FDomain>,
	{
		if self.current_layer_no() >= self.input_vars() {
			bail!(Error::TooManyRounds);
		}

		// Witness
		let current_layer = self.layers[self.current_layer_no()].clone();
		let multilinears = self.next_layer_halves[self.current_layer_no()].clone();

		GPAProver::new(
			multilinears,
			current_layer,
			self.current_layer_claim.eval,
			evaluation_domain_factory,
			&self.current_layer_claim.eval_point,
			&self.backend,
		)
		.map_err(|e| e.into())
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
		let new_eval = extrapolate_line_scalar(zero_eval, one_eval, gpa_challenge);
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
