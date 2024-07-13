// Copyright 2024 Ulvetanna Inc.

use super::{
	gkr_gpa::{GrandProductBatchProveOutput, LayerClaim, GKR_SUMCHECK_DEGREE},
	Error, GrandProductBatchProof, GrandProductClaim, GrandProductWitness,
};
use crate::{
	oracle::MultilinearPolyOracle,
	polynomial::{
		composition::BivariateProduct, extrapolate_line, IsomorphicEvaluationDomainFactory,
		MultilinearComposite, MultilinearExtension, MultilinearPoly, MultilinearQuery,
	},
	protocols::{
		evalcheck::EvalcheckMultilinearClaim,
		gkr_gpa::gkr_gpa::BatchLayerProof,
		gkr_sumcheck::{
			self, GkrSumcheckBatchProof, GkrSumcheckBatchProveOutput, GkrSumcheckClaim,
			GkrSumcheckWitness,
		},
	},
	witness::MultilinearWitness,
};
use binius_field::{
	packed::{get_packed_slice, set_packed_slice},
	ExtensionField, Field, PackedField, TowerField,
};
use binius_utils::sorting::{stable_sort, unsort};
use p3_challenger::{CanObserve, CanSample};
use std::{iter::Step, sync::Arc};

type MultilinWitnessPair<'a, P> = (MultilinearWitness<'a, P>, MultilinearWitness<'a, P>);

pub fn batch_prove<'a, F, PW, FS, CH>(
	witnesses: impl IntoIterator<Item = GrandProductWitness<'a, PW>>,
	claims: impl IntoIterator<Item = GrandProductClaim<F>>,
	mut challenger: CH,
) -> Result<GrandProductBatchProveOutput<F>, Error>
where
	FS: Field + Step,
	F: TowerField + From<PW::Scalar>,
	PW: PackedField,
	PW::Scalar: Field + From<F> + ExtensionField<FS>,
	CH: CanSample<F> + CanObserve<F>,
{
	//  Ensure witnesses and claims are of the same length, zip them together
	// 	For each witness-claim pair, create GrandProductProver
	let witness_vec = witnesses.into_iter().collect::<Vec<_>>();
	let claim_vec = claims.into_iter().collect::<Vec<_>>();

	let n_claims = claim_vec.len();
	if n_claims == 0 {
		return Ok(GrandProductBatchProveOutput::default());
	}
	if witness_vec.len() != n_claims {
		return Err(Error::MismatchedWitnessClaimLength);
	}

	// Create a vector of GrandProductProverStates
	let provers_vec = witness_vec
		.into_iter()
		.zip(claim_vec.into_iter())
		.map(|(witness, claim)| GrandProductProverState::new(&claim, witness))
		.collect::<Result<Vec<_>, _>>()?;

	let (original_indices, mut sorted_provers) =
		stable_sort(provers_vec, |prover| prover.input_vars(), true);

	let max_n_vars = sorted_provers
		.first()
		.expect("sorted_provers is not empty by invariant")
		.input_vars();

	let mut batch_layer_proofs = Vec::with_capacity(max_n_vars);
	let mut reverse_sorted_evalcheck_multilinear_claims = Vec::with_capacity(n_claims);

	for layer_no in 0..max_n_vars {
		// Step 1: Process finished provers
		process_finished_provers(
			n_claims,
			layer_no,
			&mut sorted_provers,
			&mut reverse_sorted_evalcheck_multilinear_claims,
		)?;

		// Now we must create the batch layer proof for the kth to k+1th layer reduction

		// Step 2: Create sumcheck batch proof
		let (gkr_sumcheck_batch_proof, sumcheck_challenge) = if layer_no == 0 {
			let sorted_evals = sorted_provers
				.iter()
				.map(|prover| prover.current_layer_claim.eval)
				.collect::<Vec<F>>();
			// Need to sample batching coefficients for prover-verifier challenger consistency
			let _batching_coeffs = challenger.sample_vec(sorted_evals.len() - 1);
			let proof = GkrSumcheckBatchProof {
				rounds: vec![],
				sorted_evals,
			};
			let sumcheck_challenge = vec![];
			(proof, sumcheck_challenge)
		} else {
			let domain_factory = IsomorphicEvaluationDomainFactory::<FS>::default();
			let sumcheck_claims_and_witnesses = sorted_provers
				.iter_mut()
				.map(|p| p.stage_gkr_sumcheck_claims_and_witnesses())
				.collect::<Result<Vec<_>, _>>()?;
			let GkrSumcheckBatchProveOutput {
				proof,
				reduced_claims,
			} = gkr_sumcheck::batch_prove(
				sumcheck_claims_and_witnesses,
				domain_factory,
				|_| 1, // TODO better switchover fn
				&mut challenger,
			)?;
			let sumcheck_challenge = reduced_claims[0].eval_point.clone();
			(proof, sumcheck_challenge)
		};

		// Step 3: Get (and observe) zero and one evaluations of the (k+1)th layer-multilinear
		let (zero_evals, one_evals) = sorted_provers
			.iter_mut()
			.map(|p| p.advise_sumcheck_prove(&sumcheck_challenge))
			.collect::<Result<Vec<_>, _>>()?
			.into_iter()
			.unzip::<_, _, Vec<F>, Vec<F>>();
		challenger.observe_slice(&zero_evals);
		challenger.observe_slice(&one_evals);

		// Step 4: Sample a challenge for the next layer
		let gkr_challenge = challenger.sample();

		// Step 5: Finalize each prover to update its internal current_layer_claim
		for ((prover, zero_eval), one_eval) in sorted_provers
			.iter_mut()
			.zip(zero_evals.iter())
			.zip(one_evals.iter())
		{
			prover.finalize_batch_layer_proof(
				*zero_eval,
				*one_eval,
				sumcheck_challenge.clone(),
				gkr_challenge,
			)?;
		}

		// Step 6: Create the BatchLayerProof and push it to batch_layer_proofs
		let batch_layer_proof = BatchLayerProof {
			gkr_sumcheck_batch_proof,
			zero_evals,
			one_evals,
		};
		batch_layer_proofs.push(batch_layer_proof);
	}
	process_finished_provers(
		n_claims,
		max_n_vars,
		&mut sorted_provers,
		&mut reverse_sorted_evalcheck_multilinear_claims,
	)?;

	debug_assert!(sorted_provers.is_empty());
	debug_assert_eq!(reverse_sorted_evalcheck_multilinear_claims.len(), n_claims);

	reverse_sorted_evalcheck_multilinear_claims.reverse();
	let sorted_evalcheck_multilinear_claims = reverse_sorted_evalcheck_multilinear_claims;

	let evalcheck_multilinear_claims =
		unsort(original_indices, sorted_evalcheck_multilinear_claims);

	Ok(GrandProductBatchProveOutput {
		evalcheck_multilinear_claims,
		proof: GrandProductBatchProof { batch_layer_proofs },
	})
}

fn process_finished_provers<F, PW>(
	n_claims: usize,
	layer_no: usize,
	sorted_provers: &mut Vec<GrandProductProverState<'_, F, PW>>,
	reverse_sorted_evalcheck_multilinear_claims: &mut Vec<EvalcheckMultilinearClaim<F>>,
) -> Result<(), Error>
where
	PW: PackedField,
	F: Field + From<PW::Scalar>,
	PW::Scalar: Field + From<F>,
{
	while !sorted_provers.is_empty() && sorted_provers.last().unwrap().input_vars() == layer_no {
		debug_assert!(layer_no > 0);
		let finished_prover = sorted_provers.pop().unwrap();
		let evalcheck_claim = finished_prover.finalize()?;
		reverse_sorted_evalcheck_multilinear_claims.push(evalcheck_claim);
		debug_assert_eq!(
			sorted_provers.len() + reverse_sorted_evalcheck_multilinear_claims.len(),
			n_claims
		);
	}
	Ok(())
}

/// GKR-based Grand Product Argument protocol prover state
///
/// Coordinates the proving of a grand product claim before and after
/// the sumcheck-based layer reductions.
#[derive(Debug)]
struct GrandProductProverState<'a, F, PW>
where
	F: Field + From<PW::Scalar>,
	PW: PackedField,
	PW::Scalar: Field + From<F>,
{
	// Input polynomial oracle
	poly: MultilinearPolyOracle<F>,
	// Layers of the product circuit as multilinear polynomials
	// The ith element is the ith layer of the product circuit
	layers: Vec<MultilinearWitness<'a, PW>>,
	// The ith element consists of a tuple of the
	// first and second halves of the (i+1)th layer of the product circuit
	next_layer_halves: Vec<MultilinWitnessPair<'a, PW>>,
	// The current claim about a layer multilinear of the product circuit
	current_layer_claim: LayerClaim<F>,
}

impl<'a, F, PW> GrandProductProverState<'a, F, PW>
where
	F: Field + From<PW::Scalar>,
	PW: PackedField,
	PW::Scalar: Field + From<F>,
{
	/// Create a new GrandProductProverState
	fn new(
		claim: &GrandProductClaim<F>,
		witness: GrandProductWitness<'a, PW>,
	) -> Result<Self, Error> {
		let n_vars = claim.poly.n_vars();
		if n_vars != witness.n_vars() {
			return Err(Error::ProverClaimWitnessMismatch);
		}

		// Compute the layers of the product circuit, enforce output is the claimed product
		let layer_evals = compute_product_circuit_evals(witness)?;
		debug_assert_eq!(layer_evals.len(), n_vars + 1);

		let next_layer_halves = (1..n_vars + 1)
			.map(|i| {
				let k = 1 << (i - 1);
				let first_half_evals = Arc::from(&layer_evals[i][..k]);
				let second_half_evals = Arc::from(&layer_evals[i][k..]);
				let left = MultilinearExtension::from_values_generic(first_half_evals)?
					.specialize_arc_dyn();
				let right = MultilinearExtension::from_values_generic(second_half_evals)?
					.specialize_arc_dyn();
				Ok((left, right))
			})
			.collect::<Result<Vec<_>, Error>>()?;

		let layers = layer_evals
			.iter()
			.map(|evals| {
				let mle = MultilinearExtension::from_values_generic(Arc::from(&evals[..]))?
					.specialize_arc_dyn();
				Ok(mle)
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
		Ok(GrandProductProverState {
			poly: claim.poly.clone(),
			next_layer_halves,
			layers,
			current_layer_claim: layer_claim,
		})
	}

	fn input_vars(&self) -> usize {
		self.poly.n_vars()
	}

	fn current_layer_no(&self) -> usize {
		self.current_layer_claim.eval_point.len()
	}

	// Create GKR sumcheck prover
	#[allow(clippy::type_complexity)]
	fn stage_gkr_sumcheck_claims_and_witnesses(
		&self,
	) -> Result<
		(
			GkrSumcheckClaim<F>,
			GkrSumcheckWitness<PW, BivariateProduct, MultilinearWitness<'a, PW>>,
		),
		Error,
	> {
		if self.current_layer_no() >= self.input_vars() {
			return Err(Error::TooManyRounds);
		}

		// Witness
		let current_layer = self.layers[self.current_layer_no()].clone();
		let (left_half, right_half) = self.next_layer_halves[self.current_layer_no()].clone();
		let poly = MultilinearComposite::<PW, _, _>::new(
			self.current_layer_no(),
			BivariateProduct,
			vec![left_half, right_half],
		)?;
		let witness = GkrSumcheckWitness {
			poly,
			current_layer,
		};

		// Claim
		let claim = GkrSumcheckClaim {
			n_vars: self.current_layer_no(),
			degree: GKR_SUMCHECK_DEGREE,
			sum: self.current_layer_claim.eval,
			r: self.current_layer_claim.eval_point.clone(),
		};

		Ok((claim, witness))
	}

	// Give the (k+1)th layer evaluations at the evaluation points (r'_k, 0) and (r'_k, 1)
	fn advise_sumcheck_prove(&self, sumcheck_eval_point: &[F]) -> Result<(F, F), Error> {
		if self.current_layer_no() >= self.input_vars() {
			return Err(Error::TooManyRounds);
		}

		let query = sumcheck_eval_point
			.iter()
			.cloned()
			.map(Into::into)
			.collect::<Vec<_>>();
		let multilinear_query = MultilinearQuery::with_full_query(&query)?;

		let zero_eval = self.next_layer_halves[self.current_layer_no()]
			.0
			.evaluate(&multilinear_query)?;
		let one_eval = self.next_layer_halves[self.current_layer_no()]
			.1
			.evaluate(&multilinear_query)?;

		Ok((zero_eval.into(), one_eval.into()))
	}

	fn finalize_batch_layer_proof(
		&mut self,
		zero_eval: F,
		one_eval: F,
		sumcheck_challenge: Vec<F>,
		gkr_challenge: F,
	) -> Result<(), Error> {
		if self.current_layer_no() >= self.input_vars() {
			return Err(Error::TooManyRounds);
		}

		let new_eval = extrapolate_line(zero_eval, one_eval, gkr_challenge);
		let mut layer_challenge = sumcheck_challenge;
		layer_challenge.push(gkr_challenge);

		self.current_layer_claim = LayerClaim {
			eval_point: layer_challenge,
			eval: new_eval,
		};
		Ok(())
	}

	fn finalize(self) -> Result<EvalcheckMultilinearClaim<F>, Error> {
		if self.current_layer_no() != self.input_vars() {
			return Err(Error::PrematureFinalize);
		}

		let evalcheck_multilinear_claim = EvalcheckMultilinearClaim {
			poly: self.poly,
			eval_point: self.current_layer_claim.eval_point,
			eval: self.current_layer_claim.eval,
			is_random_point: true,
		};
		Ok(evalcheck_multilinear_claim)
	}
}

// Computes the product circuit layers for a given multilinear polynomial
// The result is a vector of vectors, where the outer vector is indexed by the layer number
fn compute_product_circuit_evals<P: PackedField>(
	poly: GrandProductWitness<'_, P>,
) -> Result<Vec<Vec<P>>, Error> {
	let mut input_layer = vec![P::zero(); (1 << poly.n_vars()) / P::WIDTH];
	for (i, packed_field) in input_layer.iter_mut().enumerate() {
		poly.subcube_evals(P::LOG_WIDTH, i, std::slice::from_mut(packed_field))?;
	}

	let mut all_layers = vec![input_layer];
	for curr_n_vars in (0..poly.n_vars()).rev() {
		let layer_below = all_layers.last().expect("layers is not empty by invariant");
		let mut new_layer = vec![P::zero(); (1 << curr_n_vars) / P::WIDTH];
		for i in 0..1 << curr_n_vars {
			let left = get_packed_slice(layer_below, i);
			let right = get_packed_slice(layer_below, i + (1 << curr_n_vars));
			set_packed_slice(new_layer.as_mut_slice(), i, left * right);
		}
		all_layers.push(new_layer);
	}

	// Reverse the layers
	all_layers.reverse();
	Ok(all_layers)
}
