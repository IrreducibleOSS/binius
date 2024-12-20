// Copyright 2024 Irreducible Inc.

use std::{collections::VecDeque, iter};

use binius_field::{Field, TowerField};
use binius_utils::sorting::is_sorted_ascending;

use super::batch_prove::SumcheckProver;
use crate::{
	fiat_shamir::CanSample,
	protocols::sumcheck::{Error, RoundCoeffs},
	transcript::CanWrite,
};

/// Prover for a front-loaded batch sumcheck protocol execution.
///
/// Prover that satisfies the verification logic in
/// [`crate::protocols::sumcheck::front_loaded`]. See that module for protocol information.
///
///
/// This exposes a round-by-round interface so that the protocol call be interleaved with other
/// interactive protocols, sharing the same sequence of challenges. The verification logic must be
/// invoked with a specific sequence of calls, continuing for as many rounds as necessary until all
/// claims are finished.
///
/// 1. construct a new verifier with [`BatchProver::new`]
/// 2. if all rounds are complete, call [`BatchProver::finish`], otherwise proceed
/// 3. call [`BatchProver::send_round_proof`]
/// 4. sample a random challenge and call [`BatchProver::receive_challenge`] with it
/// 5. repeat from step 2
#[derive(Debug)]
pub struct BatchProver<F: Field, Prover> {
	provers: VecDeque<(Prover, F)>,
	round: usize,
}

impl<F, Prover> BatchProver<F, Prover>
where
	F: TowerField,
	Prover: SumcheckProver<F>,
{
	/// Constructs a new prover for the front-loaded batched sumcheck.
	///
	/// The constructor samples batching coefficients from the proof transcript.
	///
	/// ## Throws
	///
	/// * if the claims are not sorted in ascending order by number of variables
	pub fn new<Transcript>(provers: Vec<Prover>, transcript: &mut Transcript) -> Result<Self, Error>
	where
		Transcript: CanSample<F>,
	{
		if !is_sorted_ascending(provers.iter().map(|prover| prover.n_vars())) {
			return Err(Error::ClaimsOutOfOrder);
		}

		// Sample batch mixing coefficients
		let batch_coeffs = transcript.sample_vec(provers.len());
		let provers = iter::zip(provers, batch_coeffs).collect();

		Ok(Self { provers, round: 0 })
	}

	fn finish_claim_provers<Transcript>(&mut self, transcript: &mut Transcript) -> Result<(), Error>
	where
		Transcript: CanWrite,
	{
		while let Some((prover, _)) = self.provers.front() {
			if prover.n_vars() != self.round {
				break;
			}
			let (prover, _) = self.provers.pop_front().expect("front returned Some");
			let multilinear_evals = prover.finish()?;
			transcript.write_scalar_slice(&multilinear_evals);
		}
		Ok(())
	}

	/// Computes the round message and writes it to the proof transcript.
	pub fn send_round_proof<Transcript>(&mut self, transcript: &mut Transcript) -> Result<(), Error>
	where
		Transcript: CanWrite,
	{
		self.finish_claim_provers(transcript)?;

		let mut round_coeffs = RoundCoeffs::default();
		for (prover, batch_coeff) in &mut self.provers {
			let prover_coeffs = prover.execute(*batch_coeff)?;
			round_coeffs += &(prover_coeffs * *batch_coeff);
		}

		let round_proof = round_coeffs.truncate();
		transcript.write_scalar_slice(round_proof.coeffs());
		Ok(())
	}

	/// Finishes an interaction round by reducing the instance with the verifier challenge.
	pub fn receive_challenge(&mut self, challenge: F) -> Result<(), Error> {
		for (prover, _) in &mut self.provers {
			prover.fold(challenge)?;
		}
		self.round += 1;
		Ok(())
	}

	/// Finishes the remaining instance provers and checks that all rounds are completed.
	pub fn finish<Transcript>(mut self, transcript: &mut Transcript) -> Result<(), Error>
	where
		Transcript: CanWrite,
	{
		self.finish_claim_provers(transcript)?;
		if !self.provers.is_empty() {
			return Err(Error::ExpectedFold);
		}
		Ok(())
	}
}
