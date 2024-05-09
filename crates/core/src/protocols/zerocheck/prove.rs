// Copyright 2023 Ulvetanna Inc.

use super::{
	error::Error,
	zerocheck::{ZerocheckClaim, ZerocheckProof, ZerocheckProveOutput, ZerocheckWitness},
};
use crate::{
	polynomial::{extrapolate_line, CompositionPoly, MultilinearExtension},
	protocols::{
		sumcheck::prove_general::SumcheckEvaluator, zerocheck::zerocheck::reduce_zerocheck_claim,
	},
};
use binius_field::{Field, PackedField, TowerField};
use tracing::instrument;

/// Prove a zerocheck instance reduction.
#[instrument(skip_all, name = "zerocheck::prove")]
pub fn prove<'a, F, PW, CW>(
	zerocheck_claim: &ZerocheckClaim<F>,
	zerocheck_witness: ZerocheckWitness<'a, PW, CW>,
	challenge: Vec<F>,
) -> Result<ZerocheckProveOutput<'a, F, PW, CW>, Error>
where
	F: TowerField + From<PW::Scalar>,
	PW: PackedField,
	PW::Scalar: TowerField + From<F>,
	CW: CompositionPoly<PW>,
{
	let n_vars = zerocheck_witness.n_vars();

	if challenge.len() + 1 != n_vars {
		return Err(Error::ChallengeVectorMismatch);
	}

	let sumcheck_claim = reduce_zerocheck_claim(zerocheck_claim, challenge)?;

	let zerocheck_proof = ZerocheckProof;
	Ok(ZerocheckProveOutput {
		sumcheck_claim,
		sumcheck_witness: zerocheck_witness,
		zerocheck_proof,
	})
}

/// Evaluator for the first round of the zerocheck protocol.
///
/// In the first round, we do not need to evaluate at the point F::ONE, because the value is known
/// to be zero. This version of the zerocheck protocol uses the optimizations from section 3 of
/// [Gruen24].
///
/// [Gruen24]: https://eprint.iacr.org/2024/108
#[derive(Debug)]
pub struct ZerocheckFirstRoundEvaluator<'a, P, C>
where
	P: PackedField,
	C: CompositionPoly<P>,
{
	pub composition: &'a C,
	pub domain: &'a [P::Scalar],
	pub eq_ind: MultilinearExtension<'a, P::Scalar>,
}

impl<'a, F, P, C> SumcheckEvaluator<F> for ZerocheckFirstRoundEvaluator<'a, P, C>
where
	F: Field,
	P: PackedField<Scalar = F>,
	C: CompositionPoly<P>,
{
	fn n_round_evals(&self) -> usize {
		// In the very first round of a sumcheck that comes from zerocheck, we can uniquely
		// determine the degree d univariate round polynomial r with evaluations at X = 2, ..., d
		// because we know r(0) = r(1) = 0
		self.domain.len() - 2
	}

	fn process_vertex(
		&self,
		index: usize,
		evals_0: &[F],
		evals_1: &[F],
		evals_z: &mut [F],
		round_evals: &mut [F],
	) {
		debug_assert!(index < self.eq_ind.size());

		let eq_ind_factor = self.eq_ind.evaluate_on_hypercube(index).unwrap_or(F::ZERO);

		// The rest require interpolation.
		for d in 2..self.domain.len() {
			evals_0
				.iter()
				.zip(evals_1.iter())
				.zip(evals_z.iter_mut())
				.for_each(|((&evals_0_j, &evals_1_j), evals_z_j)| {
					*evals_z_j = extrapolate_line(evals_0_j, evals_1_j, self.domain[d]);
				});

			let composite_value = self
				.composition
				.evaluate(evals_z)
				.expect("evals_z is initialized with a length of poly.composition.n_vars()");

			round_evals[d - 2] += composite_value * eq_ind_factor;
		}
	}
}

/// Evaluator for the later rounds of the zerocheck protocol.
///
/// This version of the zerocheck protocol uses the optimizations from section 3 of [Gruen24].
///
/// [Gruen24]: https://eprint.iacr.org/2024/108
#[derive(Debug)]
pub struct ZerocheckLaterRoundEvaluator<'a, P, C>
where
	P: PackedField,
	C: CompositionPoly<P>,
{
	pub composition: &'a C,
	pub domain: &'a [P::Scalar],
	pub eq_ind: MultilinearExtension<'a, P::Scalar>,
}

impl<'a, F, P, C> SumcheckEvaluator<F> for ZerocheckLaterRoundEvaluator<'a, P, C>
where
	F: Field,
	P: PackedField<Scalar = F>,
	C: CompositionPoly<P>,
{
	fn n_round_evals(&self) -> usize {
		// We can uniquely derive the degree d univariate round polynomial r from evaluations at
		// X = 1, ..., d because we have an identity that relates r(0), r(1), and the current
		// round's claimed sum
		self.domain.len() - 1
	}

	fn process_vertex(
		&self,
		index: usize,
		evals_0: &[F],
		evals_1: &[F],
		evals_z: &mut [F],
		round_evals: &mut [F],
	) {
		debug_assert!(index < self.eq_ind.size());

		let eq_ind_factor = self.eq_ind.evaluate_on_hypercube(index).unwrap_or(F::ZERO);

		let composite_value = self
			.composition
			.evaluate(evals_1)
			.expect("evals_1 is initialized with a length of poly.composition.n_vars()");
		round_evals[0] += composite_value * eq_ind_factor;

		// The rest require interpolation.
		for d in 2..self.domain.len() {
			evals_0
				.iter()
				.zip(evals_1.iter())
				.zip(evals_z.iter_mut())
				.for_each(|((&evals_0_j, &evals_1_j), evals_z_j)| {
					*evals_z_j = extrapolate_line(evals_0_j, evals_1_j, self.domain[d]);
				});

			let composite_value = self
				.composition
				.evaluate(evals_z)
				.expect("evals_z is initialized with a length of poly.composition.n_vars()");

			round_evals[d - 1] += composite_value * eq_ind_factor;
		}
	}
}
