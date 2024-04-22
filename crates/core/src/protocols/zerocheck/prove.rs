// Copyright 2023 Ulvetanna Inc.

use super::{
	error::Error,
	zerocheck::{ZerocheckClaim, ZerocheckProof, ZerocheckProveOutput, ZerocheckWitness},
};
use crate::{polynomial::CompositionPoly, protocols::zerocheck::zerocheck::reduce_zerocheck_claim};
use binius_field::{PackedField, TowerField};
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

#[cfg(test)]
mod tests {
	use super::*;
	use crate::{
		oracle::{CommittedBatchSpec, CommittedId, CompositePolyOracle, MultilinearOracleSet},
		polynomial::{MultilinearComposite, MultilinearExtension},
		protocols::{test_utils::TestProductComposition, zerocheck::verify::verify},
		witness::MultilinearWitnessIndex,
	};
	use binius_field::{BinaryField32b, Field};
	use rand::{rngs::StdRng, SeedableRng};
	use std::iter::repeat_with;

	// f(x) = (x_1, x_2, x_3)) = g(h_0(x), ..., h_7(x)) where
	// g is the product composition
	// h_i(x) = 0 if x= <i>, 1 otw, defined first on boolean hypercube, then take multilinear extension
	// Specifically, f should vanish on the boolean hypercube, because some h_i will be 0.
	#[test]
	fn test_prove_verify_interaction() {
		crate::util::tracing::init_tracing();

		type F = BinaryField32b;
		let n_vars: usize = 3;
		let n_multilinears = 1 << n_vars;
		let mut rng = StdRng::seed_from_u64(0);

		// Setup witness
		let composition = TestProductComposition::new(n_multilinears);
		let multilinears = (0..1 << n_vars)
			.map(|i| {
				let values: Vec<F> = (0..1 << n_vars)
					.map(|j| if i == j { F::ZERO } else { F::ONE })
					.collect::<Vec<_>>();
				MultilinearExtension::from_values(values)
					.unwrap()
					.specialize_arc_dyn::<F>()
			})
			.collect::<Vec<_>>();

		let zerocheck_witness =
			MultilinearComposite::new(n_vars, composition, multilinears.clone()).unwrap();

		let mut oracles = MultilinearOracleSet::<F>::new();
		let batch_id = oracles.add_committed_batch(CommittedBatchSpec {
			round_id: 0,
			n_vars,
			n_polys: n_multilinears,
			tower_level: F::TOWER_LEVEL,
		});

		let mut witness = MultilinearWitnessIndex::new();
		for (i, multilinear) in multilinears.iter().cloned().enumerate() {
			let oracle_id = oracles.committed_oracle_id(CommittedId { batch_id, index: i });
			witness.set(oracle_id, multilinear);
		}

		// Setup claim
		let h = (0..n_multilinears)
			.map(|i| oracles.committed_oracle(CommittedId { batch_id, index: i }))
			.collect();
		let composite_poly =
			CompositePolyOracle::new(n_vars, h, TestProductComposition::new(n_multilinears))
				.unwrap();

		let zerocheck_claim = ZerocheckClaim {
			poly: composite_poly,
		};

		// Setup challenge
		let challenge = repeat_with(|| Field::random(&mut rng))
			.take(n_vars - 1)
			.collect::<Vec<_>>();

		// PROVER
		let prove_output = prove(&zerocheck_claim, zerocheck_witness, challenge.clone()).unwrap();

		let proof = prove_output.zerocheck_proof;
		// VERIFIER
		let sumcheck_claim = verify(&zerocheck_claim, proof, challenge).unwrap();
		assert_eq!(sumcheck_claim.sum, F::ZERO);
		assert_eq!(sumcheck_claim.poly.n_vars(), n_vars);
		assert_eq!(prove_output.sumcheck_claim.sum, F::ZERO);
		assert_eq!(prove_output.sumcheck_claim.poly.n_vars(), n_vars);
	}
}
