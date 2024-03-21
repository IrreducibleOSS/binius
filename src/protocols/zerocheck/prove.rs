// Copyright 2023 Ulvetanna Inc.

use std::{borrow::Borrow, sync::Arc};

use tracing::instrument;

use crate::{
	field::{Field, TowerField},
	oracle::MultilinearOracleSet,
	polynomial::{
		transparent::eq_ind::EqIndPartialEval, Error as PolynomialError, MultilinearComposite,
		MultilinearPoly,
	},
	protocols::zerocheck::zerocheck::reduce_zerocheck_claim,
};

use super::{
	error::Error,
	zerocheck::{
		ProductComposition, ZerocheckClaim, ZerocheckProof, ZerocheckProveOutput, ZerocheckWitness,
	},
};

fn multiply_multilinear_composite<F, M, BM>(
	composite: MultilinearComposite<F, M, BM>,
	new_multilinear: BM,
) -> Result<MultilinearComposite<F, M, BM>, PolynomialError>
where
	F: Field,
	M: MultilinearPoly<F> + ?Sized,
	BM: Borrow<M>,
{
	let n_vars: usize = composite.n_vars();
	let inner_composition = ProductComposition::new(composite.composition);
	let composition: Arc<ProductComposition<F>> = Arc::new(inner_composition);
	let mut multilinears = composite.multilinears;
	multilinears.push(new_multilinear);

	MultilinearComposite::new(n_vars, composition, multilinears)
}

/// Prove a zerocheck instance reduction.
///
/// The input polynomial is a composition of multilinear polynomials over a field F. The routine is
/// also parameterized by an operating field OF, which is isomorphic to F and over which the
/// majority of field operations are to be performed.
/// Takes a challenge vector r as input
#[instrument(skip_all, name = "zerocheck::prove")]
pub fn prove<'a, F: TowerField>(
	oracles: &mut MultilinearOracleSet<F>,
	zerocheck_witness: ZerocheckWitness<'a, F>,
	zerocheck_claim: &ZerocheckClaim<F>,
	challenge: Vec<F>,
) -> Result<ZerocheckProveOutput<'a, F>, Error> {
	let n_vars = zerocheck_witness.n_vars();

	if challenge.len() != n_vars {
		return Err(Error::ChallengeVectorMismatch);
	}

	// Step 1: Construct a multilinear polynomial eq(X, Y) on 2*n_vars variables
	// partially evaluated at r, will refer to this multilinear polynomial
	// as eq_r(X) on n_vars variables
	let eq_r = EqIndPartialEval::new(n_vars, challenge.clone())?.multilinear_extension()?;

	// Step 2: Multiply eq_r(X) by poly to get a new multivariate polynomial
	// and represent it as a Multilinear composite
	let sumcheck_witness = multiply_multilinear_composite(zerocheck_witness, Arc::new(eq_r))?;

	// Step 3: Make Sumcheck Claim on New Polynomial
	let sumcheck_claim = reduce_zerocheck_claim(oracles, zerocheck_claim, challenge)?;

	// Step 4: Wrap everything up
	let zerocheck_proof = ZerocheckProof {};
	Ok(ZerocheckProveOutput {
		sumcheck_claim,
		sumcheck_witness,
		zerocheck_proof,
	})
}

#[cfg(test)]
mod tests {
	use rand::{rngs::StdRng, SeedableRng};

	use super::*;
	use crate::{
		field::BinaryField32b,
		oracle::{CommittedBatchSpec, CommittedId, CompositePolyOracle, MultivariatePolyOracle},
		polynomial::{CompositionPoly, MultilinearExtension},
		protocols::{test_utils::TestProductComposition, zerocheck::verify::verify},
	};
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
		let composition: Arc<dyn CompositionPoly<F>> =
			Arc::new(TestProductComposition::new(n_multilinears));
		let multilinears = (0..1 << n_vars)
			.map(|i| {
				let values: Vec<F> = (0..1 << n_vars)
					.map(|j| if i == j { F::ZERO } else { F::ONE })
					.collect::<Vec<_>>();
				Arc::new(MultilinearExtension::from_values(values).unwrap())
					as Arc<dyn MultilinearPoly<F> + Send + Sync>
			})
			.collect::<Vec<_>>();
		let zerocheck_witness =
			MultilinearComposite::new(n_vars, composition, multilinears).unwrap();

		let mut oracles = MultilinearOracleSet::new();
		let batch_id = oracles.add_committed_batch(CommittedBatchSpec {
			round_id: 0,
			n_vars,
			n_polys: n_multilinears,
			tower_level: F::TOWER_LEVEL,
		});

		// Setup claim
		let h = (0..n_multilinears)
			.map(|i| oracles.committed_oracle(CommittedId { batch_id, index: i }))
			.collect();
		let composite_poly = CompositePolyOracle::new(
			n_vars,
			h,
			Arc::new(TestProductComposition::new(n_multilinears)),
		)
		.unwrap();
		let poly_oracle = MultivariatePolyOracle::Composite(composite_poly);
		let zerocheck_claim: ZerocheckClaim<F> = ZerocheckClaim { poly: poly_oracle };

		// Setup challenge
		let challenge = repeat_with(|| Field::random(&mut rng))
			.take(n_vars)
			.collect::<Vec<_>>();

		// PROVER
		let prove_output =
			prove(&mut oracles.clone(), zerocheck_witness, &zerocheck_claim, challenge.clone())
				.unwrap();
		let proof = prove_output.zerocheck_proof;
		// VERIFIER
		let sumcheck_claim =
			verify(&mut oracles.clone(), &zerocheck_claim, proof, challenge).unwrap();
		assert_eq!(sumcheck_claim.sum, F::ZERO);
		assert_eq!(sumcheck_claim.poly.n_vars(), n_vars);
		assert_eq!(prove_output.sumcheck_claim.sum, F::ZERO);
		assert_eq!(prove_output.sumcheck_claim.poly.n_vars(), n_vars);
	}
}
