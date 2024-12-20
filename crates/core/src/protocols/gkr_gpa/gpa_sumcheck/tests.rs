// Copyright 2024 Irreducible Inc.

use std::iter::{self, Step};

use binius_field::{
	arch::OptimalUnderlier512b,
	as_packed_field::{PackScalar, PackedType},
	underlier::UnderlierType,
	BinaryField128b, BinaryField8b, ExtensionField, Field, PackedField, PackedFieldIndexable,
	TowerField,
};
use binius_hal::make_portable_backend;
use binius_math::{
	DefaultEvaluationDomainFactory, MLEDirectAdapter, MultilinearExtension, MultilinearPoly,
	MultilinearQuery,
};
use groestl_crypto::Groestl256;
use rand::{rngs::StdRng, SeedableRng};

use super::prove::GPAProver;
use crate::{
	composition::BivariateProduct,
	fiat_shamir::HasherChallenger,
	protocols::{
		sumcheck::{self, zerocheck::ExtraProduct, CompositeSumClaim, SumcheckClaim},
		test_utils::AddOneComposition,
	},
	transcript::TranscriptWriter,
};

fn test_prove_verify_bivariate_product_helper<U, F, FDomain>(
	n_vars: usize,
	use_first_round_eval1_advice: bool,
) where
	U: UnderlierType + PackScalar<F> + PackScalar<FDomain>,
	F: TowerField + ExtensionField<FDomain>,
	FDomain: Field + Step,
	PackedType<U, F>: PackedFieldIndexable,
{
	let mut rng = StdRng::seed_from_u64(0);

	let packed_len = 1 << n_vars.saturating_sub(PackedType::<U, F>::LOG_WIDTH);
	let a_column = (0..packed_len)
		.map(|_| PackedType::<U, F>::random(&mut rng))
		.collect::<Vec<_>>();
	let b_column = (0..packed_len)
		.map(|_| PackedType::<U, F>::random(&mut rng))
		.collect::<Vec<_>>();
	let ab1_column = iter::zip(&a_column, &b_column)
		.map(|(&a, &b)| a * b + PackedType::<U, F>::one())
		.collect::<Vec<_>>();

	let a_mle = MLEDirectAdapter::from(MultilinearExtension::from_values_slice(&a_column).unwrap());
	let b_mle = MLEDirectAdapter::from(MultilinearExtension::from_values_slice(&b_column).unwrap());
	let ab1_mle =
		MLEDirectAdapter::from(MultilinearExtension::from_values_slice(&ab1_column).unwrap());

	let gpa_round_challenges = (0..n_vars).map(|_| F::random(&mut rng)).collect::<Vec<_>>();
	let sum = ab1_mle
		.evaluate(MultilinearQuery::expand(&gpa_round_challenges).to_ref())
		.unwrap();

	let mut transcript = TranscriptWriter::<HasherChallenger<Groestl256>>::default();

	let backend = make_portable_backend();
	let evaluation_domain_factory = DefaultEvaluationDomainFactory::<FDomain>::default();

	let composition = AddOneComposition::new(BivariateProduct {});

	let prover_composite_claim = CompositeSumClaim {
		sum,
		composition: composition.clone(),
	};

	let prover = GPAProver::<FDomain, _, _, _, _>::new(
		vec![a_mle, b_mle],
		Some(vec![ab1_mle]).filter(|_| use_first_round_eval1_advice),
		[prover_composite_claim],
		evaluation_domain_factory,
		&gpa_round_challenges,
		&backend,
	)
	.unwrap();

	let _sumcheck_proof_output = sumcheck::batch_prove(vec![prover], &mut transcript).unwrap();

	let mut verifier_transcript = transcript.into_reader();

	let verifier_composite_claim = CompositeSumClaim {
		sum,
		composition: ExtraProduct { inner: composition },
	};

	let verifier_claim = SumcheckClaim::new(n_vars, 3, vec![verifier_composite_claim]).unwrap();

	let _sumcheck_verify_output =
		sumcheck::batch_verify(&[verifier_claim], &mut verifier_transcript).unwrap();
}

#[test]
fn test_gpa_sumcheck_prove_verify_nontrivial_packing() {
	let n_vars = 8;

	// Using a 512-bit underlier with a 128-bit extension field means the packed field will have a
	// non-trivial packing width of 4.
	for use_first_round_eval1_advice in [false, true] {
		test_prove_verify_bivariate_product_helper::<
			OptimalUnderlier512b,
			BinaryField128b,
			BinaryField8b,
		>(n_vars, use_first_round_eval1_advice);
	}
}
