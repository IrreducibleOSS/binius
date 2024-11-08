// Copyright 2024 Irreducible Inc.

use super::{GrandProductClaim, GrandProductWitness};
use crate::{
	fiat_shamir::HasherChallenger,
	oracle::MultilinearOracleSet,
	protocols::gkr_gpa::{batch_prove, batch_verify, GrandProductBatchProveOutput},
	transcript::TranscriptWriter,
	witness::MultilinearExtensionIndex,
};
use binius_field::{
	as_packed_field::{PackScalar, PackedType},
	underlier::{UnderlierType, WithUnderlier},
	BinaryField128b, BinaryField32b, ExtensionField, Field, PackedField, RepackedExtension,
	TowerField,
};
use binius_math::{IsomorphicEvaluationDomainFactory, MultilinearExtension};
use groestl_crypto::Groestl256;
use rand::{rngs::StdRng, SeedableRng};
use std::iter::repeat_with;

fn generate_poly_helper<P, F>(
	rng: &mut StdRng,
	n_vars: usize,
	n_multilinears: usize,
) -> Vec<(MultilinearExtension<P>, F)>
where
	P: PackedField<Scalar: ExtensionField<F>>,
	F: Field,
{
	repeat_with(|| {
		let values = repeat_with(|| F::random(&mut *rng))
			.take(1 << n_vars)
			.collect::<Vec<F>>();
		let product = values.iter().fold(F::ONE, |acc, x| acc * *x);
		let packed_values = values
			.iter()
			.map(|x| P::set_single((*x).into()))
			.collect::<Vec<_>>();
		(MultilinearExtension::from_values(packed_values).unwrap(), product)
	})
	.take(n_multilinears)
	.collect::<Vec<_>>()
}

struct CreateClaimsWitnessesOutput<
	'a,
	U: UnderlierType + PackScalar<F, Packed = P>,
	P: PackedField<Scalar = F>,
	F: TowerField,
> {
	new_claims: Vec<GrandProductClaim<F>>,
	new_witnesses: Vec<GrandProductWitness<'a, P>>,
	oracle_set: MultilinearOracleSet<F>,
	witness_index: MultilinearExtensionIndex<'a, U, F>,
	rng: StdRng,
}

fn create_claims_witnesses_helper<
	U: UnderlierType + PackScalar<F, Packed = P>,
	P: PackedField<Scalar = F> + RepackedExtension<P>,
	F: TowerField,
>(
	mut rng: StdRng,
	mut oracle_set: MultilinearOracleSet<F>,
	mut witness_index: MultilinearExtensionIndex<'_, U, F>,
	n_vars: usize,
	n_multilins: usize,
) -> CreateClaimsWitnessesOutput<'_, U, P, F> {
	if n_vars == 0 || n_multilins == 0 {
		panic!("Require at least one variable and multilinear polynomial");
	}
	let batch_id = oracle_set.add_committed_batch(n_vars, F::TOWER_LEVEL);
	let multilin_oracles = (0..n_multilins)
		.map(|_| {
			let id = oracle_set.add_committed(batch_id);
			oracle_set.oracle(id)
		})
		.collect::<Vec<_>>();

	let mles_with_product = generate_poly_helper::<P, F>(&mut rng, n_vars, n_multilins);
	let update = (0..n_multilins).map(|index| {
		(multilin_oracles[index].id(), mles_with_product[index].0.clone().specialize_arc_dyn())
	});
	witness_index.update_multilin_poly(update).unwrap();

	let mut new_claims = Vec::with_capacity(n_multilins);
	let mut new_witnesses = Vec::with_capacity(n_multilins);
	(0..n_multilins).for_each(|index| {
		let claim = GrandProductClaim {
			n_vars,
			product: mles_with_product[index].1,
		};
		let witness_poly = witness_index
			.get_multilin_poly(multilin_oracles[index].id())
			.unwrap();
		let witness = GrandProductWitness::new(witness_poly).unwrap();
		new_claims.push(claim);
		new_witnesses.push(witness);
	});

	CreateClaimsWitnessesOutput {
		new_claims,
		new_witnesses,
		oracle_set,
		witness_index,
		rng,
	}
}

#[test]
fn test_prove_verify_batch() {
	type F = BinaryField128b;
	type U = <F as WithUnderlier>::Underlier;
	type P = PackedType<U, F>;
	type FS = BinaryField32b;
	let rng = StdRng::seed_from_u64(0);
	let oracle_set = MultilinearOracleSet::<F>::new();
	let witness_index = MultilinearExtensionIndex::<U, F>::new();
	let mut claims = Vec::new();
	let mut witnesses = Vec::new();
	let domain_factory = IsomorphicEvaluationDomainFactory::<FS>::default();
	let backend = binius_hal::make_portable_backend();

	// Setup
	let (n_vars, n_multilins) = (5, 2);
	let CreateClaimsWitnessesOutput {
		new_claims,
		new_witnesses,
		oracle_set,
		witness_index,
		rng,
	} = create_claims_witnesses_helper::<U, P, F>(rng, oracle_set, witness_index, n_vars, n_multilins);
	assert_eq!(new_claims.len(), n_multilins);
	assert_eq!(new_witnesses.len(), n_multilins);
	claims.extend(new_claims);
	witnesses.extend(new_witnesses);

	let (n_vars, n_multilins) = (4, 3);
	let CreateClaimsWitnessesOutput {
		new_claims,
		new_witnesses,
		oracle_set,
		witness_index,
		rng,
	} = create_claims_witnesses_helper::<U, P, F>(rng, oracle_set, witness_index, n_vars, n_multilins);
	assert_eq!(new_claims.len(), n_multilins);
	assert_eq!(new_witnesses.len(), n_multilins);
	claims.extend(new_claims);
	witnesses.extend(new_witnesses);

	let (n_vars, n_multilins) = (7, 5);
	let CreateClaimsWitnessesOutput {
		new_claims,
		new_witnesses,
		oracle_set,
		witness_index,
		rng,
	} = create_claims_witnesses_helper::<U, P, F>(rng, oracle_set, witness_index, n_vars, n_multilins);
	assert_eq!(new_claims.len(), n_multilins);
	assert_eq!(new_witnesses.len(), n_multilins);
	claims.extend(new_claims);
	witnesses.extend(new_witnesses);

	// Prove and Verify
	let _ = (oracle_set, witness_index, rng);

	let mut prover_transcript = TranscriptWriter::<HasherChallenger<Groestl256>>::default();
	let GrandProductBatchProveOutput {
		final_layer_claims: final_layer_claim,
		proof,
	} = batch_prove::<_, _, FS, _, _>(
		witnesses,
		&claims,
		domain_factory,
		&mut prover_transcript,
		&backend,
	)
	.unwrap();

	let mut verify_transcript = prover_transcript.into_reader();
	let verified_evalcheck_multilinear_claims =
		batch_verify(claims.clone(), proof, &mut verify_transcript).unwrap();

	assert_eq!(final_layer_claim.len(), verified_evalcheck_multilinear_claims.len());
	for (proved_eval_claim, verified_layer_laim) in final_layer_claim
		.iter()
		.zip(verified_evalcheck_multilinear_claims.iter())
	{
		// Evaluations match
		assert_eq!(proved_eval_claim.eval, verified_layer_laim.eval);
		// Evaluation Points match
		assert_eq!(proved_eval_claim.eval_point, verified_layer_laim.eval_point);
	}
}
