// Copyright 2023 Ulvetanna Inc.

use crate::{
	challenger::{CanObserve, CanSample},
	oracle::{CompositePolyOracle, Error as OracleError},
	polynomial::{
		CompositionPoly, Error as PolynomialError, IdentityCompositionPoly, MultilinearExtension,
		MultilinearExtensionSpecialized,
	},
	protocols::{
		evalcheck::{
			subclaims::{
				non_same_query_pcs_sumcheck_claim, non_same_query_pcs_sumcheck_metas,
				non_same_query_pcs_sumcheck_witness, BivariateSumcheck, MemoizedQueries,
			},
			CommittedEvalClaim, Error as EvalcheckError, EvalcheckClaim, EvalcheckProver,
			EvalcheckVerifier,
		},
		sumcheck::{
			batch_prove, Error as SumcheckError, SumcheckBatchProof, SumcheckBatchProveOutput,
			SumcheckClaim,
		},
	},
};
use binius_field::{
	as_packed_field::PackScalar, underlier::WithUnderlier, ExtensionField, Field, PackedExtension,
	PackedField, TowerField,
};
use binius_hal::ComputationBackend;
use binius_math::EvaluationDomainFactory;
use rand::Rng;
use std::ops::Deref;
use tracing::instrument;

use super::evalcheck_v2::EvalcheckMultilinearClaim;

#[derive(Clone, Debug)]
pub struct TestProductComposition {
	arity: usize,
}

impl TestProductComposition {
	pub fn new(arity: usize) -> Self {
		Self { arity }
	}
}

impl<P> CompositionPoly<P> for TestProductComposition
where
	P: PackedField,
{
	fn n_vars(&self) -> usize {
		self.arity
	}

	fn degree(&self) -> usize {
		self.arity
	}

	fn evaluate(&self, query: &[P]) -> Result<P, PolynomialError> {
		let n_vars = self.arity;
		assert_eq!(query.len(), n_vars);
		// Product of scalar values at the corresponding positions of the packed values.
		Ok(query.iter().copied().product())
	}

	fn binary_tower_level(&self) -> usize {
		0
	}
}

pub fn generate_zero_product_multilinears<P, PE>(
	mut rng: impl Rng,
	n_vars: usize,
	n_multilinears: usize,
) -> Vec<MultilinearExtensionSpecialized<P, PE>>
where
	P: PackedField,
	PE: PackedField<Scalar: ExtensionField<P::Scalar>>,
{
	(0..n_multilinears)
		.map(|j| {
			let values = (0..(1 << n_vars.saturating_sub(P::LOG_WIDTH)))
				// For every hypercube vertex, one of the multilinear values at that vertex
				// is 0, thus the composite defined by their product must be 0 over the
				// hypercube.
				.map(|i| {
					let mut packed = P::random(&mut rng);
					for k in 0..P::WIDTH {
						if (k + i * P::WIDTH) % n_multilinears == j {
							packed.set(k, P::Scalar::ZERO);
						}
					}
					packed
				})
				.collect();
			MultilinearExtension::from_values(values)
				.unwrap()
				.specialize::<PE>()
		})
		.collect()
}

pub fn transform_poly<F, OF, Data>(
	multilin: MultilinearExtension<F, Data>,
) -> Result<MultilinearExtension<OF>, PolynomialError>
where
	F: Field,
	OF: Field + From<F> + Into<F>,
	Data: Deref<Target = [F]>,
{
	let values = multilin.evals().iter().cloned().map(OF::from).collect();

	MultilinearExtension::from_values(values)
}

#[instrument(
	skip_all,
	name = "test_utils::prove_bivariate_sumchecks_with_switchover",
	level = "debug"
)]
pub fn prove_bivariate_sumchecks_with_switchover<'a, F, PW, DomainField, CH, Backend>(
	sumchecks: impl IntoIterator<Item = BivariateSumcheck<'a, F, PW>>,
	challenger: &mut CH,
	switchover_fn: impl Fn(usize) -> usize + 'static,
	domain_factory: impl EvaluationDomainFactory<DomainField>,
	backend: &Backend,
) -> Result<(SumcheckBatchProof<F>, impl IntoIterator<Item = EvalcheckClaim<F>>), SumcheckError>
where
	F: Field + From<PW::Scalar>,
	PW: PackedExtension<DomainField>,
	PW::Scalar: From<F> + ExtensionField<DomainField>,
	DomainField: Field,
	CH: CanObserve<F> + CanSample<F>,
	Backend: ComputationBackend,
{
	let SumcheckBatchProveOutput {
		proof,
		evalcheck_claims,
	} = batch_prove(sumchecks, domain_factory, switchover_fn, challenger, backend)?;

	Ok((proof, evalcheck_claims))
}

// NB. This is a compatibility hack to bridge the gap between current evalcheck (which operates on composites)
//     and sumcheck_v2 (which outputs multilinear claims). Each multilinear claim gets simply wrapped into an
//     identity composition. Once evalcheck refactors land this should get removed.
pub fn conflate_multilinear_evalchecks<F: Field>(
	claims: impl IntoIterator<Item = EvalcheckMultilinearClaim<F>>,
) -> Result<Vec<EvalcheckClaim<F>>, OracleError> {
	claims
		.into_iter()
		.map(|claim| {
			let poly = CompositePolyOracle::new(
				claim.poly.n_vars(),
				vec![claim.poly],
				IdentityCompositionPoly,
			)?;

			Ok(EvalcheckClaim {
				poly,
				eval_point: claim.eval_point,
				eval: claim.eval,
				is_random_point: claim.is_random_point,
			})
		})
		.collect::<Result<Vec<_>, _>>()
}

#[instrument(
	skip_all,
	name = "test_utils::make_non_same_query_pcs_sumcheck_claims",
	level = "debug"
)]
pub fn make_non_same_query_pcs_sumcheck_claims<'a, F: TowerField>(
	verifier: &mut EvalcheckVerifier<'a, F>,
	committed_eval_claims: &[CommittedEvalClaim<F>],
) -> Result<Vec<SumcheckClaim<F>>, EvalcheckError> {
	let metas = non_same_query_pcs_sumcheck_metas(
		verifier.oracles,
		committed_eval_claims,
		&mut verifier.batch_committed_eval_claims,
		None,
	)?;

	let claims = metas
		.into_iter()
		.map(|meta| non_same_query_pcs_sumcheck_claim(verifier.oracles, meta))
		.collect::<Result<Vec<_>, EvalcheckError>>()?;

	Ok(claims)
}

#[instrument(
	skip_all,
	name = "test_utils::make_non_same_query_pcs_sumchecks",
	level = "debug"
)]
pub fn make_non_same_query_pcs_sumchecks<'a, 'b, F, PW, Backend>(
	prover: &mut EvalcheckProver<'a, 'b, F, PW, Backend>,
	committed_eval_claims: &[CommittedEvalClaim<F>],
	backend: &Backend,
) -> Result<Vec<BivariateSumcheck<'a, F, PW>>, EvalcheckError>
where
	F: TowerField + From<PW::Scalar>,
	PW: PackedField + WithUnderlier,
	PW::Scalar: TowerField + From<F>,
	PW::Underlier: PackScalar<PW::Scalar, Packed = PW>,
	Backend: ComputationBackend,
{
	let metas = non_same_query_pcs_sumcheck_metas(
		prover.oracles,
		committed_eval_claims,
		&mut prover.batch_committed_eval_claims,
		Some(&mut prover.memoized_eq_ind),
	)?;

	let mut memoized_queries = MemoizedQueries::new();

	let sumchecks = metas
		.into_iter()
		.map(|meta| {
			let claim = non_same_query_pcs_sumcheck_claim(prover.oracles, meta.clone())?;
			let witness = non_same_query_pcs_sumcheck_witness(
				prover.witness_index,
				&mut memoized_queries,
				meta,
				backend,
			)?;
			Ok((claim, witness))
		})
		.collect::<Result<Vec<_>, EvalcheckError>>()?;

	Ok(sumchecks)
}
