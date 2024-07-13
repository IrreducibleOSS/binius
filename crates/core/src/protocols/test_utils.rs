// Copyright 2023 Ulvetanna Inc.

use crate::{
	challenger::{CanObserve, CanSample},
	polynomial::{
		CompositionPoly, Error as PolynomialError, EvaluationDomainFactory, MultilinearExtension,
		MultivariatePoly,
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
	packed::set_packed_slice, BinaryField1b, ExtensionField, Field, PackedField, TowerField,
};
use std::ops::Deref;
use tracing::instrument;

// If the macro is not used in the same module, rustc thinks it is unused for some reason
#[allow(unused_macros, unused_imports)]
pub mod macros {
	macro_rules! felts {
		($f:ident[$($elem:expr),* $(,)?]) => { vec![$($f::from($elem)),*] };
	}
	pub(crate) use felts;
}

pub fn hypercube_evals_from_oracle<F: Field>(oracle: &dyn MultivariatePoly<F>) -> Vec<F> {
	(0..(1 << oracle.n_vars()))
		.map(|i| {
			oracle
				.evaluate(&decompose_index_to_hypercube_point(oracle.n_vars(), i))
				.unwrap()
		})
		.collect()
}

pub fn decompose_index_to_hypercube_point<F: Field>(n_vars: usize, index: usize) -> Vec<F> {
	(0..n_vars)
		.map(|k| {
			if (index >> k) % 2 == 1 {
				F::ONE
			} else {
				F::ZERO
			}
		})
		.collect::<Vec<_>>()
}

pub fn packed_slice<P>(assignments: &[(std::ops::Range<usize>, u8)]) -> Vec<P>
where
	P: PackedField<Scalar = BinaryField1b>,
{
	assert_eq!(assignments[0].0.start, 0, "First assignment must start at index 0");
	assert_eq!(
		assignments[assignments.len() - 1].0.end % P::WIDTH,
		0,
		"Last assignment must end at an index divisible by packing width"
	);
	for i in 1..assignments.len() {
		assert_eq!(
			assignments[i].0.start,
			assignments[i - 1].0.end,
			"2 assignments following each other can't be overlapping or have holes in between"
		);
	}
	assignments
		.iter()
		.for_each(|(r, _)| assert!(r.end > r.start, "Range must have positive size"));
	let packed_len = (P::WIDTH - 1
		+ (assignments.iter().map(|(range, _)| range.end))
			.max()
			.unwrap_or(0))
		/ P::WIDTH;
	let mut result: Vec<P> = vec![P::default(); packed_len];
	for (range, value) in assignments.iter() {
		for i in range.clone() {
			set_packed_slice(&mut result, i, P::Scalar::from(*value));
		}
	}
	result
}

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

	fn evaluate_scalar(&self, query: &[P::Scalar]) -> Result<P::Scalar, PolynomialError> {
		let n_vars = self.arity;
		assert_eq!(query.len(), n_vars);
		// Product of scalar values at the corresponding positions of the packed values.
		Ok(query.iter().copied().product())
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
	name = "test_utils::prove_bivariate_sumchecks_with_switchover"
)]
pub fn prove_bivariate_sumchecks_with_switchover<'a, F, PW, DomainField, CH>(
	sumchecks: impl IntoIterator<Item = BivariateSumcheck<'a, F, PW>>,
	challenger: &mut CH,
	switchover_fn: impl Fn(usize) -> usize + 'static,
	domain_factory: impl EvaluationDomainFactory<DomainField>,
) -> Result<(SumcheckBatchProof<F>, impl IntoIterator<Item = EvalcheckClaim<F>>), SumcheckError>
where
	F: Field + From<PW::Scalar>,
	PW: PackedField,
	PW::Scalar: From<F> + ExtensionField<DomainField>,
	DomainField: Field,
	CH: CanObserve<F> + CanSample<F>,
{
	let SumcheckBatchProveOutput {
		proof,
		evalcheck_claims,
	} = batch_prove(sumchecks, domain_factory, switchover_fn, challenger)?;

	Ok((proof, evalcheck_claims))
}

#[instrument(skip_all, name = "test_utils::make_non_same_query_pcs_sumcheck_claims")]
pub fn make_non_same_query_pcs_sumcheck_claims<'a, F: TowerField>(
	verifier: &mut EvalcheckVerifier<'a, F>,
	committed_eval_claims: &[CommittedEvalClaim<F>],
) -> Result<Vec<SumcheckClaim<F>>, EvalcheckError> {
	let metas = non_same_query_pcs_sumcheck_metas(
		verifier.oracles,
		committed_eval_claims,
		&mut verifier.batch_committed_eval_claims,
	)?;

	let claims = metas
		.into_iter()
		.map(|meta| non_same_query_pcs_sumcheck_claim(verifier.oracles, meta))
		.collect::<Result<Vec<_>, EvalcheckError>>()?;

	Ok(claims)
}

#[instrument(skip_all, name = "test_utils::make_non_same_query_pcs_sumchecks")]
pub fn make_non_same_query_pcs_sumchecks<'a, 'b, F, PW>(
	prover: &mut EvalcheckProver<'a, 'b, F, PW>,
	committed_eval_claims: &[CommittedEvalClaim<F>],
) -> Result<Vec<BivariateSumcheck<'a, F, PW>>, EvalcheckError>
where
	F: TowerField + From<PW::Scalar>,
	PW: PackedField,
	PW::Scalar: From<F>,
{
	let metas = non_same_query_pcs_sumcheck_metas(
		prover.oracles,
		committed_eval_claims,
		&mut prover.batch_committed_eval_claims,
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
			)?;
			Ok((claim, witness))
		})
		.collect::<Result<Vec<_>, EvalcheckError>>()?;

	Ok(sumchecks)
}
