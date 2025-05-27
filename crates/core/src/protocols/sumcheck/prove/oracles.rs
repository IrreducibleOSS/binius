// Copyright 2024-2025 Irreducible Inc.

use binius_fast_compute::arith_circuit::ArithCircuitPoly;
use binius_field::{ExtensionField, Field, PackedExtension, PackedField, TowerField};
use binius_hal::ComputationBackend;
use binius_math::{EvaluationDomainFactory, EvaluationOrder, MultilinearPoly};
use binius_utils::bail;

use super::{
	RegularSumcheckProver, ZerocheckProverImpl,
	eq_ind::{EqIndSumcheckProver, EqIndSumcheckProverBuilder},
};
use crate::{
	oracle::{Constraint, ConstraintPredicate, ConstraintSet},
	protocols::{
		evalcheck::{EvalPoint, subclaims::MemoizedData},
		sumcheck::{
			CompositeSumClaim, Error, OracleClaimMeta, constraint_set_mlecheck_claim,
			constraint_set_sumcheck_claim,
		},
	},
	witness::{IndexEntry, MultilinearExtensionIndex, MultilinearWitness},
};

pub type OracleZerocheckProver<'a, P, FBase, FDomain, DomainFactory, Backend> = ZerocheckProverImpl<
	'a,
	FDomain,
	FBase,
	P,
	ArithCircuitPoly<FBase>,
	ArithCircuitPoly<<P as PackedField>::Scalar>,
	MultilinearWitness<'a, P>,
	DomainFactory,
	Backend,
>;

pub type OracleSumcheckProver<'a, FDomain, P, Backend> = RegularSumcheckProver<
	'a,
	FDomain,
	P,
	ArithCircuitPoly<<P as PackedField>::Scalar>,
	MultilinearWitness<'a, P>,
	Backend,
>;

pub type OracleMLECheckProver<'a, FDomain, P, Backend> = EqIndSumcheckProver<
	'a,
	FDomain,
	P,
	ArithCircuitPoly<<P as PackedField>::Scalar>,
	MultilinearWitness<'a, P>,
	Backend,
>;

/// Construct zerocheck prover from the constraint set. Fails when constraint set contains regular
/// sumchecks.
pub fn constraint_set_zerocheck_prover<'a, P, F, FBase, FDomain, DomainFactory, Backend>(
	constraints: Vec<Constraint<P::Scalar>>,
	multilinears: Vec<MultilinearWitness<'a, P>>,
	domain_factory: DomainFactory,
	zerocheck_challenges: &[F],
	backend: &'a Backend,
) -> Result<OracleZerocheckProver<'a, P, FBase, FDomain, DomainFactory, Backend>, Error>
where
	P: PackedField<Scalar = F>
		+ PackedExtension<F, PackedSubfield = P>
		+ PackedExtension<FDomain>
		+ PackedExtension<FBase>,
	F: TowerField,
	FBase: TowerField + ExtensionField<FDomain> + TryFrom<P::Scalar>,
	FDomain: Field,
	DomainFactory: EvaluationDomainFactory<FDomain>,
	Backend: ComputationBackend,
{
	let mut zeros = Vec::with_capacity(constraints.len());

	for Constraint {
		composition,
		predicate,
		name,
	} in constraints
	{
		let composition_base = composition
			.try_convert_field::<FBase>()
			.map_err(|_| Error::CircuitFieldDowncastFailed)?;
		match predicate {
			ConstraintPredicate::Zero => {
				zeros.push((
					name,
					ArithCircuitPoly::with_n_vars(multilinears.len(), composition_base)?,
					ArithCircuitPoly::with_n_vars(multilinears.len(), composition)?,
				));
			}
			_ => bail!(Error::MixedBatchingNotSupported),
		}
	}

	let prover = OracleZerocheckProver::<_, _, FDomain, _, _>::new(
		multilinears,
		zeros,
		zerocheck_challenges,
		domain_factory,
		backend,
	)?;

	Ok(prover)
}

/// Construct regular sumcheck prover from the constraint set. Fails when constraint set contains
/// zerochecks.
pub fn constraint_set_sumcheck_prover<'a, F, P, FDomain, Backend>(
	evaluation_order: EvaluationOrder,
	constraint_set: ConstraintSet<F>,
	witness: &MultilinearExtensionIndex<'a, P>,
	evaluation_domain_factory: impl EvaluationDomainFactory<FDomain>,
	switchover_fn: impl Fn(usize) -> usize + Clone,
	backend: &'a Backend,
) -> Result<OracleSumcheckProver<'a, FDomain, P, Backend>, Error>
where
	P: PackedField<Scalar = F> + PackedExtension<FDomain>,
	F: TowerField + ExtensionField<FDomain>,
	FDomain: Field,
	Backend: ComputationBackend,
{
	let (constraints, multilinears) = split_constraint_set::<F, P>(constraint_set, witness)?;

	let mut sums = Vec::new();

	for Constraint {
		composition,
		predicate,
		..
	} in constraints
	{
		match predicate {
			ConstraintPredicate::Sum(sum) => sums.push(CompositeSumClaim {
				composition: ArithCircuitPoly::with_n_vars(multilinears.len(), composition)?,
				sum,
			}),
			_ => bail!(Error::MixedBatchingNotSupported),
		}
	}

	let prover = RegularSumcheckProver::new(
		evaluation_order,
		multilinears,
		sums,
		evaluation_domain_factory,
		switchover_fn,
		backend,
	)?;

	Ok(prover)
}

/// Construct mlecheck prover from the constraint set. Fails when constraint set contains
/// zerochecks.
#[allow(clippy::too_many_arguments)]
pub fn constraint_set_mlecheck_prover<'a, 'b, F, P, FDomain, Backend>(
	evaluation_order: EvaluationOrder,
	constraint_set: ConstraintSet<F>,
	eq_ind_challenges: &[F],
	memoized_data: &mut MemoizedData<'b, P>,
	witness: &MultilinearExtensionIndex<'a, P>,
	evaluation_domain_factory: impl EvaluationDomainFactory<FDomain>,
	switchover_fn: impl Fn(usize) -> usize + Clone,
	backend: &'a Backend,
) -> Result<OracleMLECheckProver<'a, FDomain, P, Backend>, Error>
where
	P: PackedField<Scalar = F> + PackedExtension<FDomain>,
	F: TowerField + ExtensionField<FDomain>,
	FDomain: Field,
	Backend: ComputationBackend,
{
	let ConstraintSet {
		oracle_ids,
		constraints,
		n_vars,
	} = constraint_set;

	let mut multilinears = Vec::with_capacity(oracle_ids.len());
	let mut const_suffixes = Vec::with_capacity(oracle_ids.len());

	for id in oracle_ids {
		let IndexEntry {
			multilin_poly,
			nonzero_scalars_prefix,
		} = witness.get_index_entry(id)?;

		if multilin_poly.n_vars() != n_vars {
			bail!(Error::ConstraintSetNumberOfVariablesMismatch);
		}

		multilinears.push(multilin_poly);
		const_suffixes.push((F::ZERO, ((1 << n_vars) - nonzero_scalars_prefix)))
	}

	let mut sums = Vec::new();

	for Constraint {
		composition,
		predicate,
		..
	} in constraints
	{
		match predicate {
			ConstraintPredicate::Sum(sum) => sums.push(CompositeSumClaim {
				composition: ArithCircuitPoly::with_n_vars(multilinears.len(), composition)?,
				sum,
			}),
			_ => bail!(Error::MixedBatchingNotSupported),
		}
	}

	let n_vars = eq_ind_challenges.len();

	let eq_ind_partial = match evaluation_order {
		EvaluationOrder::LowToHigh => &eq_ind_challenges[n_vars.min(1)..],
		EvaluationOrder::HighToLow => &eq_ind_challenges[..n_vars.saturating_sub(1)],
	};

	let eq_ind_partial_evals = memoized_data
		.full_query(eq_ind_partial)?
		.expansion()
		.to_vec();

	let prover = EqIndSumcheckProverBuilder::with_switchover(multilinears, switchover_fn, backend)?
		.with_eq_ind_partial_evals(Backend::to_hal_slice(eq_ind_partial_evals))
		.with_const_suffixes(&const_suffixes)?
		.build(evaluation_order, eq_ind_challenges, sums, evaluation_domain_factory)?;

	Ok(prover)
}

type ConstraintsAndMultilinears<'a, F, P> = (Vec<Constraint<F>>, Vec<MultilinearWitness<'a, P>>);

#[allow(clippy::type_complexity)]
pub fn split_constraint_set<'a, F, P>(
	constraint_set: ConstraintSet<F>,
	witness: &MultilinearExtensionIndex<'a, P>,
) -> Result<ConstraintsAndMultilinears<'a, F, P>, Error>
where
	F: Field,
	P: PackedField,
	P::Scalar: ExtensionField<F>,
{
	let ConstraintSet {
		oracle_ids,
		constraints,
		n_vars,
	} = constraint_set;

	let multilinears = oracle_ids
		.iter()
		.map(|&oracle_id| witness.get_multilin_poly(oracle_id))
		.collect::<Result<Vec<_>, _>>()?;

	if multilinears
		.iter()
		.any(|multilin| multilin.n_vars() != n_vars)
	{
		bail!(Error::ConstraintSetNumberOfVariablesMismatch);
	}

	Ok((constraints, multilinears))
}

pub struct SumcheckProversWithMetas<'a, P, FDomain, Backend>
where
	P: PackedField,
	FDomain: Field,
	Backend: ComputationBackend,
{
	pub provers: Vec<OracleSumcheckProver<'a, FDomain, P, Backend>>,
	pub metas: Vec<OracleClaimMeta>,
}

/// Constructs sumcheck provers and metas from the vector of [`ConstraintSet`]
pub fn constraint_sets_sumcheck_provers_metas<'a, P, FDomain, Backend>(
	evaluation_order: EvaluationOrder,
	constraint_sets: Vec<ConstraintSet<P::Scalar>>,
	witness: &MultilinearExtensionIndex<'a, P>,
	evaluation_domain_factory: impl EvaluationDomainFactory<FDomain>,
	switchover_fn: impl Fn(usize) -> usize,
	backend: &'a Backend,
) -> Result<SumcheckProversWithMetas<'a, P, FDomain, Backend>, Error>
where
	P: PackedExtension<FDomain>,
	P::Scalar: TowerField + ExtensionField<FDomain>,
	FDomain: Field,
	Backend: ComputationBackend,
{
	let mut provers = Vec::with_capacity(constraint_sets.len());
	let mut metas = Vec::with_capacity(constraint_sets.len());

	for constraint_set in constraint_sets {
		let (_, meta) = constraint_set_sumcheck_claim(constraint_set.clone())?;
		let prover = constraint_set_sumcheck_prover(
			evaluation_order,
			constraint_set,
			witness,
			evaluation_domain_factory.clone(),
			&switchover_fn,
			backend,
		)?;
		metas.push(meta);
		provers.push(prover);
	}
	Ok(SumcheckProversWithMetas { provers, metas })
}

pub struct MLECheckProverWithMeta<'a, P, FDomain, Backend>
where
	P: PackedField,
	FDomain: Field,
	Backend: ComputationBackend,
{
	pub prover: OracleMLECheckProver<'a, FDomain, P, Backend>,
	pub meta: OracleClaimMeta,
}

/// Constructs sumcheck provers and metas from the vector of [`ConstraintSet`]
#[allow(clippy::too_many_arguments)]
pub fn constraint_sets_mlecheck_prover_meta<'a, 'b, P, FDomain, Backend>(
	evaluation_order: EvaluationOrder,
	constraint_set: ConstraintSet<P::Scalar>,
	eq_ind_challenges: EvalPoint<P::Scalar>,
	memoized_data: &mut MemoizedData<'b, P>,
	witness: &MultilinearExtensionIndex<'a, P>,
	evaluation_domain_factory: impl EvaluationDomainFactory<FDomain>,
	switchover_fn: impl Fn(usize) -> usize,
	backend: &'a Backend,
) -> Result<MLECheckProverWithMeta<'a, P, FDomain, Backend>, Error>
where
	P: PackedExtension<FDomain>,
	P::Scalar: TowerField + ExtensionField<FDomain>,
	FDomain: Field,
	Backend: ComputationBackend,
{
	let (_, meta) = constraint_set_mlecheck_claim(constraint_set.clone())?;
	let prover = constraint_set_mlecheck_prover(
		evaluation_order,
		constraint_set,
		&eq_ind_challenges,
		memoized_data,
		witness,
		evaluation_domain_factory,
		&switchover_fn,
		backend,
	)?;

	Ok(MLECheckProverWithMeta { prover, meta })
}
