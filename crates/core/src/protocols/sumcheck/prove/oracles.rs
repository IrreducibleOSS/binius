// Copyright 2024 Irreducible Inc.

use super::{RegularSumcheckProver, UnivariateZerocheck};
use crate::{
	oracle::{Constraint, ConstraintPredicate, ConstraintSet, TypeErasedComposition},
	protocols::sumcheck::{
		constraint_set_sumcheck_claim, CompositeSumClaim, Error, OracleClaimMeta,
	},
	witness::{MultilinearExtensionIndex, MultilinearWitness},
};
use binius_field::{
	as_packed_field::{PackScalar, PackedType},
	underlier::UnderlierType,
	ExtensionField, Field, PackedFieldIndexable, TowerField,
};
use binius_hal::ComputationBackend;
use binius_math::EvaluationDomainFactory;
use binius_utils::bail;
use itertools::izip;
use std::sync::Arc;

pub type OracleZerocheckProver<'a, FDomain, PBase, P, Backend> = UnivariateZerocheck<
	'a,
	FDomain,
	PBase,
	P,
	TypeErasedComposition<PBase>,
	TypeErasedComposition<P>,
	MultilinearWitness<'a, P>,
	Backend,
>;

pub type OracleSumcheckProver<'a, FDomain, P, Backend> = RegularSumcheckProver<
	'a,
	FDomain,
	P,
	TypeErasedComposition<P>,
	MultilinearWitness<'a, P>,
	Backend,
>;

/// Construct zerocheck prover from the constraint set. Fails when constraint set contains regular sumchecks.
#[allow(clippy::type_complexity)]
pub fn constraint_set_zerocheck_prover<'a, U, FBase, FW, FDomain, Backend>(
	constraint_set_base: ConstraintSet<PackedType<U, FBase>>,
	constraint_set: ConstraintSet<PackedType<U, FW>>,
	witness: &MultilinearExtensionIndex<'a, U, FW>,
	evaluation_domain_factory: impl EvaluationDomainFactory<FDomain>,
	switchover_fn: impl Fn(usize) -> usize + Clone,
	zerocheck_challenges: &[FW],
	backend: &'a Backend,
) -> Result<
	OracleZerocheckProver<'a, FDomain, PackedType<U, FBase>, PackedType<U, FW>, Backend>,
	Error,
>
where
	U: UnderlierType + PackScalar<FBase> + PackScalar<FW> + PackScalar<FDomain>,
	FBase: ExtensionField<FDomain>,
	FW: ExtensionField<FDomain> + ExtensionField<FBase>,
	FDomain: Field,
	PackedType<U, FW>: PackedFieldIndexable,
	Backend: ComputationBackend,
{
	let (constraints_base, multilinears_base) =
		split_constraint_set::<_, FBase, _>(constraint_set_base, witness)?;
	let (constraints, multilinears) = split_constraint_set::<_, FW, _>(constraint_set, witness)?;

	if izip!(&multilinears, multilinears_base)
		.any(|(multilinear, multilinear_base)| !Arc::ptr_eq(multilinear, &multilinear_base))
	{
		bail!(Error::BaseAndExtensionFieldConstraintSetsMismatch);
	}

	let mut zeros = Vec::new();

	for (
		Constraint {
			composition,
			predicate,
		},
		Constraint {
			composition: composition_base,
			predicate: predicate_base,
		},
	) in izip!(constraints, constraints_base)
	{
		match (predicate, predicate_base) {
			(ConstraintPredicate::Zero, ConstraintPredicate::Zero) => {
				zeros.push((composition_base, composition));
			}
			_ => bail!(Error::MixedBatchingNotSupported),
		}
	}

	let prover = UnivariateZerocheck::new(
		multilinears,
		zeros,
		zerocheck_challenges,
		evaluation_domain_factory,
		switchover_fn,
		backend,
	)?;

	Ok(prover)
}

/// Construct regular sumcheck prover from the constraint set. Fails when constraint set contains zerochecks.
pub fn constraint_set_sumcheck_prover<'a, U, FW, FDomain, Backend>(
	constraint_set: ConstraintSet<PackedType<U, FW>>,
	witness: &MultilinearExtensionIndex<'a, U, FW>,
	evaluation_domain_factory: impl EvaluationDomainFactory<FDomain>,
	switchover_fn: impl Fn(usize) -> usize + Clone,
	backend: &'a Backend,
) -> Result<OracleSumcheckProver<'a, FDomain, PackedType<U, FW>, Backend>, Error>
where
	U: UnderlierType + PackScalar<FW> + PackScalar<FDomain>,
	FW: ExtensionField<FDomain>,
	FDomain: Field,
	Backend: ComputationBackend,
{
	let (constraints, multilinears) = split_constraint_set::<_, FW, _>(constraint_set, witness)?;

	let mut sums = Vec::new();

	for Constraint {
		composition,
		predicate,
	} in constraints
	{
		match predicate {
			ConstraintPredicate::Sum(sum) => sums.push(CompositeSumClaim { composition, sum }),
			_ => bail!(Error::MixedBatchingNotSupported),
		}
	}

	let prover = RegularSumcheckProver::new(
		multilinears,
		sums,
		evaluation_domain_factory,
		switchover_fn,
		backend,
	)?;

	Ok(prover)
}

type ConstraintsAndMultilinears<'a, P, PW> = (Vec<Constraint<P>>, Vec<MultilinearWitness<'a, PW>>);

#[allow(clippy::type_complexity)]
fn split_constraint_set<'a, U, F, FW>(
	constraint_set: ConstraintSet<PackedType<U, F>>,
	witness: &MultilinearExtensionIndex<'a, U, FW>,
) -> Result<ConstraintsAndMultilinears<'a, PackedType<U, F>, PackedType<U, FW>>, Error>
where
	U: UnderlierType + PackScalar<F> + PackScalar<FW>,
	F: Field,
	FW: ExtensionField<F>,
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

pub struct SumcheckProversWithMetas<'a, U, FW, FDomain, Backend>
where
	U: UnderlierType + PackScalar<FW>,
	FW: TowerField,
	FDomain: Field,
	Backend: ComputationBackend,
{
	pub provers: Vec<OracleSumcheckProver<'a, FDomain, PackedType<U, FW>, Backend>>,
	pub metas: Vec<OracleClaimMeta>,
}

/// Constructs sumcheck provers and metas from the vector of [`ConstraintSet`]
pub fn constraint_sets_sumcheck_provers_metas<'a, U, FW, FDomain, Backend>(
	constraint_sets: Vec<ConstraintSet<PackedType<U, FW>>>,
	witness: &MultilinearExtensionIndex<'a, U, FW>,
	evaluation_domain_factory: impl EvaluationDomainFactory<FDomain>,
	switchover_fn: impl Fn(usize) -> usize,
	backend: &'a Backend,
) -> Result<SumcheckProversWithMetas<'a, U, FW, FDomain, Backend>, Error>
where
	U: UnderlierType + PackScalar<FW> + PackScalar<FDomain>,
	FW: TowerField + ExtensionField<FDomain>,
	FDomain: Field,
	Backend: ComputationBackend,
{
	let mut provers = Vec::with_capacity(constraint_sets.len());
	let mut metas = Vec::with_capacity(constraint_sets.len());

	for constraint_set in constraint_sets {
		let (_, meta) = constraint_set_sumcheck_claim(constraint_set.clone())?;
		let prover = constraint_set_sumcheck_prover(
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
