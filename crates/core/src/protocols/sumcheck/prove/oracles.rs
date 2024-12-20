// Copyright 2024 Irreducible Inc.

use binius_field::{
	as_packed_field::{PackScalar, PackedType},
	underlier::UnderlierType,
	ExtensionField, Field, PackedField, PackedFieldIndexable, TowerField,
};
use binius_hal::ComputationBackend;
use binius_math::EvaluationDomainFactory;
use binius_utils::bail;

use super::{RegularSumcheckProver, UnivariateZerocheck};
use crate::{
	oracle::{Constraint, ConstraintPredicate, ConstraintSet},
	polynomial::ArithCircuitPoly,
	protocols::sumcheck::{
		constraint_set_sumcheck_claim, CompositeSumClaim, Error, OracleClaimMeta,
	},
	witness::{MultilinearExtensionIndex, MultilinearWitness},
};

pub type OracleZerocheckProver<'a, FDomain, PBase, P, Backend> = UnivariateZerocheck<
	'a,
	'a,
	FDomain,
	PBase,
	P,
	ArithCircuitPoly<<PBase as PackedField>::Scalar>,
	ArithCircuitPoly<<P as PackedField>::Scalar>,
	MultilinearWitness<'a, P>,
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

/// Construct zerocheck prover from the constraint set. Fails when constraint set contains regular sumchecks.
#[allow(clippy::type_complexity)]
pub fn constraint_set_zerocheck_prover<'a, U, FBase, FW, FDomain, Backend>(
	constraint_set: ConstraintSet<FW>,
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
	FBase: TowerField + ExtensionField<FDomain> + TryFrom<FW>,
	FW: TowerField + ExtensionField<FDomain> + ExtensionField<FBase>,
	FDomain: Field,
	PackedType<U, FW>: PackedFieldIndexable,
	Backend: ComputationBackend,
{
	let (constraints, multilinears) = split_constraint_set::<_, FW, _>(constraint_set, witness)?;

	let mut zeros = Vec::new();

	for Constraint {
		composition,
		predicate,
	} in constraints
	{
		let composition_base = composition
			.clone()
			.try_convert_field()
			.map_err(|_| Error::CircuitFieldDowncastFailed)?;
		match predicate {
			ConstraintPredicate::Zero => {
				zeros.push((
					ArithCircuitPoly::with_n_vars(multilinears.len(), composition_base)?,
					ArithCircuitPoly::with_n_vars(multilinears.len(), composition)?,
				));
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
	constraint_set: ConstraintSet<FW>,
	witness: &MultilinearExtensionIndex<'a, U, FW>,
	evaluation_domain_factory: impl EvaluationDomainFactory<FDomain>,
	switchover_fn: impl Fn(usize) -> usize + Clone,
	backend: &'a Backend,
) -> Result<OracleSumcheckProver<'a, FDomain, PackedType<U, FW>, Backend>, Error>
where
	U: UnderlierType + PackScalar<FW> + PackScalar<FDomain>,
	FW: TowerField + ExtensionField<FDomain>,
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
			ConstraintPredicate::Sum(sum) => sums.push(CompositeSumClaim {
				composition: ArithCircuitPoly::with_n_vars(multilinears.len(), composition)?,
				sum,
			}),
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

type ConstraintsAndMultilinears<'a, F, PW> = (Vec<Constraint<F>>, Vec<MultilinearWitness<'a, PW>>);

#[allow(clippy::type_complexity)]
fn split_constraint_set<'a, U, F, FW>(
	constraint_set: ConstraintSet<F>,
	witness: &MultilinearExtensionIndex<'a, U, FW>,
) -> Result<ConstraintsAndMultilinears<'a, F, PackedType<U, FW>>, Error>
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
	constraint_sets: Vec<ConstraintSet<FW>>,
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
