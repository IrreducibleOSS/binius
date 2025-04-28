// Copyright 2024-2025 Irreducible Inc.

use binius_field::{ExtensionField, Field, PackedExtension, PackedField, TowerField};
use binius_hal::ComputationBackend;
use binius_math::{EvaluationDomainFactory, EvaluationOrder};
use binius_utils::bail;

use super::{RegularSumcheckProver, ZerocheckProverImpl};
use crate::{
	oracle::{Constraint, ConstraintPredicate, ConstraintSet},
	polynomial::ArithCircuitPoly,
	protocols::sumcheck::{
		constraint_set_sumcheck_claim, CompositeSumClaim, Error, OracleClaimMeta,
	},
	witness::{MultilinearExtensionIndex, MultilinearWitness},
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

/// Construct zerocheck prover from the constraint set. Fails when constraint set contains regular sumchecks.
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
					ArithCircuitPoly::with_n_vars_circuit(multilinears.len(), composition_base)?,
					ArithCircuitPoly::with_n_vars_circuit(multilinears.len(), composition)?,
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

/// Construct regular sumcheck prover from the constraint set. Fails when constraint set contains zerochecks.
pub fn constraint_set_sumcheck_prover<'a, FW, PW, FDomain, Backend>(
	evaluation_order: EvaluationOrder,
	constraint_set: ConstraintSet<FW>,
	witness: &MultilinearExtensionIndex<'a, PW>,
	evaluation_domain_factory: impl EvaluationDomainFactory<FDomain>,
	switchover_fn: impl Fn(usize) -> usize + Clone,
	backend: &'a Backend,
) -> Result<OracleSumcheckProver<'a, FDomain, PW, Backend>, Error>
where
	PW: PackedField<Scalar = FW> + PackedExtension<FDomain>,
	FW: TowerField + ExtensionField<FDomain>,
	FDomain: Field,
	Backend: ComputationBackend,
{
	let (constraints, multilinears) = split_constraint_set::<FW, PW>(constraint_set, witness)?;

	let mut sums = Vec::new();

	for Constraint {
		composition,
		predicate,
		..
	} in constraints
	{
		match predicate {
			ConstraintPredicate::Sum(sum) => sums.push(CompositeSumClaim {
				composition: ArithCircuitPoly::with_n_vars_circuit(
					multilinears.len(),
					composition,
				)?,
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

type ConstraintsAndMultilinears<'a, F, PW> = (Vec<Constraint<F>>, Vec<MultilinearWitness<'a, PW>>);

#[allow(clippy::type_complexity)]
pub fn split_constraint_set<'a, F, PW>(
	constraint_set: ConstraintSet<F>,
	witness: &MultilinearExtensionIndex<'a, PW>,
) -> Result<ConstraintsAndMultilinears<'a, F, PW>, Error>
where
	F: Field,
	PW: PackedField,
	PW::Scalar: ExtensionField<F>,
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

pub struct SumcheckProversWithMetas<'a, PW, FDomain, Backend>
where
	PW: PackedField,
	FDomain: Field,
	Backend: ComputationBackend,
{
	pub provers: Vec<OracleSumcheckProver<'a, FDomain, PW, Backend>>,
	pub metas: Vec<OracleClaimMeta>,
}

/// Constructs sumcheck provers and metas from the vector of [`ConstraintSet`]
pub fn constraint_sets_sumcheck_provers_metas<'a, PW, FDomain, Backend>(
	evaluation_order: EvaluationOrder,
	constraint_sets: Vec<ConstraintSet<PW::Scalar>>,
	witness: &MultilinearExtensionIndex<'a, PW>,
	evaluation_domain_factory: impl EvaluationDomainFactory<FDomain>,
	switchover_fn: impl Fn(usize) -> usize,
	backend: &'a Backend,
) -> Result<SumcheckProversWithMetas<'a, PW, FDomain, Backend>, Error>
where
	PW: PackedExtension<FDomain>,
	PW::Scalar: TowerField + ExtensionField<FDomain>,
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
