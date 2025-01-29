// Copyright 2024-2025 Irreducible Inc.

use binius_field::{ExtensionField, Field, PackedExtension, PackedField, PackedFieldIndexable};
use binius_hal::ComputationBackend;
use binius_math::{CompositionPolyOS, MultilinearPoly};

use super::{batch_prove::SumcheckProver, RegularSumcheckProver, ZerocheckProver};
use crate::protocols::sumcheck::{common::RoundCoeffs, error::Error};

/// A sum type that is used to put both regular sumchecks and zerochecks into the same `batch_prove` call.
pub enum ConcreteProver<'a, FDomain, PBase, P, CompositionBase, Composition, M, Backend>
where
	FDomain: Field,
	PBase: PackedField,
	P: PackedField,
	M: MultilinearPoly<P> + Send + Sync,
	Backend: ComputationBackend,
{
	Sumcheck(RegularSumcheckProver<'a, FDomain, P, Composition, M, Backend>),
	Zerocheck(ZerocheckProver<'a, FDomain, PBase, P, CompositionBase, Composition, M, Backend>),
}

impl<F, FDomain, FBase, P, CompositionBase, Composition, M, Backend> SumcheckProver<F>
	for ConcreteProver<'_, FDomain, FBase, P, CompositionBase, Composition, M, Backend>
where
	F: Field + ExtensionField<FBase> + ExtensionField<FDomain>,
	FDomain: Field,
	FBase: ExtensionField<FDomain>,
	P: PackedFieldIndexable<Scalar = F>
		+ PackedExtension<F, PackedSubfield = P>
		+ PackedExtension<FDomain>
		+ PackedExtension<FBase>,
	CompositionBase: CompositionPolyOS<<P as PackedExtension<FBase>>::PackedSubfield>,
	Composition: CompositionPolyOS<P>,
	M: MultilinearPoly<P> + Send + Sync,
	Backend: ComputationBackend,
{
	fn n_vars(&self) -> usize {
		match self {
			ConcreteProver::Sumcheck(prover) => prover.n_vars(),
			ConcreteProver::Zerocheck(prover) => prover.n_vars(),
		}
	}

	fn execute(&mut self, batch_coeff: F) -> Result<RoundCoeffs<F>, Error> {
		match self {
			ConcreteProver::Sumcheck(prover) => prover.execute(batch_coeff),
			ConcreteProver::Zerocheck(prover) => prover.execute(batch_coeff),
		}
	}

	fn fold(&mut self, challenge: F) -> Result<(), Error> {
		match self {
			ConcreteProver::Sumcheck(prover) => prover.fold(challenge),
			ConcreteProver::Zerocheck(prover) => prover.fold(challenge),
		}
	}

	fn finish(self: Box<Self>) -> Result<Vec<F>, Error> {
		match *self {
			ConcreteProver::Sumcheck(prover) => Box::new(prover).finish(),
			ConcreteProver::Zerocheck(prover) => Box::new(prover).finish(),
		}
	}
}
