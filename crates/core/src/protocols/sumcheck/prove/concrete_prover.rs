// Copyright 2024 Irreducible Inc.

use super::{batch_prove::SumcheckProver, RegularSumcheckProver, ZerocheckProver};
use crate::protocols::sumcheck::{common::RoundCoeffs, error::Error};
use binius_field::{
	ExtensionField, Field, PackedExtension, PackedField, PackedFieldIndexable, RepackedExtension,
};
use binius_hal::ComputationBackend;
use binius_math::{CompositionPoly, MultilinearPoly};

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

impl<'a, F, FDomain, PBase, P, CompositionBase, Composition, M, Backend> SumcheckProver<F>
	for ConcreteProver<'a, FDomain, PBase, P, CompositionBase, Composition, M, Backend>
where
	F: Field + ExtensionField<PBase::Scalar> + ExtensionField<FDomain>,
	FDomain: Field,
	PBase: PackedField<Scalar: ExtensionField<FDomain>> + PackedExtension<FDomain>,
	P: PackedFieldIndexable<Scalar = F> + PackedExtension<FDomain> + RepackedExtension<PBase>,
	CompositionBase: CompositionPoly<PBase>,
	Composition: CompositionPoly<P>,
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

	fn finish(self) -> Result<Vec<F>, Error> {
		match self {
			ConcreteProver::Sumcheck(prover) => prover.finish(),
			ConcreteProver::Zerocheck(prover) => prover.finish(),
		}
	}
}
