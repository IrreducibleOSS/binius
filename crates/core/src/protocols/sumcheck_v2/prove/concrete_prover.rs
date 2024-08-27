// Copyright 2024 Ulvetanna Inc.

use super::{batch_prove::SumcheckProver, RegularSumcheckProver, ZerocheckProver};
use crate::protocols::sumcheck_v2::{common::RoundCoeffs, error::Error};
use binius_field::{ExtensionField, Field, PackedExtension, PackedField, PackedFieldIndexable};
use binius_math::polynomial::{CompositionPoly, MultilinearPoly};

/// A sum type that is used to put both regular sumchecks and zerochecks into the same `batch_prove` call.
pub enum ConcreteProver<FDomain, P, Composition, M>
where
	FDomain: Field,
	P: PackedField,
	M: MultilinearPoly<P> + Send + Sync,
{
	Sumcheck(RegularSumcheckProver<FDomain, P, Composition, M>),
	Zerocheck(ZerocheckProver<FDomain, P, Composition, M>),
}

impl<F, FDomain, P, Composition, M> SumcheckProver<F> for ConcreteProver<FDomain, P, Composition, M>
where
	F: Field + ExtensionField<FDomain>,
	FDomain: Field,
	P: PackedField<Scalar = F> + PackedExtension<FDomain> + PackedFieldIndexable,
	Composition: CompositionPoly<P>,
	M: MultilinearPoly<P> + Send + Sync,
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
