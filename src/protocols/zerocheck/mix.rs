// Copyright 2024 Ulvetanna Inc.

use crate::{
	challenger::CanSample,
	field::TowerField,
	iopoly::{CompositePolyOracle, MultilinearPolyOracle, MultivariatePolyOracle},
	polynomial::{MultilinearComposite, MultilinearPoly},
	protocols::{
		sumcheck::MixComposition,
		zerocheck::{
			zerocheck::{ZerocheckClaim, ZerocheckWitness},
			Error,
		},
	},
};
use std::sync::Arc;

/// Securely mix several zerocheck claims into a single claim.
///
/// Given a group of zerocheck claims, construct a new composition polynomial for a zerocheck
/// constraint that is zero if and only if all the underlying constraints are zero over of the
/// hypercube. The mix composition polynomial takes a linear combination of the input constraints,
/// using a combination of tower basis elements and verifier challenges as coefficients.
pub fn mix_claims<'a, F: TowerField, CH>(
	n_vars: usize,
	multilinears: Vec<MultilinearPolyOracle<F>>,
	constraints: impl Iterator<Item = &'a ZerocheckClaim<F>>,
	challenger: CH,
) -> Result<ZerocheckClaim<F>, Error>
where
	CH: CanSample<F>,
{
	let composition =
		MixComposition::new(&multilinears, constraints.map(|claim| &claim.poly), challenger)?;
	Ok(ZerocheckClaim {
		poly: MultivariatePolyOracle::Composite(CompositePolyOracle::new(
			n_vars,
			multilinears,
			Arc::new(composition),
		)?),
	})
}

/// Construct the zerocheck witness for a mixed zerocheck claim.
pub fn mix_witness<'a, F: TowerField>(
	claim: ZerocheckClaim<F>,
	multilinears: Vec<Arc<dyn MultilinearPoly<F> + Send + Sync + 'a>>,
) -> Result<ZerocheckWitness<'a, F>, Error> {
	let composite = claim.poly.into_composite();
	let witness =
		MultilinearComposite::new(composite.n_vars(), composite.composition(), multilinears)?;
	Ok(witness)
}
