// Copyright 2024 Ulvetanna Inc.

use crate::{oracle::Error as OracleError, protocols::gkr_gpa::Error as GrandProductError};
use binius_math::polynomial::Error as PolynomialError;

#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error("claim is inconsistent, T and U have different number of variables")]
	InconsistentClaim,
	#[error("witneses and claims have mismatched lengths")]
	MismatchedWitnessClaimLength,
	#[error("witness is inconsistent, T and U have different number of variables")]
	InconsistentWitness,
	#[error("claim and witness are inconsistent")]
	InconsistentClaimWitness,
	#[error("invalid instance - the products of T and U polynomials differ")]
	ProductsDiffer,
	#[error("oracle error: {0}")]
	Oracle(#[from] OracleError),
	#[error("polynomial error: {0}")]
	Polynomial(#[from] PolynomialError),
	#[error("verification error: {0}")]
	Verification(#[from] VerificationError),
	#[error("gkr-based grand product failure: {0}")]
	GrandProductError(#[from] GrandProductError),
}

#[derive(Debug, thiserror::Error)]
pub enum VerificationError {
	#[error("The batch proof is not consistent with the number of batch claims")]
	MismatchedClaimsAndBatchProof,
}
