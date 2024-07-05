// Copyright 2024 Ulvetanna Inc.

use crate::{
	oracle::Error as OracleError, polynomial::Error as PolynomialError,
	protocols::msetcheck::Error as MsetcheckError, witness::Error as WitnessError,
};

#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error("the number of variables in read count oracles is incorrect")]
	CountsNumVariablesMismatch,
	#[error("the number of variables in multilinear oracles does not match")]
	NumVariablesMismatch,
	#[error("witnesses smaller than the underlier are not supported")]
	WitnessSmallerThanUnderlier,
	#[error("provided mapping length does not conform to multilinear witnesses size")]
	MappingSizeMismatch,
	#[error("provided mapping indexes out of T bounds")]
	MappingIndexOutOfBounds,
	#[error("the number of variables in some witness multilinear does not match the claim")]
	WitnessNumVariablesMismatch,
	#[error("the number of variables in one of the merged polynomials does not match the other")]
	MergedWitnessNumVariablesMismatch,
	#[error("actual Lasso counts may not fit into the chosen count integer type")]
	LassoCountTypeTooSmall,
	#[error("oracle error: {0}")]
	Oracle(#[from] OracleError),
	#[error("polynomial error: {0}")]
	Polynomial(#[from] PolynomialError),
	#[error("witness error: {0}")]
	Witness(#[from] WitnessError),
	#[error("multiset check error: {0}")]
	Msetcheck(#[from] MsetcheckError),
}
