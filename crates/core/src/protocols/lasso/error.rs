// Copyright 2024 Irreducible Inc.

use crate::{
	oracle::Error as OracleError, polynomial::Error as PolynomialError,
	protocols::gkr_gpa::Error as GrandProductError, witness::Error as WitnessError,
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
	#[error("actual Lasso counts may not fit into the chosen count integer type")]
	LassoCountTypeTooSmall,
	#[error("oracle error: {0}")]
	Oracle(#[from] OracleError),
	#[error("polynomial error: {0}")]
	Polynomial(#[from] PolynomialError),
	#[error("witness error: {0}")]
	Witness(#[from] WitnessError),
	#[error("vectors of looker tables and u_to_t_mappings must have the same size")]
	MappingsLookerTablesLenMismatch,
	#[error("claim and witness must have the same amount of u_oracles and u_polynomials")]
	ClaimWitnessTablesLenMismatch,
	#[error("gkr-based grand product failure: {0}")]
	GrandProductError(#[from] GrandProductError),
	#[error("invalid instance - the products of T and U polynomials differ")]
	ProductsDiffer,
	#[error("lasso counts contain zeros")]
	ZeroCounts,
	#[error("grand_products arrays have different len")]
	ProductsArraysLenMismatch,
	#[error("grand_products and claims arrays have different len")]
	ProductsClaimsArraysLenMismatch,
}
