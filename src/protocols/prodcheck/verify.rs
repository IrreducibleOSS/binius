// Copyright 2024 Ulvetanna Inc.

use crate::{field::Field, oracle::MultilinearPolyOracle};

use super::{
	error::Error,
	prodcheck::{reduce_prodcheck_claim, ProdcheckClaim, ReducedProductCheckClaims},
};

/// Verify a product check instance reduction.
pub fn verify<F>(
	claim: &ProdcheckClaim<F>,
	grand_prod_oracle: MultilinearPolyOracle<F>,
) -> Result<ReducedProductCheckClaims<F>, Error>
where
	F: Field,
{
	reduce_prodcheck_claim(claim, grand_prod_oracle)
}
