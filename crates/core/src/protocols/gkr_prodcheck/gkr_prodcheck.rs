// Copyright 2024 Ulvetanna Inc.

use crate::{
	oracle::MultilinearPolyOracle,
	protocols::gkr_gpa::{GrandProductClaim, GrandProductWitness},
	witness::MultilinearWitness,
};
use binius_field::{Field, PackedField};

#[derive(Debug, Clone)]
pub struct ProdcheckClaim<F: Field> {
	/// Oracle to the multilinear polynomial T
	pub t_oracle: MultilinearPolyOracle<F>,
	/// Oracle to the multilinear polynomial U
	pub u_oracle: MultilinearPolyOracle<F>,
}

#[derive(Debug, Clone)]
pub struct ProdcheckWitness<'a, PW: PackedField> {
	pub t_poly: MultilinearWitness<'a, PW>,
	pub u_poly: MultilinearWitness<'a, PW>,
}

#[derive(Debug, Default)]
pub struct ProdcheckBatchProof<F: Field> {
	pub products: Vec<F>,
}

#[derive(Debug, Default)]
pub struct ProdcheckBatchProveOutput<'a, F, PW>
where
	F: Field,
	PW: PackedField,
	PW::Scalar: Field + From<F> + Into<F>,
{
	pub reduced_witnesses: Vec<GrandProductWitness<'a, PW>>,
	pub reduced_claims: Vec<GrandProductClaim<F>>,
	pub batch_proof: ProdcheckBatchProof<F>,
}
