// Copyright 2024 Irreducible Inc.

use crate::{
	oracle::{BatchId, MultilinearOracleSet, MultilinearPolyOracle, OracleId},
	protocols::{
		gkr_gpa::{GrandProductClaim, GrandProductWitness},
		lasso::error::Error,
	},
	witness::{MultilinearExtensionIndex, MultilinearWitness},
};
use binius_field::{
	as_packed_field::{PackScalar, PackedType},
	underlier::UnderlierType,
	ExtensionField, Field, PackedField, TowerField,
};

use crate::transparent::constant::Constant;
use binius_utils::bail;
use getset::Getters;
use itertools::izip;

pub struct LassoBatches {
	pub counts_batch_ids: Vec<BatchId>,
	pub final_counts_batch_id: BatchId,

	pub counts: Vec<OracleId>,
	pub final_counts: OracleId,
}

impl LassoBatches {
	pub fn new_in<C: TowerField, F: TowerField>(
		oracles: &mut MultilinearOracleSet<F>,
		u_n_vars: &[usize],
		lookup_t_n_vars: usize,
	) -> Self {
		let (counts_batch_ids, counts): (Vec<_>, Vec<_>) = u_n_vars
			.iter()
			.map(|u_n_vars| {
				let counts_batch_id = oracles.add_committed_batch(*u_n_vars, C::TOWER_LEVEL);
				(counts_batch_id, oracles.add_committed(counts_batch_id))
			})
			.unzip();

		let final_counts_batch_id = oracles.add_committed_batch(lookup_t_n_vars, C::TOWER_LEVEL);
		let final_counts = oracles.add_committed(final_counts_batch_id);
		Self {
			counts_batch_ids,
			final_counts_batch_id,
			counts,
			final_counts,
		}
	}
}

#[derive(Debug, Getters)]
pub struct LassoClaim<F: Field> {
	/// T polynomial - the table being "looked up"
	#[get = "pub"]
	t_oracle: MultilinearPolyOracle<F>,
	/// U polynomials - each element of U must equal some element of T
	#[get = "pub"]
	u_oracles: Vec<MultilinearPolyOracle<F>>,
}

impl<F: Field> LassoClaim<F> {
	pub fn new(
		t_oracle: MultilinearPolyOracle<F>,
		u_oracles: Vec<MultilinearPolyOracle<F>>,
	) -> Result<Self, Error> {
		Ok(Self {
			t_oracle,
			u_oracles,
		})
	}
}

#[derive(Debug, Getters)]
pub struct LassoWitness<'a, PW: PackedField, L: AsRef<[usize]>> {
	#[get = "pub"]
	t_polynomial: MultilinearWitness<'a, PW>,
	#[get = "pub"]
	u_polynomials: Vec<MultilinearWitness<'a, PW>>,
	#[get = "pub"]
	u_to_t_mappings: Vec<L>,
}

impl<'a, PW: PackedField, L: AsRef<[usize]>> LassoWitness<'a, PW, L> {
	pub fn new(
		t_polynomial: MultilinearWitness<'a, PW>,
		u_polynomials: Vec<MultilinearWitness<'a, PW>>,
		u_to_t_mappings: Vec<L>,
	) -> Result<Self, Error> {
		if u_polynomials.len() != u_to_t_mappings.len() {
			bail!(Error::MappingsLookerTablesLenMismatch);
		}

		if u_polynomials
			.iter()
			.zip(&u_to_t_mappings)
			.any(|(poly, mapping)| poly.size() != mapping.as_ref().len())
		{
			bail!(Error::ClaimWitnessTablesLenMismatch);
		}

		let size = t_polynomial.size();

		if u_to_t_mappings
			.iter()
			.flat_map(|mapping| mapping.as_ref())
			.any(|&index| index >= size)
		{
			bail!(Error::MappingIndexOutOfBounds);
		}

		Ok(Self {
			t_polynomial,
			u_polynomials,
			u_to_t_mappings,
		})
	}
}

#[derive(Debug, Default)]
pub struct LassoProof<F: Field> {
	pub left_grand_products: Vec<F>,
	pub right_grand_products: Vec<F>,
	pub counts_grand_products: Vec<F>,
}

pub struct LassoProveOutput<'a, U: UnderlierType + PackScalar<F>, F: Field> {
	pub reduced_gpa_claims: Vec<GrandProductClaim<F>>,
	pub reduced_gpa_witnesses: Vec<GrandProductWitness<'a, PackedType<U, F>>>,
	pub gpa_metas: Vec<OracleId>,
	pub lasso_proof: LassoProof<F>,
	pub witness_index: MultilinearExtensionIndex<'a, U, F>,
}

pub struct LassoReducedClaimOracleIds {
	pub ones_oracle_id: OracleId,
	pub mixed_t_final_counts_oracle_id: OracleId,
	pub mixed_t_one_oracle_id: OracleId,
	pub mixed_u_counts_oracle_ids: Vec<OracleId>,
	pub mixed_u_counts_plus_one_oracle_ids: Vec<OracleId>,
}

#[derive(Debug, Default)]
pub struct GkrClaimOracleIds {
	pub left: Vec<OracleId>,
	pub right: Vec<OracleId>,
	pub counts: Vec<OracleId>,
}

pub fn reduce_lasso_claim<C: TowerField, F: TowerField + ExtensionField<C> + From<C>>(
	oracles: &mut MultilinearOracleSet<F>,
	lasso_claim: &LassoClaim<F>,
	lasso_batches: &LassoBatches,
	gamma: F,
	alpha: F,
) -> Result<(GkrClaimOracleIds, LassoReducedClaimOracleIds), Error> {
	let t_n_vars = lasso_claim.t_oracle.n_vars();

	let final_counts_oracle = oracles.oracle(lasso_batches.final_counts);

	if final_counts_oracle.n_vars() != t_n_vars {
		bail!(Error::CountsNumVariablesMismatch);
	}

	let alpha_gen = alpha * C::MULTIPLICATIVE_GENERATOR;

	let mut mixed_u_counts_oracle_ids = Vec::new();
	let mut mixed_u_counts_plus_one_oracle_ids = Vec::new();

	let mut gkr_claim_oracle_ids = GkrClaimOracleIds::default();

	for (counts_oracle_id, u_oracle) in izip!(&lasso_batches.counts, &lasso_claim.u_oracles) {
		let u_n_vars = u_oracle.n_vars();

		let counts_oracle = oracles.oracle(*counts_oracle_id);

		if counts_oracle.n_vars() != u_n_vars {
			bail!(Error::CountsNumVariablesMismatch);
		}

		let mixed_u_counts_oracle_id = oracles.add_linear_combination_with_offset(
			u_n_vars,
			gamma,
			[(u_oracle.id(), F::ONE), (*counts_oracle_id, alpha)],
		)?;

		mixed_u_counts_oracle_ids.push(mixed_u_counts_oracle_id);

		let mixed_u_counts_plus_one_oracle_id = oracles.add_linear_combination_with_offset(
			u_n_vars,
			gamma,
			[(u_oracle.id(), F::ONE), (*counts_oracle_id, alpha_gen)],
		)?;

		mixed_u_counts_plus_one_oracle_ids.push(mixed_u_counts_plus_one_oracle_id);

		gkr_claim_oracle_ids.left.push(mixed_u_counts_oracle_id);
		gkr_claim_oracle_ids
			.right
			.push(mixed_u_counts_plus_one_oracle_id);
		gkr_claim_oracle_ids.counts.push(*counts_oracle_id);
	}

	let ones_oracle_id = oracles.add_transparent(Constant {
		n_vars: t_n_vars,
		value: F::ONE,
	})?;

	let mixed_t_final_counts_oracle_id = oracles.add_linear_combination_with_offset(
		t_n_vars,
		gamma,
		[
			(lasso_claim.t_oracle.id(), F::ONE),
			(lasso_batches.final_counts, alpha),
		],
	)?;

	let mixed_t_one_oracle_id = oracles.add_linear_combination_with_offset(
		t_n_vars,
		gamma,
		[(lasso_claim.t_oracle.id(), F::ONE), (ones_oracle_id, alpha)],
	)?;

	let lasso_claim_oracles = LassoReducedClaimOracleIds {
		ones_oracle_id,
		mixed_t_final_counts_oracle_id,
		mixed_t_one_oracle_id,
		mixed_u_counts_oracle_ids,
		mixed_u_counts_plus_one_oracle_ids,
	};

	gkr_claim_oracle_ids
		.left
		.push(mixed_t_final_counts_oracle_id);
	gkr_claim_oracle_ids.right.push(mixed_t_one_oracle_id);
	gkr_claim_oracle_ids.counts.push(lasso_batches.final_counts);

	Ok((gkr_claim_oracle_ids, lasso_claim_oracles))
}
