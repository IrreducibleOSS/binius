// Copyright 2024 Ulvetanna Inc.

use super::error::Error;
use crate::{
	oracle::{
		BatchId, CompositePolyOracle, MultilinearOracleSet, MultilinearPolyOracle, OracleId,
		ShiftVariant,
	},
	polynomial::{transparent::step_down::StepDown, CompositionPoly, Error as PolynomialError},
	protocols::{
		gkr_prodcheck::{ProdcheckClaim, ProdcheckWitness},
		zerocheck::{ZerocheckClaim, ZerocheckWitnessTypeErased},
	},
	witness::{MultilinearExtensionIndex, MultilinearWitness},
};
use binius_field::{
	as_packed_field::{PackScalar, PackedType},
	underlier::UnderlierType,
	BinaryField16b, BinaryField32b, Field, PackedField, TowerField,
};
use binius_utils::bail;
use getset::{CopyGetters, Getters};
use itertools::izip;

pub trait LassoCount: TowerField {
	fn overflowing_add(self, rhs: Self) -> (Self, bool);
	fn shr1(self) -> Self;
}

impl LassoCount for BinaryField32b {
	fn overflowing_add(self, rhs: BinaryField32b) -> (BinaryField32b, bool) {
		let (sum, overflow) = u32::from(self).overflowing_add(u32::from(rhs));
		(BinaryField32b::new(sum), overflow)
	}

	fn shr1(self) -> BinaryField32b {
		BinaryField32b::new(u32::from(self) >> 1)
	}
}

impl LassoCount for BinaryField16b {
	fn overflowing_add(self, rhs: BinaryField16b) -> (BinaryField16b, bool) {
		let (sum, overflow) = u16::from(self).overflowing_add(u16::from(rhs));
		(BinaryField16b::new(sum), overflow)
	}

	fn shr1(self) -> BinaryField16b {
		BinaryField16b::new(u16::from(self) >> 1)
	}
}

#[derive(CopyGetters)]
pub struct LassoBatches {
	#[get_copy = "pub"]
	counts_batch_id: BatchId,
	#[get_copy = "pub"]
	final_counts_batch_id: BatchId,

	pub counts: Vec<OracleId>,
	pub carry_out: Vec<OracleId>,
	pub final_counts: OracleId,
}

impl LassoBatches {
	pub fn new_in<C: LassoCount, F: TowerField>(
		oracles: &mut MultilinearOracleSet<F>,
		u_n_vars: usize,
		lookup_table_n_vars: usize,
		number_u_tables: usize,
	) -> Self {
		let counts_batch_id = oracles.add_committed_batch(u_n_vars + C::TOWER_LEVEL, 0);
		let final_counts_batch_id =
			oracles.add_committed_batch(lookup_table_n_vars + C::TOWER_LEVEL, 0);

		let counts = vec![oracles.add_committed(counts_batch_id); number_u_tables];
		let carry_out = vec![oracles.add_committed(counts_batch_id); number_u_tables];

		let final_counts = oracles.add_committed(final_counts_batch_id);
		Self {
			counts_batch_id,
			final_counts_batch_id,
			counts,
			carry_out,
			final_counts,
		}
	}
}

#[derive(Debug, Getters)]
pub struct LassoClaim<F: Field> {
	/// T polynomial - the table being "looked up"
	#[get = "pub"]
	t_oracle: MultilinearPolyOracle<F>,
	/// U polynomial - each element of U must equal some element of T
	#[get = "pub"]
	u_oracles: Vec<MultilinearPolyOracle<F>>,
}

impl<F: Field> LassoClaim<F> {
	pub fn new(
		t_oracle: MultilinearPolyOracle<F>,
		u_oracles: Vec<MultilinearPolyOracle<F>>,
	) -> Result<Self, Error> {
		if let Some(first) = u_oracles.first() {
			if u_oracles
				.iter()
				.any(|u_oracle| u_oracle.n_vars() != first.n_vars())
			{
				bail!(Error::NumVariablesMismatch);
			}
		}

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

		if let Some(first) = u_polynomials.first() {
			let size = t_polynomial.size();
			if u_polynomials
				.iter()
				.any(|u_polynomial| u_polynomial.n_vars() != first.n_vars())
			{
				bail!(Error::NumVariablesMismatch);
			}

			if u_to_t_mappings
				.iter()
				.any(|mapping| first.size() != mapping.as_ref().len())
			{
				bail!(Error::NumVariablesMismatch);
			}

			if u_to_t_mappings
				.iter()
				.flat_map(|mapping| mapping.as_ref())
				.any(|&index| index >= size)
			{
				bail!(Error::MappingIndexOutOfBounds);
			}
		}

		Ok(Self {
			t_polynomial,
			u_polynomials,
			u_to_t_mappings,
		})
	}
}

/// Composition for unary carry application: f(x, carry_in, carry_out) := x * carry_in - carry_out
#[derive(Clone, Debug)]
pub struct UnaryCarryConstraint;

impl<P: PackedField> CompositionPoly<P> for UnaryCarryConstraint {
	fn n_vars(&self) -> usize {
		3
	}

	fn degree(&self) -> usize {
		2
	}

	fn evaluate(&self, query: &[P]) -> Result<P, PolynomialError> {
		if query.len() != 3 {
			bail!(PolynomialError::IncorrectQuerySize { expected: 3 });
		}

		Ok(query[0] * query[1] - query[2])
	}

	fn binary_tower_level(&self) -> usize {
		0
	}
}

pub struct ReducedLassoClaims<F: Field> {
	pub zerocheck_claims: Vec<ZerocheckClaim<F>>,
	pub prodcheck_claims: Vec<ProdcheckClaim<F>>,
}

pub struct LassoProveOutput<'a, U: UnderlierType + PackScalar<FW>, FW: TowerField, F: Field> {
	pub reduced_lasso_claims: ReducedLassoClaims<F>,
	pub prodcheck_witnesses: Vec<ProdcheckWitness<'a, PackedType<U, FW>>>,
	pub zerocheck_witnesses:
		Vec<ZerocheckWitnessTypeErased<'a, PackedType<U, FW>, UnaryCarryConstraint>>,
	pub witness_index: MultilinearExtensionIndex<'a, U, FW>,
}

pub(super) struct LassoReducedClaimOracleIds {
	pub packed_counts_oracle_id: Vec<OracleId>,
	pub packed_counts_plus_one_oracle_id: Vec<OracleId>,
	pub packed_final_counts_oracle_id: OracleId,
	pub counts_plus_one_oracle_id: Vec<OracleId>,
	pub carry_in_oracle_id: Vec<OracleId>,
	pub carry_out_shifted_oracle_id: Vec<OracleId>,
	pub ones_repeating_oracle_id: Option<OracleId>,
	pub mixed_t_final_counts_oracle_id: OracleId,
	pub mixed_t_zero_oracle_id: OracleId,
	pub mixed_u_counts_oracle_id: Vec<OracleId>,
	pub mixed_u_counts_plus_one_oracle_id: Vec<OracleId>,
}

pub fn reduce_lasso_claim<C: LassoCount, F: TowerField>(
	oracles: &mut MultilinearOracleSet<F>,
	lasso_claim: &LassoClaim<F>,
	lasso_batches: &LassoBatches,
	gamma: F,
	alpha: F,
) -> Result<(ReducedLassoClaims<F>, LassoReducedClaimOracleIds), Error> {
	let t_n_vars = lasso_claim.t_oracle.n_vars();
	let t_n_vars_gf2 = t_n_vars + C::TOWER_LEVEL;

	// Extract Lasso committed column oracles
	let final_counts_oracle = oracles.oracle(lasso_batches.final_counts);

	if final_counts_oracle.n_vars() != t_n_vars_gf2 {
		bail!(Error::CountsNumVariablesMismatch);
	}

	let mut packed_counts_oracle_id = Vec::new();
	let mut packed_counts_plus_one_oracle_id = Vec::new();
	let mut counts_plus_one_oracle_id = Vec::new();
	let mut carry_in_oracle_id = Vec::new();
	let mut carry_out_shifted_oracle_id = Vec::new();
	let mut mixed_u_counts_oracle_id = Vec::new();
	let mut mixed_u_counts_plus_one_oracle_id = Vec::new();

	let mut prodcheck_claims = Vec::new();

	let mut zerocheck_claims = Vec::new();

	let mut ones_repeating_oracle_id = None;

	if !lasso_claim.u_oracles.is_empty() {
		let u_n_vars = lasso_claim.u_oracles[0].n_vars();
		let u_n_vars_gf2 = u_n_vars + C::TOWER_LEVEL;

		let counts_oracle_first = oracles.oracle(lasso_batches.counts[0]);
		let carry_out_oracle_first = oracles.oracle(lasso_batches.carry_out[0]);

		if counts_oracle_first.n_vars() != u_n_vars_gf2
			|| carry_out_oracle_first.n_vars() != u_n_vars_gf2
		{
			bail!(Error::CountsNumVariablesMismatch);
		}

		// Representing a column of ones as a repeating oracle of 10...0
		let one = StepDown::new(C::TOWER_LEVEL, 1)?;
		let one_oracle_id = oracles.add_transparent(one)?;
		ones_repeating_oracle_id = Some(oracles.add_repeating(one_oracle_id, u_n_vars)?);

		for (i, (counts_oracle_id, carry_out_oracle_id, u_oracle)) in izip!(
			lasso_batches.counts.iter(),
			lasso_batches.carry_out.iter(),
			lasso_claim.u_oracles.iter()
		)
		.enumerate()
		{
			// carry_in = ([carry_out] << 1) + 1
			carry_out_shifted_oracle_id.push(oracles.add_shifted(
				*carry_out_oracle_id,
				1,
				C::TOWER_LEVEL,
				ShiftVariant::LogicalLeft,
			)?);

			carry_in_oracle_id.push(
				oracles.add_linear_combination(
					u_n_vars_gf2,
					[
						(carry_out_shifted_oracle_id[i], F::ONE),
						(
							ones_repeating_oracle_id
								.expect("ones_repeating_oracle_id was created above"),
							F::ONE,
						),
					],
				)?,
			);

			// counts_plus_one = [counts] + carry_in
			counts_plus_one_oracle_id.push(oracles.add_linear_combination(
				u_n_vars_gf2,
				[(*counts_oracle_id, F::ONE), (carry_in_oracle_id[i], F::ONE)],
			)?);

			// [counts] * carry_in - [carry_out] = 0,
			let unary_carry_constraint_oracle = CompositePolyOracle::new(
				u_n_vars_gf2,
				vec![
					oracles.oracle(*counts_oracle_id),
					oracles.oracle(carry_in_oracle_id[i]),
					oracles.oracle(*carry_out_oracle_id),
				],
				UnaryCarryConstraint,
			)?;

			zerocheck_claims.push(ZerocheckClaim {
				poly: unary_carry_constraint_oracle,
			});

			// Packed oracles for the multiset check
			packed_counts_oracle_id.push(oracles.add_packed(*counts_oracle_id, C::TOWER_LEVEL)?);
			packed_counts_plus_one_oracle_id
				.push(oracles.add_packed(counts_plus_one_oracle_id[i], C::TOWER_LEVEL)?);

			mixed_u_counts_oracle_id.push(oracles.add_linear_combination_with_offset(
				u_n_vars,
				gamma,
				[(u_oracle.id(), F::ONE), (packed_counts_oracle_id[i], alpha)],
			)?);

			mixed_u_counts_plus_one_oracle_id.push(oracles.add_linear_combination_with_offset(
				u_n_vars,
				gamma,
				[
					(u_oracle.id(), F::ONE),
					(packed_counts_plus_one_oracle_id[i], alpha),
				],
			)?);

			prodcheck_claims.push(ProdcheckClaim {
				t_oracle: oracles.oracle(mixed_u_counts_oracle_id[i]),
				u_oracle: oracles.oracle(mixed_u_counts_plus_one_oracle_id[i]),
			});
		}
	}

	let packed_final_counts_oracle_id =
		oracles.add_packed(final_counts_oracle.id(), C::TOWER_LEVEL)?;

	let mixed_t_final_counts_oracle_id = oracles.add_linear_combination_with_offset(
		t_n_vars,
		gamma,
		[
			(lasso_claim.t_oracle.id(), F::ONE),
			(packed_final_counts_oracle_id, alpha),
		],
	)?;

	let mixed_t_zero_oracle_id = oracles.add_linear_combination_with_offset(
		t_n_vars,
		gamma,
		[(lasso_claim.t_oracle.id(), F::ONE)],
	)?;

	let lasso_claim_oracles = LassoReducedClaimOracleIds {
		packed_counts_oracle_id,
		packed_counts_plus_one_oracle_id,
		packed_final_counts_oracle_id,
		counts_plus_one_oracle_id,
		carry_in_oracle_id,
		carry_out_shifted_oracle_id,
		ones_repeating_oracle_id,
		mixed_t_final_counts_oracle_id,
		mixed_t_zero_oracle_id,
		mixed_u_counts_oracle_id,
		mixed_u_counts_plus_one_oracle_id,
	};

	prodcheck_claims.push(ProdcheckClaim {
		t_oracle: oracles.oracle(mixed_t_final_counts_oracle_id),
		u_oracle: oracles.oracle(mixed_t_zero_oracle_id),
	});

	let reduced_lasso_claims = ReducedLassoClaims {
		zerocheck_claims,
		prodcheck_claims,
	};

	Ok((reduced_lasso_claims, lasso_claim_oracles))
}
