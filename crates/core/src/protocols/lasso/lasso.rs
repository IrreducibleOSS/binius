// Copyright 2024 Ulvetanna Inc.

use super::error::Error;
use crate::{
	oracle::{
		BatchId, CompositePolyOracle, MultilinearOracleSet, MultilinearPolyOracle, OracleId,
		ShiftVariant,
	},
	polynomial::{transparent::step_down::StepDown, CompositionPoly, Error as PolynomialError},
	protocols::{
		msetcheck::{MsetcheckClaim, MsetcheckWitness},
		zerocheck::{ZerocheckClaim, ZerocheckWitnessTypeErased},
	},
	witness::{MultilinearExtensionIndex, MultilinearWitness},
};
use binius_field::{
	as_packed_field::{PackScalar, PackedType},
	underlier::UnderlierType,
	BinaryField16b, BinaryField32b, Field, PackedField, TowerField,
};
use getset::{CopyGetters, Getters};

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
pub struct LassoBatch {
	#[get_copy = "pub"]
	batch_id: BatchId,

	pub counts: OracleId,
	pub carry_out: OracleId,
	pub final_counts: OracleId,
}

impl LassoBatch {
	pub fn new_in<C: LassoCount, F: TowerField>(
		oracles: &mut MultilinearOracleSet<F>,
		n_vars: usize,
	) -> Self {
		let batch_id = oracles.add_committed_batch(n_vars + C::TOWER_LEVEL, 0);
		Self {
			batch_id,
			counts: oracles.add_committed(batch_id),
			carry_out: oracles.add_committed(batch_id),
			final_counts: oracles.add_committed(batch_id),
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
	u_oracle: MultilinearPolyOracle<F>,
}

impl<F: Field> LassoClaim<F> {
	pub fn new(
		t_oracle: MultilinearPolyOracle<F>,
		u_oracle: MultilinearPolyOracle<F>,
	) -> Result<Self, Error> {
		if t_oracle.n_vars() != u_oracle.n_vars() {
			return Err(Error::NumVariablesMismatch);
		}

		Ok(Self { t_oracle, u_oracle })
	}

	pub fn n_vars(&self) -> usize {
		self.t_oracle.n_vars()
	}
}

#[derive(Debug, Getters)]
pub struct LassoWitness<'a, PW: PackedField, L: AsRef<[usize]>> {
	#[get = "pub"]
	t_polynomial: MultilinearWitness<'a, PW>,
	#[get = "pub"]
	u_polynomial: MultilinearWitness<'a, PW>,
	#[get = "pub"]
	u_to_t_mapping: L,
}

impl<'a, PW: PackedField, L: AsRef<[usize]>> LassoWitness<'a, PW, L> {
	pub fn new(
		t_polynomial: MultilinearWitness<'a, PW>,
		u_polynomial: MultilinearWitness<'a, PW>,
		u_to_t_mapping: L,
	) -> Result<Self, Error> {
		if t_polynomial.n_vars() != u_polynomial.n_vars() {
			return Err(Error::NumVariablesMismatch);
		}

		let size = t_polynomial.size();
		if size != u_to_t_mapping.as_ref().len() {
			return Err(Error::MappingSizeMismatch);
		}

		if u_to_t_mapping.as_ref().iter().any(|&index| index >= size) {
			return Err(Error::MappingIndexOutOfBounds);
		}

		Ok(Self {
			t_polynomial,
			u_polynomial,
			u_to_t_mapping,
		})
	}

	pub fn n_vars(&self) -> usize {
		self.t_polynomial.n_vars()
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
			return Err(PolynomialError::IncorrectQuerySize { expected: 3 });
		}

		Ok(query[0] * query[1] - query[2])
	}

	fn binary_tower_level(&self) -> usize {
		0
	}
}

pub struct ReducedLassoClaims<F: Field> {
	pub zerocheck_claim: ZerocheckClaim<F>,
	pub msetcheck_claim: MsetcheckClaim<F>,
}

pub struct LassoProveOutput<'a, U: UnderlierType + PackScalar<FW>, F: Field, FW: TowerField> {
	pub reduced_lasso_claims: ReducedLassoClaims<F>,
	pub zerocheck_witness: ZerocheckWitnessTypeErased<'a, PackedType<U, FW>, UnaryCarryConstraint>,
	pub msetcheck_witness: MsetcheckWitness<'a, PackedType<U, FW>>,
	pub witness_index: MultilinearExtensionIndex<'a, U, FW>,
}

pub(super) struct LassoReducedClaimOracleIds {
	pub tu_merged_oracle_id: OracleId,
	pub final_counts_and_counts_oracle_id: OracleId,
	pub zeros_counts_plus_one_oracle_id: OracleId,
	pub packed_counts_oracle_id: OracleId,
	pub packed_counts_plus_one_oracle_id: OracleId,
	pub packed_final_counts_oracle_id: OracleId,
	pub counts_plus_one_oracle_id: OracleId,
	pub carry_in_oracle_id: OracleId,
	pub carry_out_shifted_oracle_id: OracleId,
	pub ones_repeating_oracle_id: OracleId,
}

pub fn reduce_lasso_claim<C: LassoCount, F: TowerField>(
	oracles: &mut MultilinearOracleSet<F>,
	lasso_claim: &LassoClaim<F>,
	lasso_batch: &LassoBatch,
) -> Result<(ReducedLassoClaims<F>, LassoReducedClaimOracleIds), Error> {
	let n_vars = lasso_claim.n_vars();
	let n_vars_gf2 = n_vars + C::TOWER_LEVEL;

	// Extract Lasso committed column oracles
	let counts_oracle = oracles.oracle(lasso_batch.counts);
	let carry_out_oracle = oracles.oracle(lasso_batch.carry_out);
	let final_counts_oracle = oracles.oracle(lasso_batch.final_counts);

	if counts_oracle.n_vars() != n_vars_gf2
		|| carry_out_oracle.n_vars() != n_vars_gf2
		|| final_counts_oracle.n_vars() != n_vars_gf2
	{
		return Err(Error::CountsNumVariablesMismatch);
	}

	// Representing a column of ones as a repeating oracle of 10...0
	let one = StepDown::new(C::TOWER_LEVEL, 1)?;
	let one_oracle_id = oracles.add_transparent(one)?;
	let ones_repeating_oracle_id = oracles.add_repeating(one_oracle_id, n_vars)?;

	// carry_in = ([carry_out] << 1) + 1
	let carry_out_shifted_oracle_id =
		oracles.add_shifted(carry_out_oracle.id(), 1, C::TOWER_LEVEL, ShiftVariant::LogicalLeft)?;

	let carry_in_oracle_id = oracles.add_linear_combination(
		n_vars_gf2,
		[
			(carry_out_shifted_oracle_id, F::ONE),
			(ones_repeating_oracle_id, F::ONE),
		],
	)?;
	let carry_in_oracle = oracles.oracle(carry_in_oracle_id);

	// counts_plus_one = [counts] + carry_in
	let counts_plus_one_oracle_id = oracles.add_linear_combination(
		n_vars_gf2,
		[(counts_oracle.id(), F::ONE), (carry_in_oracle_id, F::ONE)],
	)?;

	// Packed oracles for the multiset check
	let packed_counts_oracle_id = oracles.add_packed(counts_oracle.id(), C::TOWER_LEVEL)?;
	let packed_final_counts_oracle_id =
		oracles.add_packed(final_counts_oracle.id(), C::TOWER_LEVEL)?;
	let packed_counts_plus_one_oracle_id =
		oracles.add_packed(counts_plus_one_oracle_id, C::TOWER_LEVEL)?;

	// [counts] * carry_in - [carry_out] = 0,
	let unary_carry_constraint_oracle = CompositePolyOracle::new(
		n_vars_gf2,
		vec![counts_oracle, carry_in_oracle, carry_out_oracle],
		UnaryCarryConstraint,
	)?;

	let zerocheck_claim = ZerocheckClaim {
		poly: unary_carry_constraint_oracle,
	};

	// merge([T], [U])
	let tu_merged_oracle_id =
		oracles.add_merged(lasso_claim.t_oracle().id(), lasso_claim.u_oracle().id())?;
	let tu_merged_oracle = oracles.oracle(tu_merged_oracle_id);

	// merge([final_counts], [counts])
	let final_counts_and_counts_oracle_id =
		oracles.add_merged(packed_final_counts_oracle_id, packed_counts_oracle_id)?;
	let final_counts_and_counts_oracle = oracles.oracle(final_counts_and_counts_oracle_id);

	// merge(zeros, counts_plus_one)
	let zeros_counts_plus_one_oracle_id =
		oracles.add_zero_padded(packed_counts_plus_one_oracle_id, n_vars + 1)?;
	let zeros_counts_plus_one_oracle = oracles.oracle(zeros_counts_plus_one_oracle_id);

	let msetcheck_claim = MsetcheckClaim::new(
		[tu_merged_oracle.clone(), final_counts_and_counts_oracle],
		[tu_merged_oracle, zeros_counts_plus_one_oracle],
	)?;

	let reduced_lasso_claims = ReducedLassoClaims {
		zerocheck_claim,
		msetcheck_claim,
	};

	let lasso_claim_oracles = LassoReducedClaimOracleIds {
		tu_merged_oracle_id,
		final_counts_and_counts_oracle_id,
		zeros_counts_plus_one_oracle_id,
		packed_counts_oracle_id,
		packed_counts_plus_one_oracle_id,
		packed_final_counts_oracle_id,
		counts_plus_one_oracle_id,
		carry_in_oracle_id,
		carry_out_shifted_oracle_id,
		ones_repeating_oracle_id,
	};

	Ok((reduced_lasso_claims, lasso_claim_oracles))
}
