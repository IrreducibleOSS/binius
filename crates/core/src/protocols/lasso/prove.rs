// Copyright 2024 Ulvetanna Inc.

use super::{
	error::Error,
	lasso::{
		reduce_lasso_claim, LassoBatch, LassoClaim, LassoCount, LassoProveOutput,
		LassoReducedClaimOracleIds, LassoWitness, UnaryCarryConstraint,
	},
};
use crate::{
	oracle::MultilinearOracleSet,
	polynomial::{MultilinearComposite, MultilinearPoly},
	protocols::msetcheck::MsetcheckWitness,
	witness::MultilinearExtensionIndex,
};
use binius_field::{
	as_packed_field::{PackScalar, PackedType},
	underlier::{UnderlierType, WithUnderlier},
	BinaryField1b, ExtensionField, Field, PackedField, PackedFieldIndexable, TowerField,
};
use itertools::izip;
use std::{array, sync::Arc};
use tracing::instrument;

fn multilin_poly_to_underliers_ref_mut<U: UnderlierType + PackScalar<F>, F: Field>(
	poly: impl MultilinearPoly<PackedType<U, F>>,
	dest: &mut [U],
) -> Result<(), Error> {
	let packed_dest = PackedType::<U, F>::from_underliers_ref_mut(dest);
	Ok(poly.subcube_evals(poly.n_vars(), 0, packed_dest)?)
}

/// Prove a Lasso instance reduction.
///
/// Given $\nu$-variate multilinears $T$ (the "table") and $U$ and a mapping $U \mapsto T$ from hypercube
/// vertices of $U$ to hypercube vertices of $T$, represented by an array `u_to_t_mapping` of vertex indices,
/// we define GF(2) multilinears $C$ (for "counts") and $F$ (for "final counts"), initialize them with zero
/// constant polynomials, and perform the following algorithm, treating consecutive bits as integers:
///
/// ```text
///  for i in 0..2**n_vars
///    j := u_to_t_mapping[i]
///    C[i] = F[j]
///    F[j] += 1
/// ```
///
/// The number of bits in each "group" must be sufficient to represent $2^{n\\_vars}$, and is specified via
/// `PC` type parameter. Prover then commits an additional GF(2) polynomial $O$ (for "carry Out"), which
/// represents carry propagation in a simplified addition gadget that only supports increments. We obtain
/// "carry in" as virtual oracle $I = (O << 1) + 1$ and zerocheck the following constraint:
///
/// $$C \times I - O = 0$$
///
/// which is a simplified addition gadget constraint when second addend is zero, and "carry in" equals
/// one in the lowest bit.
///
/// We then proceed by defining a virtual oracle $P = C + I$ (for "Plus one") and performing a multiset check
/// between `(merge(T, U), merge(0, P))` and `(merge(T, U), merge(F, C))`. See Section 4.4 of [DP23] for the
/// proofs. Please note that the implemented addition gadget approach differs from the one described in the
/// paper (which uses multiplicative group to represent counts), as well as uses slightly different notation.
///
/// [DP23]: <https://eprint.iacr.org/2023/1784>
#[instrument(skip_all, name = "lasso::prove")]
pub fn prove<'a, FC, U, F, FW, L>(
	oracles: &mut MultilinearOracleSet<F>,
	witness_index: MultilinearExtensionIndex<'a, U, FW>,
	lasso_claim: &LassoClaim<F>,
	lasso_witness: LassoWitness<'a, PackedType<U, FW>, L>,
	lasso_batch: &LassoBatch,
) -> Result<LassoProveOutput<'a, U, F, FW>, Error>
where
	U: UnderlierType + PackScalar<FW> + PackScalar<FC> + PackScalar<BinaryField1b>,
	FC: LassoCount,
	PackedType<U, FC>: PackedFieldIndexable,
	F: TowerField,
	FW: TowerField + ExtensionField<FC>,
	L: AsRef<[usize]>,
{
	let n_vars = lasso_claim.n_vars();
	let n_vars_gf2 = n_vars + FC::TOWER_LEVEL;

	// Check that counts actually fit into the chosen data type
	// NB. Need one more bit because 1 << n_vars is a valid count.
	if n_vars >= FC::N_BITS {
		return Err(Error::LassoCountTypeTooSmall);
	}

	if n_vars != lasso_witness.n_vars() {
		return Err(Error::WitnessNumVariablesMismatch);
	}

	let bit_packing_log_width = PackedType::<U, BinaryField1b>::LOG_WIDTH;
	if n_vars_gf2 < bit_packing_log_width {
		return Err(Error::WitnessSmallerThanUnderlier);
	}

	// underliers for read counts vectors, a total of seven
	let mut underlier_vecs: [_; 7] =
		array::from_fn(|_| vec![U::default(); 1 << (n_vars_gf2 - bit_packing_log_width)]);

	// cast underliers into Lasso counts for addition gadget computations
	let [counts, counts_plus_one, carry_in, carry_out, carry_out_shifted, ones, final_counts] =
		underlier_vecs.each_mut().map(|underliers| {
			let packed_slice =
				PackedType::<U, FC>::from_underliers_ref_mut(underliers.as_mut_slice());
			PackedType::<U, FC>::unpack_scalars_mut(packed_slice)
		});

	// addition gadget computing count+1 via a carry_in XORed with 1 at the lowest bit
	let t_indice = lasso_witness.u_to_t_mapping().as_ref();
	for (&t_index, counts, counts_plus_one, carry_in, carry_out, carry_out_shifted, ones) in
		izip!(t_indice, counts, counts_plus_one, carry_in, carry_out, carry_out_shifted, ones)
	{
		let count = final_counts[t_index];
		let (count_plus_one, overflow) = count.overflowing_add(FC::ONE);
		assert!(!overflow, "Lasso count overflowed!");

		final_counts[t_index] = count_plus_one;

		// u32 addition gadget with y = 1
		*counts = count;
		*counts_plus_one = count_plus_one;
		*carry_in = count + count_plus_one;
		*ones = FC::ONE;
		*carry_out_shifted = *carry_in + *ones;
		*carry_out = (*carry_in).shr1();
	}

	// construct virtual polynomial oracles

	let (reduced_lasso_claims, reduced_claim_oracle_ids) =
		reduce_lasso_claim::<FC, _>(oracles, lasso_claim, lasso_batch)?;

	let LassoReducedClaimOracleIds {
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
	} = reduced_claim_oracle_ids;

	// generate msetcheck witness
	let [counts, counts_plus_one, carry_in, carry_out, carry_out_shifted, ones, final_counts] =
		underlier_vecs;

	let fw_packing_log_width = PackedType::<U, FW>::LOG_WIDTH;
	let fc_packing_log_width = PackedType::<U, FC>::LOG_WIDTH;

	// [t, u]
	let mut tu_merged = vec![U::default(); 1 << (n_vars + 1 - fw_packing_log_width)];
	let half_tu_merged = tu_merged.len() / 2;
	let (t_half, u_half) = tu_merged.split_at_mut(half_tu_merged);
	multilin_poly_to_underliers_ref_mut::<U, FW>(lasso_witness.t_polynomial(), t_half)?;
	multilin_poly_to_underliers_ref_mut::<U, FW>(lasso_witness.u_polynomial(), u_half)?;

	// [final_counts, counts]
	let mut final_counts_and_counts = final_counts.clone();
	final_counts_and_counts.extend(counts.as_slice());

	// [0, counts+1]
	let zeros = vec![U::default(); 1 << (n_vars - fc_packing_log_width)];
	let mut zeros_counts_plus_one = zeros.clone();
	zeros_counts_plus_one.extend(counts_plus_one.as_slice());
	debug_assert!(zeros.len() * 2 == zeros_counts_plus_one.len());

	// add 1-bit witnesses to the index
	let counts_oracle_id = lasso_batch.counts;
	let carry_out_oracle_id = lasso_batch.carry_out;
	let final_counts_oracle_id = lasso_batch.final_counts;

	let counts = Arc::<[U]>::from(counts);
	let counts_plus_one = Arc::<[U]>::from(counts_plus_one);
	let final_counts = Arc::<[U]>::from(final_counts);

	let witness_index = witness_index.update_owned::<BinaryField1b, _>([
		(counts_oracle_id, counts.clone()),
		(counts_plus_one_oracle_id, counts_plus_one.clone()),
		(carry_in_oracle_id, carry_in.into()),
		(carry_out_oracle_id, carry_out.into()),
		(carry_out_shifted_oracle_id, carry_out_shifted.into()),
		(ones_repeating_oracle_id, ones.into()),
		(final_counts_oracle_id, final_counts.clone()),
	])?;

	// add FC witnesses to the index
	let witness_index = witness_index.update_owned::<FC, _>([
		(packed_counts_oracle_id, counts),
		(packed_counts_plus_one_oracle_id, counts_plus_one),
		(packed_final_counts_oracle_id, final_counts),
		(final_counts_and_counts_oracle_id, final_counts_and_counts.into()),
		(zeros_counts_plus_one_oracle_id, zeros_counts_plus_one.into()),
	])?;

	// ...and concatenated [t, u] one
	let witness_index = witness_index.update_owned::<FW, _>([(tu_merged_oracle_id, tu_merged)])?;

	// reduced witnesses
	let counts = witness_index.get_multilin_poly(counts_oracle_id)?;
	let carry_in = witness_index.get_multilin_poly(carry_in_oracle_id)?;
	let carry_out = witness_index.get_multilin_poly(carry_out_oracle_id)?;

	let zerocheck_witness = MultilinearComposite::new(
		n_vars_gf2,
		UnaryCarryConstraint,
		vec![counts, carry_in, carry_out],
	)?;

	let tu_merged = witness_index.get_multilin_poly(tu_merged_oracle_id)?;
	let final_counts_and_counts =
		witness_index.get_multilin_poly(final_counts_and_counts_oracle_id)?;
	let zeros_counts_plus_one = witness_index.get_multilin_poly(zeros_counts_plus_one_oracle_id)?;

	let msetcheck_witness = MsetcheckWitness::new(
		[tu_merged.clone(), final_counts_and_counts],
		[tu_merged, zeros_counts_plus_one],
	)?;

	Ok(LassoProveOutput {
		reduced_lasso_claims,
		zerocheck_witness,
		msetcheck_witness,
		witness_index,
	})
}
