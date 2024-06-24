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
	polynomial::{MultilinearComposite, MultilinearExtension, MultilinearPoly},
	protocols::msetcheck::MsetcheckWitness,
	witness::{MultilinearWitness, MultilinearWitnessIndex},
};
use binius_field::{
	BinaryField, BinaryField1b, ExtensionField, Field, PackedField, PackedFieldIndexable,
	RepackedExtension, TowerField,
};
use bytemuck::{must_cast_slice_mut, Pod};
use itertools::izip;
use std::{array, borrow::Borrow, fmt::Debug};
use tracing::instrument;

/// Returns merge(x, y) where x, y are multilinear polynomials
fn construct_large_field_merged_witness<F: Field>(
	x: impl MultilinearPoly<F>,
	y: impl MultilinearPoly<F>,
) -> Result<MultilinearWitness<'static, F>, Error> {
	let x = x.borrow();
	let y = y.borrow();

	let n_vars = x.n_vars();
	if y.n_vars() != n_vars {
		return Err(Error::MergedWitnessNumVariablesMismatch);
	}

	// TODO: Find a way to avoid these copies
	let mut values = vec![F::ZERO; 1 << (n_vars + 1)];
	let (x_values, y_values) = values.split_at_mut(1 << n_vars);
	x.subcube_evals(x.n_vars(), 0, x_values)?;
	y.subcube_evals(y.n_vars(), 0, y_values)?;
	let merge_poly = MultilinearExtension::from_values(values)?;
	Ok(merge_poly.specialize_arc_dyn())
}

// Returns merge(x, y) where x, y are MLEs over packed fields
fn construct_packed_merged_witness<P: PackedField>(
	x: &MultilinearExtension<P>,
	y: &MultilinearExtension<P>,
) -> Result<MultilinearExtension<P>, Error> {
	let n_vars = x.n_vars();
	if y.n_vars() != n_vars || n_vars < P::LOG_WIDTH {
		return Err(Error::MergedWitnessNumVariablesMismatch);
	}

	// TODO: Find a way to avoid these copies
	// (will be achieved once the MultilinearPoly wrapper impl for merged polynomials lands)
	let mut values = vec![P::default(); 1 << (n_vars + 1 - P::LOG_WIDTH)];
	let (x_values, y_values) = values.split_at_mut(1 << (n_vars - P::LOG_WIDTH));
	x_values.copy_from_slice(x.evals());
	y_values.copy_from_slice(y.evals());
	Ok(MultilinearExtension::from_values(values)?)
}

// one day array::try_map would be a thing
fn array_try_map<T, U: Debug, const N: usize>(
	arr: [T; N],
	closure: impl FnMut(T) -> Result<U, Error>,
) -> Result<[U; N], Error> {
	let vec = arr
		.into_iter()
		.map(closure)
		.collect::<Result<Vec<_>, Error>>()?;
	Ok(<[U; N]>::try_from(vec).expect("infallible by construction"))
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
pub fn prove<'a, PC, PB, F, FW, L>(
	oracles: &mut MultilinearOracleSet<F>,
	witness_index: &mut MultilinearWitnessIndex<'a, FW>,
	lasso_claim: &LassoClaim<F>,
	lasso_witness: LassoWitness<'a, FW, L>,
	lasso_batch: &LassoBatch,
) -> Result<LassoProveOutput<'a, F, FW, PB>, Error>
where
	PC: RepackedExtension<PB, Scalar: LassoCount> + PackedFieldIndexable + Pod,
	PB: PackedField<Scalar = BinaryField1b> + Pod,
	F: TowerField,
	FW: TowerField + ExtensionField<PC::Scalar>,
	L: AsRef<[usize]>,
{
	let n_vars = lasso_claim.n_vars();
	let n_vars_gf2 = n_vars + PC::Scalar::TOWER_LEVEL;

	// Check that counts actually fit into the chosen data type
	// NB. Need one more bit because 1 << n_vars is a valid count.
	if n_vars >= <PC::Scalar as BinaryField>::N_BITS {
		return Err(Error::LassoCountTypeTooSmall);
	}

	if n_vars != lasso_witness.n_vars() {
		return Err(Error::WitnessNumVariablesMismatch);
	}

	// GF(2) vectors for read counts, a total of seven
	let mut gf2_vectors: [_; 7] =
		array::from_fn(|_| vec![PB::default(); 1 << (n_vars_gf2 - PB::LOG_WIDTH)]);

	// recast GF(2) vectors into Lasso counts for addition gadget computations
	let [counts, counts_plus_one, carry_in, carry_out, carry_out_shifted, ones, final_counts] =
		gf2_vectors
			.each_mut()
			// FIXME this is a hack that should be factored out when PackedDivisible lands
			.map(|values_gf2| {
				let counts_slice = must_cast_slice_mut::<PB, PC>(values_gf2.as_mut_slice());
				PC::unpack_scalars_mut(counts_slice)
			});

	// addition gadget computing count+1 via a carry_in XORed with 1 at the lowest bit
	let t_indice = lasso_witness.u_to_t_mapping().as_ref();
	for (&t_index, counts, counts_plus_one, carry_in, carry_out, carry_out_shifted, ones) in
		izip!(t_indice, counts, counts_plus_one, carry_in, carry_out, carry_out_shifted, ones)
	{
		let count = final_counts[t_index];
		let (count_plus_one, overflow) = count.overflowing_add(PC::Scalar::ONE);
		assert!(!overflow, "Lasso count overflowed!");

		final_counts[t_index] = count_plus_one;

		// u32 addition gadget with y = 1
		*counts = count;
		*counts_plus_one = count_plus_one;
		*carry_in = count + count_plus_one;
		*ones = PC::Scalar::ONE;
		*carry_out_shifted = *carry_in + *ones;
		*carry_out = (*carry_in).shr1();
	}

	// create Lasso count witnesses for packed columns: counts, counts+1, and final counts
	// (TODO: actually share evals with corresponding GF(2) witnesses once a better field vec  is available)
	let [counts, counts_plus_one, _carry_in, _carry_out, _carry_out_shifted, _ones, final_counts] =
		gf2_vectors.each_mut();
	let witnesses_to_pack = [counts, counts_plus_one, final_counts];

	let [counts, counts_plus_one, final_counts] = array_try_map(witnesses_to_pack, |values_gf2| {
		// TODO this is a hack that should be factored out when PackedDivisible lands
		let counts_slice = must_cast_slice_mut::<PB, PC>(values_gf2.as_mut_slice());
		Ok(MultilinearExtension::from_values(counts_slice.to_vec())?)
	})?;

	// All five GF(2) columns are needed in the witness index
	// TODO once better field vec lands and we get rid of Cow within MultilinearExtension this
	// can be greatly simplified.
	let [(counts_gf2_arc_dyn, counts_gf2), (counts_plus_one_gf2_arc_dyn, _counts_plus_one_gf2), (carry_in_gf2_arc_dyn, _carry_in_gf2), (carry_out_gf2_arc_dyn, carry_out_gf2), (carry_out_shifted_gf2_arc_dyn, _carry_out_shifted_gf2), (ones_gf2_arc_dyn, _ones_gf2), (final_counts_gf2_arc_dyn, final_counts_gf2)] =
		array_try_map(gf2_vectors, |values_gf2| {
			let mle = MultilinearExtension::from_values(values_gf2)?;
			let mle_borrowed = mle.clone();
			let mle_arc_dyn = mle.specialize_arc_dyn();
			Ok((mle_arc_dyn, mle_borrowed))
		})?;

	let counts_oracle = lasso_batch.counts_oracle(oracles);
	let carry_out_oracle = lasso_batch.carry_out_oracle(oracles);
	let final_counts_oracle = lasso_batch.final_counts_oracle(oracles);

	witness_index.set_many([
		(counts_oracle.id(), counts_gf2_arc_dyn.clone()),
		(carry_out_oracle.id(), carry_out_gf2_arc_dyn.clone()),
		(final_counts_oracle.id(), final_counts_gf2_arc_dyn.clone()),
	]);

	let (reduced_lasso_claims, reduced_claim_oracle_ids) =
		reduce_lasso_claim::<PC::Scalar, _>(oracles, lasso_claim, lasso_batch)?;

	let LassoReducedClaimOracleIds {
		tu_merged_oracle_id,
		final_counts_and_counts_oracle_id,
		zeros_counts_plus_one_oracle_id,
		zeros_oracle_id,
		packed_counts_oracle_id,
		packed_counts_plus_one_oracle_id,
		packed_final_counts_oracle_id,
		counts_plus_one_oracle_id,
		carry_in_oracle_id,
		carry_out_shifted_oracle_id,
		ones_repeating_oracle_id,
	} = reduced_claim_oracle_ids;

	witness_index.set_many([
		(carry_in_oracle_id, carry_in_gf2_arc_dyn.clone()),
		(carry_out_shifted_oracle_id, carry_out_shifted_gf2_arc_dyn),
		(ones_repeating_oracle_id, ones_gf2_arc_dyn),
		(counts_plus_one_oracle_id, counts_plus_one_gf2_arc_dyn),
	]);

	let zerocheck_witness = MultilinearComposite::new(
		n_vars_gf2,
		UnaryCarryConstraint,
		vec![
			counts_gf2_arc_dyn,
			carry_in_gf2_arc_dyn,
			carry_out_gf2_arc_dyn,
		],
	)?;

	let tu_merged = construct_large_field_merged_witness(
		lasso_witness.t_polynomial(),
		lasso_witness.u_polynomial(),
	)?;

	let final_counts_and_counts =
		construct_packed_merged_witness(&final_counts, &counts)?.specialize_arc_dyn();

	let zeros = MultilinearExtension::<PC>::zeros(n_vars)?;
	let zeros_counts_plus_one =
		construct_packed_merged_witness(&zeros, &counts_plus_one)?.specialize_arc_dyn();

	witness_index.set_many([
		(tu_merged_oracle_id, tu_merged.clone()),
		(final_counts_and_counts_oracle_id, final_counts_and_counts.clone()),
		(zeros_counts_plus_one_oracle_id, zeros_counts_plus_one.clone()),
		(zeros_oracle_id, zeros.specialize_arc_dyn()),
		(packed_counts_oracle_id, counts.specialize_arc_dyn()),
		(packed_counts_plus_one_oracle_id, counts_plus_one.specialize_arc_dyn()),
		(packed_final_counts_oracle_id, final_counts.specialize_arc_dyn()),
	]);

	let msetcheck_witness = MsetcheckWitness::new(
		[tu_merged.clone(), final_counts_and_counts],
		[tu_merged, zeros_counts_plus_one],
	)?;

	let committed_polys = [counts_gf2, carry_out_gf2, final_counts_gf2];

	Ok(LassoProveOutput {
		reduced_lasso_claims,
		zerocheck_witness,
		msetcheck_witness,
		committed_polys,
	})
}
