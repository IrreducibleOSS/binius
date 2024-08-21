// Copyright 2024 Ulvetanna Inc.

use super::{
	error::Error,
	lasso::{
		reduce_lasso_claim, LassoBatches, LassoClaim, LassoCount, LassoProveOutput,
		LassoReducedClaimOracleIds, LassoWitness, UnaryCarryConstraint,
	},
};
use crate::{
	oracle::MultilinearOracleSet,
	polynomial::{Error as PolynomialError, MultilinearComposite, MultilinearPoly},
	protocols::gkr_prodcheck::ProdcheckWitness,
	witness::{MultilinearExtensionIndex, MultilinearWitness},
};
use binius_field::{
	as_packed_field::{PackScalar, PackedType},
	underlier::{UnderlierType, WithUnderlier},
	BinaryField1b, ExtensionField, Field, PackedField, PackedFieldIndexable, TowerField,
};
use binius_utils::bail;
use itertools::izip;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};
use std::{array, sync::Arc};
use tracing::instrument;

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
/// We then proceed by defining a virtual oracle $P = C + I$ (for "Plus one") and performing a gkr prodcheck
///
/// $\prod_{v∈B_{l}}(\gamma + T(v) + \alpha * F(v))\prod_{v∈B_{l}}(\gamma + U_{0}(v) + \alpha * P_{0}(v)) \dots \prod_{v∈B_{l}}(\gamma + U_{n}(v) + \alpha * P_{n}(v)) =$   
/// $\prod_{v∈B_{l}}(\gamma + T(v)+\alpha*0)\prod_{v∈B_{l}}(\gamma + U_{0}(v) + \alpha * C_{0}(v)) \dots \prod_{v∈B_{l}}(\gamma + U_{n}(v) + \alpha * C_{n}(v))$    
///
/// See Section 4.4 of [DP23] for the
/// proofs. Please note that the implemented addition gadget approach differs from the one described in the
/// paper (which uses multiplicative group to represent counts), as well as uses slightly different notation.
///
/// [DP23]: <https://eprint.iacr.org/2023/1784>
#[instrument(skip_all, name = "lasso::prove", level = "debug")]
pub fn prove<'a, FC, U, F, FW, L>(
	oracles: &mut MultilinearOracleSet<F>,
	witness_index: MultilinearExtensionIndex<'a, U, FW>,
	lasso_claim: &LassoClaim<F>,
	lasso_witness: LassoWitness<'a, PackedType<U, FW>, L>,
	lasso_batches: &LassoBatches,
	gamma: F,
	alpha: F,
) -> Result<LassoProveOutput<'a, U, FW, F>, Error>
where
	U: UnderlierType + PackScalar<FW> + PackScalar<FC> + PackScalar<BinaryField1b>,
	FC: LassoCount,
	PackedType<U, FC>: PackedFieldIndexable,
	PackedType<U, FW>: PackedFieldIndexable,
	F: TowerField + From<FW>,
	FW: TowerField + ExtensionField<FC> + From<F>,
	L: AsRef<[usize]>,
{
	let t_n_vars = lasso_claim.t_oracle().n_vars();
	let t_n_vars_gf2 = t_n_vars + FC::TOWER_LEVEL;

	if lasso_claim.u_oracles().len() != lasso_witness.u_polynomials().len() {
		bail!(Error::ClaimWitnessTablesLenMismatch);
	}

	let bit_packing_log_width = PackedType::<U, BinaryField1b>::LOG_WIDTH;

	let mut final_counts_underlier_vecs =
		vec![U::default(); 1 << (t_n_vars_gf2 - bit_packing_log_width)];

	let final_counts = {
		let packed_slice = PackedType::<U, FC>::from_underliers_ref_mut(
			final_counts_underlier_vecs.as_mut_slice(),
		);
		PackedType::<U, FC>::unpack_scalars_mut(packed_slice)
	};

	// Check that counts actually fit into the chosen data type
	// NB. Need one more bit because 1 << n_vars is a valid count.
	if t_n_vars >= FC::N_BITS {
		bail!(Error::LassoCountTypeTooSmall);
	}

	if t_n_vars_gf2 < bit_packing_log_width {
		bail!(Error::WitnessSmallerThanUnderlier);
	}

	let (reduced_lasso_claims, reduced_claim_oracle_ids) =
		reduce_lasso_claim::<FC, _>(oracles, lasso_claim, lasso_batches, gamma, alpha)?;

	let LassoReducedClaimOracleIds {
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
	} = reduced_claim_oracle_ids;

	let mut prodcheck_witnesses = Vec::new();
	let mut zerocheck_witnesses = Vec::new();

	let mut witness_index = witness_index;

	if !lasso_claim.u_oracles().is_empty() {
		let u_n_vars = lasso_claim.u_oracles()[0].n_vars();
		let u_n_vars_gf2 = u_n_vars + FC::TOWER_LEVEL;

		if u_n_vars >= FC::N_BITS {
			bail!(Error::LassoCountTypeTooSmall);
		}

		if u_n_vars_gf2 < bit_packing_log_width {
			bail!(Error::WitnessSmallerThanUnderlier);
		}

		for (i, (u_polynomial, u_to_t_mapping)) in lasso_witness
			.u_polynomials()
			.iter()
			.zip(lasso_witness.u_to_t_mappings())
			.enumerate()
		{
			// underliers for read counts vectors, a total of seven
			let mut underlier_vecs: [_; 5] =
				array::from_fn(|_| vec![U::default(); 1 << (u_n_vars_gf2 - bit_packing_log_width)]);

			// cast underliers into Lasso counts for addition gadget computations
			let [counts, counts_plus_one, carry_in, carry_out, carry_out_shifted] =
				underlier_vecs.each_mut().map(|underliers| {
					let packed_slice =
						PackedType::<U, FC>::from_underliers_ref_mut(underliers.as_mut_slice());
					PackedType::<U, FC>::unpack_scalars_mut(packed_slice)
				});

			let t_indice = u_to_t_mapping.as_ref();
			// addition gadget computing count+1 via a carry_in XORed with 1 at the lowest bit
			for (&t_index, counts, counts_plus_one, carry_in, carry_out, carry_out_shifted) in
				izip!(t_indice, counts, counts_plus_one, carry_in, carry_out, carry_out_shifted)
			{
				let count = final_counts[t_index];
				let (count_plus_one, overflow) = count.overflowing_add(FC::ONE);
				assert!(!overflow, "Lasso count overflowed!");

				final_counts[t_index] = count_plus_one;

				// u32 addition gadget with y = 1
				*counts = count;
				*counts_plus_one = count_plus_one;
				*carry_in = count + count_plus_one;
				*carry_out_shifted = *carry_in + FC::ONE;
				*carry_out = (*carry_in).shr1();
			}

			// construct virtual polynomial oracles

			let [counts, counts_plus_one, carry_in, carry_out, carry_out_shifted] = underlier_vecs;

			// add 1-bit witnesses to the index
			let counts_oracle_id = &lasso_batches.counts;
			let carry_out_oracle_id = &lasso_batches.carry_out;

			let counts = Arc::<[U]>::from(counts);
			let counts_plus_one = Arc::<[U]>::from(counts_plus_one);

			witness_index = witness_index.update_owned::<BinaryField1b, _>([
				(counts_oracle_id[i], counts.clone()),
				(counts_plus_one_oracle_id[i], counts_plus_one.clone()),
				(carry_in_oracle_id[i], carry_in.into()),
				(carry_out_oracle_id[i], carry_out.into()),
				(carry_out_shifted_oracle_id[i], carry_out_shifted.into()),
			])?;

			// add FC witnesses to the index
			witness_index = witness_index.update_owned::<FC, _>([
				(packed_counts_oracle_id[i], counts),
				(packed_counts_plus_one_oracle_id[i], counts_plus_one),
			])?;

			let packed_counts_poly = witness_index.get_multilin_poly(packed_counts_oracle_id[i])?;
			let packed_counts_plus_one_poly =
				witness_index.get_multilin_poly(packed_counts_plus_one_oracle_id[i])?;

			let mixed_u_counts = linciom(u_polynomial, Some(&packed_counts_poly), gamma, alpha)?;
			let mixed_u_counts_plus_one =
				linciom(u_polynomial, Some(&packed_counts_plus_one_poly), gamma, alpha)?;

			witness_index = witness_index.update_owned::<FW, _>([
				(mixed_u_counts_oracle_id[i], mixed_u_counts),
				(mixed_u_counts_plus_one_oracle_id[i], mixed_u_counts_plus_one),
			])?;

			let mixed_u_counts_poly =
				witness_index.get_multilin_poly(mixed_u_counts_oracle_id[i])?;
			let mixed_u_counts_plus_one_poly =
				witness_index.get_multilin_poly(mixed_u_counts_plus_one_oracle_id[i])?;

			prodcheck_witnesses.push(ProdcheckWitness {
				t_poly: mixed_u_counts_poly,
				u_poly: mixed_u_counts_plus_one_poly,
			});

			let counts_poly = witness_index.get_multilin_poly(counts_oracle_id[i])?;
			let carry_in_poly = witness_index.get_multilin_poly(carry_in_oracle_id[i])?;
			let carry_out_poly = witness_index.get_multilin_poly(carry_out_oracle_id[i])?;

			zerocheck_witnesses.push(MultilinearComposite::new(
				u_n_vars_gf2,
				UnaryCarryConstraint,
				vec![counts_poly, carry_in_poly, carry_out_poly],
			)?);
		}

		let mut ones_underlier_vecs =
			vec![U::default(); 1 << (u_n_vars_gf2 - bit_packing_log_width)];

		let ones = {
			let packed_slice =
				PackedType::<U, FC>::from_underliers_ref_mut(ones_underlier_vecs.as_mut_slice());
			PackedType::<U, FC>::unpack_scalars_mut(packed_slice)
		};

		ones.fill(FC::ONE);

		witness_index = witness_index.update_owned::<BinaryField1b, _>([(
			ones_repeating_oracle_id.expect("ones_repeating_oracle_id must created "),
			ones_underlier_vecs,
		)])?;
	}

	let final_counts_oracle_id = lasso_batches.final_counts;

	witness_index = witness_index.update_owned::<BinaryField1b, _>([(
		final_counts_oracle_id,
		final_counts_underlier_vecs.clone(),
	)])?;

	witness_index = witness_index.update_owned::<FC, _>([(
		packed_final_counts_oracle_id,
		final_counts_underlier_vecs.clone(),
	)])?;

	let packed_final_counts_poly =
		witness_index.get_multilin_poly(packed_final_counts_oracle_id)?;

	let mixed_t_final_counts =
		linciom(lasso_witness.t_polynomial(), Some(&packed_final_counts_poly), gamma, alpha)?;
	let mixed_t_zero = linciom(lasso_witness.t_polynomial(), None, gamma, alpha)?;

	witness_index = witness_index.update_owned::<FW, _>([
		(mixed_t_final_counts_oracle_id, mixed_t_final_counts),
		(mixed_t_zero_oracle_id, mixed_t_zero),
	])?;

	let mixed_t_final_counts_poly =
		witness_index.get_multilin_poly(mixed_t_final_counts_oracle_id)?;
	let mixed_t_zero_poly = witness_index.get_multilin_poly(mixed_t_zero_oracle_id)?;

	prodcheck_witnesses.push(ProdcheckWitness {
		t_poly: mixed_t_final_counts_poly,
		u_poly: mixed_t_zero_poly,
	});

	Ok(LassoProveOutput {
		reduced_lasso_claims,
		prodcheck_witnesses,
		zerocheck_witnesses,
		witness_index,
	})
}

fn linciom<'a, U, FW, PW, F>(
	trace: &MultilinearWitness<'a, PW>,
	counts: Option<&MultilinearWitness<'a, PW>>,
	gamma: F,
	alpha: F,
) -> Result<Arc<[U]>, Error>
where
	U: UnderlierType + PackScalar<FW>,
	PackedType<U, FW>: PackedFieldIndexable,
	PW: PackedField<Scalar = FW>,
	FW: Field + From<F>,
{
	let n_vars = trace.n_vars();

	let packing_log_width = PackedType::<U, FW>::LOG_WIDTH;

	let mut underliers = vec![U::default(); 1 << (n_vars - packing_log_width)];
	let values = PackedType::<U, FW>::unpack_scalars_mut(
		PackedType::<U, FW>::from_underliers_ref_mut(underliers.as_mut_slice()),
	);

	values.fill(FW::from(gamma));

	// first dimension of the relation is not weighted
	values.par_iter_mut().enumerate().try_for_each(
		|(i, values_i)| -> Result<_, PolynomialError> {
			*values_i += trace.evaluate_on_hypercube(i)?;
			Ok(())
		},
	)?;

	let alpha = FW::from(alpha);

	if let Some(counts) = counts {
		values.par_iter_mut().enumerate().try_for_each(
			|(i, values_i)| -> Result<_, PolynomialError> {
				*values_i += counts.evaluate_on_hypercube_and_scale(i, alpha)?;
				Ok(())
			},
		)?;
	}

	Ok(underliers.into())
}
