// Copyright 2024 Irreducible Inc.

use super::lasso::{
	reduce_lasso_claim, LassoBatches, LassoClaim, LassoProof, LassoProveOutput,
	LassoReducedClaimOracleIds, LassoWitness,
};

use crate::{
	polynomial::Error as PolynomialError,
	protocols::{
		gkr_gpa::{
			construct_grand_product_claims, construct_grand_product_witnesses,
			get_grand_products_from_witnesses,
		},
		lasso::Error,
	},
};

use crate::{
	oracle::MultilinearOracleSet,
	witness::{MultilinearExtensionIndex, MultilinearWitness},
};
use binius_field::{
	as_packed_field::{PackScalar, PackedType},
	underlier::{UnderlierType, WithUnderlier},
	ExtensionField, Field, PackedField, PackedFieldIndexable, TowerField,
};
use binius_utils::bail;
use itertools::izip;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};
use std::{array, sync::Arc};
use tracing::instrument;

/// Prove a Lasso instance reduction.
///
/// Given $\nu$-variate multilinear $T$ (the "lookup table") and array of $U$ (the "looker columns") and a array of mappings $U \mapsto T$ from hypercube
/// vertices of $U$ to hypercube vertices of $T$, represented by an array `u_to_t_mapping` of vertex indices,
/// we define multilinears $C$ (for "counts") as identically-0 polynomial and $F$ (for "final counts") as identically-1 polynomial and $g$ as the Multiplicative generator of subgroup,
/// and perform the following algorithm:
///
///
///
/// ```text
///  for i in len(U)
///    for j in len(U[i])
/// 	   m := u_to_t_mappings[j]
/// 	   C[i][j] := F[m]
/// 	   F[m] *= g
/// ```
///
/// The order of multiplicative subgroup must be sufficient to represent $\sum len(U_{i})$ elements, and is specified via
/// `FC` type parameter.
///
/// We then proceed by defining a virtual oracles $O$ as identically-1 polynomial and $P = g \times C$, and reduce tables to grand product claims:
///
/// $\prod_{v∈B}(\gamma + U_{0}(v) + \alpha * C_{0}(v)) = L_{0}$  
/// $\dots$  
/// $\prod_{v∈B}(\gamma + U_{n}(v) + \alpha * C_{n}(v)) = L_{n}$   
/// $\prod_{v∈B}(\gamma + T(v) + \alpha * F(v)) = L_{n+1}$
///
/// $\prod_{v∈B}(\gamma + U_{0}(v) + \alpha * P_{0}(v)) = R_{0}$
/// $\dots$  
/// $\prod_{v∈B}(\gamma + U_{n}(v) + \alpha * P_{n}(v)) = R_{n}$   
/// $\prod_{v∈B}(\gamma + T(v)+\alpha * O(v)) = R_{n+1}$  
///
/// $\prod_{v∈B} C_{0}(v) = Z_{0}$   
/// $\dots$   
/// $\prod_{v∈B} C_{n}(v) = Z_{n}$   
/// $\prod_{v∈B} F(v) = Z_{n+1}$
///
/// And check that $\prod_{i=0}^{n+1} L_{i} = \prod_{i=0}^{n+1} R_{i}$ and $\prod_{i=0}^{n+1} Z_{i}$ ≠ $0$    
/// See Section 4.4 of [DP23] for the
/// proofs.
///
/// [DP23]: <https://eprint.iacr.org/2023/1784>
#[allow(clippy::too_many_arguments)]
#[instrument(skip_all, name = "lasso::prove", level = "debug")]
pub fn prove<'a, FC, U, F, L>(
	oracles: &mut MultilinearOracleSet<F>,
	witness_index: MultilinearExtensionIndex<'a, U, F>,
	lasso_claim: &LassoClaim<F>,
	lasso_witness: LassoWitness<'a, PackedType<U, F>, L>,
	lasso_batches: &LassoBatches,
	gamma: F,
	alpha: F,
) -> Result<LassoProveOutput<'a, U, F>, Error>
where
	U: UnderlierType + PackScalar<F> + PackScalar<FC>,
	FC: TowerField,
	PackedType<U, FC>: PackedFieldIndexable,
	PackedType<U, F>: PackedFieldIndexable,
	F: TowerField + ExtensionField<FC>,
	L: AsRef<[usize]>,
{
	let t_n_vars = lasso_claim.t_oracle().n_vars();

	if lasso_claim.u_oracles().len() != lasso_witness.u_polynomials().len() {
		bail!(Error::ClaimWitnessTablesLenMismatch);
	}

	let bit_packing_log_width = PackedType::<U, FC>::LOG_WIDTH;

	let mut final_counts_underlier_vecs: [_; 2] =
		array::from_fn(|_| vec![U::default(); 1 << (t_n_vars - bit_packing_log_width)]);

	let [final_counts, ones_repeating] = final_counts_underlier_vecs.each_mut().map(|underliers| {
		let packed_slice = PackedType::<U, FC>::from_underliers_ref_mut(underliers.as_mut_slice());
		PackedType::<U, FC>::unpack_scalars_mut(packed_slice)
	});

	final_counts.fill(FC::ONE);
	ones_repeating.fill(FC::ONE);

	let common_counts_len = lasso_claim
		.u_oracles()
		.iter()
		.map(|oracle| 1 << oracle.n_vars())
		.sum::<usize>();

	if common_counts_len >= 1 << FC::N_BITS {
		bail!(Error::LassoCountTypeTooSmall);
	}

	let (gkr_claim_oracle_ids, reduced_claim_oracle_ids) =
		reduce_lasso_claim::<FC, _>(oracles, lasso_claim, lasso_batches, gamma, alpha)?;

	let LassoReducedClaimOracleIds {
		ones_oracle_id,
		mixed_t_final_counts_oracle_id,
		mixed_t_one_oracle_id,
		mixed_u_counts_oracle_ids,
		mixed_u_counts_plus_one_oracle_ids,
	} = reduced_claim_oracle_ids;

	let mut witness_index = witness_index;

	let alpha_gen = alpha * FC::MULTIPLICATIVE_GENERATOR;

	for (i, (u_polynomial, u_to_t_mapping)) in lasso_witness
		.u_polynomials()
		.iter()
		.zip(lasso_witness.u_to_t_mappings())
		.enumerate()
	{
		let u_n_vars = u_polynomial.n_vars();

		let mut counts_underlier_vec = vec![U::default(); 1 << (u_n_vars - bit_packing_log_width)];

		let counts = {
			let packed_slice =
				PackedType::<U, FC>::from_underliers_ref_mut(counts_underlier_vec.as_mut_slice());
			PackedType::<U, FC>::unpack_scalars_mut(packed_slice)
		};

		let t_indice = u_to_t_mapping.as_ref();

		for (&t_index, counts) in izip!(t_indice, counts) {
			let count = final_counts[t_index];

			final_counts[t_index] = count * FC::MULTIPLICATIVE_GENERATOR;

			*counts = count;
		}

		let counts = {
			let packed_slice =
				PackedType::<U, FC>::from_underliers_ref_mut(counts_underlier_vec.as_mut_slice());
			PackedType::<U, FC>::unpack_scalars_mut(packed_slice)
		};

		let mixed_u_counts = lincom::<U, FC, F>(u_polynomial, counts, gamma, alpha)?;

		let mixed_u_counts_plus_one = lincom(u_polynomial, counts, gamma, alpha_gen)?;

		witness_index.set_owned::<F, _>([
			(mixed_u_counts_oracle_ids[i], mixed_u_counts),
			(mixed_u_counts_plus_one_oracle_ids[i], mixed_u_counts_plus_one),
		])?;
		witness_index.set_owned::<FC, _>([(lasso_batches.counts[i], counts_underlier_vec)])?;
	}

	let mixed_t_final_counts = lincom(lasso_witness.t_polynomial(), final_counts, gamma, alpha)?;

	let mixed_t_ones = lincom(lasso_witness.t_polynomial(), ones_repeating, gamma, alpha)?;

	let [final_counts_underlier_vecs, ones_repeating] = final_counts_underlier_vecs;

	witness_index.set_owned::<FC, _>([
		(lasso_batches.final_counts, final_counts_underlier_vecs),
		(ones_oracle_id, ones_repeating),
	])?;
	witness_index.set_owned::<F, _>([
		(mixed_t_final_counts_oracle_id, mixed_t_final_counts),
		(mixed_t_one_oracle_id, mixed_t_ones),
	])?;

	let left_witnesses =
		construct_grand_product_witnesses(&gkr_claim_oracle_ids.left, &witness_index)?;
	let left_grand_products = get_grand_products_from_witnesses(&left_witnesses);
	let left_claims =
		construct_grand_product_claims(&gkr_claim_oracle_ids.left, oracles, &left_grand_products)?;

	let right_witnesses =
		construct_grand_product_witnesses(&gkr_claim_oracle_ids.right, &witness_index)?;
	let right_grand_products = get_grand_products_from_witnesses(&right_witnesses);
	let right_claims = construct_grand_product_claims(
		&gkr_claim_oracle_ids.right,
		oracles,
		&right_grand_products,
	)?;

	let counts_witnesses =
		construct_grand_product_witnesses(&gkr_claim_oracle_ids.counts, &witness_index)?;
	let counts_grand_products = get_grand_products_from_witnesses(&counts_witnesses);
	let counts_claims = construct_grand_product_claims(
		&gkr_claim_oracle_ids.counts,
		oracles,
		&counts_grand_products,
	)?;

	let left_product: F = left_grand_products.iter().product();
	let right_product: F = right_grand_products.iter().product();

	if left_product != right_product {
		bail!(Error::ProductsDiffer);
	}

	if counts_grand_products.iter().any(|count| *count == F::ZERO) {
		bail!(Error::ZeroCounts);
	}

	let lasso_proof = LassoProof {
		left_grand_products,
		right_grand_products,
		counts_grand_products,
	};

	let reduced_gpa_claims = [left_claims, right_claims, counts_claims].concat();

	let reduced_gpa_witnesses = [left_witnesses, right_witnesses, counts_witnesses].concat();

	let gpa_metas = [
		gkr_claim_oracle_ids.left,
		gkr_claim_oracle_ids.right,
		gkr_claim_oracle_ids.counts,
	]
	.concat();

	Ok(LassoProveOutput {
		reduced_gpa_claims,
		reduced_gpa_witnesses,
		gpa_metas,
		lasso_proof,
		witness_index,
	})
}

fn lincom<U, FC, F>(
	trace: &MultilinearWitness<PackedType<U, F>>,
	counts: &[FC],
	gamma: F,
	alpha: F,
) -> Result<Arc<[U]>, Error>
where
	U: UnderlierType + PackScalar<F>,
	PackedType<U, F>: PackedFieldIndexable,
	F: Field + ExtensionField<FC>,
	FC: Field,
{
	let n_vars = trace.n_vars();

	let packing_log_width = PackedType::<U, F>::LOG_WIDTH;

	let mut underliers = vec![U::default(); 1 << (n_vars - packing_log_width)];
	let values = PackedType::<U, F>::unpack_scalars_mut(
		PackedType::<U, F>::from_underliers_ref_mut(underliers.as_mut_slice()),
	);

	values.par_iter_mut().enumerate().for_each(|(i, values_i)| {
		*values_i = alpha * counts[i] + gamma;
	});

	values.par_iter_mut().enumerate().try_for_each(
		|(i, values_i)| -> Result<_, PolynomialError> {
			*values_i += trace.evaluate_on_hypercube(i)?;
			Ok(())
		},
	)?;

	Ok(underliers.into())
}
