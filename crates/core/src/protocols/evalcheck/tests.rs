// Copyright 2024 Irreducible Inc.

use crate::{
	oracle::{MultilinearOracleSet, ShiftVariant},
	polynomial::MultivariatePoly,
	protocols::evalcheck::{
		EvalcheckMultilinearClaim, EvalcheckProof, EvalcheckProver, EvalcheckVerifier,
	},
	transparent::select_row::SelectRow,
	witness::MultilinearExtensionIndex,
};
use assert_matches::assert_matches;
use binius_field::{
	packed::{get_packed_slice, len_packed_slice, set_packed_slice},
	underlier::WithUnderlier,
	BinaryField128b, Field, PackedBinaryField128x1b, PackedBinaryField16x8b,
	PackedBinaryField1x128b, PackedBinaryField4x32b, PackedField, TowerField,
};
use binius_hal::{make_portable_backend, ComputationBackendExt};
use binius_math::{extrapolate_line, MultilinearExtension, MultilinearPoly, MultilinearQuery};
use bytemuck::cast_slice_mut;
use itertools::{Either, Itertools};
use rand::{rngs::StdRng, SeedableRng};
use std::iter::repeat_with;

type FExtension = BinaryField128b;
type PExtension = PackedBinaryField1x128b;
type U = <PExtension as WithUnderlier>::Underlier;
type PF = PackedBinaryField4x32b;

#[test]
fn test_evaluation_point_batching() {
	let mut rng = StdRng::seed_from_u64(0);
	let backend = make_portable_backend();

	let log_size = 8;
	let tower_level = 5;

	// random multilinear polys in BF32
	let multilins = repeat_with(|| {
		let evals = repeat_with(|| PF::random(&mut rng))
			.take(1 << (log_size - 2))
			.collect();
		MultilinearExtension::from_values(evals)
			.unwrap()
			.specialize::<PExtension>()
	})
	.take(4)
	.collect::<Vec<_>>();

	// eval point & eval in BF128
	let eval_point = repeat_with(|| <FExtension as Field>::random(&mut rng))
		.take(log_size)
		.collect::<Vec<_>>();

	let query = backend.multilinear_query(&eval_point).unwrap();
	let batch_evals = multilins
		.iter()
		.map(|multilin| multilin.evaluate(query.to_ref()).unwrap())
		.collect::<Vec<_>>();

	let mut oracles = MultilinearOracleSet::new();

	let batch_0_id = oracles.add_committed_batch(log_size, tower_level);
	let [x0, x1] = oracles.add_committed_multiple::<2>(batch_0_id);

	let batch_1_id = oracles.add_committed_batch(log_size, tower_level);
	let [y0, y1] = oracles.add_committed_multiple::<2>(batch_1_id);

	let suboracles = vec![
		oracles.oracle(x0),
		oracles.oracle(y0),
		oracles.oracle(x1),
		oracles.oracle(y1),
	];

	let multilins = suboracles
		.iter()
		.zip_eq(multilins)
		.map(|(s, m)| (s.id(), m.upcast_arc_dyn()));
	let mut witness_index = MultilinearExtensionIndex::<U, FExtension>::new();
	witness_index.update_multilin_poly(multilins).unwrap();

	let claims: Vec<_> = suboracles
		.into_iter()
		.zip(batch_evals.clone())
		.map(|(poly, eval)| EvalcheckMultilinearClaim {
			poly,
			eval_point: eval_point.clone(),
			eval,
		})
		.collect();

	let mut prover_state = EvalcheckProver::new(&mut oracles, &mut witness_index, &backend);
	let proof = prover_state.prove(claims.clone()).unwrap();

	let prove_batch0 = prover_state
		.batch_committed_eval_claims()
		.try_extract_same_query_pcs_claim(0)
		.unwrap()
		.unwrap();

	let prove_batch1 = prover_state
		.batch_committed_eval_claims()
		.try_extract_same_query_pcs_claim(1)
		.unwrap()
		.unwrap();

	let mut verifier_oracles = oracles;
	let mut verifier_state = EvalcheckVerifier::<FExtension>::new(&mut verifier_oracles);
	verifier_state.verify(claims, proof).unwrap();

	let verify_batch0 = verifier_state
		.batch_committed_eval_claims()
		.try_extract_same_query_pcs_claim(0)
		.unwrap()
		.unwrap();

	let verify_batch1 = verifier_state
		.batch_committed_eval_claims()
		.try_extract_same_query_pcs_claim(1)
		.unwrap()
		.unwrap();

	assert_eq!(prove_batch0.eval_point, eval_point);
	assert_eq!(verify_batch0.eval_point, eval_point);

	assert_eq!(prove_batch0.evals, [batch_evals[0], batch_evals[2]]);
	assert_eq!(verify_batch0.evals, [batch_evals[0], batch_evals[2]]);

	assert_eq!(prove_batch1.eval_point, eval_point);
	assert_eq!(verify_batch1.eval_point, eval_point);

	assert_eq!(prove_batch1.evals, [batch_evals[1], batch_evals[3]]);
	assert_eq!(verify_batch1.evals, [batch_evals[1], batch_evals[3]]);
}

fn shift_one<P: PackedField>(evals: &mut [P], bits: usize, variant: ShiftVariant) {
	let len = len_packed_slice(evals);
	assert_eq!(len % (1 << bits), 0);

	for block_idx in 0..len >> bits {
		let range = (block_idx << bits)..((block_idx + 1) << bits);
		let (range, mut last) = match variant {
			ShiftVariant::LogicalLeft => (Either::Left(range), P::Scalar::ZERO),
			ShiftVariant::LogicalRight => (Either::Right(range.rev()), P::Scalar::ZERO),
			ShiftVariant::CircularLeft => {
				let last = get_packed_slice(evals, range.end - 1);
				(Either::Left(range), last)
			}
		};

		for i in range {
			let next = get_packed_slice(evals, i);
			set_packed_slice(evals, i, last);
			last = next;
		}
	}
}

#[test]
fn test_shifted_evaluation_whole_cube() {
	type P = PackedBinaryField16x8b;

	let n_vars = 8;

	let mut oracles = MultilinearOracleSet::<FExtension>::new();
	let batch_id = oracles.add_committed_batch(n_vars, <P as PackedField>::Scalar::TOWER_LEVEL);
	let poly_id = oracles.add_committed(batch_id);

	let shifted_id = oracles
		.add_shifted(poly_id, 1, n_vars, ShiftVariant::CircularLeft)
		.unwrap();

	let mut rng = StdRng::seed_from_u64(0);
	let eval_point = repeat_with(|| <FExtension as Field>::random(&mut rng))
		.take(n_vars)
		.collect::<Vec<_>>();

	let poly_witness = MultilinearExtension::from_values(
		repeat_with(|| P::random(&mut rng))
			.take(1 << (n_vars - P::LOG_WIDTH))
			.collect(),
	)
	.unwrap();

	let mut shifted_evals = poly_witness.evals().to_vec();
	shift_one(&mut shifted_evals, n_vars, ShiftVariant::CircularLeft);
	let shifted_witness = MultilinearExtension::from_values(shifted_evals).unwrap();

	let backend = make_portable_backend();
	let query: MultilinearQuery<BinaryField128b, _> =
		backend.multilinear_query(&eval_point).unwrap();
	let evals =
		[poly_witness.clone(), shifted_witness.clone()].map(|w| w.evaluate(&query).unwrap());

	let claims: Vec<_> = [oracles.oracle(poly_id), oracles.oracle(shifted_id)]
		.into_iter()
		.zip(evals)
		.map(|(poly, eval)| EvalcheckMultilinearClaim {
			poly,
			eval_point: eval_point.clone(),
			eval,
		})
		.collect();

	let mut witness_index = MultilinearExtensionIndex::<U, FExtension>::new();
	witness_index
		.update_multilin_poly(vec![
			(poly_id, poly_witness.to_ref().specialize_arc_dyn::<PExtension>()),
			(shifted_id, shifted_witness.to_ref().specialize_arc_dyn::<PExtension>()),
		])
		.unwrap();

	let mut prover_state = EvalcheckProver::new(&mut oracles, &mut witness_index, &backend);
	let proof = prover_state.prove(claims.clone()).unwrap();
	assert_eq!(
		prover_state
			.batch_committed_eval_claims_mut()
			.take_claims(batch_id)
			.unwrap()
			.len(),
		1
	);

	let mut verifier_state = EvalcheckVerifier::<FExtension>::new(&mut oracles);
	verifier_state.verify(claims.clone(), proof).unwrap();
	assert_eq!(
		verifier_state
			.batch_committed_eval_claims_mut()
			.take_claims(batch_id)
			.unwrap()
			.len(),
		1
	);
}

#[test]
fn test_shifted_evaluation_subcube() {
	type P = PackedBinaryField16x8b;

	let n_vars = 8;

	let mut oracles = MultilinearOracleSet::<FExtension>::new();

	let batch_id = oracles.add_committed_batch(n_vars, <P as PackedField>::Scalar::TOWER_LEVEL);
	let poly_id = oracles.add_committed(batch_id);

	let shifted_id = oracles
		.add_shifted(poly_id, 3, 4, ShiftVariant::CircularLeft)
		.unwrap();

	let mut rng = StdRng::seed_from_u64(0);
	let eval_point = repeat_with(|| <FExtension as Field>::random(&mut rng))
		.take(n_vars)
		.collect::<Vec<_>>();

	let poly_witness = MultilinearExtension::from_values(
		repeat_with(|| P::random(&mut rng))
			.take(1 << (n_vars - P::LOG_WIDTH))
			.collect(),
	)
	.unwrap();

	let mut shifted_evals = poly_witness.evals().to_vec();
	for subcube in cast_slice_mut::<_, u16>(&mut shifted_evals).iter_mut() {
		*subcube = subcube.wrapping_shl(3);
	}
	let shifted_witness = MultilinearExtension::from_values(shifted_evals).unwrap();

	let backend = make_portable_backend();
	let query = backend
		.multilinear_query::<BinaryField128b>(&eval_point)
		.unwrap();
	let evals =
		[poly_witness.clone(), shifted_witness.clone()].map(|w| w.evaluate(&query).unwrap());

	let claims: Vec<_> = [oracles.oracle(poly_id), oracles.oracle(shifted_id)]
		.into_iter()
		.zip(evals)
		.map(|(poly, eval)| EvalcheckMultilinearClaim {
			poly,
			eval_point: eval_point.clone(),
			eval,
		})
		.collect();

	let mut witness_index = MultilinearExtensionIndex::<U, FExtension>::new();
	witness_index
		.update_multilin_poly(vec![
			(poly_id, poly_witness.to_ref().specialize_arc_dyn::<PExtension>()),
			(shifted_id, shifted_witness.to_ref().specialize_arc_dyn::<PExtension>()),
		])
		.unwrap();

	let mut prover_state = EvalcheckProver::new(&mut oracles, &mut witness_index, &backend);
	let proof = prover_state.prove(claims.clone()).unwrap();
	assert_eq!(
		prover_state
			.batch_committed_eval_claims_mut()
			.take_claims(batch_id)
			.unwrap()
			.len(),
		1
	);

	let mut verifier_state = EvalcheckVerifier::<FExtension>::new(&mut oracles);
	verifier_state.verify(claims, proof).unwrap();
	assert_eq!(
		verifier_state
			.batch_committed_eval_claims_mut()
			.take_claims(batch_id)
			.unwrap()
			.len(),
		1
	);
}

#[test]
fn test_evalcheck_linear_combination() {
	let n_vars = 8;

	let select_row1 = SelectRow::new(n_vars, 0).unwrap();
	let select_row2 = SelectRow::new(n_vars, 5).unwrap();
	let select_row3 = SelectRow::new(n_vars, 10).unwrap();

	let mut oracles = MultilinearOracleSet::new();

	let select_row1_oracle_id = oracles.add_transparent(select_row1.clone()).unwrap();
	let select_row2_oracle_id = oracles.add_transparent(select_row2.clone()).unwrap();
	let select_row3_oracle_id = oracles.add_transparent(select_row3.clone()).unwrap();

	let lin_com_id = oracles
		.add_linear_combination_with_offset(
			n_vars,
			FExtension::new(1),
			[
				(select_row1_oracle_id, FExtension::new(2)),
				(select_row2_oracle_id, FExtension::new(3)),
				(select_row3_oracle_id, FExtension::new(4)),
			],
		)
		.unwrap();
	let lin_com = oracles.oracle(lin_com_id);

	let mut rng = StdRng::seed_from_u64(0);
	let eval_point = repeat_with(|| <FExtension as Field>::random(&mut rng))
		.take(n_vars)
		.collect::<Vec<_>>();

	let eval = select_row1.evaluate(&eval_point).unwrap() * FExtension::new(2)
		+ select_row2.evaluate(&eval_point).unwrap() * FExtension::new(3)
		+ select_row3.evaluate(&eval_point).unwrap() * FExtension::new(4)
		+ FExtension::new(1);

	let select_row1_witness = select_row1
		.multilinear_extension::<PackedBinaryField128x1b>()
		.unwrap();
	let select_row2_witness = select_row2
		.multilinear_extension::<PackedBinaryField128x1b>()
		.unwrap();
	let select_row3_witness = select_row3
		.multilinear_extension::<PackedBinaryField128x1b>()
		.unwrap();

	let lin_com_values = (0..1 << n_vars)
		.map(|i| {
			select_row1_witness.evaluate_on_hypercube(i).unwrap() * FExtension::new(2)
				+ select_row2_witness.evaluate_on_hypercube(i).unwrap() * FExtension::new(3)
				+ select_row3_witness.evaluate_on_hypercube(i).unwrap() * FExtension::new(4)
				+ FExtension::new(1)
		})
		.map(PackedBinaryField1x128b::set_single)
		.collect();
	let lin_com_witness = MultilinearExtension::from_values(lin_com_values).unwrap();

	// Make the claim a composite oracle over a linear combination, in order to test the case
	// of requiring nested composite evalcheck proofs.
	let claim_oracle = lin_com.clone();

	let claim = EvalcheckMultilinearClaim {
		poly: claim_oracle,
		eval_point,
		eval,
	};

	let mut witness_index = MultilinearExtensionIndex::<U, FExtension>::new();
	witness_index
		.update_multilin_poly(vec![
			(select_row1_oracle_id, select_row1_witness.specialize_arc_dyn()),
			(select_row2_oracle_id, select_row2_witness.specialize_arc_dyn()),
			(select_row3_oracle_id, select_row3_witness.specialize_arc_dyn()),
			(lin_com_id, lin_com_witness.specialize_arc_dyn()),
		])
		.unwrap();

	let backend = make_portable_backend();
	let mut prover_state = EvalcheckProver::new(&mut oracles, &mut witness_index, &backend);
	let proof = prover_state.prove(vec![claim.clone()]).unwrap();

	let mut verifier_state = EvalcheckVerifier::<FExtension>::new(&mut oracles);
	verifier_state.verify(vec![claim], proof).unwrap();
}

#[test]
fn test_evalcheck_repeating() {
	let n_vars = 7;
	let row_id = 11;

	let mut oracles = MultilinearOracleSet::new();

	let select_row = SelectRow::new(n_vars, row_id).unwrap();
	let select_row_oracle_id = oracles.add_transparent(select_row.clone()).unwrap();

	let select_row_subwitness = select_row
		.multilinear_extension::<PackedBinaryField128x1b>()
		.unwrap();
	let repeated_values = (0..4)
		.flat_map(|_| select_row_subwitness.evals().iter().copied())
		.collect::<Vec<_>>();

	let select_row_witness = MultilinearExtension::from_values(repeated_values)
		.unwrap()
		.specialize_arc_dyn();

	let repeating_id = oracles.add_repeating(select_row_oracle_id, 2).unwrap();
	let repeating = oracles.oracle(repeating_id);

	let mut rng = StdRng::seed_from_u64(0);
	let eval_point = repeat_with(|| <FExtension as Field>::random(&mut rng))
		.take(n_vars + 2)
		.collect::<Vec<_>>();

	let eval = select_row.evaluate(&eval_point[..n_vars]).unwrap();

	let claim = EvalcheckMultilinearClaim {
		poly: repeating.clone(),
		eval_point,
		eval,
	};

	let mut witness_index = MultilinearExtensionIndex::<U, FExtension>::new();
	witness_index
		.update_multilin_poly(vec![(repeating_id, select_row_witness)])
		.unwrap();

	let backend = make_portable_backend();
	let mut prover_state = EvalcheckProver::new(&mut oracles, &mut witness_index, &backend);
	let proof = prover_state.prove(vec![claim.clone()]).unwrap();

	assert_matches!(proof[0], EvalcheckProof::Repeating(..));

	let mut verifier_state = EvalcheckVerifier::<FExtension>::new(&mut oracles);
	verifier_state.verify(vec![claim.clone()], proof).unwrap();
}

#[test]
/// Constructs a small Merged oracle, proves and verifies it.
fn test_evalcheck_merged() {
	let n_vars = 7;
	let row1_id = 9;
	let row2_id = 10;

	let mut oracles = MultilinearOracleSet::new();

	let select_row1 = SelectRow::new(n_vars, row1_id).unwrap();
	let select_row2 = SelectRow::new(n_vars, row2_id).unwrap();

	let select_row1_subwitness = select_row1
		.multilinear_extension::<PackedBinaryField128x1b>()
		.unwrap();
	let select_row2_subwitness = select_row2
		.multilinear_extension::<PackedBinaryField128x1b>()
		.unwrap();

	let merged_witness = {
		let x = select_row1_subwitness
			.clone()
			.specialize::<PackedBinaryField128x1b>();
		let y = select_row2_subwitness
			.clone()
			.specialize::<PackedBinaryField128x1b>();
		assert!(x.n_vars() >= PackedBinaryField128x1b::LOG_WIDTH);
		assert_eq!(x.n_vars(), y.n_vars());
		let mut values = vec![
			PackedBinaryField128x1b::zero();
			1 << (x.n_vars() + 1 - PackedBinaryField128x1b::LOG_WIDTH)
		];
		let (x_values, y_values) =
			values.split_at_mut(1 << (x.n_vars() - PackedBinaryField128x1b::LOG_WIDTH));
		x.subcube_evals(x.n_vars(), 0, 0, x_values).unwrap();
		y.subcube_evals(y.n_vars(), 0, 0, y_values).unwrap();
		let merge_poly = MultilinearExtension::from_values(values).unwrap();
		merge_poly.specialize()
	};

	let select_row1_oracle_id = oracles.add_transparent(select_row1.clone()).unwrap();
	let select_row2_oracle_id = oracles.add_transparent(select_row2.clone()).unwrap();

	let merged_id = oracles
		.add_merged(select_row1_oracle_id, select_row2_oracle_id)
		.unwrap();
	let merged = oracles.oracle(merged_id);

	let mut witness_index = MultilinearExtensionIndex::<U, FExtension>::new();
	witness_index
		.update_multilin_poly(vec![
			(select_row1_oracle_id, select_row1_subwitness.specialize_arc_dyn()),
			(select_row2_oracle_id, select_row2_subwitness.specialize_arc_dyn()),
			(merged_id, merged_witness.upcast_arc_dyn()),
		])
		.unwrap();

	let mut rng = StdRng::seed_from_u64(0);
	let eval_point = repeat_with(|| <FExtension as Field>::random(&mut rng))
		.take(n_vars + 1)
		.collect::<Vec<_>>();

	let inner_eval_point = &eval_point[..n_vars];
	let eval1 = select_row1.evaluate(inner_eval_point).unwrap();
	let eval2 = select_row2.evaluate(inner_eval_point).unwrap();

	let eval = extrapolate_line(eval1, eval2, eval_point[n_vars]);
	let claim = EvalcheckMultilinearClaim {
		poly: merged,
		eval_point,
		eval,
	};

	let backend = make_portable_backend();
	let mut prover_state = EvalcheckProver::new(&mut oracles, &mut witness_index, &backend);
	let proof = prover_state.prove(vec![claim.clone()]).unwrap();

	assert_matches!(proof[0], EvalcheckProof::Merged { .. });

	let mut verifier_state = EvalcheckVerifier::<FExtension>::new(&mut oracles);
	verifier_state.verify(vec![claim], proof).unwrap();
}

#[test]
/// Constructs a small Interleaved oracle, proves and verifies it.
fn test_evalcheck_interleaved() {
	let n_vars = 7;
	let row1_id = 0;
	let row2_id = 1;

	let mut oracles = MultilinearOracleSet::new();

	let select_row1 = SelectRow::new(n_vars, row1_id).unwrap();
	let select_row2 = SelectRow::new(n_vars, row2_id).unwrap();

	let select_row1_subwitness = select_row1
		.multilinear_extension::<PackedBinaryField128x1b>()
		.unwrap();
	let select_row2_subwitness = select_row2
		.multilinear_extension::<PackedBinaryField128x1b>()
		.unwrap();

	let interleaved_witness = {
		let x = select_row1_subwitness
			.clone()
			.specialize::<PackedBinaryField128x1b>();
		let y = select_row2_subwitness
			.clone()
			.specialize::<PackedBinaryField128x1b>();
		assert!(x.n_vars() >= PackedBinaryField128x1b::LOG_WIDTH);
		assert_eq!(x.n_vars(), y.n_vars());
		let mut values = vec![
			PackedBinaryField128x1b::zero();
			1 << (x.n_vars() + 1 - PackedBinaryField128x1b::LOG_WIDTH)
		];
		let (x_values, y_values) =
			values.split_at_mut(1 << (x.n_vars() - PackedBinaryField128x1b::LOG_WIDTH));
		x.subcube_evals(x.n_vars(), 0, 0, x_values).unwrap();
		y.subcube_evals(y.n_vars(), 0, 0, y_values).unwrap();
		let mut values2 = vec![
			PackedBinaryField128x1b::zero();
			1 << (x.n_vars() + 1 - PackedBinaryField128x1b::LOG_WIDTH)
		];
		for i in 0..(1 << x.n_vars()) {
			set_packed_slice(&mut values2, i * 2, get_packed_slice(x_values, i));
			set_packed_slice(&mut values2, i * 2 + 1, get_packed_slice(y_values, i));
		}
		let poly = MultilinearExtension::from_values(values2).unwrap();
		poly.specialize()
	};

	let select_row1_oracle_id = oracles.add_transparent(select_row1.clone()).unwrap();
	let select_row2_oracle_id = oracles.add_transparent(select_row2.clone()).unwrap();

	let interleaved_id = oracles
		.add_interleaved(select_row1_oracle_id, select_row2_oracle_id)
		.unwrap();
	let interleaved = oracles.oracle(interleaved_id);

	let mut witness_index = MultilinearExtensionIndex::<U, FExtension>::new();
	witness_index
		.update_multilin_poly(vec![
			(select_row1_oracle_id, select_row1_subwitness.specialize_arc_dyn()),
			(select_row2_oracle_id, select_row2_subwitness.specialize_arc_dyn()),
			(interleaved_id, interleaved_witness.upcast_arc_dyn()),
		])
		.unwrap();

	let mut rng = StdRng::seed_from_u64(0);
	let eval_point = repeat_with(|| <FExtension as Field>::random(&mut rng))
		.take(n_vars + 1)
		.collect::<Vec<_>>();

	let inner_eval_point = &eval_point[1..];
	let eval1 = select_row1.evaluate(inner_eval_point).unwrap();
	let eval2 = select_row2.evaluate(inner_eval_point).unwrap();

	let eval = extrapolate_line(eval1, eval2, eval_point[0]);
	let claim = EvalcheckMultilinearClaim {
		poly: interleaved,
		eval_point,
		eval,
	};

	let backend = make_portable_backend();
	let mut prover_state = EvalcheckProver::new(&mut oracles, &mut witness_index, &backend);
	let proof = prover_state.prove(vec![claim.clone()]).unwrap();

	assert_matches!(proof[0], EvalcheckProof::Interleaved { .. });

	let mut verifier_state = EvalcheckVerifier::<FExtension>::new(&mut oracles);
	verifier_state.verify(vec![claim], proof).unwrap();
}

#[test]
/// Constructs a small ZeroPadded oracle, proves and verifies it.
fn test_evalcheck_zero_padded() {
	let inner_n_vars = 7;
	let n_vars = 10;
	let row_id = 9;

	let mut oracles = MultilinearOracleSet::new();

	let select_row = SelectRow::new(inner_n_vars, row_id).unwrap();

	let select_row_subwitness = select_row
		.multilinear_extension::<PackedBinaryField128x1b>()
		.unwrap();

	let x = select_row_subwitness
		.clone()
		.specialize::<PackedBinaryField128x1b>();

	assert!(x.n_vars() >= PackedBinaryField128x1b::LOG_WIDTH);

	let mut values =
		vec![PackedBinaryField128x1b::zero(); 1 << (n_vars - PackedBinaryField128x1b::LOG_WIDTH)];

	let values_len = values.len();

	let (_, x_values) =
		values.split_at_mut(values_len - (1 << (x.n_vars() - PackedBinaryField128x1b::LOG_WIDTH)));

	x.subcube_evals(x.n_vars(), 0, 0, x_values).unwrap();

	let zero_padded_poly = MultilinearExtension::from_values(values).unwrap();

	let select_row_oracle_id = oracles.add_transparent(select_row.clone()).unwrap();

	let zero_padded_id = oracles
		.add_zero_padded(select_row_oracle_id, n_vars)
		.unwrap();

	let zero_padded = oracles.oracle(zero_padded_id);

	let mut witness_index = MultilinearExtensionIndex::<U, FExtension>::new();
	witness_index
		.update_multilin_poly(vec![
			(select_row_oracle_id, select_row_subwitness.specialize_arc_dyn()),
			(zero_padded_id, zero_padded_poly.specialize_arc_dyn()),
		])
		.unwrap();

	let mut rng = StdRng::seed_from_u64(0);

	let eval_point = repeat_with(|| <FExtension as Field>::random(&mut rng))
		.take(n_vars)
		.collect::<Vec<_>>();

	let inner_eval_point = &eval_point[..inner_n_vars];
	let eval = select_row.evaluate(inner_eval_point).unwrap();

	let mut inner_eval = eval;

	for i in 0..n_vars - inner_n_vars {
		inner_eval = extrapolate_line::<BinaryField128b, BinaryField128b>(
			BinaryField128b::zero(),
			inner_eval,
			eval_point[inner_n_vars + i],
		);
	}

	let claim = EvalcheckMultilinearClaim {
		poly: zero_padded,
		eval_point: eval_point.clone(),
		eval: inner_eval,
	};

	let backend = make_portable_backend();
	let mut prover_state = EvalcheckProver::new(&mut oracles, &mut witness_index, &backend);
	let proof = prover_state.prove(vec![claim.clone()]).unwrap();

	assert_matches!(proof[0], EvalcheckProof::ZeroPadded { .. });

	let mut verifier_state = EvalcheckVerifier::<FExtension>::new(&mut oracles);
	verifier_state.verify(vec![claim], proof).unwrap();
}
