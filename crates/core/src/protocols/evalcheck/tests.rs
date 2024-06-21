// Copyright 2024 Ulvetanna Inc.

use crate::{
	oracle::{
		CommittedBatchSpec, CommittedId, CompositePolyOracle, MultilinearOracleSet,
		MultilinearPolyOracle, ProjectionVariant, ShiftVariant,
	},
	polynomial::{
		composition::BivariateProduct, extrapolate_line, transparent::select_row::SelectRow,
		CompositionPoly, Error as PolynomialError, MultilinearComposite, MultilinearExtension,
		MultilinearPoly, MultilinearQuery, MultivariatePoly,
	},
	protocols::{
		evalcheck::{EvalcheckClaim, EvalcheckProof, EvalcheckProver, EvalcheckVerifier},
		sumcheck::SumcheckClaim,
	},
	witness::MultilinearWitnessIndex,
};
use assert_matches::assert_matches;
use binius_field::{
	packed::{get_packed_slice, len_packed_slice, set_packed_slice},
	BinaryField128b, Field, PackedBinaryField128x1b, PackedBinaryField16x8b,
	PackedBinaryField4x32b, PackedField, TowerField,
};
use bytemuck::cast_slice_mut;
use itertools::Either;
use rand::{rngs::StdRng, SeedableRng};
use std::iter::repeat_with;

type EF = BinaryField128b;
type PF = PackedBinaryField4x32b;

#[derive(Clone, Debug)]
struct QuadProduct;

impl CompositionPoly<EF> for QuadProduct {
	fn n_vars(&self) -> usize {
		4
	}

	fn degree(&self) -> usize {
		4
	}

	fn evaluate_scalar(&self, query: &[EF]) -> Result<EF, PolynomialError> {
		if query.len() != 4 {
			return Err(PolynomialError::IncorrectQuerySize { expected: 4 });
		}
		let (a, b, c, d) = (query[0], query[1], query[2], query[3]);
		Ok(a * b * c * d)
	}

	fn evaluate(&self, query: &[EF]) -> Result<EF, PolynomialError> {
		if query.len() != 4 {
			return Err(PolynomialError::IncorrectQuerySize { expected: 4 });
		}
		let (a, b, c, d) = (query[0], query[1], query[2], query[3]);
		Ok(a * b * c * d)
	}

	fn binary_tower_level(&self) -> usize {
		0
	}
}

#[test]
fn test_evaluation_point_batching() {
	binius_utils::tracing::init_tracing();

	let mut rng = StdRng::seed_from_u64(0);

	let log_size = 8;
	let tower_level = 5;

	// random multilinear polys in BF32
	let multilins = repeat_with(|| {
		let evals = repeat_with(|| PF::random(&mut rng))
			.take(1 << (log_size - 2))
			.collect();
		MultilinearExtension::from_values(evals)
			.unwrap()
			.specialize::<EF>()
	})
	.take(4)
	.collect::<Vec<_>>();

	// eval point & eval in BF128
	let eval_point = repeat_with(|| <EF as PackedField>::random(&mut rng))
		.take(log_size)
		.collect::<Vec<_>>();

	let query = MultilinearQuery::with_full_query(&eval_point).unwrap();
	let batch_evals = multilins
		.iter()
		.map(|multilin| multilin.evaluate(&query).unwrap())
		.collect::<Vec<_>>();

	let eval = batch_evals.iter().fold(EF::ONE, |acc, cur| acc * cur);

	let mut oracles = MultilinearOracleSet::new();
	let batch_0_id = oracles.add_committed_batch(CommittedBatchSpec {
		n_vars: log_size,
		n_polys: 2,
		tower_level,
	});
	let batch_1_id = oracles.add_committed_batch(CommittedBatchSpec {
		n_vars: log_size,
		n_polys: 2,
		tower_level,
	});

	let suboracles = vec![
		oracles.committed_oracle(CommittedId {
			batch_id: batch_0_id,
			index: 0,
		}),
		oracles.committed_oracle(CommittedId {
			batch_id: batch_1_id,
			index: 0,
		}),
		oracles.committed_oracle(CommittedId {
			batch_id: batch_0_id,
			index: 1,
		}),
		oracles.committed_oracle(CommittedId {
			batch_id: batch_1_id,
			index: 1,
		}),
	];

	let mut witness_index = MultilinearWitnessIndex::new();
	for (oracle, multilin) in suboracles.iter().zip(multilins.into_iter()) {
		witness_index.set(oracle.id(), multilin.upcast_arc_dyn());
	}

	let oracle = CompositePolyOracle::new(log_size, suboracles.clone(), QuadProduct).unwrap();

	let claim = EvalcheckClaim {
		poly: oracle,
		eval_point: eval_point.clone(),
		eval,
		is_random_point: true,
	};

	let mut prover_oracles = oracles.clone();
	let mut prover_state = EvalcheckProver::new(&mut prover_oracles, &mut witness_index);
	let proof = prover_state.prove(claim.clone()).unwrap();

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
	let mut verifier_state = EvalcheckVerifier::new(&mut verifier_oracles);
	verifier_state.verify(claim, proof).unwrap();

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

	let mut oracles = MultilinearOracleSet::<EF>::new();

	let batch_id = oracles.add_committed_batch(CommittedBatchSpec {
		n_vars,
		n_polys: 1,
		tower_level: <P as PackedField>::Scalar::TOWER_LEVEL,
	});

	let poly_id = oracles.committed_oracle_id(CommittedId { batch_id, index: 0 });
	let shifted_id = oracles
		.add_shifted(poly_id, 1, n_vars, ShiftVariant::CircularLeft)
		.unwrap();

	let composite = CompositePolyOracle::new(
		n_vars,
		vec![oracles.oracle(poly_id), oracles.oracle(shifted_id)],
		BivariateProduct,
	)
	.unwrap();

	let mut rng = StdRng::seed_from_u64(0);
	let eval_point = repeat_with(|| <EF as Field>::random(&mut rng))
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

	let composite_witness = MultilinearComposite::new(
		n_vars,
		BivariateProduct,
		vec![
			poly_witness.to_ref().specialize::<EF>(),
			shifted_witness.to_ref().specialize::<EF>(),
		],
	)
	.unwrap();

	let query = MultilinearQuery::with_full_query(&eval_point).unwrap();
	let eval = composite_witness.evaluate(&query).unwrap();

	let claim = EvalcheckClaim {
		poly: composite,
		eval_point,
		eval,
		is_random_point: true,
	};

	let mut witness_index = MultilinearWitnessIndex::new();

	witness_index.set(poly_id, poly_witness.to_ref().specialize_arc_dyn::<EF>());
	witness_index.set(shifted_id, shifted_witness.to_ref().specialize_arc_dyn::<EF>());

	let mut prover_state = EvalcheckProver::new(&mut oracles, &mut witness_index);
	let proof = prover_state.prove(claim.clone()).unwrap();
	assert_eq!(
		prover_state
			.batch_committed_eval_claims_mut()
			.take_claims(batch_id)
			.unwrap()
			.len(),
		1
	);

	let mut verifier_state = EvalcheckVerifier::new(&mut oracles);
	verifier_state.verify(claim, proof).unwrap();
	assert_eq!(verifier_state.new_sumcheck_claims().len(), 1);
	assert_eq!(
		verifier_state
			.batch_committed_eval_claims_mut()
			.take_claims(batch_id)
			.unwrap()
			.len(),
		1
	);

	let SumcheckClaim {
		poly: composite,
		sum,
	} = verifier_state.new_sumcheck_claims().first().unwrap();

	assert_eq!(composite.inner_polys()[0].id(), poly_id);
	assert_eq!(*sum, shifted_witness.evaluate(&query).unwrap());
}

#[test]
fn test_shifted_evaluation_subcube() {
	type P = PackedBinaryField16x8b;

	let n_vars = 8;

	let mut oracles = MultilinearOracleSet::<EF>::new();

	let batch_id = oracles.add_committed_batch(CommittedBatchSpec {
		n_vars,
		n_polys: 1,
		tower_level: <P as PackedField>::Scalar::TOWER_LEVEL,
	});

	let poly_id = oracles.committed_oracle_id(CommittedId { batch_id, index: 0 });
	let shifted_id = oracles
		.add_shifted(poly_id, 3, 4, ShiftVariant::CircularLeft)
		.unwrap();

	let composite = CompositePolyOracle::new(
		n_vars,
		vec![oracles.oracle(poly_id), oracles.oracle(shifted_id)],
		BivariateProduct,
	)
	.unwrap();

	let mut rng = StdRng::seed_from_u64(0);
	let eval_point = repeat_with(|| <EF as Field>::random(&mut rng))
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

	let composite_witness = MultilinearComposite::new(
		n_vars,
		BivariateProduct,
		vec![
			poly_witness.to_ref().specialize::<EF>(),
			shifted_witness.to_ref().specialize::<EF>(),
		],
	)
	.unwrap();

	let query = MultilinearQuery::with_full_query(&eval_point).unwrap();
	let eval = composite_witness.evaluate(&query).unwrap();

	let claim = EvalcheckClaim {
		poly: composite,
		eval_point: eval_point.clone(),
		eval,
		is_random_point: true,
	};

	let mut witness_index = MultilinearWitnessIndex::new();

	witness_index.set(poly_id, poly_witness.to_ref().specialize_arc_dyn::<EF>());
	witness_index.set(shifted_id, shifted_witness.to_ref().specialize_arc_dyn::<EF>());

	let mut prover_state = EvalcheckProver::new(&mut oracles, &mut witness_index);
	let proof = prover_state.prove(claim.clone()).unwrap();
	assert_eq!(
		prover_state
			.batch_committed_eval_claims_mut()
			.take_claims(batch_id)
			.unwrap()
			.len(),
		1
	);

	let mut verifier_state = EvalcheckVerifier::new(&mut oracles);
	verifier_state.verify(claim, proof).unwrap();
	assert_eq!(
		verifier_state
			.batch_committed_eval_claims_mut()
			.take_claims(batch_id)
			.unwrap()
			.len(),
		1
	);
	assert_eq!(verifier_state.new_sumcheck_claims().len(), 1);

	let SumcheckClaim {
		poly: composite,
		sum,
	} = verifier_state.new_sumcheck_claims().first().unwrap();

	match composite.inner_polys()[0] {
		MultilinearPolyOracle::Projected(_, ref projected) => {
			assert_eq!(projected.inner().id(), poly_id);
			assert_eq!(projected.values(), &eval_point[4..]);
			assert_eq!(projected.projection_variant(), ProjectionVariant::LastVars);
		}
		_ => panic!("expected sumcheck on projection"),
	}

	assert_eq!(*sum, shifted_witness.evaluate(&query).unwrap());
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
			EF::new(1),
			[
				(select_row1_oracle_id, EF::new(2)),
				(select_row2_oracle_id, EF::new(3)),
				(select_row3_oracle_id, EF::new(4)),
			],
		)
		.unwrap();
	let lin_com = oracles.oracle(lin_com_id);

	let mut rng = StdRng::seed_from_u64(0);
	let eval_point = repeat_with(|| <EF as Field>::random(&mut rng))
		.take(n_vars)
		.collect::<Vec<_>>();

	let eval = select_row1.evaluate(&eval_point).unwrap() * EF::new(2)
		+ select_row2.evaluate(&eval_point).unwrap() * EF::new(3)
		+ select_row3.evaluate(&eval_point).unwrap() * EF::new(4)
		+ EF::new(1);

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
			select_row1_witness.evaluate_on_hypercube(i).unwrap() * EF::new(2)
				+ select_row2_witness.evaluate_on_hypercube(i).unwrap() * EF::new(3)
				+ select_row3_witness.evaluate_on_hypercube(i).unwrap() * EF::new(4)
				+ EF::new(1)
		})
		.collect();
	let lin_com_witness = MultilinearExtension::from_values(lin_com_values).unwrap();

	// Make the claim a composite oracle over a linear combination, in order to test the case
	// of requiring nested composite evalcheck proofs.
	let claim_oracle = lin_com.clone();
	let claim_oracle = claim_oracle.into_composite();

	let claim = EvalcheckClaim {
		poly: claim_oracle,
		eval_point,
		eval,
		is_random_point: true,
	};

	let mut witness = MultilinearWitnessIndex::<EF>::new();
	witness.set(select_row1_oracle_id, select_row1_witness.specialize_arc_dyn());
	witness.set(select_row2_oracle_id, select_row2_witness.specialize_arc_dyn());
	witness.set(select_row3_oracle_id, select_row3_witness.specialize_arc_dyn());
	witness.set(lin_com_id, lin_com_witness.specialize_arc_dyn());

	let mut prover_state = EvalcheckProver::new(&mut oracles, &mut witness);
	let proof = prover_state.prove(claim.clone()).unwrap();

	let mut verifier_state = EvalcheckVerifier::new(&mut oracles);
	verifier_state.verify(claim, proof).unwrap();
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

	let mut witness_index = MultilinearWitnessIndex::<EF>::new();

	let mut rng = StdRng::seed_from_u64(0);
	let eval_point = repeat_with(|| <EF as Field>::random(&mut rng))
		.take(n_vars + 2)
		.collect::<Vec<_>>();

	let eval = select_row.evaluate(&eval_point[..n_vars]).unwrap();

	let claim = EvalcheckClaim {
		poly: repeating.clone().into_composite(),
		eval_point,
		eval,
		is_random_point: true,
	};

	witness_index.set(repeating_id, select_row_witness);

	let mut prover_state = EvalcheckProver::new(&mut oracles, &mut witness_index);
	let proof = prover_state.prove(claim.clone()).unwrap();

	if let EvalcheckProof::Composite { ref subproofs } = proof {
		assert_eq!(subproofs.len(), 1);
		assert_matches!(subproofs[0].1, EvalcheckProof::Repeating(..));
	} else {
		panic!("Proof should be Composite.");
	}

	let mut verifier_state = EvalcheckVerifier::new(&mut oracles);
	verifier_state.verify(claim, proof).unwrap();
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
		x.subcube_evals(x.n_vars(), 0, x_values).unwrap();
		y.subcube_evals(y.n_vars(), 0, y_values).unwrap();
		let merge_poly = MultilinearExtension::from_values(values).unwrap();
		let res = merge_poly.specialize();
		res
	};

	let select_row1_oracle_id = oracles.add_transparent(select_row1.clone()).unwrap();
	let select_row2_oracle_id = oracles.add_transparent(select_row2.clone()).unwrap();

	let merged_id = oracles
		.add_merged(select_row1_oracle_id, select_row2_oracle_id)
		.unwrap();
	let merged = oracles.oracle(merged_id);

	let mut witness_index = MultilinearWitnessIndex::<EF>::new();

	let select_row1_subwitness = select_row1_subwitness.specialize_arc_dyn();
	let select_row2_subwitness = select_row2_subwitness.specialize_arc_dyn();

	witness_index.set(select_row1_oracle_id, select_row1_subwitness);
	witness_index.set(select_row2_oracle_id, select_row2_subwitness);
	witness_index.set(merged_id, merged_witness.upcast_arc_dyn());

	let mut rng = StdRng::seed_from_u64(0);
	let eval_point = repeat_with(|| <EF as Field>::random(&mut rng))
		.take(n_vars + 1)
		.collect::<Vec<_>>();

	let inner_eval_point = &eval_point[..n_vars];
	let eval1 = select_row1.evaluate(inner_eval_point).unwrap();
	let eval2 = select_row2.evaluate(inner_eval_point).unwrap();

	let eval = extrapolate_line(eval1, eval2, eval_point[n_vars]);
	let claim = EvalcheckClaim {
		poly: merged.into_composite(),
		eval_point,
		eval,
		is_random_point: true,
	};

	let mut prover_state = EvalcheckProver::new(&mut oracles, &mut witness_index);
	let proof = prover_state.prove(claim.clone()).unwrap();

	if let EvalcheckProof::Composite { ref subproofs } = proof {
		assert_eq!(subproofs.len(), 1);
		assert_matches!(subproofs[0].1, EvalcheckProof::Merged { .. });
	} else {
		panic!("Proof should be Composite.");
	}

	let mut verifier_state = EvalcheckVerifier::new(&mut oracles);
	verifier_state.verify(claim, proof).unwrap();
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
		x.subcube_evals(x.n_vars(), 0, x_values).unwrap();
		y.subcube_evals(y.n_vars(), 0, y_values).unwrap();
		let mut values2 = vec![
			PackedBinaryField128x1b::zero();
			1 << (x.n_vars() + 1 - PackedBinaryField128x1b::LOG_WIDTH)
		];
		for i in 0..(1 << x.n_vars()) {
			set_packed_slice(&mut values2, i * 2, get_packed_slice(x_values, i));
			set_packed_slice(&mut values2, i * 2 + 1, get_packed_slice(y_values, i));
		}
		let poly = MultilinearExtension::from_values(values2).unwrap();
		let res = poly.specialize();
		res
	};

	let select_row1_oracle_id = oracles.add_transparent(select_row1.clone()).unwrap();
	let select_row2_oracle_id = oracles.add_transparent(select_row2.clone()).unwrap();

	let interleaved_id = oracles
		.add_interleaved(select_row1_oracle_id, select_row2_oracle_id)
		.unwrap();
	let interleaved = oracles.oracle(interleaved_id);

	let mut witness_index = MultilinearWitnessIndex::<EF>::new();

	let select_row1_subwitness = select_row1_subwitness.specialize_arc_dyn();
	let select_row2_subwitness = select_row2_subwitness.specialize_arc_dyn();

	witness_index.set(select_row1_oracle_id, select_row1_subwitness);
	witness_index.set(select_row2_oracle_id, select_row2_subwitness);
	witness_index.set(interleaved_id, interleaved_witness.upcast_arc_dyn());

	let mut rng = StdRng::seed_from_u64(0);
	let eval_point = repeat_with(|| <EF as Field>::random(&mut rng))
		.take(n_vars + 1)
		.collect::<Vec<_>>();

	let inner_eval_point = &eval_point[1..];
	let eval1 = select_row1.evaluate(inner_eval_point).unwrap();
	let eval2 = select_row2.evaluate(inner_eval_point).unwrap();

	let eval = extrapolate_line(eval1, eval2, eval_point[0]);
	let claim = EvalcheckClaim {
		poly: interleaved.into_composite(),
		eval_point,
		eval,
		is_random_point: true,
	};

	let mut prover_state = EvalcheckProver::new(&mut oracles, &mut witness_index);
	let proof = prover_state.prove(claim.clone()).unwrap();

	if let EvalcheckProof::Composite { ref subproofs } = proof {
		assert_eq!(subproofs.len(), 1);
		assert_matches!(subproofs[0].1, EvalcheckProof::Interleaved { .. });
	} else {
		panic!("Proof should be Composite.");
	}

	let mut verifier_state = EvalcheckVerifier::new(&mut oracles);
	verifier_state.verify(claim, proof).unwrap();
}
