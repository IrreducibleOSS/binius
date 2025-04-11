// Copyright 2024-2025 Irreducible Inc.

use std::iter::{repeat, repeat_with};

use assert_matches::assert_matches;
use binius_field::{
	as_packed_field::PackedType,
	packed::{get_packed_slice, len_packed_slice, set_packed_slice},
	underlier::WithUnderlier,
	BinaryField128b, Field, PackedBinaryField128x1b, PackedBinaryField16x8b,
	PackedBinaryField1x128b, PackedField, TowerField,
};
use binius_hal::{make_portable_backend, ComputationBackendExt};
use binius_hash::groestl::Groestl256;
use binius_macros::arith_expr;
use binius_math::{
	extrapolate_line, CompositionPoly, MultilinearExtension, MultilinearPoly, MultilinearQuery,
};
use bytemuck::cast_slice_mut;
use itertools::Either;
use rand::{rngs::StdRng, SeedableRng};

use crate::{
	oracle::{MultilinearOracleSet, ShiftVariant},
	polynomial::{ArithCircuitPoly, MultivariatePoly},
	protocols::evalcheck::{
		deserialize_evalcheck_proof, serialize_evalcheck_proof, EvalcheckMultilinearClaim,
		EvalcheckProofAdvice, EvalcheckProofEnum, EvalcheckProver, EvalcheckVerifier, Subclaim,
	},
	transparent::{select_row::SelectRow, step_down::StepDown},
	witness::MultilinearExtensionIndex,
};

type FExtension = BinaryField128b;
type PExtension = PackedBinaryField1x128b;
type U = <PExtension as WithUnderlier>::Underlier;

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
	let poly_id = oracles.add_committed(n_vars, <P as PackedField>::Scalar::TOWER_LEVEL);

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

	let claims: Vec<_> = [poly_id, shifted_id]
		.into_iter()
		.zip(evals)
		.map(|(id, eval)| EvalcheckMultilinearClaim {
			id,
			eval_point: eval_point.clone().into(),
			eval,
		})
		.collect();

	let mut witness_index = MultilinearExtensionIndex::<PackedType<U, FExtension>>::new();
	witness_index
		.update_multilin_poly(vec![
			(poly_id, poly_witness.to_ref().specialize_arc_dyn::<PExtension>()),
			(shifted_id, shifted_witness.to_ref().specialize_arc_dyn::<PExtension>()),
		])
		.unwrap();

	let mut prover_state = EvalcheckProver::new(&mut oracles, &mut witness_index, &backend);
	let (proofs, advices) = prover_state.prove(claims.clone()).unwrap();
	assert_eq!(prover_state.committed_eval_claims().len(), 1);

	let mut verifier_state = EvalcheckVerifier::<FExtension>::new(&mut oracles);
	verifier_state.verify(claims, proofs, advices).unwrap();
	assert_eq!(verifier_state.committed_eval_claims().len(), 1);
}

#[test]
fn test_shifted_evaluation_subcube() {
	type P = PackedBinaryField16x8b;

	let n_vars = 8;

	let mut oracles = MultilinearOracleSet::<FExtension>::new();

	let poly_id = oracles.add_committed(n_vars, <P as PackedField>::Scalar::TOWER_LEVEL);

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

	let claims: Vec<_> = [poly_id, shifted_id]
		.into_iter()
		.zip(evals)
		.map(|(id, eval)| EvalcheckMultilinearClaim {
			id,
			eval_point: eval_point.clone().into(),
			eval,
		})
		.collect();

	let mut witness_index = MultilinearExtensionIndex::<PackedType<U, FExtension>>::new();
	witness_index
		.update_multilin_poly(vec![
			(poly_id, poly_witness.to_ref().specialize_arc_dyn::<PExtension>()),
			(shifted_id, shifted_witness.to_ref().specialize_arc_dyn::<PExtension>()),
		])
		.unwrap();

	let mut prover_state = EvalcheckProver::new(&mut oracles, &mut witness_index, &backend);
	let (proofs, advices) = prover_state.prove(claims.clone()).unwrap();
	assert_eq!(prover_state.committed_eval_claims().len(), 1);

	let mut verifier_state = EvalcheckVerifier::<FExtension>::new(&mut oracles);
	verifier_state.verify(claims, proofs, advices).unwrap();
	assert_eq!(verifier_state.committed_eval_claims().len(), 1);
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
	let claim = EvalcheckMultilinearClaim {
		id: lin_com_id,
		eval_point: eval_point.into(),
		eval,
	};

	let mut witness_index = MultilinearExtensionIndex::<PackedType<U, FExtension>>::new();
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
	let (proofs, advices) = prover_state.prove(vec![claim.clone()]).unwrap();

	let mut verifier_state = EvalcheckVerifier::<FExtension>::new(&mut oracles);
	verifier_state.verify(vec![claim], proofs, advices).unwrap();
}

#[test]
fn test_evalcheck_linear_combination_size_one() {
	let n_vars = 8;

	let select_row = SelectRow::new(n_vars, 0).unwrap();

	let mut oracles = MultilinearOracleSet::new();

	let select_row_oracle_id = oracles.add_transparent(select_row.clone()).unwrap();
	let offset = FExtension::new(12345);
	let coef = FExtension::new(67890);

	let lin_com_id = oracles
		.add_linear_combination_with_offset(n_vars, offset, [(select_row_oracle_id, coef)])
		.unwrap();

	let mut rng = StdRng::seed_from_u64(0);
	let eval_point = repeat_with(|| <FExtension as Field>::random(&mut rng))
		.take(n_vars)
		.collect::<Vec<_>>();

	let eval = select_row.evaluate(&eval_point).unwrap() * coef + offset;

	let select_row_witness = select_row
		.multilinear_extension::<PackedBinaryField128x1b>()
		.unwrap();

	let lin_com_values = (0..1 << n_vars)
		.map(|i| select_row_witness.evaluate_on_hypercube(i).unwrap() * coef + offset)
		.map(PackedBinaryField1x128b::set_single)
		.collect();
	let lin_com_witness = MultilinearExtension::from_values(lin_com_values).unwrap();

	// Make the claim a composite oracle over a linear combination, in order to test the case
	// of requiring nested composite evalcheck proofs.
	let claim = EvalcheckMultilinearClaim {
		id: lin_com_id,
		eval_point: eval_point.into(),
		eval,
	};

	let mut witness_index = MultilinearExtensionIndex::<PackedType<U, FExtension>>::new();
	witness_index
		.update_multilin_poly(vec![
			(select_row_oracle_id, select_row_witness.specialize_arc_dyn()),
			(lin_com_id, lin_com_witness.specialize_arc_dyn()),
		])
		.unwrap();

	let backend = make_portable_backend();
	let mut prover_state = EvalcheckProver::new(&mut oracles, &mut witness_index, &backend);
	let (proofs, advices) = prover_state.prove(vec![claim.clone()]).unwrap();

	let mut verifier_state = EvalcheckVerifier::<FExtension>::new(&mut oracles);
	verifier_state.verify(vec![claim], proofs, advices).unwrap();
}

#[test]
fn test_evalcheck_composite() {
	let n_vars = 8;

	let select_row1 = SelectRow::new(n_vars, 0).unwrap();
	let select_row2 = SelectRow::new(n_vars, 5).unwrap();
	let select_row3 = SelectRow::new(n_vars, 10).unwrap();

	let mut oracles = MultilinearOracleSet::new();

	let select_row1_oracle_id = oracles.add_transparent(select_row1.clone()).unwrap();
	let select_row2_oracle_id = oracles.add_transparent(select_row2.clone()).unwrap();
	let select_row3_oracle_id = oracles.add_transparent(select_row3.clone()).unwrap();

	let comp = arith_expr!(BinaryField128b[x, y, z] = x * y * 15 + z * y * 8 + z * 2 + 77);

	let composite_id = oracles
		.add_composite_mle(
			n_vars,
			[
				select_row1_oracle_id,
				select_row2_oracle_id,
				select_row3_oracle_id,
			],
			comp.clone(),
		)
		.unwrap();

	let mut rng = StdRng::seed_from_u64(0);
	let eval_point = repeat_with(|| <FExtension as Field>::random(&mut rng))
		.take(n_vars)
		.collect::<Vec<_>>();

	let eval = ArithCircuitPoly::new(comp)
		.evaluate(&[
			select_row1.evaluate(&eval_point).unwrap(),
			select_row2.evaluate(&eval_point).unwrap(),
			select_row3.evaluate(&eval_point).unwrap(),
		])
		.unwrap();

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
	let claim = EvalcheckMultilinearClaim {
		id: composite_id,
		eval_point: eval_point.into(),
		eval,
	};

	let mut witness_index = MultilinearExtensionIndex::<PackedType<U, FExtension>>::new();
	witness_index
		.update_multilin_poly(vec![
			(select_row1_oracle_id, select_row1_witness.specialize_arc_dyn()),
			(select_row2_oracle_id, select_row2_witness.specialize_arc_dyn()),
			(select_row3_oracle_id, select_row3_witness.specialize_arc_dyn()),
			(composite_id, lin_com_witness.specialize_arc_dyn()),
		])
		.unwrap();

	let backend = make_portable_backend();
	let mut prover_state = EvalcheckProver::new(&mut oracles, &mut witness_index, &backend);
	let (proofs, advices) = prover_state.prove(vec![claim.clone()]).unwrap();

	let mut verifier_state = EvalcheckVerifier::<FExtension>::new(&mut oracles);
	verifier_state.verify(vec![claim], proofs, advices).unwrap();
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

	let mut rng = StdRng::seed_from_u64(0);
	let eval_point = repeat_with(|| <FExtension as Field>::random(&mut rng))
		.take(n_vars + 2)
		.collect::<Vec<_>>();

	let eval = select_row.evaluate(&eval_point[..n_vars]).unwrap();

	let claim = EvalcheckMultilinearClaim {
		id: repeating_id,
		eval_point: eval_point.into(),
		eval,
	};

	let mut witness_index = MultilinearExtensionIndex::<PackedType<U, FExtension>>::new();
	witness_index
		.update_multilin_poly(vec![(repeating_id, select_row_witness)])
		.unwrap();

	let backend = make_portable_backend();
	let mut prover_state = EvalcheckProver::new(&mut oracles, &mut witness_index, &backend);
	let (proofs, advices) = prover_state.prove(vec![claim.clone()]).unwrap();

	assert_matches!(proofs[0], EvalcheckProofEnum::Repeating);

	let mut verifier_state = EvalcheckVerifier::<FExtension>::new(&mut oracles);
	verifier_state.verify(vec![claim], proofs, advices).unwrap();
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

	let mut witness_index = MultilinearExtensionIndex::<PackedType<U, FExtension>>::new();
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
		id: zero_padded_id,
		eval_point: eval_point.into(),
		eval: inner_eval,
	};

	let backend = make_portable_backend();
	let mut prover_state = EvalcheckProver::new(&mut oracles, &mut witness_index, &backend);
	let (proofs, advices) = prover_state.prove(vec![claim.clone()]).unwrap();

	assert_matches!(proofs[0], EvalcheckProofEnum::ZeroPadded { .. });

	let mut verifier_state = EvalcheckVerifier::<FExtension>::new(&mut oracles);
	verifier_state.verify(vec![claim], proofs, advices).unwrap();
}

#[test]
fn test_evalcheck_serialization() {
	let mut rng = StdRng::seed_from_u64(0xdeadbeef);

	let linear_combination = EvalcheckProofEnum::LinearCombination {
		subproofs: vec![
			Subclaim::ExistingClaim(8),
			Subclaim::ExistingClaim(2),
			Subclaim::ExistingClaim(4),
			Subclaim::NewClaim(<BinaryField128b as Field>::random(&mut rng)),
			Subclaim::NewClaim(<BinaryField128b as Field>::random(&mut rng)),
		],
	};

	let mut transcript = crate::transcript::ProverTranscript::<
		crate::fiat_shamir::HasherChallenger<Groestl256>,
	>::new();

	let mut writer = transcript.message();
	serialize_evalcheck_proof(&mut writer, &linear_combination);
	let mut transcript = transcript.into_verifier();
	let mut reader = transcript.message();

	let out_deserialized = deserialize_evalcheck_proof::<_, BinaryField128b>(&mut reader).unwrap();

	assert_eq!(out_deserialized, linear_combination);

	transcript.finalize().unwrap()
}

#[test]
fn test_evalcheck_duplicate_claims() {
	let n_vars = 8;
	let mut oracles = MultilinearOracleSet::new();

	let stepdown1 = StepDown::new(n_vars, 1 << 2).unwrap();
	let stepdown2 = StepDown::new(n_vars, 1 << 4).unwrap();

	let stepdown1_witness = stepdown1
		.multilinear_extension::<PackedBinaryField128x1b>()
		.unwrap();
	let stepdown2_witness = stepdown2
		.multilinear_extension::<PackedBinaryField128x1b>()
		.unwrap();

	let transp1_id = oracles.add_transparent(stepdown1.clone()).unwrap();
	let transp2_id = oracles.add_transparent(stepdown2.clone()).unwrap();

	let mut rng = StdRng::seed_from_u64(0);
	let field_gen = repeat(<BinaryField128b as Field>::random(&mut rng));
	let eval_point = field_gen.take(n_vars).collect::<Vec<_>>();

	let eval_stepdown1 = stepdown1.evaluate(&eval_point).unwrap();
	let eval_stepdown2 = stepdown2.evaluate(&eval_point).unwrap();

	let claims = vec![
		EvalcheckMultilinearClaim {
			id: transp1_id,
			eval_point: eval_point.clone().into(),
			eval: eval_stepdown1,
		},
		EvalcheckMultilinearClaim {
			id: transp2_id,
			eval_point: eval_point.clone().into(),
			eval: eval_stepdown2,
		},
		EvalcheckMultilinearClaim {
			id: transp1_id,
			eval_point: eval_point.into(),
			eval: eval_stepdown1,
		},
	];

	let mut witness_index = MultilinearExtensionIndex::<PackedType<U, FExtension>>::new();
	witness_index
		.update_multilin_poly(vec![
			(transp1_id, stepdown1_witness.specialize_arc_dyn()),
			(transp2_id, stepdown2_witness.specialize_arc_dyn()),
		])
		.unwrap();

	let backend = make_portable_backend();
	let mut prover_state = EvalcheckProver::new(&mut oracles, &mut witness_index, &backend);
	let (proofs, advices) = prover_state.prove(claims.clone()).unwrap();

	assert_eq!(
		advices.as_slice(),
		&[
			EvalcheckProofAdvice::DuplicateClaim(2),
			EvalcheckProofAdvice::HandleClaim,
			EvalcheckProofAdvice::HandleClaim
		]
	);

	assert_eq!(
		proofs.as_slice(),
		&[
			EvalcheckProofEnum::Transparent,
			EvalcheckProofEnum::Transparent
		]
	);

	let mut verifier_state = EvalcheckVerifier::<FExtension>::new(&mut oracles);
	verifier_state.verify(claims, proofs, advices).unwrap();
}

#[test]
fn test_evalcheck_existing_claims() {
	let n_vars = 8;
	let mut oracles = MultilinearOracleSet::new();

	let stepdown1 = StepDown::new(n_vars, 1 << 4).unwrap();
	let stepdown2 = StepDown::new(n_vars, 1 << 5).unwrap();
	let stepdown3 = StepDown::new(n_vars, 1 << 6).unwrap();

	let stepdown1_witness = stepdown1
		.multilinear_extension::<PackedBinaryField128x1b>()
		.unwrap();
	let stepdown2_witness = stepdown2
		.multilinear_extension::<PackedBinaryField128x1b>()
		.unwrap();
	let stepdown3_witness = stepdown3
		.multilinear_extension::<PackedBinaryField128x1b>()
		.unwrap();

	let transp1_id = oracles.add_transparent(stepdown1.clone()).unwrap();
	let transp2_id = oracles.add_transparent(stepdown2.clone()).unwrap();
	let transp3_id = oracles.add_transparent(stepdown3.clone()).unwrap();

	let mut rng = StdRng::seed_from_u64(0);
	let field_gen = repeat(<BinaryField128b as Field>::random(&mut rng));
	let fs = field_gen.take(6).collect::<Vec<_>>();

	let lin_comb1_id = oracles
		.add_linear_combination_with_offset(
			n_vars,
			fs[0],
			[(transp1_id, fs[1]), (transp2_id, fs[2])],
		)
		.unwrap();

	let lin_comb2_id = oracles
		.add_linear_combination_with_offset(
			n_vars,
			fs[3],
			[(lin_comb1_id, fs[4]), (transp3_id, fs[5])],
		)
		.unwrap();

	let mut rng = StdRng::seed_from_u64(0xdeadbeef);
	let field_gen = repeat(<BinaryField128b as Field>::random(&mut rng));
	let eval_point = field_gen.take(n_vars).collect::<Vec<_>>();

	let eval_stepdown1 = stepdown1.evaluate(&eval_point).unwrap();
	let eval_stepdown2 = stepdown2.evaluate(&eval_point).unwrap();
	let eval_stepdown3 = stepdown3.evaluate(&eval_point).unwrap();
	let eval_lin_comb1 = fs[0] + eval_stepdown1 * fs[1] + eval_stepdown2 * fs[2];

	let eval_lin_comb2 = fs[3] + eval_lin_comb1 * fs[4] + eval_stepdown3 * fs[5];

	let lin_comb1_values = (0..1 << n_vars)
		.map(|i| {
			fs[0]
				+ stepdown1_witness.evaluate_on_hypercube(i).unwrap() * fs[1]
				+ stepdown2_witness.evaluate_on_hypercube(i).unwrap() * fs[2]
		})
		.map(PackedBinaryField1x128b::set_single)
		.collect();
	let lin_comb1_witness = MultilinearExtension::from_values(lin_comb1_values).unwrap();

	let lin_comb2_values = (0..1 << n_vars)
		.map(|i| {
			fs[3]
				+ lin_comb1_witness.evaluate_on_hypercube(i).unwrap() * fs[4]
				+ stepdown3_witness.evaluate_on_hypercube(i).unwrap() * fs[5]
		})
		.map(PackedBinaryField1x128b::set_single)
		.collect();
	let lin_comb2_witness = MultilinearExtension::from_values(lin_comb2_values).unwrap();

	let claims = vec![
		EvalcheckMultilinearClaim {
			id: lin_comb2_id,
			eval_point: eval_point.clone().into(),
			eval: eval_lin_comb2,
		},
		EvalcheckMultilinearClaim {
			id: lin_comb1_id,
			eval_point: eval_point.into(),
			eval: eval_lin_comb1,
		},
	];

	let mut witness_index = MultilinearExtensionIndex::<PackedType<U, FExtension>>::new();
	witness_index
		.update_multilin_poly(vec![
			(transp1_id, stepdown1_witness.specialize_arc_dyn()),
			(transp2_id, stepdown2_witness.specialize_arc_dyn()),
			(transp3_id, stepdown3_witness.specialize_arc_dyn()),
			(lin_comb1_id, lin_comb1_witness.specialize_arc_dyn()),
			(lin_comb2_id, lin_comb2_witness.specialize_arc_dyn()),
		])
		.unwrap();

	let backend = make_portable_backend();
	let mut prover_state = EvalcheckProver::new(&mut oracles, &mut witness_index, &backend);
	let (proofs, advices) = prover_state.prove(claims.clone()).unwrap();

	for advice in &advices {
		assert!(matches!(advice, EvalcheckProofAdvice::HandleClaim));
	}

	assert_eq!(
		proofs.as_slice(),
		&[
			EvalcheckProofEnum::LinearCombination {
				subproofs: vec![
					Subclaim::ExistingClaim(1),
					Subclaim::NewClaim(eval_stepdown3)
				]
			},
			EvalcheckProofEnum::LinearCombination {
				subproofs: vec![
					Subclaim::NewClaim(eval_stepdown1),
					Subclaim::NewClaim(eval_stepdown2)
				]
			},
			EvalcheckProofEnum::Transparent,
			EvalcheckProofEnum::Transparent,
			EvalcheckProofEnum::Transparent
		]
	);

	let mut verifier_state = EvalcheckVerifier::<FExtension>::new(&mut oracles);
	verifier_state.verify(claims, proofs, advices).unwrap();
}
