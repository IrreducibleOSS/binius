// Copyright 2024-2025 Irreducible Inc.

use std::{array, iter::repeat_with};

use binius_field::{
	packed::{get_packed_slice, len_packed_slice, set_packed_slice},
	AESTowerField128b, BinaryField128b, BinaryField1b, ByteSliced16x128x1b, ByteSlicedAES16x128b,
	ByteSlicedAES16x16x8b, ExtensionField, Field, PackedBinaryField128x1b, PackedBinaryField16x8b,
	PackedBinaryField1x128b, PackedField, RepackedExtension, TowerField,
};
use binius_hal::{make_portable_backend, ComputationBackendExt};
use binius_hash::groestl::Groestl256;
use binius_macros::arith_expr;
use binius_math::{
	extrapolate_line, CompositionPoly, MLEDirectAdapter, MultilinearExtension, MultilinearPoly,
	MultilinearQuery,
};
use bytemuck::{cast_slice_mut, Pod};
use itertools::Either;
use rand::{rngs::StdRng, SeedableRng};

use crate::{
	fiat_shamir::HasherChallenger,
	oracle::{MultilinearOracleSet, ShiftVariant},
	polynomial::{ArithCircuitPoly, MultivariatePoly},
	protocols::evalcheck::{
		deserialize_evalcheck_proof, serialize_evalcheck_proof, EvalcheckHint,
		EvalcheckMultilinearClaim, EvalcheckProver, EvalcheckVerifier,
	},
	transcript::ProverTranscript,
	transparent::select_row::SelectRow,
	witness::MultilinearExtensionIndex,
};

type FExtension = BinaryField128b;
type PExtension = PackedBinaryField1x128b;

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

fn run_test_shifted_evaluation_whole_cube<P, FExtension, PExtension>(n_vars: usize)
where
	P: PackedField,
	P::Scalar: TowerField,
	FExtension: TowerField + ExtensionField<P::Scalar>,
	PExtension: PackedField<Scalar = FExtension> + RepackedExtension<P>,
{
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
	let query: MultilinearQuery<FExtension, _> = backend.multilinear_query(&eval_point).unwrap();
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

	let mut witness_index = MultilinearExtensionIndex::<PExtension>::new();
	witness_index
		.update_multilin_poly(vec![
			(poly_id, poly_witness.to_ref().specialize_arc_dyn::<PExtension>()),
			(shifted_id, shifted_witness.to_ref().specialize_arc_dyn::<PExtension>()),
		])
		.unwrap();

	let mut transcript = ProverTranscript::<HasherChallenger<Groestl256>>::new();
	let mut prover_state = EvalcheckProver::new(&mut oracles, &mut witness_index, &backend);
	prover_state.prove(claims.clone(), &mut transcript).unwrap();
	assert_eq!(prover_state.committed_eval_claims().len(), 1);

	let mut transcript = transcript.into_verifier();
	let mut verifier_state = EvalcheckVerifier::<FExtension>::new(&mut oracles);

	verifier_state.verify(claims, &mut transcript).unwrap();
	assert_eq!(verifier_state.committed_eval_claims().len(), 1);
}

#[test]
fn test_shifted_evaluation_whole_cube() {
	run_test_shifted_evaluation_whole_cube::<PackedBinaryField16x8b, FExtension, PExtension>(8);
	run_test_shifted_evaluation_whole_cube::<
		ByteSlicedAES16x16x8b,
		AESTowerField128b,
		ByteSlicedAES16x128b,
	>(16);
}

fn run_test_shifted_evaluation_subcube<P, FExtension, PExtension>(n_vars: usize)
where
	P: PackedField + Pod,
	P::Scalar: TowerField,
	FExtension: TowerField + ExtensionField<P::Scalar>,
	PExtension: PackedField<Scalar = FExtension> + RepackedExtension<P>,
{
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
		.multilinear_query::<FExtension>(&eval_point)
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

	let mut witness_index = MultilinearExtensionIndex::<PExtension>::new();
	witness_index
		.update_multilin_poly(vec![
			(poly_id, poly_witness.to_ref().specialize_arc_dyn::<PExtension>()),
			(shifted_id, shifted_witness.to_ref().specialize_arc_dyn::<PExtension>()),
		])
		.unwrap();

	let mut transcript = ProverTranscript::<HasherChallenger<Groestl256>>::new();
	let mut prover_state = EvalcheckProver::new(&mut oracles, &mut witness_index, &backend);
	prover_state.prove(claims.clone(), &mut transcript).unwrap();
	assert_eq!(prover_state.committed_eval_claims().len(), 1);

	let mut transcript = transcript.into_verifier();
	let mut verifier_state = EvalcheckVerifier::<FExtension>::new(&mut oracles);
	verifier_state.verify(claims, &mut transcript).unwrap();
	assert_eq!(verifier_state.committed_eval_claims().len(), 1);
}

#[test]
fn test_shifted_evaluation_subcube() {
	run_test_shifted_evaluation_subcube::<PackedBinaryField16x8b, FExtension, PExtension>(8);
	run_test_shifted_evaluation_subcube::<
		ByteSlicedAES16x16x8b,
		AESTowerField128b,
		ByteSlicedAES16x128b,
	>(16);
}

fn run_test_evalcheck_linear_combination<P, FExtension, PExtension>(n_vars: usize)
where
	P: PackedField<Scalar = BinaryField1b> + Pod,
	P::Scalar: TowerField,
	FExtension: TowerField + ExtensionField<BinaryField1b>,
	PExtension:
		PackedField<Scalar = FExtension> + RepackedExtension<P> + RepackedExtension<PExtension>,
{
	let mut rng = StdRng::seed_from_u64(0);

	let values: [FExtension; 4] = array::from_fn(|_| <FExtension as Field>::random(&mut rng));

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
			values[0],
			[
				(select_row1_oracle_id, values[1]),
				(select_row2_oracle_id, values[2]),
				(select_row3_oracle_id, values[3]),
			],
		)
		.unwrap();

	let eval_point = repeat_with(|| <FExtension as Field>::random(&mut rng))
		.take(n_vars)
		.collect::<Vec<_>>();

	let eval = select_row1.evaluate(&eval_point).unwrap() * values[1]
		+ select_row2.evaluate(&eval_point).unwrap() * values[2]
		+ select_row3.evaluate(&eval_point).unwrap() * values[3]
		+ values[0];

	let select_row1_witness = select_row1.multilinear_extension::<P>().unwrap();
	let select_row2_witness = select_row2.multilinear_extension::<P>().unwrap();
	let select_row3_witness = select_row3.multilinear_extension::<P>().unwrap();

	let lin_com_values = (0..1 << n_vars)
		.map(|i| {
			values[1] * select_row1_witness.evaluate_on_hypercube(i).unwrap()
				+ values[2] * select_row2_witness.evaluate_on_hypercube(i).unwrap()
				+ values[3] * select_row3_witness.evaluate_on_hypercube(i).unwrap()
				+ values[0]
		})
		.map(PExtension::set_single)
		.collect();
	let lin_com_witness = MultilinearExtension::from_values(lin_com_values).unwrap();

	// Make the claim a composite oracle over a linear combination, in order to test the case
	// of requiring nested composite evalcheck proofs.
	let claim = EvalcheckMultilinearClaim {
		id: lin_com_id,
		eval_point: eval_point.into(),
		eval,
	};

	let mut witness_index = MultilinearExtensionIndex::<PExtension>::new();
	witness_index
		.update_multilin_poly(vec![
			(select_row1_oracle_id, select_row1_witness.specialize_arc_dyn()),
			(select_row2_oracle_id, select_row2_witness.specialize_arc_dyn()),
			(select_row3_oracle_id, select_row3_witness.specialize_arc_dyn()),
			(lin_com_id, lin_com_witness.specialize_arc_dyn()),
		])
		.unwrap();

	let backend = make_portable_backend();

	let mut transcript = ProverTranscript::<HasherChallenger<Groestl256>>::new();
	let mut prover_state = EvalcheckProver::new(&mut oracles, &mut witness_index, &backend);
	prover_state
		.prove(vec![claim.clone()], &mut transcript)
		.unwrap();

	let mut transcript = transcript.into_verifier();
	let mut verifier_state = EvalcheckVerifier::<FExtension>::new(&mut oracles);
	verifier_state.verify(vec![claim], &mut transcript).unwrap();
}

#[test]
fn test_evalcheck_linear_combination() {
	run_test_evalcheck_linear_combination::<PackedBinaryField128x1b, FExtension, PExtension>(8);
	run_test_evalcheck_linear_combination::<
		ByteSliced16x128x1b,
		AESTowerField128b,
		ByteSlicedAES16x128b,
	>(16);
}

fn run_test_evalcheck_linear_combination_size_one<P, FExtension, PExtension>(n_vars: usize)
where
	P: PackedField<Scalar = BinaryField1b> + Pod,
	P::Scalar: TowerField,
	FExtension: TowerField + ExtensionField<BinaryField1b>,
	PExtension:
		PackedField<Scalar = FExtension> + RepackedExtension<P> + RepackedExtension<PExtension>,
{
	let mut rng = StdRng::seed_from_u64(0);

	let [offset, coef] = array::from_fn(|_| <FExtension as Field>::random(&mut rng));

	let select_row = SelectRow::new(n_vars, 0).unwrap();

	let mut oracles = MultilinearOracleSet::new();

	let select_row_oracle_id = oracles.add_transparent(select_row.clone()).unwrap();

	let lin_com_id = oracles
		.add_linear_combination_with_offset(n_vars, offset, [(select_row_oracle_id, coef)])
		.unwrap();

	let mut rng = StdRng::seed_from_u64(0);
	let eval_point = repeat_with(|| <FExtension as Field>::random(&mut rng))
		.take(n_vars)
		.collect::<Vec<_>>();

	let eval = select_row.evaluate(&eval_point).unwrap() * coef + offset;

	let select_row_witness = select_row.multilinear_extension::<P>().unwrap();

	let lin_com_values = (0..1 << n_vars)
		.map(|i| coef * select_row_witness.evaluate_on_hypercube(i).unwrap() + offset)
		.map(PExtension::set_single)
		.collect();
	let lin_com_witness = MultilinearExtension::from_values(lin_com_values).unwrap();

	// Make the claim a composite oracle over a linear combination, in order to test the case
	// of requiring nested composite evalcheck proofs.
	let claim = EvalcheckMultilinearClaim {
		id: lin_com_id,
		eval_point: eval_point.into(),
		eval,
	};

	let mut witness_index = MultilinearExtensionIndex::<PExtension>::new();
	witness_index
		.update_multilin_poly(vec![
			(select_row_oracle_id, select_row_witness.specialize_arc_dyn()),
			(lin_com_id, lin_com_witness.specialize_arc_dyn()),
		])
		.unwrap();

	let backend = make_portable_backend();

	let mut transcript = ProverTranscript::<HasherChallenger<Groestl256>>::new();
	let mut prover_state = EvalcheckProver::new(&mut oracles, &mut witness_index, &backend);
	prover_state
		.prove(vec![claim.clone()], &mut transcript)
		.unwrap();

	let mut transcript = transcript.into_verifier();
	let mut verifier_state = EvalcheckVerifier::<FExtension>::new(&mut oracles);
	verifier_state.verify(vec![claim], &mut transcript).unwrap();
}

#[test]
fn test_evalcheck_linear_combination_size_one() {
	run_test_evalcheck_linear_combination_size_one::<PackedBinaryField128x1b, FExtension, PExtension>(
		8,
	);
	run_test_evalcheck_linear_combination_size_one::<
		ByteSliced16x128x1b,
		AESTowerField128b,
		ByteSlicedAES16x128b,
	>(16);
}

fn run_test_evalcheck_composite<P, FExtension, PExtension>(n_vars: usize)
where
	P: PackedField<Scalar = BinaryField1b> + Pod,
	P::Scalar: TowerField,
	FExtension: TowerField + ExtensionField<BinaryField1b>,
	PExtension:
		PackedField<Scalar = FExtension> + RepackedExtension<P> + RepackedExtension<PExtension>,
{
	let mut rng = StdRng::seed_from_u64(0);

	let select_row1 = SelectRow::new(n_vars, 0).unwrap();
	let select_row2 = SelectRow::new(n_vars, 5).unwrap();
	let select_row3 = SelectRow::new(n_vars, 10).unwrap();

	let mut oracles = MultilinearOracleSet::new();

	let select_row1_oracle_id = oracles.add_transparent(select_row1.clone()).unwrap();
	let select_row2_oracle_id = oracles.add_transparent(select_row2.clone()).unwrap();
	let select_row3_oracle_id = oracles.add_transparent(select_row3.clone()).unwrap();

	let comp = arith_expr!(FExtension[x, y, z] = x * y + z * y + z);

	let values: [FExtension; 4] = array::from_fn(|_| <FExtension as Field>::random(&mut rng));

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

	let select_row1_witness = select_row1.multilinear_extension::<P>().unwrap();
	let select_row2_witness = select_row2.multilinear_extension::<P>().unwrap();
	let select_row3_witness = select_row3.multilinear_extension::<P>().unwrap();

	let lin_com_values = (0..1 << n_vars)
		.map(|i| {
			values[1] * select_row1_witness.evaluate_on_hypercube(i).unwrap()
				+ values[2] * select_row2_witness.evaluate_on_hypercube(i).unwrap()
				+ values[3] * select_row3_witness.evaluate_on_hypercube(i).unwrap()
				+ values[0]
		})
		.map(PExtension::set_single)
		.collect();
	let lin_com_witness = MultilinearExtension::from_values(lin_com_values).unwrap();

	// Make the claim a composite oracle over a linear combination, in order to test the case
	// of requiring nested composite evalcheck proofs.
	let claim = EvalcheckMultilinearClaim {
		id: composite_id,
		eval_point: eval_point.into(),
		eval,
	};

	let mut witness_index = MultilinearExtensionIndex::<PExtension>::new();
	witness_index
		.update_multilin_poly(vec![
			(select_row1_oracle_id, select_row1_witness.specialize_arc_dyn()),
			(select_row2_oracle_id, select_row2_witness.specialize_arc_dyn()),
			(select_row3_oracle_id, select_row3_witness.specialize_arc_dyn()),
			(composite_id, lin_com_witness.specialize_arc_dyn()),
		])
		.unwrap();

	let backend = make_portable_backend();

	let mut transcript = ProverTranscript::<HasherChallenger<Groestl256>>::new();
	let mut prover_state = EvalcheckProver::new(&mut oracles, &mut witness_index, &backend);
	prover_state
		.prove(vec![claim.clone()], &mut transcript)
		.unwrap();

	let mut transcript = transcript.into_verifier();
	let mut verifier_state = EvalcheckVerifier::<FExtension>::new(&mut oracles);
	verifier_state.verify(vec![claim], &mut transcript).unwrap();
}

#[test]
fn test_evalcheck_composite() {
	run_test_evalcheck_composite::<PackedBinaryField128x1b, FExtension, PExtension>(8);
	run_test_evalcheck_composite::<ByteSliced16x128x1b, AESTowerField128b, ByteSlicedAES16x128b>(
		16,
	);
}

fn run_test_evalcheck_repeating<P, FExtension, PExtension>(n_vars: usize)
where
	P: PackedField<Scalar = BinaryField1b> + Pod,
	P::Scalar: TowerField,
	FExtension: TowerField + ExtensionField<BinaryField1b>,
	PExtension:
		PackedField<Scalar = FExtension> + RepackedExtension<P> + RepackedExtension<PExtension>,
{
	let row_id = 11;

	let mut oracles = MultilinearOracleSet::new();

	let select_row = SelectRow::new(n_vars, row_id).unwrap();
	let select_row_oracle_id = oracles.add_transparent(select_row.clone()).unwrap();

	let select_row_subwitness = select_row.multilinear_extension::<P>().unwrap();
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

	let mut witness_index = MultilinearExtensionIndex::<PExtension>::new();
	witness_index
		.update_multilin_poly(vec![(repeating_id, select_row_witness)])
		.unwrap();

	let backend = make_portable_backend();

	let mut transcript = ProverTranscript::<HasherChallenger<Groestl256>>::new();
	let mut prover_state = EvalcheckProver::new(&mut oracles, &mut witness_index, &backend);
	prover_state
		.prove(vec![claim.clone()], &mut transcript)
		.unwrap();

	let mut transcript = transcript.into_verifier();
	let mut verifier_state = EvalcheckVerifier::<FExtension>::new(&mut oracles);
	verifier_state.verify(vec![claim], &mut transcript).unwrap();
}

#[test]
fn test_evalcheck_repeating() {
	run_test_evalcheck_repeating::<PackedBinaryField128x1b, FExtension, PExtension>(8);
	run_test_evalcheck_repeating::<ByteSliced16x128x1b, AESTowerField128b, ByteSlicedAES16x128b>(
		16,
	);
}

fn run_test_evalcheck_zero_padded<P, FExtension, PExtension>(n_vars: usize, inner_n_vars: usize)
where
	P: PackedField<Scalar = BinaryField1b> + Pod + RepackedExtension<P>,
	P::Scalar: TowerField,
	FExtension: TowerField + ExtensionField<BinaryField1b>,
	PExtension:
		PackedField<Scalar = FExtension> + RepackedExtension<P> + RepackedExtension<PExtension>,
{
	let row_id = 9;

	let mut oracles = MultilinearOracleSet::new();

	let select_row = SelectRow::new(inner_n_vars, row_id).unwrap();

	let select_row_subwitness = select_row.multilinear_extension::<P>().unwrap();

	let x = select_row_subwitness.clone().specialize::<P>();

	assert!(x.n_vars() >= P::LOG_WIDTH);

	let mut values = vec![P::zero(); 1 << (n_vars - P::LOG_WIDTH)];

	let values_len = values.len();

	let (_, x_values) = values.split_at_mut(values_len - (1 << (x.n_vars() - P::LOG_WIDTH)));

	x.subcube_evals(x.n_vars(), 0, 0, x_values).unwrap();

	let zero_padded_poly = MultilinearExtension::from_values(values).unwrap();

	let select_row_oracle_id = oracles.add_transparent(select_row.clone()).unwrap();

	let start_index = inner_n_vars;
	let n_pad_vars = n_vars - inner_n_vars;
	let nonzero_index = (1 << n_pad_vars) - 1;
	let zero_padded_id = oracles
		.add_zero_padded(select_row_oracle_id, n_pad_vars, nonzero_index, start_index)
		.unwrap();

	let mut witness_index = MultilinearExtensionIndex::<PExtension>::new();
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
		inner_eval = extrapolate_line::<FExtension, FExtension>(
			FExtension::zero(),
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

	let mut transcript = ProverTranscript::<HasherChallenger<Groestl256>>::new();
	let mut prover_state = EvalcheckProver::new(&mut oracles, &mut witness_index, &backend);
	prover_state
		.prove(vec![claim.clone()], &mut transcript)
		.unwrap();

	let mut transcript = transcript.into_verifier();
	let mut verifier_state = EvalcheckVerifier::<FExtension>::new(&mut oracles);
	verifier_state.verify(vec![claim], &mut transcript).unwrap();
}

#[test]
/// Constructs a small ZeroPadded oracle, proves and verifies it.
fn test_evalcheck_zero_padded() {
	run_test_evalcheck_zero_padded::<PackedBinaryField128x1b, FExtension, PExtension>(10, 8);
	run_test_evalcheck_zero_padded::<ByteSliced16x128x1b, AESTowerField128b, ByteSlicedAES16x128b>(
		18, 16,
	);
}

// Test evalcheck serialization
#[test]
fn test_evalcheck_serialization() {
	let mut transcript = ProverTranscript::<HasherChallenger<Groestl256>>::new();
	let mut writer = transcript.message();
	serialize_evalcheck_proof(&mut writer, &EvalcheckHint::NewClaim);
	serialize_evalcheck_proof(&mut writer, &EvalcheckHint::DuplicateClaim(6));

	let mut transcript = transcript.into_verifier();
	let mut reader = transcript.message();

	let out_1 = deserialize_evalcheck_proof(&mut reader).unwrap();
	let out_2 = deserialize_evalcheck_proof(&mut reader).unwrap();

	assert_eq!(out_1, EvalcheckHint::NewClaim);
	assert_eq!(out_2, EvalcheckHint::DuplicateClaim(6));

	transcript.finalize().unwrap()
}

#[test]
pub fn tets_zero_vars_zeropadding() {
	let mut oracles = MultilinearOracleSet::<BinaryField128b>::new();
	let mut witness_index = MultilinearExtensionIndex::<BinaryField128b>::new();

	let committed = oracles.add_committed(0, BinaryField128b::TOWER_LEVEL);
	let mut values = vec![BinaryField128b::from(32)];
	let committed_mle = MultilinearExtension::from_values(values.clone()).unwrap();
	let committed_poly = MLEDirectAdapter::from(committed_mle);

	witness_index
		.update_multilin_poly([(committed, committed_poly.upcast_arc_dyn())])
		.unwrap();

	let zero_padded = oracles.add_zero_padded(committed, 3, 0, 0).unwrap();

	values.resize(1 << 3, BinaryField128b::ZERO);

	let zero_padded_mle = MultilinearExtension::from_values(values).unwrap();
	let zero_padded_poly = MLEDirectAdapter::from(zero_padded_mle);

	let backend = make_portable_backend();

	let eval_point = vec![
		BinaryField128b::from(3),
		BinaryField128b::from(2),
		BinaryField128b::ONE,
	];

	let query = backend.multilinear_query(&eval_point).unwrap();
	let eval = zero_padded_poly.evaluate(query.to_ref()).unwrap();
	assert_eq!(eval, BinaryField128b::zero());

	let zero_eval_claim = EvalcheckMultilinearClaim {
		id: zero_padded,
		eval_point: eval_point.into(),
		eval,
	};

	let eval_point = vec![
		BinaryField128b::from(3),
		BinaryField128b::from(2),
		BinaryField128b::ZERO,
	];

	let query = backend.multilinear_query(&eval_point).unwrap();
	let eval = zero_padded_poly.evaluate(query.to_ref()).unwrap();

	assert_ne!(eval, BinaryField128b::zero());

	let non_zero_eval_claim = EvalcheckMultilinearClaim {
		id: zero_padded,
		eval_point: eval_point.into(),
		eval,
	};

	let mut transcript = ProverTranscript::<HasherChallenger<Groestl256>>::new();
	let mut prover_state = EvalcheckProver::new(&mut oracles, &mut witness_index, &backend);
	prover_state
		.prove(vec![zero_eval_claim.clone(), non_zero_eval_claim.clone()], &mut transcript)
		.unwrap();

	let mut transcript = transcript.into_verifier();
	let mut verifier_state = EvalcheckVerifier::<FExtension>::new(&mut oracles);
	verifier_state
		.verify(vec![zero_eval_claim, non_zero_eval_claim], &mut transcript)
		.unwrap();
}
