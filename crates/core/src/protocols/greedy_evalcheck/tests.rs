// Copyright 2024-2025 Irreducible Inc.
use std::iter::repeat_with;

use binius_compute::{
	ComputeLayer, FSliceMut,
	alloc::{BumpAllocator, HostBumpAllocator},
	cpu::CpuLayer,
};
use binius_field::{
	BinaryField1b, BinaryField32b, BinaryField128b, ExtensionField, Field, PackedBinaryField1x128b,
	PackedBinaryField128x1b, PackedExtension, PackedField, RepackedExtension, TowerField,
	packed::{get_packed_slice, len_packed_slice, pack_slice, set_packed_slice},
	tower::{CanonicalTowerFamily, TowerFamily},
};
use binius_hal::{ComputationBackendExt, make_portable_backend};
use binius_hash::groestl::Groestl256;
use binius_macros::arith_expr;
use binius_math::{DefaultEvaluationDomainFactory, MultilinearExtension};
use bytemuck::{Pod, zeroed_vec};
use either::Either;
use rand::{SeedableRng, rngs::StdRng};

use crate::{
	fiat_shamir::HasherChallenger,
	oracle::{MultilinearOracleSet, ShiftVariant},
	polynomial::MultivariatePoly,
	protocols::{
		evalcheck::EvalcheckMultilinearClaim,
		greedy_evalcheck::{prove, verify},
		sumcheck::standard_switchover_heuristic,
	},
	transcript::ProverTranscript,
	transparent::select_row::SelectRow,
	witness::{HalMultilinearExtensionIndex, MultilinearExtensionIndex},
};

type FExtension = BinaryField128b;
type PExtension = PackedBinaryField1x128b;
type FDomain = BinaryField32b;

pub fn shift_one<P: PackedField>(evals: &mut [P], bits: usize, variant: ShiftVariant) {
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

fn run_test_evalcheck_composite_projected<P, FExtension, PExtension>(n_vars: usize)
where
	P: PackedField<Scalar = BinaryField1b> + Pod,
	P::Scalar: TowerField,
	FExtension: TowerField + ExtensionField<BinaryField1b> + ExtensionField<FDomain>,
	CanonicalTowerFamily: TowerFamily<B128 = FExtension>,
	PExtension: PackedField<Scalar = FExtension>
		+ PackedExtension<FDomain>
		+ RepackedExtension<P>
		+ RepackedExtension<PExtension>
		+ Pod,
{
	let mut rng = StdRng::seed_from_u64(0);

	let select_row1 = SelectRow::new(n_vars, 0).unwrap();
	let select_row2 = SelectRow::new(n_vars, 5).unwrap();
	let select_row3 = SelectRow::new(n_vars, 10).unwrap();

	let mut oracles = MultilinearOracleSet::new();

	let select_row1_oracle_id = oracles.add_transparent(select_row1.clone()).unwrap();
	let select_row2_oracle_id = oracles.add_transparent(select_row2.clone()).unwrap();
	let select_row3_oracle_id = oracles.add_transparent(select_row3.clone()).unwrap();

	#[allow(deprecated)]
	let comp = arith_expr!(FExtension[x, y, z] = x*y + x*z +  z);

	let composite_id = oracles
		.add_composite_mle(
			n_vars,
			[
				select_row1_oracle_id,
				select_row2_oracle_id,
				select_row3_oracle_id,
			],
			comp,
		)
		.unwrap();

	let eval_point = repeat_with(|| <FExtension as Field>::random(&mut rng))
		.take(n_vars)
		.collect::<Vec<_>>();

	let eval = select_row3.evaluate(&eval_point).unwrap();

	let composite_claim = EvalcheckMultilinearClaim {
		id: composite_id,
		eval_point: eval_point.clone().into(),
		eval,
	};

	let select_row1_witness = select_row1.multilinear_extension::<P>().unwrap();
	let select_row2_witness = select_row2.multilinear_extension::<P>().unwrap();
	let select_row3_witness = select_row3.multilinear_extension::<P>().unwrap();

	let composite_scalars = (0..1 << n_vars)
		.map(|i| {
			select_row1_witness.evaluate_on_hypercube(i).unwrap()
				* select_row2_witness.evaluate_on_hypercube(i).unwrap()
				+ select_row2_witness.evaluate_on_hypercube(i).unwrap()
					* select_row3_witness.evaluate_on_hypercube(i).unwrap()
				+ select_row3_witness.evaluate_on_hypercube(i).unwrap()
		})
		.collect::<Vec<P::Scalar>>();

	let composite_values: Vec<P> = pack_slice(&composite_scalars);

	let composite_witness = MultilinearExtension::from_values(composite_values).unwrap();

	let mut shifted_evals = composite_witness.evals().to_vec();
	shift_one(&mut shifted_evals, n_vars, ShiftVariant::CircularLeft);
	let shifted_witness = MultilinearExtension::from_values(shifted_evals).unwrap();

	let shifted_id = oracles
		.add_shifted(composite_id, 1, n_vars, ShiftVariant::CircularLeft)
		.unwrap();

	let backend = make_portable_backend();

	let query = backend
		.multilinear_query::<FExtension>(&eval_point)
		.unwrap();

	let eval = shifted_witness.evaluate(query.to_ref()).unwrap();

	let mut witness_index = MultilinearExtensionIndex::<PExtension>::new();
	witness_index
		.update_multilin_poly(vec![
			(select_row1_oracle_id, select_row1_witness.specialize_arc_dyn()),
			(select_row2_oracle_id, select_row2_witness.specialize_arc_dyn()),
			(select_row3_oracle_id, select_row3_witness.specialize_arc_dyn()),
			(composite_id, composite_witness.specialize_arc_dyn()),
			(shifted_id, shifted_witness.specialize_arc_dyn()),
		])
		.unwrap();

	let shifted_claim = EvalcheckMultilinearClaim {
		id: shifted_id,
		eval_point: eval_point.into(),
		eval,
	};

	let domain_factory = DefaultEvaluationDomainFactory::<FDomain>::default();

	let hal = <CpuLayer<CanonicalTowerFamily>>::default();
	let mut dev_mem = zeroed_vec(1 << 20);
	let mut host_mem = hal.host_alloc(1 << 20);
	let host_alloc = HostBumpAllocator::new(host_mem.as_mut());
	let dev_alloc =
		BumpAllocator::new((&mut dev_mem) as FSliceMut<FExtension, CpuLayer<CanonicalTowerFamily>>);
	let hal_witness = HalMultilinearExtensionIndex::new(&dev_alloc, &hal);

	let mut transcript = ProverTranscript::<HasherChallenger<Groestl256>>::new();
	let _ = prove::<_, _, FDomain, _, _, _>(
		&mut oracles,
		&mut witness_index,
		&hal_witness,
		[composite_claim.clone(), shifted_claim.clone()],
		standard_switchover_heuristic(-2),
		&mut transcript,
		&domain_factory,
		&backend,
		&dev_alloc,
		&host_alloc,
	)
	.unwrap();

	let mut transcript = transcript.into_verifier();
	verify(&mut oracles, [composite_claim, shifted_claim], &mut transcript).unwrap();
}

#[test]
fn test_evalcheck_composite_projected() {
	run_test_evalcheck_composite_projected::<PackedBinaryField128x1b, FExtension, PExtension>(8);
}
