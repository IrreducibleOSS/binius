// Copyright 2023 Ulvetanna Inc.

use rayon::{
	iter::{
		Fold, IndexedParallelIterator, IntoParallelIterator, IntoParallelRefMutIterator,
		ParallelIterator,
	},
	range::Iter,
};

use crate::{
	field::{ExtensionField, Field, PackedField},
	polynomial::{extrapolate_line, EvaluationDomain, MultilinearComposite, MultilinearPoly},
	protocols::sumcheck::{Error, SumcheckProof, SumcheckRound},
};

use super::tensor::Tensor;

#[derive(Clone)]
pub struct PreSwitchoverWitness<'a, F: Field, FE: ExtensionField<F>> {
	pub polynomial: MultilinearComposite<'a, F, FE>,
	pub tensor: Tensor<FE>,
}

#[derive(Clone)]
pub struct PreSwitchoverRoundOutput<'a, F: Field, FE: ExtensionField<F>> {
	pub current_proof: SumcheckProof<FE>,
	pub current_witness: PreSwitchoverWitness<'a, F, FE>,
}

#[derive(Clone)]
pub struct PostSwitchoverWitness<'a, F: Field> {
	pub polynomial: MultilinearComposite<'a, F, F>,
}

#[derive(Clone)]
pub struct PostSwitchoverRoundOutput<'a, F: Field, OF: Field + Into<F> + From<F>> {
	pub current_proof: SumcheckProof<F>,
	pub current_witness: PostSwitchoverWitness<'a, OF>,
}

fn process_round_evals<P: PackedField, FE: ExtensionField<P::Scalar>>(
	poly: &MultilinearComposite<P, FE>,
	evals_0: Vec<FE>,
	evals_1: Vec<FE>,
	mut evals_z: Vec<FE>,
	mut round_evals: Vec<FE>,
	degree: usize,
	domain: &[FE],
) -> (Vec<FE>, Vec<FE>, Vec<FE>, Vec<FE>) {
	round_evals[0] = poly
		.composition
		.evaluate(&evals_1)
		.expect("evals_1 is initialized with a length of poly.composition.n_vars()");

	for d in 2..degree + 1 {
		evals_0
			.iter()
			.zip(evals_1.iter())
			.zip(evals_z.iter_mut())
			.for_each(|((&evals_0_j, &evals_1_j), evals_z_j)| {
				*evals_z_j = extrapolate_line(evals_0_j, evals_1_j, domain[d]);
			});
		round_evals[d - 1] = poly
			.composition
			.evaluate(&evals_z)
			.expect("evals_z is initialized with a length of poly.composition.n_vars()");
	}

	(evals_0, evals_1, evals_z, round_evals)
}

fn calculate_round_evals_from_fold_result<F: Field>(
	degree: usize,
	fold_result: Fold<
		Iter<usize>,
		impl Fn() -> (Vec<F>, Vec<F>, Vec<F>, Vec<F>) + Send + Sync,
		impl Fn((Vec<F>, Vec<F>, Vec<F>, Vec<F>), usize) -> (Vec<F>, Vec<F>, Vec<F>, Vec<F>)
			+ Send
			+ Sync,
	>,
) -> Vec<F> {
	fold_result
		.map(|(_, _, _, round_evals)| round_evals)
		.reduce(
			|| vec![F::ZERO; degree],
			|mut overall_round_evals, partial_round_evals| {
				overall_round_evals
					.iter_mut()
					.zip(partial_round_evals.iter())
					.for_each(|(f, s)| *f += s);
				overall_round_evals
			},
		)
}

// Called for round 0 only
pub fn compute_round_coeffs_first<'a, F: Field, FE: ExtensionField<F>>(
	current_witness: PreSwitchoverWitness<'a, F, FE>,
	domain: &EvaluationDomain<FE>,
) -> Result<PreSwitchoverRoundOutput<'a, F, FE>, Error> {
	let poly = current_witness.polynomial;
	let degree = poly.degree();
	let domain = domain.points();

	let n_multilinears = poly.composition.n_vars();
	let rd_vars = poly.n_vars();

	let fold_result = (0..1 << (rd_vars - 1)).into_par_iter().fold(
		|| {
			(
				vec![FE::ZERO; n_multilinears],
				vec![FE::ZERO; n_multilinears],
				vec![FE::ZERO; n_multilinears],
				vec![FE::ZERO; degree],
			)
		},
		|(mut evals_0, mut evals_1, evals_z, round_evals), i| {
			for (j, multilin) in poly.iter_multilinear_polys().enumerate() {
				evals_0[j] = multilin
					.evaluate_on_hypercube(i << 1)
					.unwrap_or_else(|_| panic!("tried to evaluate on hypercube vertex {}, but multilin has n_vars = {}, rd_vars is {}", 1 << i, multilin.n_vars(), rd_vars))
					.into();
				evals_1[j] = multilin
					.evaluate_on_hypercube((i << 1) + 1)
    				.unwrap_or_else(|_| panic!("tried to evaluate on hypercube vertex {}, but multilin has n_vars = {}, rd_vars is {}", (1 << i) + 1, multilin.n_vars(), rd_vars))
					.into();
			}
			process_round_evals(
				&poly,
				evals_0,
				evals_1,
				evals_z,
				round_evals,
				degree,
				domain,
			)
		},
	);

	let round_evals = calculate_round_evals_from_fold_result(degree, fold_result);

	let coeffs = round_evals;
	let current_proof = SumcheckProof {
		rounds: vec![SumcheckRound { coeffs }],
	};

	let result = PreSwitchoverRoundOutput {
		current_proof,
		current_witness: PreSwitchoverWitness {
			polynomial: poly,
			tensor: current_witness.tensor,
		},
	};
	Ok(result)
}

// Called for rounds 1 through s - 1 where s is the last round before the switchover
pub fn compute_round_coeffs_pre_switchover<'a, F: Field, FE: ExtensionField<F>>(
	prev_rd_output: PreSwitchoverRoundOutput<'a, F, FE>,
	domain: &EvaluationDomain<FE>,
) -> Result<PreSwitchoverRoundOutput<'a, F, FE>, Error> {
	let PreSwitchoverRoundOutput {
		current_proof,
		current_witness,
	} = prev_rd_output;

	let tensor = current_witness.tensor;
	let poly = current_witness.polynomial;
	let degree = poly.degree();
	let domain = domain.points();

	let mut updated_proof = current_proof;

	let n_multilinears = poly.composition.n_vars();
	let rd_vars = poly.n_vars() - tensor.round();

	let fold_result = (0..1 << (rd_vars - 1)).into_par_iter().fold(
		|| {
			(
				vec![FE::ZERO; n_multilinears],
				vec![FE::ZERO; n_multilinears],
				vec![FE::ZERO; n_multilinears],
				vec![FE::ZERO; degree],
			)
		},
		|(mut evals_0, mut evals_1, evals_z, round_evals), i| {
			for (j, multilin) in poly.iter_multilinear_polys().enumerate() {
				evals_0[j] = tensor
					.tensor_query(multilin, i << 1)
					.expect("Failed to query tensor");
				evals_1[j] = tensor
					.tensor_query(multilin, (i << 1) + 1)
					.expect("Failed to query tensor");
			}
			process_round_evals(&poly, evals_0, evals_1, evals_z, round_evals, degree, domain)
		},
	);

	let round_evals = calculate_round_evals_from_fold_result(degree, fold_result);

	// round_evals and round_claim, if honest, gives verifier enough information to
	// determine r(X) := \sum_{i \in B_{n-1}} poly(X, i)
	let coeffs = round_evals;
	updated_proof.rounds.push(SumcheckRound { coeffs });
	let result = PreSwitchoverRoundOutput {
		current_proof: updated_proof,
		current_witness: PreSwitchoverWitness {
			polynomial: poly,
			tensor,
		},
	};
	Ok(result)
}

fn fold_multilinear_with_tensor<'a, F: Field, FE: ExtensionField<F>>(
	multilin: &MultilinearPoly<'a, F>,
	tensor: &Tensor<FE>,
) -> Result<MultilinearPoly<'a, FE>, Error> {
	let rd_vars = multilin.n_vars() - tensor.round();
	let mut result_evals = vec![FE::default(); 1 << rd_vars];

	result_evals
		.par_iter_mut()
		.enumerate()
		.for_each(|(i, result_eval)| {
			*result_eval = tensor
				.tensor_query(multilin, i)
				.expect("Failed to query tensor");
		});

	Ok(MultilinearPoly::from_values(result_evals)?)
}

pub fn switchover<F: Field, FE: ExtensionField<F>>(
	pre_switchover_witness: PreSwitchoverWitness<'_, F, FE>,
) -> Result<PostSwitchoverWitness<FE>, Error> {
	let PreSwitchoverWitness { polynomial, tensor } = pre_switchover_witness;

	let rd_vars = polynomial.n_vars() - tensor.round();

	let new_multilinears = polynomial
		.iter_multilinear_polys()
		.map(|multilin| fold_multilinear_with_tensor(multilin, &tensor))
		.collect::<Result<Vec<_>, _>>()?;

	let new_poly =
		<MultilinearComposite<FE, _>>::new(rd_vars, polynomial.composition, new_multilinears)?;

	Ok(PostSwitchoverWitness {
		polynomial: new_poly,
	})
}

pub fn compute_round_coeffs_post_switchover<'a, F, OF>(
	prev_rd_output: PostSwitchoverRoundOutput<'a, F, OF>,
	domain: &EvaluationDomain<F>,
) -> Result<PostSwitchoverRoundOutput<'a, F, OF>, Error>
where
	F: Field,
	OF: Field + Into<F> + From<F>,
{
	let PostSwitchoverRoundOutput {
		current_proof,
		current_witness,
	} = prev_rd_output;

	let poly = current_witness.polynomial;
	let degree = poly.degree();
	let operating_domain = domain
		.points()
		.iter()
		.cloned()
		.map(OF::from)
		.collect::<Vec<_>>();

	let mut updated_proof = current_proof;

	let n_multilinears = poly.composition.n_vars();
	let rd_vars = poly.n_vars();

	let fold_result = (0..1 << (rd_vars - 1)).into_par_iter().fold(
		|| {
			(
				vec![OF::ZERO; n_multilinears],
				vec![OF::ZERO; n_multilinears],
				vec![OF::ZERO; n_multilinears],
				vec![OF::ZERO; degree],
			)
		},
		|(mut evals_0, mut evals_1, evals_z, round_evals), i| {
			for (j, multilin) in poly.iter_multilinear_polys().enumerate() {
				evals_0[j] = multilin.evaluate_on_hypercube(i << 1)
				.unwrap_or_else(|_| panic!("tried to evaluate on hypercube vertex {}, but multilin has n_vars = {}, rd_vars is {}", 1 << i, multilin.n_vars(), rd_vars));

				evals_1[j] = multilin.evaluate_on_hypercube((i << 1) + 1)
				.unwrap_or_else(|_| panic!("tried to evaluate on hypercube vertex {}, but multilin has n_vars = {}, rd_vars is {}", 1 << i, multilin.n_vars(), rd_vars));

			}
			process_round_evals(
				&poly,
				evals_0,
				evals_1,
				evals_z,
				round_evals,
				degree,
				&operating_domain,
			)
		},
	);

	let round_evals = calculate_round_evals_from_fold_result(degree, fold_result);

	let coeffs = round_evals
		.iter()
		.map(|&elem| elem.into())
		.collect::<Vec<_>>();

	updated_proof.rounds.push(SumcheckRound { coeffs });
	let result = PostSwitchoverRoundOutput {
		current_proof: updated_proof,
		current_witness: PostSwitchoverWitness { polynomial: poly },
	};
	Ok(result)
}
