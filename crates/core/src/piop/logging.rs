// Copyright 2025 Irreducible Inc.

use std::collections::HashMap;

use binius_field::TowerField;

use crate::protocols::sumcheck::prove::SumcheckProver;

#[derive(Debug)]
#[allow(dead_code)]
pub(super) struct SumcheckBatchProverDimensionsData {
	prover_n_vars: HashMap<usize, usize>,
	round: usize,
}

impl SumcheckBatchProverDimensionsData {
	pub(super) fn new<'a, F, Prover>(
		round: usize,
		provers: impl IntoIterator<Item = &'a Prover>,
	) -> Self
	where
		F: TowerField,
		Prover: SumcheckProver<F> + 'a,
	{
		let mut prover_n_vars = HashMap::new();
		for prover in provers {
			prover_n_vars.insert(prover.n_vars(), prover.n_vars());
		}
		Self {
			prover_n_vars,
			round,
		}
	}
}

#[derive(Debug)]
#[allow(dead_code)]
pub(super) struct FriFoldRoundsData {
	round: usize,
	log_batch_size: usize,
	codeword_len: usize,
}

impl FriFoldRoundsData {
	pub(super) fn new(round: usize, log_batch_size: usize, codeword_len: usize) -> Self {
		Self {
			round,
			log_batch_size,
			codeword_len,
		}
	}
}
