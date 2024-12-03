// Copyright 2024 Irreducible Inc.

use crate::{
	polynomial::{Error, MultivariatePoly},
	tensor_algebra::TensorAlgebra,
	tower::TowerFamily,
};
use binius_field::{
	util::inner_product_unchecked, ExtensionField, Field, PackedExtension, PackedField,
	PackedFieldIndexable, TowerField,
};
use binius_hal::ComputationBackend;
use binius_math::{MultilinearExtension, MultilinearQuery};
use binius_utils::bail;
use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};
use std::{fmt::Debug, iter, marker::PhantomData};
use tracing::instrument;

/// This struct provides generic functionality for the multilinear function $A$ from [DP24] Section 5.
///
/// Unlike the `RingSwitchEqInd` struct, this struct is parameterized by the tower height, which is
/// a runtime parameter.
///
/// Recall that the "ring-switched eq indicator" is a transparent multilinear required for the ring-switching argument.
/// The principal purpose of the below abstraction is for `fri_pcs` to not need to know about the small field.
///
/// [DP24]: <https://eprint.iacr.org/2024/504>
#[derive(Debug)]
pub enum TowerRingSwitchEqInd<Tower>
where
	Tower: TowerFamily + Debug,
	Tower::B128: PackedField<Scalar = Tower::B128>
		+ PackedExtension<Tower::B1>
		+ PackedExtension<Tower::B8>
		+ PackedExtension<Tower::B16>
		+ PackedExtension<Tower::B32>
		+ PackedExtension<Tower::B64>
		+ PackedExtension<Tower::B128>,
{
	B1(RingSwitchEqInd<Tower::B1, Tower::B128>),
	B8(RingSwitchEqInd<Tower::B8, Tower::B128>),
	B16(RingSwitchEqInd<Tower::B16, Tower::B128>),
	B32(RingSwitchEqInd<Tower::B32, Tower::B128>),
	B64(RingSwitchEqInd<Tower::B64, Tower::B128>),
	B128(RingSwitchEqInd<Tower::B128, Tower::B128>),
}

impl<Tower> TowerRingSwitchEqInd<Tower>
where
	Tower: TowerFamily + Debug,
	Tower::B128: PackedField<Scalar = Tower::B128>
		+ PackedExtension<Tower::B1>
		+ PackedExtension<Tower::B8>
		+ PackedExtension<Tower::B16>
		+ PackedExtension<Tower::B32>
		+ PackedExtension<Tower::B64>
		+ PackedExtension<Tower::B128>,
{
	fn n_vars(&self) -> usize {
		match self {
			TowerRingSwitchEqInd::B1(rs_eq) => rs_eq.n_vars(),
			TowerRingSwitchEqInd::B8(rs_eq) => rs_eq.n_vars(),
			TowerRingSwitchEqInd::B16(rs_eq) => rs_eq.n_vars(),
			TowerRingSwitchEqInd::B32(rs_eq) => rs_eq.n_vars(),
			TowerRingSwitchEqInd::B64(rs_eq) => rs_eq.n_vars(),
			TowerRingSwitchEqInd::B128(rs_eq) => rs_eq.n_vars(),
		}
	}

	pub fn new(
		tower_height: usize,
		r_evals: Vec<Tower::B128>,
		r_mixing_challenges: Vec<Tower::B128>,
	) -> Result<Self, Error> {
		let tower_rs_eq_id = match tower_height {
			0 => TowerRingSwitchEqInd::B1(RingSwitchEqInd::<Tower::B1, _>::new(
				r_evals,
				r_mixing_challenges,
			)?),
			3 => TowerRingSwitchEqInd::B8(RingSwitchEqInd::<Tower::B8, _>::new(
				r_evals,
				r_mixing_challenges,
			)?),
			4 => TowerRingSwitchEqInd::B16(RingSwitchEqInd::<Tower::B16, _>::new(
				r_evals,
				r_mixing_challenges,
			)?),
			5 => TowerRingSwitchEqInd::B32(RingSwitchEqInd::<Tower::B32, _>::new(
				r_evals,
				r_mixing_challenges,
			)?),
			6 => TowerRingSwitchEqInd::B64(RingSwitchEqInd::<Tower::B64, _>::new(
				r_evals,
				r_mixing_challenges,
			)?),
			7 => TowerRingSwitchEqInd::B128(RingSwitchEqInd::<Tower::B128, _>::new(
				r_evals,
				r_mixing_challenges,
			)?),
			_ => Err(Error::InvalidTowerHeight {
				actual: tower_height,
			})?,
		};
		Ok(tower_rs_eq_id)
	}

	pub fn multilinear_extension<
		P: PackedFieldIndexable<Scalar = Tower::B128>,
		Backend: ComputationBackend,
	>(
		&self,
		backend: &Backend,
	) -> Result<MultilinearExtension<P, Backend::Vec<P>>, Error> {
		match self {
			TowerRingSwitchEqInd::B1(rs_eq) => rs_eq.multilinear_extension(backend),
			TowerRingSwitchEqInd::B8(rs_eq) => rs_eq.multilinear_extension(backend),
			TowerRingSwitchEqInd::B16(rs_eq) => rs_eq.multilinear_extension(backend),
			TowerRingSwitchEqInd::B32(rs_eq) => rs_eq.multilinear_extension(backend),
			TowerRingSwitchEqInd::B64(rs_eq) => rs_eq.multilinear_extension(backend),
			TowerRingSwitchEqInd::B128(rs_eq) => rs_eq.multilinear_extension(backend),
		}
	}

	fn evaluate(&self, query: &[Tower::B128]) -> Result<Tower::B128, Error> {
		match self {
			TowerRingSwitchEqInd::B1(rs_eq) => rs_eq.evaluate(query),
			TowerRingSwitchEqInd::B8(rs_eq) => rs_eq.evaluate(query),
			TowerRingSwitchEqInd::B16(rs_eq) => rs_eq.evaluate(query),
			TowerRingSwitchEqInd::B32(rs_eq) => rs_eq.evaluate(query),
			TowerRingSwitchEqInd::B64(rs_eq) => rs_eq.evaluate(query),
			TowerRingSwitchEqInd::B128(rs_eq) => rs_eq.evaluate(query),
		}
	}
}

impl<Tower> MultivariatePoly<Tower::B128> for TowerRingSwitchEqInd<Tower>
where
	Tower: TowerFamily + Debug,
	Tower::B128: PackedField<Scalar = Tower::B128>
		+ PackedExtension<Tower::B1>
		+ PackedExtension<Tower::B8>
		+ PackedExtension<Tower::B16>
		+ PackedExtension<Tower::B32>
		+ PackedExtension<Tower::B64>
		+ PackedExtension<Tower::B128>,
{
	fn n_vars(&self) -> usize {
		self.n_vars()
	}

	fn degree(&self) -> usize {
		self.n_vars()
	}

	fn evaluate(&self, query: &[Tower::B128]) -> Result<Tower::B128, Error> {
		// query is typically going to be `r_sumcheck_challenges`
		let n_vars = MultivariatePoly::<Tower::B128>::n_vars(self);
		if query.len() != n_vars {
			bail!(Error::IncorrectQuerySize { expected: n_vars });
		};

		self.evaluate(query)
	}

	fn binary_tower_level(&self) -> usize {
		Tower::B128::TOWER_LEVEL
	}
}

/// This struct provides functionality for the multilinear function $A$ from [DP24] Section 5.
///
/// The function $A$ is $\ell':= \ell - \kappa$-variate and depends on the last $\ell'$ coordinates
/// of the evaluation point as well as the $\kappa$ mixing challenges. The struct takes as generics both
/// a small and a big field.
///
/// [DP24]: <https://eprint.iacr.org/2024/504>
#[derive(Debug)]
pub struct RingSwitchEqInd<FS, F>
where
	FS: Field,
	F: ExtensionField<FS>,
{
	n_vars: usize,
	r_evals: Vec<F>, // r_{\kappa}, \ldots, r_{\ell-1} (in particular, we forget about r_0, \ldots, r_{\kappa-1})
	r_mixing_challenges: Vec<F>, // r''_0, \ldots, r''_{\kappa-1}
	_phantom: PhantomData<FS>,
}

impl<FS, F> RingSwitchEqInd<FS, F>
where
	FS: Field,
	F: ExtensionField<FS> + PackedField<Scalar = F> + PackedExtension<FS>,
{
	pub fn new(r_evals: Vec<F>, r_mixing_challenges: Vec<F>) -> Result<Self, Error> {
		let n_vars = r_evals.len();

		if r_mixing_challenges.len() != <TensorAlgebra<FS, F>>::kappa() {
			bail!(Error::RingSwitchWrongLength {
				expected: <TensorAlgebra<FS, F>>::kappa(),
				actual: r_mixing_challenges.len()
			});
		}

		Ok(Self {
			n_vars,
			r_evals,
			r_mixing_challenges,
			_phantom: PhantomData,
		})
	}

	#[instrument("RingSwitchEqInd::multilinear_extension", skip_all, level = "trace")]
	pub fn multilinear_extension<P, Backend>(
		&self,
		backend: &Backend,
	) -> Result<MultilinearExtension<P, Backend::Vec<P>>, Error>
	where
		P: PackedFieldIndexable<Scalar = F>,
		Backend: ComputationBackend,
	{
		let r_evals = &self.r_evals;
		let r_mixing_challenges = &self.r_mixing_challenges;
		let expanded_mixing_coeffs = MultilinearQuery::expand(r_mixing_challenges);
		let mut evals = backend.tensor_product_full_query(r_evals)?;
		P::unpack_scalars_mut(&mut evals)
			.par_iter_mut()
			.for_each(|val| {
				let vert = *val;
				*val = inner_product_unchecked(
					expanded_mixing_coeffs.expansion().iter().copied(),
					ExtensionField::<FS>::iter_bases(&vert),
				);
			});
		Ok(MultilinearExtension::from_values_generic(evals)?)
	}
}

impl<FS, F> MultivariatePoly<F> for RingSwitchEqInd<FS, F>
where
	FS: TowerField,
	F: ExtensionField<FS> + PackedField<Scalar = F> + PackedExtension<FS> + TowerField,
{
	fn n_vars(&self) -> usize {
		self.n_vars
	}

	fn degree(&self) -> usize {
		self.n_vars
	}

	fn evaluate(&self, query: &[F]) -> Result<F, Error> {
		// query is typically going to be `r_sumcheck_challenges`
		let n_vars = MultivariatePoly::<F>::n_vars(self);
		if query.len() != n_vars {
			bail!(Error::IncorrectQuerySize { expected: n_vars });
		};
		let r_evals = &self.r_evals;
		let r_mixing_challenges = &self.r_mixing_challenges;

		let tensor_eval = iter::zip(r_evals.iter().copied(), query.iter().copied()).fold(
			TensorAlgebra::one(),
			|eval, (vert_i, hztl_i)| {
				// This formula is specific to characteristic 2 fields
				// Here we know that $h v + (1 - h) (1 - v) = 1 + h + v$.
				let vert_scaled = eval.clone().scale_vertical(vert_i);
				let hztl_scaled = eval.clone().scale_horizontal(hztl_i);
				eval + &vert_scaled + &hztl_scaled
			},
		);
		let expanded_mixing_coeffs = MultilinearQuery::expand(r_mixing_challenges);
		let folded_eval = inner_product_unchecked::<F, _>(
			tensor_eval.transpose().vertical_elems().iter().copied(),
			expanded_mixing_coeffs.into_expansion(),
		);
		Ok(folded_eval)
	}

	// realistically this will be 7.
	fn binary_tower_level(&self) -> usize {
		F::TOWER_LEVEL
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::tower::{self, AESTowerFamily};
	use binius_field::{BinaryField128b, BinaryField8b};
	use binius_hal::{make_portable_backend, ComputationBackendExt};
	use iter::repeat_with;
	use rand::{prelude::StdRng, SeedableRng};

	#[test]
	fn test_ring_switch_eq_ind() {
		type FS = BinaryField8b;
		type F = BinaryField128b;
		let kappa = <TensorAlgebra<FS, F>>::kappa();
		let ell = 10;
		let mut rng = StdRng::seed_from_u64(0);

		let n_vars = ell - kappa;
		let r_evals = repeat_with(|| <F as Field>::random(&mut rng))
			.take(n_vars)
			.collect::<Vec<_>>();

		let r_mixing_challenges = repeat_with(|| <F as Field>::random(&mut rng))
			.take(kappa)
			.collect::<Vec<_>>();
		let backend = make_portable_backend();

		let r_sumcheck_challenges = repeat_with(|| <F as Field>::random(&mut rng))
			.take(n_vars)
			.collect::<Vec<_>>();

		let rs_eq =
			RingSwitchEqInd::<FS, _>::new(r_evals.clone(), r_mixing_challenges.clone()).unwrap();

		let val1 = rs_eq.evaluate(&r_sumcheck_challenges).unwrap();

		let partial_evals = rs_eq.multilinear_extension::<F, _>(&backend).unwrap();
		let val2 = partial_evals
			.evaluate(
				&backend
					.multilinear_query::<F>(&r_sumcheck_challenges)
					.unwrap(),
			)
			.unwrap();

		assert_eq!(val1, val2);
	}

	#[test]
	fn test_tower_ring_switch_eq_ind() {
		type F = <tower::AESTowerFamily as tower::TowerFamily>::B128;
		let tower_height = 4;
		type FS = <tower::AESTowerFamily as tower::TowerFamily>::B16;
		let kappa = 7 - tower_height;
		let ell = 10;
		let mut rng = StdRng::seed_from_u64(0);

		let n_vars = ell - kappa;
		let r_evals = repeat_with(|| <F as Field>::random(&mut rng))
			.take(n_vars)
			.collect::<Vec<_>>();

		let r_mixing_challenges = repeat_with(|| <F as Field>::random(&mut rng))
			.take(kappa)
			.collect::<Vec<_>>();

		let tower_rs_eq = TowerRingSwitchEqInd::<AESTowerFamily>::new(
			tower_height,
			r_evals.clone(),
			r_mixing_challenges.clone(),
		)
		.unwrap();

		let backend = make_portable_backend();

		let r_sumcheck_challenges = repeat_with(|| <F as Field>::random(&mut rng))
			.take(n_vars)
			.collect::<Vec<_>>();

		let rs_eq: RingSwitchEqInd<FS, F> =
			RingSwitchEqInd::new(r_evals.clone(), r_mixing_challenges.clone()).unwrap();

		let val1 = tower_rs_eq.evaluate(&r_sumcheck_challenges).unwrap();
		// we run a similar test to our usual ring_switch_eq_ind, where we first build the `MultilinearExtension` and
		// then run `.evaluate()`.
		let partial_evals: MultilinearExtension<F> = rs_eq.multilinear_extension(&backend).unwrap();
		let val2 = partial_evals
			.evaluate(
				&backend
					.multilinear_query::<F>(&r_sumcheck_challenges)
					.unwrap(),
			)
			.unwrap();

		assert_eq!(val1, val2);
	}
}
