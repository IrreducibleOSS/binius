// Copyright 2024-2025 Irreducible Inc.

use std::iter::successors;

use binius_field::{BinaryField128b, PackedField, TowerField};
use binius_macros::{DeserializeBytes, SerializeBytes, erased_serialize_bytes};
use binius_math::MultilinearExtension;
use binius_maybe_rayon::prelude::*;
use binius_utils::{DeserializeBytes, bail};
use bytemuck::zeroed_vec;
use itertools::{Itertools, izip};

use crate::polynomial::{Error, MultivariatePoly};

/// A transparent multilinear polynomial whose evaluation at index $i$ is $g^i$ for
/// some field element $g$.
#[derive(Debug, SerializeBytes, DeserializeBytes)]
pub struct Powers<F: TowerField> {
	n_vars: usize,
	base: F,
}

inventory::submit! {
	<dyn MultivariatePoly<BinaryField128b>>::register_deserializer(
		"Powers",
		|buf, mode| Ok(Box::new(Powers::<BinaryField128b>::deserialize(&mut *buf, mode)?))
	)
}

impl<F: TowerField> Powers<F> {
	pub const fn new(n_vars: usize, base: F) -> Self {
		Self { n_vars, base }
	}

	pub fn multilinear_extension<P: PackedField<Scalar = F>>(
		&self,
	) -> Result<MultilinearExtension<P>, Error> {
		let mut values = zeroed_vec(1 << self.n_vars.saturating_sub(P::LOG_WIDTH));

		const CHUNK_LOG_SIZE: usize = 10;
		values
			.par_chunks_mut(1 << CHUNK_LOG_SIZE)
			.enumerate()
			.for_each(|(chunk_index, chunk)| {
				let start_power = (chunk_index as u64) << CHUNK_LOG_SIZE;
				let powers = successors(Some(self.base.pow_vartime([start_power])), |&power| {
					Some(power * self.base)
				})
				.take(1 << self.n_vars);

				for (dest, values) in izip!(chunk, &powers.chunks(P::WIDTH)) {
					*dest = P::from_scalars(values);
				}
			});

		Ok(MultilinearExtension::new(self.n_vars, values)?)
	}
}

#[erased_serialize_bytes]
impl<F: TowerField, P: PackedField<Scalar = F>> MultivariatePoly<P> for Powers<F> {
	fn n_vars(&self) -> usize {
		self.n_vars
	}

	fn degree(&self) -> usize {
		self.n_vars
	}

	fn evaluate(&self, query: &[P]) -> Result<P, Error> {
		let n_vars = self.n_vars;
		if query.len() != self.n_vars {
			bail!(Error::IncorrectQuerySize { expected: n_vars });
		}

		let mut result = P::one();
		let mut base_power = self.base;
		for &q_i in query {
			result *= P::one() - q_i + q_i * base_power;
			base_power = base_power.square();
		}

		Ok(result)
	}

	fn binary_tower_level(&self) -> usize {
		F::TOWER_LEVEL
	}
}

#[cfg(test)]
mod tests {
	use binius_field::{BinaryField32b, Field, PackedBinaryField4x32b, PackedField, TowerField};
	use rand::{SeedableRng, prelude::StdRng};

	use super::Powers;
	use crate::polynomial::MultivariatePoly;

	fn test_consistency_helper<P: PackedField<Scalar: TowerField>>(n_vars: usize, base: P::Scalar) {
		let powers = Powers::new(n_vars, base);
		let mle = powers.multilinear_extension::<P>().unwrap();
		let mut power = P::Scalar::ONE;
		for i in 0..1 << n_vars {
			let query = (0..n_vars)
				.map(|j| {
					if (i >> j) & 1 == 0 {
						P::zero()
					} else {
						P::one()
					}
				})
				.collect::<Vec<_>>();
			assert_eq!(powers.evaluate(&query).unwrap(), P::broadcast(power));
			assert_eq!(mle.evaluate_on_hypercube(i).unwrap(), power);
			power *= base;
		}
	}

	#[test]
	fn test_consistency() {
		type F = BinaryField32b;
		type P = PackedBinaryField4x32b;
		let mut rng = StdRng::seed_from_u64(0);
		for n_vars in 0..12 {
			test_consistency_helper::<P>(n_vars, <F as Field>::random(&mut rng));
		}
	}
}
