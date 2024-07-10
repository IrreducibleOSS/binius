// Copyright 2024 Ulvetanna Inc.

//! Simple reference implementations of forward and inverse transforms on non-packed field elements
//! for testing.

use super::{error::Error, twiddle::TwiddleAccess};
use binius_field::{BinaryField, ExtensionField};

pub fn forward_transform_simple<F, FF>(
	log_domain_size: usize,
	s_evals: &[impl TwiddleAccess<F>],
	data: &mut [FF],
	coset: u32,
) -> Result<(), Error>
where
	F: BinaryField,
	FF: ExtensionField<F>,
{
	let n = data.len();
	assert!(n.is_power_of_two());

	let log_n = n.trailing_zeros() as usize;
	let coset_bits = 32 - coset.leading_zeros() as usize;
	if log_n + coset_bits > log_domain_size {
		return Err(Error::DomainTooSmall {
			log_required_domain_size: log_n + coset_bits,
		});
	}

	for i in (0..log_n).rev() {
		let s_evals_i = &s_evals[i];
		for j in 0..1 << (log_n - 1 - i) {
			let twiddle = s_evals_i.get((coset as usize) << (log_n - 1 - i) | j);
			for k in 0..1 << i {
				let idx0 = j << (i + 1) | k;
				let idx1 = idx0 | 1 << i;
				data[idx0] += data[idx1] * twiddle;
				data[idx1] += data[idx0];
			}
		}
	}

	Ok(())
}

pub fn inverse_transform_simple<F, FF>(
	log_domain_size: usize,
	s_evals: &[impl TwiddleAccess<F>],
	data: &mut [FF],
	coset: u32,
) -> Result<(), Error>
where
	F: BinaryField,
	FF: ExtensionField<F>,
{
	let n = data.len();
	assert!(n.is_power_of_two());

	let log_n = n.trailing_zeros() as usize;
	let coset_bits = 32 - coset.leading_zeros() as usize;
	if log_n + coset_bits > log_domain_size {
		return Err(Error::DomainTooSmall {
			log_required_domain_size: log_n + coset_bits,
		});
	}

	#[allow(clippy::needless_range_loop)]
	for i in 0..log_n {
		let s_evals_i = &s_evals[i];
		for j in 0..1 << (log_n - 1 - i) {
			let twiddle = s_evals_i.get((coset as usize) << (log_n - 1 - i) | j);
			for k in 0..1 << i {
				let idx0 = j << (i + 1) | k;
				let idx1 = idx0 | 1 << i;
				data[idx1] += data[idx0];
				data[idx0] += data[idx1] * twiddle;
			}
		}
	}

	Ok(())
}
