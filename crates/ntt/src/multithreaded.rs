// Copyright 2024-2025 Irreducible Inc.

use binius_field::{BinaryField, PackedField};
use binius_math::BinarySubspace;
use binius_maybe_rayon::prelude::*;
use binius_utils::rayon::get_log_max_threads;

use super::{
	error::Error,
	single_threaded::{self, check_batch_transform_inputs_and_params},
	strided_array::StridedArray2DViewMut,
	twiddle::TwiddleAccess,
	AdditiveNTT, SingleThreadedNTT,
};
use crate::twiddle::OnTheFlyTwiddleAccess;

/// Implementation of `AdditiveNTT` that performs the computation multithreaded.
#[derive(Debug)]
pub struct MultithreadedNTT<F: BinaryField, TA: TwiddleAccess<F> = OnTheFlyTwiddleAccess<F, Vec<F>>>
{
	single_threaded: SingleThreadedNTT<F, TA>,
	log_max_threads: usize,
}

impl<F: BinaryField, TA: TwiddleAccess<F> + Sync> SingleThreadedNTT<F, TA> {
	/// Returns multithreaded NTT implementation which uses default number of threads.
	pub fn multithreaded(self) -> MultithreadedNTT<F, TA> {
		let log_max_threads = get_log_max_threads();
		self.multithreaded_with_max_threads(log_max_threads as _)
	}

	/// Returns multithreaded NTT implementation which uses `1 << log_max_threads` threads.
	pub const fn multithreaded_with_max_threads(
		self,
		log_max_threads: usize,
	) -> MultithreadedNTT<F, TA> {
		MultithreadedNTT {
			single_threaded: self,
			log_max_threads,
		}
	}
}

impl<F, TA> AdditiveNTT<F> for MultithreadedNTT<F, TA>
where
	F: BinaryField,
	TA: TwiddleAccess<F> + Sync,
{
	fn log_domain_size(&self) -> usize {
		self.single_threaded.log_domain_size()
	}

	fn subspace(&self, i: usize) -> BinarySubspace<F> {
		self.single_threaded.subspace(i)
	}

	fn get_subspace_eval(&self, i: usize, j: usize) -> F {
		self.single_threaded.get_subspace_eval(i, j)
	}

	fn forward_transform<P: PackedField<Scalar = F>>(
		&self,
		data: &mut [P],
		coset: u32,
		log_batch_size: usize,
		log_n: usize,
	) -> Result<(), Error> {
		forward_transform(
			self.log_domain_size(),
			self.single_threaded.twiddles(),
			data,
			coset,
			log_batch_size,
			log_n,
			self.log_max_threads,
		)
	}

	fn inverse_transform<P: PackedField<Scalar = F>>(
		&self,
		data: &mut [P],
		coset: u32,
		log_batch_size: usize,
		log_n: usize,
	) -> Result<(), Error> {
		inverse_transform(
			self.log_domain_size(),
			self.single_threaded.twiddles(),
			data,
			coset,
			log_batch_size,
			log_n,
			self.log_max_threads,
		)
	}
}

fn forward_transform<F: BinaryField, P: PackedField<Scalar = F>>(
	log_domain_size: usize,
	s_evals: &[impl TwiddleAccess<F> + Sync],
	data: &mut [P],
	coset: u32,
	log_batch_size: usize,
	log_n: usize,
	log_max_threads: usize,
) -> Result<(), Error> {
	match data.len() {
		0 => return Ok(()),
		1 => {
			return match P::WIDTH {
				1 => Ok(()),
				_ => single_threaded::forward_transform(
					log_domain_size,
					s_evals,
					data,
					coset,
					log_batch_size,
					P::LOG_WIDTH - log_batch_size,
				),
			};
		}
		_ => {}
	};

	let log_w = P::LOG_WIDTH;

	let log_b = log_batch_size;

	check_batch_transform_inputs_and_params(log_domain_size, data, coset, log_b, log_n)?;

	// Cutoff is the stage of the NTT where each the butterfly units are contained within
	// packed base field elements.
	let cutoff = log_w.saturating_sub(log_b);

	let par_rounds = (log_n - cutoff).min(log_max_threads);
	let log_height = par_rounds;
	let log_width = log_n + log_b - log_w - log_height;

	// Perform the column-wise NTTs in parallel over vertical strides of the matrix.
	{
		let matrix = StridedArray2DViewMut::without_stride(data, 1 << log_height, 1 << log_width)
			.expect("dimensions are correct");

		let log_strides = log_max_threads.min(log_width);
		let log_stride_len = log_width - log_strides;

		matrix
			.into_par_strides(1 << log_stride_len)
			.for_each(|mut stride| {
				for i in (0..par_rounds).rev() {
					let coset_twiddle = s_evals[log_n - par_rounds + i]
						.coset(log_domain_size - log_n, coset as usize);

					for j in 0..1 << (par_rounds - 1 - i) {
						let twiddle = P::broadcast(coset_twiddle.get(j));
						for k in 0..1 << i {
							for l in 0..1 << log_stride_len {
								let idx0 = j << (i + 1) | k;
								let idx1 = idx0 | 1 << i;

								let mut u = stride[(idx0, l)];
								let mut v = stride[(idx1, l)];
								u += v * twiddle;
								v += u;
								stride[(idx0, l)] = u;
								stride[(idx1, l)] = v;
							}
						}
					}
				}
			});
	}

	let single_thread_log_n = log_width + P::LOG_WIDTH - log_batch_size;

	data.par_chunks_mut(1 << log_width)
		.enumerate()
		.try_for_each(|(inner_coset, chunk)| {
			single_threaded::forward_transform(
				log_domain_size,
				&s_evals[0..log_n - par_rounds],
				chunk,
				coset << par_rounds | (inner_coset as u32),
				log_batch_size,
				single_thread_log_n,
			)
		})?;

	Ok(())
}

fn inverse_transform<F: BinaryField, P: PackedField<Scalar = F>>(
	log_domain_size: usize,
	s_evals: &[impl TwiddleAccess<F> + Sync],
	data: &mut [P],
	coset: u32,
	log_batch_size: usize,
	log_n: usize,
	log_max_threads: usize,
) -> Result<(), Error> {
	match data.len() {
		0 => return Ok(()),
		1 => {
			return match P::WIDTH {
				1 => Ok(()),
				_ => single_threaded::inverse_transform(
					log_domain_size,
					s_evals,
					data,
					coset,
					log_batch_size,
					log_n.saturating_sub(log_batch_size),
				),
			};
		}
		_ => {}
	};

	let log_w = P::LOG_WIDTH;

	let log_b = log_batch_size;

	check_batch_transform_inputs_and_params(log_domain_size, data, coset, log_b, log_n)?;

	// Cutoff is the stage of the NTT where each the butterfly units are contained within
	// packed base field elements.
	let cutoff = log_w.saturating_sub(log_b);

	let par_rounds = (log_n - cutoff).min(log_max_threads);
	let log_height = par_rounds;
	let log_width = log_n + log_b - log_w - log_height;
	let single_thread_log_n = log_width + P::LOG_WIDTH - log_batch_size;

	data.par_chunks_mut(1 << log_width)
		.enumerate()
		.try_for_each(|(inner_coset, chunk)| {
			single_threaded::inverse_transform(
				log_domain_size,
				&s_evals[0..log_n - par_rounds],
				chunk,
				coset << par_rounds | (inner_coset as u32),
				log_batch_size,
				single_thread_log_n,
			)
		})?;

	// Perform the column-wise NTTs in parallel over vertical strides of the matrix.
	let matrix = StridedArray2DViewMut::without_stride(data, 1 << log_height, 1 << log_width)
		.expect("dimensions are correct");

	let log_strides = log_max_threads.min(log_width);
	let log_stride_len = log_width - log_strides;

	matrix
		.into_par_strides(1 << log_stride_len)
		.for_each(|mut stride| {
			for i in 0..par_rounds {
				let coset_twiddle =
					s_evals[log_n - par_rounds + i].coset(log_domain_size - log_n, coset as usize);

				for j in 0..1 << (par_rounds - 1 - i) {
					let twiddle = P::broadcast(coset_twiddle.get(j));
					for k in 0..1 << i {
						for l in 0..1 << log_stride_len {
							let idx0 = j << (i + 1) | k;
							let idx1 = idx0 | 1 << i;

							let mut u = stride[(idx0, l)];
							let mut v = stride[(idx1, l)];
							v += u;
							u += v * twiddle;
							stride[(idx0, l)] = u;
							stride[(idx1, l)] = v;
						}
					}
				}
			}
		});

	if log_n + log_b < log_w {
		for i in 1 << (log_n + log_b)..P::WIDTH {
			data[0].set(i, F::ZERO);
		}
	}

	Ok(())
}
