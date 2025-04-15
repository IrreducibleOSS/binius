// Copyright 2024-2025 Irreducible Inc.

use binius_field::{BinaryField, PackedField};
use binius_math::BinarySubspace;
use binius_maybe_rayon::prelude::*;
use binius_utils::rayon::get_log_max_threads;

use super::{
	additive_ntt::{AdditiveNTT, NTTShape},
	error::Error,
	single_threaded::{self, check_batch_transform_inputs_and_params, SingleThreadedNTT},
	strided_array::StridedArray2DViewMut,
	twiddle::TwiddleAccess,
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
		shape: NTTShape,
		coset: u32,
	) -> Result<(), Error> {
		forward_transform(
			self.log_domain_size(),
			self.single_threaded.twiddles(),
			data,
			shape,
			coset,
			self.log_max_threads,
		)
	}

	fn inverse_transform<P: PackedField<Scalar = F>>(
		&self,
		data: &mut [P],
		shape: NTTShape,
		coset: u32,
	) -> Result<(), Error> {
		inverse_transform(
			self.log_domain_size(),
			self.single_threaded.twiddles(),
			data,
			shape,
			coset,
			self.log_max_threads,
		)
	}
}

#[allow(clippy::too_many_arguments)]
fn forward_transform<F: BinaryField, P: PackedField<Scalar = F>>(
	log_domain_size: usize,
	s_evals: &[impl TwiddleAccess<F> + Sync],
	data: &mut [P],
	shape: NTTShape,
	coset: u32,
	log_max_threads: usize,
) -> Result<(), Error> {
	check_batch_transform_inputs_and_params(log_domain_size, data, shape, coset)?;

	match data.len() {
		0 => return Ok(()),
		1 => {
			return match P::WIDTH {
				1 => Ok(()),
				_ => {
					single_threaded::forward_transform(log_domain_size, s_evals, data, shape, coset)
				}
			};
		}
		_ => {}
	};

	let NTTShape {
		log_x,
		log_y,
		log_z,
	} = shape;

	let log_w = P::LOG_WIDTH;

	// Cutoff is the stage of the NTT where each the butterfly units are contained within
	// packed base field elements.
	let cutoff = log_w.saturating_sub(log_x);

	let log_height = (log_y + log_z).saturating_sub(cutoff).min(log_max_threads);
	let log_width = (log_x + log_y + log_z).saturating_sub(log_w + log_height);

	let par_rounds = log_height.saturating_sub(log_z);

	// Perform the column-wise NTTs in parallel over vertical strides of the matrix.
	{
		let matrix = StridedArray2DViewMut::without_stride(data, 1 << log_height, 1 << log_width)
			.expect("dimensions are correct");

		let log_strides = log_max_threads.min(log_width);
		let log_stride_len = log_width - log_strides;

		matrix
			.into_par_strides(1 << log_stride_len)
			.for_each(|mut stride| {
				// i indexes the layer of the NTT network, also the binary subspace.
				for i in (0..par_rounds).rev() {
					let s_evals_par_i = &s_evals[log_y - par_rounds + i];
					let coset_offset = (coset as usize) << (par_rounds - 1 - i);

					// j indexes the outer Z tensor axis.
					for j in 0..1 << log_z {
						// k indexes the block within the layer. Each block performs butterfly operations with
						// the same twiddle factor.
						for k in 0..1 << (par_rounds - 1 - i) {
							let twiddle = P::broadcast(s_evals_par_i.get(coset_offset | k));
							// l indexes parallel stride columns
							for l in 0..1 << i {
								for m in 0..1 << log_stride_len {
									let idx0 = j << par_rounds | k << (i + 1) | l;
									let idx1 = idx0 | 1 << i;

									let mut u = stride[(idx0, m)];
									let mut v = stride[(idx1, m)];
									u += v * twiddle;
									v += u;
									stride[(idx0, m)] = u;
									stride[(idx1, m)] = v;
								}
							}
						}
					}
				}
			});
	}

	let log_row_z = log_z.saturating_sub(log_height);
	let single_thread_log_y = log_width + log_w - log_x - log_row_z;

	data.par_chunks_mut(1 << (log_width + par_rounds))
		.flat_map(|large_chunk| large_chunk.par_chunks_mut(1 << log_width).enumerate())
		.try_for_each(|(inner_coset, chunk)| {
			single_threaded::forward_transform(
				log_domain_size,
				&s_evals[0..log_y - par_rounds],
				chunk,
				NTTShape {
					log_x,
					log_y: single_thread_log_y,
					log_z: log_row_z,
				},
				coset << par_rounds | inner_coset as u32,
			)
		})?;

	Ok(())
}

#[allow(clippy::too_many_arguments)]
fn inverse_transform<F: BinaryField, P: PackedField<Scalar = F>>(
	log_domain_size: usize,
	s_evals: &[impl TwiddleAccess<F> + Sync],
	data: &mut [P],
	shape: NTTShape,
	coset: u32,
	log_max_threads: usize,
) -> Result<(), Error> {
	check_batch_transform_inputs_and_params(log_domain_size, data, shape, coset)?;

	match data.len() {
		0 => return Ok(()),
		1 => {
			return match P::WIDTH {
				1 => Ok(()),
				_ => {
					single_threaded::inverse_transform(log_domain_size, s_evals, data, shape, coset)
				}
			};
		}
		_ => {}
	};

	let NTTShape {
		log_x,
		log_y,
		log_z,
	} = shape;

	let log_w = P::LOG_WIDTH;

	// Cutoff is the stage of the NTT where each the butterfly units are contained within
	// packed base field elements.
	let cutoff = log_w.saturating_sub(log_x);

	let log_height = (log_y + log_z).saturating_sub(cutoff).min(log_max_threads);
	let log_width = (log_x + log_y + log_z).saturating_sub(log_w + log_height);

	let par_rounds = log_height.saturating_sub(log_z);
	let log_row_z = log_z.saturating_sub(log_height);

	let single_thread_log_y = log_width + log_w - log_x - log_row_z;

	data.par_chunks_mut(1 << (log_width + par_rounds))
		.flat_map(|large_chunk| large_chunk.par_chunks_mut(1 << log_width).enumerate())
		.try_for_each(|(inner_coset, chunk)| {
			single_threaded::inverse_transform(
				log_domain_size,
				&s_evals[0..log_y - par_rounds],
				chunk,
				NTTShape {
					log_x,
					log_y: single_thread_log_y,
					log_z: log_row_z,
				},
				coset << par_rounds | inner_coset as u32,
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
			// i indexes the layer of the NTT network, also the binary subspace.
			for i in 0..par_rounds {
				let s_evals_par_i = &s_evals[log_y - par_rounds + i];
				let coset_offset = (coset as usize) << (par_rounds - 1 - i);

				// j indexes the outer Z tensor axis.
				for j in 0..1 << log_z {
					// k indexes the block within the layer. Each block performs butterfly operations with
					// the same twiddle factor.
					for k in 0..1 << (par_rounds - 1 - i) {
						let twiddle = P::broadcast(s_evals_par_i.get(coset_offset | k));
						// l indexes parallel stride columns
						for l in 0..1 << i {
							for m in 0..1 << log_stride_len {
								let idx0 = j << par_rounds | k << (i + 1) | l;
								let idx1 = idx0 | 1 << i;

								let mut u = stride[(idx0, m)];
								let mut v = stride[(idx1, m)];
								v += u;
								u += v * twiddle;
								stride[(idx0, m)] = u;
								stride[(idx1, m)] = v;
							}
						}
					}
				}
			}
		});

	Ok(())
}
