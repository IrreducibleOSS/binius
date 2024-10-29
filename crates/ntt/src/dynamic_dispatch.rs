// Copyright 2024 Irreducible Inc.

use crate::{twiddle::PrecomputedTwiddleAccess, AdditiveNTT, MultithreadedNTT, SingleThreadedNTT};
use binius_field::{BinaryField, PackedField};
use binius_utils::rayon::get_log_max_threads;

/// How many threads to use (threads number is a power of 2).
#[derive(Default, Debug, Clone, Copy)]
pub enum ThreadingSettings {
	/// Use a single thread for calculations.
	#[default]
	SingleThreaded,
	/// Use the default number of threads based on the number of cores.
	MultithreadedDefault,
	/// Explicitly set the logarithm of number of threads.
	ExplicitThreadsCount { log_threads: usize },
}

impl ThreadingSettings {
	/// Get the log2 of the number of threads to use.
	pub fn log_threads_count(&self) -> usize {
		match self {
			ThreadingSettings::SingleThreaded => 0,
			ThreadingSettings::MultithreadedDefault => get_log_max_threads(),
			ThreadingSettings::ExplicitThreadsCount { log_threads } => *log_threads,
		}
	}

	/// Check if settings imply multithreading.
	pub fn is_multithreaded(&self) -> bool {
		match self {
			ThreadingSettings::SingleThreaded => false,
			ThreadingSettings::MultithreadedDefault => true,
			ThreadingSettings::ExplicitThreadsCount { log_threads } => *log_threads > 0,
		}
	}
}

#[derive(Default)]
pub struct NTTOptions {
	pub precompute_twiddles: bool,
	pub thread_settings: ThreadingSettings,
}

/// An enum that can be used to switch between different NTT implementations without passing AdditiveNTT as a type parameter.
#[derive(Debug)]
pub enum DynamicDispatchNTT<F: BinaryField> {
	SingleThreaded(SingleThreadedNTT<F>),
	SingleThreadedPrecompute(SingleThreadedNTT<F, PrecomputedTwiddleAccess<F>>),
	MultiThreaded(MultithreadedNTT<F>),
	MultiThreadedPrecompute(MultithreadedNTT<F, PrecomputedTwiddleAccess<F>>),
}

impl<F: BinaryField> DynamicDispatchNTT<F> {
	/// Create a new AdditiveNTT based on the given settings.
	pub fn new(log_domain_size: usize, options: NTTOptions) -> Result<Self, crate::error::Error> {
		let log_threads = options.thread_settings.log_threads_count();
		let result = match (options.precompute_twiddles, log_threads) {
			(false, 0) => {
				DynamicDispatchNTT::SingleThreaded(SingleThreadedNTT::new(log_domain_size)?)
			}
			(true, 0) => DynamicDispatchNTT::SingleThreadedPrecompute(
				SingleThreadedNTT::new(log_domain_size)?.precompute_twiddles(),
			),
			(false, _) => DynamicDispatchNTT::MultiThreaded(
				SingleThreadedNTT::new(log_domain_size)?
					.multithreaded_with_max_threads(log_threads),
			),
			(true, _) => DynamicDispatchNTT::MultiThreadedPrecompute(
				SingleThreadedNTT::new(log_domain_size)?
					.precompute_twiddles()
					.multithreaded_with_max_threads(log_threads),
			),
		};

		Ok(result)
	}
}

impl<F, P> AdditiveNTT<P> for DynamicDispatchNTT<F>
where
	F: BinaryField,
	P: PackedField<Scalar = F>,
{
	fn log_domain_size(&self) -> usize {
		match self {
			DynamicDispatchNTT::SingleThreaded(ntt) => ntt.log_domain_size(),
			DynamicDispatchNTT::SingleThreadedPrecompute(ntt) => ntt.log_domain_size(),
			DynamicDispatchNTT::MultiThreaded(ntt) => ntt.log_domain_size(),
			DynamicDispatchNTT::MultiThreadedPrecompute(ntt) => ntt.log_domain_size(),
		}
	}

	fn get_subspace_eval(&self, i: usize, j: usize) -> F {
		match self {
			DynamicDispatchNTT::SingleThreaded(ntt) => ntt.get_subspace_eval(i, j),
			DynamicDispatchNTT::SingleThreadedPrecompute(ntt) => ntt.get_subspace_eval(i, j),
			DynamicDispatchNTT::MultiThreaded(ntt) => ntt.get_subspace_eval(i, j),
			DynamicDispatchNTT::MultiThreadedPrecompute(ntt) => ntt.get_subspace_eval(i, j),
		}
	}

	fn forward_transform(
		&self,
		data: &mut [P],
		coset: u32,
		log_batch_size: usize,
	) -> Result<(), crate::error::Error> {
		match self {
			DynamicDispatchNTT::SingleThreaded(ntt) => {
				ntt.forward_transform(data, coset, log_batch_size)
			}
			DynamicDispatchNTT::SingleThreadedPrecompute(ntt) => {
				ntt.forward_transform(data, coset, log_batch_size)
			}
			DynamicDispatchNTT::MultiThreaded(ntt) => {
				ntt.forward_transform(data, coset, log_batch_size)
			}
			DynamicDispatchNTT::MultiThreadedPrecompute(ntt) => {
				ntt.forward_transform(data, coset, log_batch_size)
			}
		}
	}

	fn inverse_transform(
		&self,
		data: &mut [P],
		coset: u32,
		log_batch_size: usize,
	) -> Result<(), crate::error::Error> {
		match self {
			DynamicDispatchNTT::SingleThreaded(ntt) => {
				ntt.inverse_transform(data, coset, log_batch_size)
			}
			DynamicDispatchNTT::SingleThreadedPrecompute(ntt) => {
				ntt.inverse_transform(data, coset, log_batch_size)
			}
			DynamicDispatchNTT::MultiThreaded(ntt) => {
				ntt.inverse_transform(data, coset, log_batch_size)
			}
			DynamicDispatchNTT::MultiThreadedPrecompute(ntt) => {
				ntt.inverse_transform(data, coset, log_batch_size)
			}
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use binius_field::BinaryField8b;

	#[test]
	fn test_creation() {
		fn make_ntt(options: NTTOptions) -> DynamicDispatchNTT<BinaryField8b> {
			DynamicDispatchNTT::<BinaryField8b>::new(6, options).unwrap()
		}

		let ntt = make_ntt(NTTOptions {
			precompute_twiddles: false,
			thread_settings: ThreadingSettings::SingleThreaded,
		});
		assert!(matches!(ntt, DynamicDispatchNTT::SingleThreaded(_)));

		let ntt = make_ntt(NTTOptions {
			precompute_twiddles: true,
			thread_settings: ThreadingSettings::SingleThreaded,
		});
		assert!(matches!(ntt, DynamicDispatchNTT::SingleThreadedPrecompute(_)));

		let multithreaded = get_log_max_threads() > 0;
		let ntt = make_ntt(NTTOptions {
			precompute_twiddles: false,
			thread_settings: ThreadingSettings::MultithreadedDefault,
		});
		if multithreaded {
			assert!(matches!(ntt, DynamicDispatchNTT::MultiThreaded(_)));
		} else {
			assert!(matches!(ntt, DynamicDispatchNTT::SingleThreaded(_)));
		}

		let ntt = make_ntt(NTTOptions {
			precompute_twiddles: true,
			thread_settings: ThreadingSettings::MultithreadedDefault,
		});
		if multithreaded {
			assert!(matches!(ntt, DynamicDispatchNTT::MultiThreadedPrecompute(_)));
		} else {
			assert!(matches!(ntt, DynamicDispatchNTT::SingleThreadedPrecompute(_)));
		}

		let ntt = make_ntt(NTTOptions {
			precompute_twiddles: false,
			thread_settings: ThreadingSettings::ExplicitThreadsCount { log_threads: 2 },
		});
		assert!(matches!(ntt, DynamicDispatchNTT::MultiThreaded(_)));

		let ntt = make_ntt(NTTOptions {
			precompute_twiddles: true,
			thread_settings: ThreadingSettings::ExplicitThreadsCount { log_threads: 0 },
		});
		assert!(matches!(ntt, DynamicDispatchNTT::SingleThreadedPrecompute(_)));

		let ntt = make_ntt(NTTOptions {
			precompute_twiddles: false,
			thread_settings: ThreadingSettings::ExplicitThreadsCount { log_threads: 0 },
		});
		assert!(matches!(ntt, DynamicDispatchNTT::SingleThreaded(_)));
	}
}
