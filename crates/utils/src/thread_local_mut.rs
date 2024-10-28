// Copyright 2024 Irreducible Inc.

use std::cell::UnsafeCell;
use thread_local::ThreadLocal;

/// Creates a "scratch space" within each thread with mutable access.
///
/// This is mainly meant to be used as an optimization to avoid unneccesary allocs/frees within rayon code.
/// You only pay for allocation of this scratch space once per thread.
///
/// Since the space is local to each thread you also don't have to worry about atomicity.
#[derive(Debug, Default)]
pub struct ThreadLocalMut<T: Send>(ThreadLocal<UnsafeCell<T>>);

impl<T: Send> ThreadLocalMut<T> {
	pub fn new() -> Self {
		Self(ThreadLocal::new())
	}

	#[inline]
	pub fn with_mut<U>(&self, init: impl FnOnce() -> T, run_scope: impl FnOnce(&mut T) -> U) -> U {
		let data = self.0.get_or(|| UnsafeCell::new(init()));
		run_scope(unsafe { data.get().as_mut().unwrap_unchecked() })
	}
}
