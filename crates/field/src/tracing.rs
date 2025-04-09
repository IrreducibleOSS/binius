// Copyright 2024-2025 Irreducible Inc.
use cfg_if::cfg_if;

cfg_if! {
	if #[cfg(feature = "trace_multiplications")] {
		use std::cell::Cell;

		thread_local! {
			static IS_IN_MULT_FUNCTION: Cell<bool> = Cell::new(false);
		}

		/// This guard is used to remove the duplicated items when one field multiplication
		/// re-uses several other ones.
		pub(crate) struct TraceGuard(bool);

		impl TraceGuard {
			pub fn new(lhs: &'static str, rhs: &'static str) -> Self {
				let val = IS_IN_MULT_FUNCTION.with(|v| {
					if !v.get() {
						v.set(true);
						tracing::event!(name: "mul", tracing::Level::TRACE, {lhs, rhs});

						true
					} else {
						false
					}
				});
				Self(val)
			}
		}

		impl Drop for TraceGuard {
			fn drop(&mut self) {
				if self.0 {
					IS_IN_MULT_FUNCTION.with(|v| {
						v.set(false);
					});
				};
			}
		}

		macro_rules! trace_multiplication {
			($name: ty) => {
				let _guard = $crate::tracing::TraceGuard::new(stringify!($name), stringify!($name));
			};
			($lhs: ty, $rhs: ty) => {
				let _guard = $crate::tracing::TraceGuard::new(stringify!($lhs), stringify!($rhs));
			};
		}

	} else {
		macro_rules! trace_multiplication {
			($name: ty) => {};
			($lhs: ty, $rhs: ty) => {};
		}
	}
}

pub(crate) use trace_multiplication;
