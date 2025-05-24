// Copyright 2024-2025 Irreducible Inc.

macro_rules! maybe_impl_broadcast {
	($underlier:ty, $scalar:ident) => {};
}

macro_rules! maybe_impl_tower_constants {
	($scalar:ident, $underlier:ty, _) => {};
	($scalar:ident, $underlier:ty, $alpha_idx:tt) => {
		impl_tower_constants!($scalar, $underlier, {
			<$underlier>::from_u128(alphas!(u128, $alpha_idx))
		});
	};
}

macro_rules! impl_strategy {
	($impl_macro:ident $name:ident, None) => {};
	($impl_macro:ident $name:ident, $delegate:ty, $fallback:ty) => {
		cfg_if! {
			if #[cfg(target_feature = "gfni")] {
				$impl_macro!($name @ $delegate);
			} else {
				$impl_macro!($name @ $fallback);
			}
		}
	};
	($impl_macro:ident $name:ident, $strategy:ty) => {
		$impl_macro!($name @ $strategy);
	};
}

macro_rules! impl_transformation {
	($name:ident, $num:literal, $fallback:ty) => {
		cfg_if::cfg_if! {
			if #[cfg(target_feature = "gfni")] {
				impl_transformation_with_gfni_nxn!($name, $num);
			} else {
				impl_transformation_with_strategy!($name, $fallback);
			}
		}
	};
	($name:ident, $delegate:ty, $fallback:ty) => {
		cfg_if::cfg_if! {
			if #[cfg(target_feature = "gfni")] {
				impl_transformation_with_strategy!($name, $delegate);
			} else {
				impl_transformation_with_strategy!($name, $fallback);
			}
		}
	};
	($name:ident, $strategy:ty) => {
		impl_transformation_with_strategy!($name, $strategy);
	};
}

pub(crate) use impl_strategy;
pub(crate) use impl_transformation;
pub(crate) use maybe_impl_broadcast;
pub(crate) use maybe_impl_tower_constants;
