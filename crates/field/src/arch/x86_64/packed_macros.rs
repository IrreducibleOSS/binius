// Copyright 2024-2025 Irreducible Inc.

macro_rules! maybe_impl_broadcast {
	(M128, $scalar:ident) => {};
	(M256, $scalar:ident) => {};
	(M512, $scalar:ident) => {};
}

macro_rules! maybe_impl_tower_constants {
	($scalar:ident, $underlier:ty, _) => {};
	($scalar:ident, M128, $alpha_idx:tt) => {
		impl_tower_constants!($scalar, M128, { M128::from_u128(alphas!(u128, $alpha_idx)) });
	};
	($scalar:ident, M256, $alpha_idx:tt) => {
		impl_tower_constants!($scalar, M256, { M256::from_equal_u128s(M256, $alpha_idx) });
	};
	($scalar:ident, M512, $alpha_idx:tt) => {
		impl_tower_constants!($scalar, M512, { M512::from_equal_u128s(M512, $alpha_idx) });
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

pub(crate) use maybe_impl_broadcast;
pub(crate) use maybe_impl_tower_constants;
pub(crate) use impl_strategy;
pub(crate) use impl_transformation;

