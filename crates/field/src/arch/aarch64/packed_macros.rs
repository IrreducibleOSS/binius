// Copyright 2024-2025 Irreducible Inc.

macro_rules! maybe_impl_broadcast {
	(M128, $scalar:ident) => {};
}

macro_rules! maybe_impl_tower_constants {
	($scalar:ident, $underlier:ty, _) => {};
	($scalar:ident, M128, $alpha_idx:tt) => {
		impl_tower_constants!($scalar, M128, { M128(alphas!(u128, $alpha_idx)) });
	};
}

macro_rules! impl_strategy {
	($impl_macro:ident $name:ident, (None)) => {};
	($impl_macro:ident $name:ident, ($strategy:ty)) => {
		$impl_macro!($name @ $strategy);
	};
}

macro_rules! impl_transformation {
	($name:ident, ($strategy:ty)) => {
		impl_transformation_with_strategy!($name, $strategy);
	};
}

pub(crate) use impl_strategy;
pub(crate) use impl_transformation;
pub(crate) use maybe_impl_broadcast;
pub(crate) use maybe_impl_tower_constants;
