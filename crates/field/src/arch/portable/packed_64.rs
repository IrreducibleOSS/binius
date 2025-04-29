// Copyright 2024-2025 Irreducible Inc.

use std::marker::PhantomData;

use binius_utils::{
	bytes::{Buf, BufMut},
	DeserializeBytes, SerializationError, SerializationMode, SerializeBytes,
};
use cfg_if::cfg_if;

use super::{
	packed::{
		impl_broadcast, impl_ops_for_zero_height, serialize_deserialize, PackedPrimitiveType,
	},
	packed_arithmetic::{alphas, impl_tower_constants},
};
use crate::{
	arch::{PackedStrategy, PairwiseRecursiveStrategy, PairwiseStrategy},
	arithmetic_traits::{
		impl_invert_with, impl_mul_alpha_with, impl_mul_with, impl_square_with,
		impl_transformation_with_strategy,
	},
	BinaryField16b, BinaryField1b, BinaryField2b, BinaryField32b, BinaryField4b, BinaryField64b,
	BinaryField8b,
};

// // Define 64 bit packed field types
// pub type PackedBinaryField64x1b = PackedPrimitiveType<u64, BinaryField1b>;
// pub type PackedBinaryField32x2b = PackedPrimitiveType<u64, BinaryField2b>;
// pub type PackedBinaryField16x4b = PackedPrimitiveType<u64, BinaryField4b>;
// pub type PackedBinaryField8x8b = PackedPrimitiveType<u64, BinaryField8b>;
// pub type PackedBinaryField4x16b = PackedPrimitiveType<u64, BinaryField16b>;
// pub type PackedBinaryField2x32b = PackedPrimitiveType<u64, BinaryField32b>;
// pub type PackedBinaryField1x64b = PackedPrimitiveType<u64, BinaryField64b>;

// serialize_deserialize!(PackedBinaryField64x1b);
// serialize_deserialize!(PackedBinaryField32x2b);
// serialize_deserialize!(PackedBinaryField16x4b);
// serialize_deserialize!(PackedBinaryField8x8b);
// serialize_deserialize!(PackedBinaryField4x16b);
// serialize_deserialize!(PackedBinaryField2x32b);
// serialize_deserialize!(PackedBinaryField1x64b);

// // Define broadcast
// impl_broadcast!(u64, BinaryField1b);
// impl_broadcast!(u64, BinaryField2b);
// impl_broadcast!(u64, BinaryField4b);
// impl_broadcast!(u64, BinaryField8b);
// impl_broadcast!(u64, BinaryField16b);
// impl_broadcast!(u64, BinaryField32b);
// impl_broadcast!(u64, BinaryField64b);

// // Define operations for height 0
// impl_ops_for_zero_height!(PackedBinaryField64x1b);

// // Define constants
// impl_tower_constants!(BinaryField1b, u64, { alphas!(u64, 0) });
// impl_tower_constants!(BinaryField2b, u64, { alphas!(u64, 1) });
// impl_tower_constants!(BinaryField4b, u64, { alphas!(u64, 2) });
// impl_tower_constants!(BinaryField8b, u64, { alphas!(u64, 3) });
// impl_tower_constants!(BinaryField16b, u64, { alphas!(u64, 4) });
// impl_tower_constants!(BinaryField32b, u64, { alphas!(u64, 5) });

// // Define multiplication
// impl_mul_with!(PackedBinaryField32x2b @ PackedStrategy);
// impl_mul_with!(PackedBinaryField16x4b @ PackedStrategy);
// cfg_if! {
// 	if #[cfg(all(target_arch = "x86_64", target_feature = "sse2", target_feature = "gfni", feature = "nightly_features"))] {
// 		impl_mul_with!(PackedBinaryField8x8b => crate::PackedBinaryField16x8b);
// 		impl_mul_with!(PackedBinaryField4x16b => crate::PackedBinaryField8x16b);
// 		impl_mul_with!(PackedBinaryField2x32b => crate::PackedBinaryField4x32b);
// 		impl_mul_with!(PackedBinaryField1x64b => crate::PackedBinaryField2x64b);
// 	} else {
// 		impl_mul_with!(PackedBinaryField8x8b @ crate::arch::PairwiseTableStrategy);
// 		impl_mul_with!(PackedBinaryField4x16b @ PairwiseRecursiveStrategy);
// 		impl_mul_with!(PackedBinaryField2x32b @ PairwiseRecursiveStrategy);
// 		impl_mul_with!(PackedBinaryField1x64b @ PairwiseRecursiveStrategy);
// 	}
// }

macro_rules! define_packed_field {
    ($(
        $name:ident, $bits:ident, $prim:ty, $alpha_idx:tt, ($strategy:ident $(, $args:tt)*)
    );* $(;)?) => {
        $(
            // Define packed field types
			pub type $name = PackedPrimitiveType<$prim, $bits>;

			// Define serialization and deserialization
			serialize_deserialize!($name);

			// Define broadcast
            impl_broadcast!($prim, $bits);

			// Define operations for height 0
            define_packed_field!(@maybe_impl_ops $name, $alpha_idx);

			// Define constants
			impl_tower_constants!($bits, $prim, { alphas!($prim, $alpha_idx) });

			// Define multiplication
            define_packed_field!(@impl_mul $name, $strategy $(, $args)*);
        )*
    };

    // Conditional ops impl if alpha_idx == 0
    (@maybe_impl_ops $name:ident, 0) => {
        impl_ops_for_zero_height!($name);
    };
    (@maybe_impl_ops $name:ident, $other_idx:tt) => {};

    // Multiplication logic 
    (@impl_mul $name:ident, None) => {};
    (@impl_mul $name:ident, $strategy:ty) => {
        impl_mul_with!($name @ $strategy);
    };
    (@impl_mul $name:ident, CfgSwitch, $delegate:ty, $fallback:ty) => {
        cfg_if! {
            if #[cfg(all(target_arch = "x86_64", target_feature = "sse2", target_feature = "gfni", feature = "nightly_features"))] {
                impl_mul_with!($name => $delegate);
            } else {
                impl_mul_with!($name @ $fallback);
            }
        }
    };
}

#[cfg(all(target_arch = "x86_64", target_feature = "sse2", target_feature = "gfni", feature = "nightly_features"))]
use crate::{PackedBinaryField16x8b, PackedBinaryField8x16b, PackedBinaryField4x32b, PackedBinaryField2x64b};
use crate::arch::PairwiseTableStrategy;

define_packed_field!(
    PackedBinaryField64x1b, BinaryField1b, u64, 0, (None);
    PackedBinaryField32x2b, BinaryField2b, u64, 1, (PackedStrategy);
    PackedBinaryField16x4b, BinaryField4b, u64, 2, (PackedStrategy);
    PackedBinaryField8x8b,  BinaryField8b,  u64, 3, (CfgSwitch, PackedBinaryField16x8b, PairwiseTableStrategy);
    PackedBinaryField4x16b, BinaryField16b, u64, 4, (CfgSwitch, PackedBinaryField8x16b, PairwiseRecursiveStrategy);
    PackedBinaryField2x32b, BinaryField32b, u64, 5, (CfgSwitch, PackedBinaryField4x32b, PairwiseRecursiveStrategy);
    PackedBinaryField1x64b, BinaryField64b, u64, 6, (CfgSwitch, PackedBinaryField2x64b, PairwiseRecursiveStrategy);
);

// Define square
impl_square_with!(PackedBinaryField32x2b @ PackedStrategy);
impl_square_with!(PackedBinaryField16x4b @ PackedStrategy);
cfg_if! {
	if #[cfg(all(target_arch = "x86_64", target_feature = "sse2", target_feature = "gfni", feature = "nightly_features"))] {
		impl_square_with!(PackedBinaryField8x8b => crate::PackedBinaryField16x8b);
		impl_square_with!(PackedBinaryField4x16b => crate::PackedBinaryField8x16b);
		impl_square_with!(PackedBinaryField2x32b => crate::PackedBinaryField4x32b);
		impl_square_with!(PackedBinaryField1x64b => crate::PackedBinaryField2x64b);
	} else {
		impl_square_with!(PackedBinaryField8x8b @ crate::arch::PairwiseTableStrategy);
		impl_square_with!(PackedBinaryField4x16b @ PairwiseStrategy);
		impl_square_with!(PackedBinaryField2x32b @ PairwiseRecursiveStrategy);
		impl_square_with!(PackedBinaryField1x64b @ crate::arch::HybridRecursiveStrategy);
	}
}

// Define invert
impl_invert_with!(PackedBinaryField32x2b @ PackedStrategy);
impl_invert_with!(PackedBinaryField16x4b @ PackedStrategy);
cfg_if! {
	if #[cfg(all(target_arch = "x86_64", target_feature = "sse2", target_feature = "gfni", feature = "nightly_features"))] {
		impl_invert_with!(PackedBinaryField8x8b => crate::PackedBinaryField16x8b);
		impl_invert_with!(PackedBinaryField4x16b => crate::PackedBinaryField8x16b);
		impl_invert_with!(PackedBinaryField2x32b => crate::PackedBinaryField4x32b);
		impl_invert_with!(PackedBinaryField1x64b => crate::PackedBinaryField2x64b);
	} else {
		impl_invert_with!(PackedBinaryField8x8b @ crate::arch::PairwiseTableStrategy);
		impl_invert_with!(PackedBinaryField4x16b @ PairwiseStrategy);
		impl_invert_with!(PackedBinaryField2x32b @ PairwiseStrategy);
		impl_invert_with!(PackedBinaryField1x64b @ PairwiseRecursiveStrategy);
	}
}

// Define multiply by alpha
impl_mul_alpha_with!(PackedBinaryField32x2b @ PackedStrategy);
impl_mul_alpha_with!(PackedBinaryField16x4b @ PackedStrategy);
impl_mul_alpha_with!(PackedBinaryField8x8b @ PackedStrategy);
impl_mul_alpha_with!(PackedBinaryField4x16b @ PackedStrategy);
impl_mul_alpha_with!(PackedBinaryField2x32b @ PackedStrategy);
impl_mul_alpha_with!(PackedBinaryField1x64b @ PairwiseRecursiveStrategy);

// Define linear transformations
impl_transformation_with_strategy!(PackedBinaryField64x1b, PackedStrategy);
impl_transformation_with_strategy!(PackedBinaryField32x2b, PackedStrategy);
impl_transformation_with_strategy!(PackedBinaryField16x4b, PackedStrategy);
impl_transformation_with_strategy!(PackedBinaryField8x8b, PackedStrategy);
impl_transformation_with_strategy!(PackedBinaryField4x16b, PackedStrategy);
impl_transformation_with_strategy!(PackedBinaryField2x32b, PackedStrategy);
impl_transformation_with_strategy!(PackedBinaryField1x64b, PairwiseStrategy);
