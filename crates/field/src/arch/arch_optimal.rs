// Copyright 2024 Ulvetanna Inc.

use crate::{aes_field::*, binary_field::*, Field, PackedField};
use bytemuck::Pod;
use cfg_if::cfg_if;

pub trait ArchOptimal: Field {
	type OptimalThroughputPacked: PackedField<Scalar = Self> + Pod;
}

macro_rules! set_optimal_packed_types {
	($field:ty, $optimal_throughput_packed:ty) => {
		impl ArchOptimal for $field {
			type OptimalThroughputPacked = $optimal_throughput_packed;
		}
	};
}

cfg_if! {
	if #[cfg(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "gfni"))] {
		use crate::arch::packed_512::*;
		use crate::arch::packed_aes_512::*;
		use crate::polyval::BinaryField128bPolyval;
		use crate::arch::packed_polyval_512::PackedBinaryPolyval4x128b;

		pub const OPTIMAL_ALIGNMENT: usize = 512;

		set_optimal_packed_types!(BinaryField1b, PackedBinaryField512x1b);
		set_optimal_packed_types!(BinaryField2b, PackedBinaryField256x2b);
		set_optimal_packed_types!(BinaryField4b, PackedBinaryField128x4b);
		set_optimal_packed_types!(BinaryField8b, PackedBinaryField64x8b);
		set_optimal_packed_types!(BinaryField16b, PackedBinaryField32x16b);
		set_optimal_packed_types!(BinaryField32b, PackedBinaryField16x32b);
		set_optimal_packed_types!(BinaryField64b, PackedBinaryField8x64b);
		set_optimal_packed_types!(BinaryField128b, PackedBinaryField4x128b);

		set_optimal_packed_types!(AESTowerField8b, PackedAESBinaryField64x8b);
		set_optimal_packed_types!(AESTowerField16b, PackedAESBinaryField32x16b);
		set_optimal_packed_types!(AESTowerField32b, PackedAESBinaryField16x32b);
		set_optimal_packed_types!(AESTowerField64b, PackedAESBinaryField8x64b);
		set_optimal_packed_types!(AESTowerField128b, PackedAESBinaryField4x128b);

		set_optimal_packed_types!(BinaryField128bPolyval, PackedBinaryPolyval4x128b);
	} else if #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "gfni"))] {
		use crate::arch::packed_256::*;
		use crate::arch::packed_aes_256::*;
		use crate::arch::packed_polyval_256::PackedBinaryPolyval2x128b;
		use crate::polyval::BinaryField128bPolyval;

		pub const OPTIMAL_ALIGNMENT: usize = 256;

		set_optimal_packed_types!(BinaryField1b, PackedBinaryField256x1b);
		set_optimal_packed_types!(BinaryField2b, PackedBinaryField128x2b);
		set_optimal_packed_types!(BinaryField4b, PackedBinaryField64x4b);
		set_optimal_packed_types!(BinaryField8b, PackedBinaryField32x8b);
		set_optimal_packed_types!(BinaryField16b, PackedBinaryField16x16b);
		set_optimal_packed_types!(BinaryField32b, PackedBinaryField8x32b);
		set_optimal_packed_types!(BinaryField64b, PackedBinaryField4x64b);
		set_optimal_packed_types!(BinaryField128b, PackedBinaryField2x128b);

		 set_optimal_packed_types!(AESTowerField16b, PackedAESBinaryField16x16b);
		set_optimal_packed_types!(AESTowerField32b, PackedAESBinaryField8x32b);
		set_optimal_packed_types!(AESTowerField64b, PackedAESBinaryField4x64b);
		set_optimal_packed_types!(AESTowerField128b, PackedAESBinaryField2x128b);

		set_optimal_packed_types!(BinaryField128bPolyval, PackedBinaryPolyval2x128b);
	} else if #[cfg(all(target_arch = "x86_64", target_feature = "sse2", target_feature = "gfni"))] {
		use crate::arch::packed_128::*;
		use crate::arch::packed_aes_128::*;
		use crate::polyval::BinaryField128bPolyval;

		pub const OPTIMAL_ALIGNMENT: usize = 128;

		set_optimal_packed_types!(BinaryField1b, PackedBinaryField256x1b);
		set_optimal_packed_types!(BinaryField2b, PackedBinaryField128x2b);
		set_optimal_packed_types!(BinaryField4b, PackedBinaryField64x4b);
		set_optimal_packed_types!(BinaryField8b, PackedBinaryField32x8b);
		set_optimal_packed_types!(BinaryField16b, PackedBinaryField16x16b);
		set_optimal_packed_types!(BinaryField32b, PackedBinaryField8x32b);
		set_optimal_packed_types!(BinaryField64b, PackedBinaryField4x64b);
		set_optimal_packed_types!(BinaryField128b, PackedBinaryField2x128b);

		 set_optimal_packed_types!(AESTowerField16b, PackedAESBinaryField16x16b);
		set_optimal_packed_types!(AESTowerField32b, PackedAESBinaryField8x32b);
		set_optimal_packed_types!(AESTowerField64b, PackedAESBinaryField4x64b);
		set_optimal_packed_types!(AESTowerField128b, PackedAESBinaryField2x128b);

		set_optimal_packed_types!(BinaryField128bPolyval, BinaryField128bPolyval);
	} else if #[cfg(all(target_arch = "aarch64", target_feature = "neon", target_feature = "aes"))] {
		use crate::arch::packed_128::*;
		use crate::arch::packed_aes_128::*;
		use crate::polyval::BinaryField128bPolyval;

		pub const OPTIMAL_ALIGNMENT: usize = 128;

		set_optimal_packed_types!(BinaryField1b, PackedBinaryField128x1b);
		set_optimal_packed_types!(BinaryField2b, PackedBinaryField64x2b);
		set_optimal_packed_types!(BinaryField4b, PackedBinaryField32x4b);
		set_optimal_packed_types!(BinaryField8b, PackedBinaryField16x8b);
		set_optimal_packed_types!(BinaryField16b, PackedBinaryField8x16b);
		set_optimal_packed_types!(BinaryField32b, PackedBinaryField4x32b);
		set_optimal_packed_types!(BinaryField64b, PackedBinaryField2x64b);
		set_optimal_packed_types!(BinaryField128b, PackedBinaryField1x128b);

		set_optimal_packed_types!(AESTowerField16b, PackedAESBinaryField8x16b);
		set_optimal_packed_types!(AESTowerField32b, PackedAESBinaryField4x32b);
		set_optimal_packed_types!(AESTowerField64b, PackedAESBinaryField2x64b);
		set_optimal_packed_types!(AESTowerField128b, PackedAESBinaryField1x128b);

		set_optimal_packed_types!(BinaryField128bPolyval, BinaryField128bPolyval);
	} else {
		use crate::arch::packed_64::*;
		use crate::arch::packed_128::*;
		use crate::arch::packed_aes_128::*;

		use crate::polyval::BinaryField128bPolyval;

		pub const OPTIMAL_ALIGNMENT: usize = 128;

		set_optimal_packed_types!(BinaryField1b, PackedBinaryField64x1b);
		set_optimal_packed_types!(BinaryField2b, PackedBinaryField32x2b);
		set_optimal_packed_types!(BinaryField4b, PackedBinaryField16x4b);
		set_optimal_packed_types!(BinaryField8b, PackedBinaryField8x8b);
		set_optimal_packed_types!(BinaryField16b, PackedBinaryField4x16b);
		set_optimal_packed_types!(BinaryField32b, PackedBinaryField2x32b);
		set_optimal_packed_types!(BinaryField64b, PackedBinaryField1x64b);
		set_optimal_packed_types!(BinaryField128b, PackedBinaryField1x128b);

		set_optimal_packed_types!(AESTowerField8b, AESTowerField8b);
		set_optimal_packed_types!(AESTowerField16b, PackedAESBinaryField8x16b);
		set_optimal_packed_types!(AESTowerField32b, PackedAESBinaryField4x32b);
		set_optimal_packed_types!(AESTowerField64b, PackedAESBinaryField2x64b);
		set_optimal_packed_types!(AESTowerField128b, PackedAESBinaryField1x128b);

		set_optimal_packed_types!(BinaryField128bPolyval, BinaryField128bPolyval);
	}
}
