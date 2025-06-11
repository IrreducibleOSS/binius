// Copyright 2024-2025 Irreducible Inc.

//! Traits for working with field towers.

use std::mem::{MaybeUninit, transmute};

use binius_maybe_rayon::{iter::ParallelIterator, slice::ParallelSlice};
use bytemuck::Pod;
use trait_set::trait_set;

use crate::{
	AESTowerField8b, AESTowerField16b, AESTowerField32b, AESTowerField64b, AESTowerField128b,
	BinaryField1b, BinaryField8b, BinaryField16b, BinaryField32b, BinaryField64b, BinaryField128b,
	BinaryField128bPolyval, ByteSlicedUnderlier, ExtensionField, PackedExtension, PackedField,
	TowerField,
	as_packed_field::{PackScalar, PackedType},
	linear_transformation::{PackedTransformationFactory, Transformation},
	make_binary_to_aes_packed_transformer,
	polyval::{
		AES_TO_POLYVAL_TRANSFORMATION, BINARY_TO_POLYVAL_TRANSFORMATION,
		POLYVAL_TO_AES_TRANSFORMARION, POLYVAL_TO_BINARY_TRANSFORMATION,
	},
	tower_levels::TowerLevel16,
	underlier::{NumCast, UnderlierType, UnderlierWithBitOps},
};

/// A trait that groups a family of related [`TowerField`]s as associated types.
pub trait TowerFamily: Sized + 'static + Sync + Send {
	type B1: TowerField + TryFrom<Self::B128>;
	type B8: TowerField + TryFrom<Self::B128> + ExtensionField<Self::B1>;
	type B16: TowerField + TryFrom<Self::B128> + ExtensionField<Self::B1> + ExtensionField<Self::B8>;
	type B32: TowerField
		+ TryFrom<Self::B128>
		+ ExtensionField<Self::B1>
		+ ExtensionField<Self::B8>
		+ ExtensionField<Self::B16>;
	type B64: TowerField
		+ TryFrom<Self::B128>
		+ ExtensionField<Self::B1>
		+ ExtensionField<Self::B8>
		+ ExtensionField<Self::B16>
		+ ExtensionField<Self::B32>;
	type B128: TowerField
		+ ExtensionField<Self::B1>
		+ ExtensionField<Self::B8>
		+ ExtensionField<Self::B16>
		+ ExtensionField<Self::B32>
		+ ExtensionField<Self::B64>;
}

pub trait ProverTowerFamily: TowerFamily {
	type FastB128: TowerField + From<Self::B128> + Into<Self::B128> + ExtensionField<Self::B1>;

	fn packed_transformation_to_fast<Top, FastTop>() -> impl Transformation<Top, FastTop>
	where
		Top: PackedTop<Self> + PackedTransformationFactory<FastTop>,
		FastTop: PackedField<Scalar = Self::FastB128>;

	fn packed_transformation_from_fast<FastTop, Top>() -> impl Transformation<FastTop, Top>
	where
		FastTop: PackedTransformationFactory<Top>,
		Top: PackedField<Scalar = Self::B128>;

	fn transform_1b_to_bytesliced<U>(
		evals: Vec<PackedType<U, Self::B1>>,
	) -> Vec<PackedType<ByteSlicedUnderlier<U, 16>, BinaryField1b>>
	where
		U: TowerUnderlier<Self> + UnderlierWithBitOps + From<u8> + Pod,
		ByteSlicedUnderlier<U, 16>: PackScalar<BinaryField1b, Packed: Pod>,
		u8: NumCast<U>;

	fn transform_32b_to_bytesliced<U>(
		evals: Vec<PackedType<U, Self::B32>>,
	) -> Vec<PackedType<ByteSlicedUnderlier<U, 16>, AESTowerField32b>>
	where
		U: ProverTowerUnderlier<Self> + UnderlierWithBitOps + From<u8> + Pod,
		ByteSlicedUnderlier<U, 16>: PackScalar<AESTowerField32b, Packed: Pod>,
		u8: NumCast<U>,
		PackedType<U, BinaryField8b>: PackedTransformationFactory<PackedType<U, AESTowerField8b>>;

	fn transform_128b_to_bytesliced<U>(
		evals: Vec<PackedType<U, Self::B128>>,
	) -> Vec<PackedType<ByteSlicedUnderlier<U, 16>, AESTowerField128b>>
	where
		U: ProverTowerUnderlier<Self> + UnderlierWithBitOps + From<u8> + Pod,
		ByteSlicedUnderlier<U, 16>: PackScalar<AESTowerField128b, Packed: Pod>,
		u8: NumCast<U>,
		PackedType<U, BinaryField8b>: PackedTransformationFactory<PackedType<U, AESTowerField8b>>;
}

/// The canonical Fan-Paar tower family.
#[derive(Debug, Default)]
pub struct CanonicalTowerFamily;

impl TowerFamily for CanonicalTowerFamily {
	type B1 = BinaryField1b;
	type B8 = BinaryField8b;
	type B16 = BinaryField16b;
	type B32 = BinaryField32b;
	type B64 = BinaryField64b;
	type B128 = BinaryField128b;
}

impl ProverTowerFamily for CanonicalTowerFamily {
	type FastB128 = BinaryField128bPolyval;

	fn packed_transformation_to_fast<Top, FastTop>() -> impl Transformation<Top, FastTop>
	where
		Top: PackedTop<Self> + PackedTransformationFactory<FastTop>,
		FastTop: PackedField<Scalar = Self::FastB128>,
	{
		Top::make_packed_transformation(BINARY_TO_POLYVAL_TRANSFORMATION)
	}

	fn packed_transformation_from_fast<FastTop, Top>() -> impl Transformation<FastTop, Top>
	where
		FastTop: PackedTransformationFactory<Top>,
		Top: PackedField<Scalar = Self::B128>,
	{
		FastTop::make_packed_transformation(POLYVAL_TO_BINARY_TRANSFORMATION)
	}

	fn transform_1b_to_bytesliced<U>(
		evals: Vec<PackedType<U, Self::B1>>,
	) -> Vec<PackedType<ByteSlicedUnderlier<U, 16>, BinaryField1b>>
	where
		U: TowerUnderlier<Self> + UnderlierWithBitOps + From<u8> + Pod,
		ByteSlicedUnderlier<U, 16>: PackScalar<BinaryField1b, Packed: Pod>,
		u8: NumCast<U>,
	{
		const BYTES_COUNT: usize = 16;
		evals
			.par_chunks(BYTES_COUNT)
			.map(|values| {
				let mut aes = [MaybeUninit::<PackedType<U, BinaryField1b>>::uninit(); BYTES_COUNT];
				for (x0_aes, x0) in aes.iter_mut().zip(values.iter()) {
					_ = *x0_aes.write(*x0);
				}

				let aes = unsafe {
					transmute::<
						&mut [MaybeUninit<PackedType<U, BinaryField1b>>; BYTES_COUNT],
						&mut [U; BYTES_COUNT],
					>(&mut aes)
				};

				U::transpose_bytes_to_byte_sliced::<TowerLevel16>(aes);

				let x0_bytes_sliced = bytemuck::must_cast_mut::<
					_,
					PackedType<ByteSlicedUnderlier<U, 16>, BinaryField1b>,
				>(aes);
				x0_bytes_sliced.to_owned()
			})
			.collect::<Vec<_>>()
	}

	fn transform_128b_to_bytesliced<U>(
		evals: Vec<PackedType<U, Self::B128>>,
	) -> Vec<PackedType<ByteSlicedUnderlier<U, 16>, AESTowerField128b>>
	where
		U: ProverTowerUnderlier<Self> + UnderlierWithBitOps + From<u8> + Pod,
		ByteSlicedUnderlier<U, 16>: PackScalar<AESTowerField128b, Packed: Pod>,
		u8: NumCast<U>,
		PackedType<U, BinaryField8b>: PackedTransformationFactory<PackedType<U, AESTowerField8b>>,
	{
		const BYTES_COUNT: usize = 16;

		let fwd_transform = make_binary_to_aes_packed_transformer::<
			PackedType<U, BinaryField128b>,
			PackedType<U, AESTowerField128b>,
		>();

		evals
			.par_chunks_exact(BYTES_COUNT)
			.map(|values| {
				let mut aes =
					[MaybeUninit::<PackedType<U, AESTowerField128b>>::uninit(); BYTES_COUNT];
				for (x0_aes, x0) in aes.iter_mut().zip(values.iter()) {
					_ = *x0_aes.write(fwd_transform.transform(x0));
				}

				let aes = unsafe {
					transmute::<
						&mut [MaybeUninit<PackedType<U, AESTowerField128b>>; BYTES_COUNT],
						&mut [U; BYTES_COUNT],
					>(&mut aes)
				};

				U::transpose_bytes_to_byte_sliced::<TowerLevel16>(aes);

				let x0_bytes_sliced = bytemuck::must_cast_mut::<
					_,
					PackedType<ByteSlicedUnderlier<U, 16>, AESTowerField128b>,
				>(aes);
				x0_bytes_sliced.to_owned()
			})
			.collect::<Vec<_>>()
	}

	fn transform_32b_to_bytesliced<U>(
		evals: Vec<PackedType<U, Self::B32>>,
	) -> Vec<PackedType<ByteSlicedUnderlier<U, 16>, AESTowerField32b>>
	where
		U: ProverTowerUnderlier<Self> + UnderlierWithBitOps + From<u8> + Pod,
		ByteSlicedUnderlier<U, 16>: PackScalar<AESTowerField32b, Packed: Pod>,
		u8: NumCast<U>,
		PackedType<U, BinaryField8b>: PackedTransformationFactory<PackedType<U, AESTowerField8b>>,
	{
		const BYTES_COUNT: usize = 16;

		let fwd_transform = make_binary_to_aes_packed_transformer::<
			PackedType<U, BinaryField32b>,
			PackedType<U, AESTowerField32b>,
		>();

		evals
			.par_chunks_exact(BYTES_COUNT)
			.map(|values| {
				let mut aes =
					[MaybeUninit::<PackedType<U, AESTowerField32b>>::uninit(); BYTES_COUNT];
				for (x0_aes, x0) in aes.iter_mut().zip(values.iter()) {
					_ = *x0_aes.write(fwd_transform.transform(x0));
				}

				let aes = unsafe {
					transmute::<
						&mut [MaybeUninit<PackedType<U, AESTowerField32b>>; BYTES_COUNT],
						&mut [U; BYTES_COUNT],
					>(&mut aes)
				};

				U::transpose_bytes_to_byte_sliced::<TowerLevel16>(aes);

				let x0_bytes_sliced = bytemuck::must_cast_mut::<
					_,
					PackedType<ByteSlicedUnderlier<U, 16>, AESTowerField32b>,
				>(aes);
				x0_bytes_sliced.to_owned()
			})
			.collect::<Vec<_>>()
	}
}

/// The tower defined by Fan-Paar extensions built on top of the Rijndael field.
#[derive(Debug)]
pub struct AESTowerFamily;

impl TowerFamily for AESTowerFamily {
	type B1 = BinaryField1b;
	type B8 = AESTowerField8b;
	type B16 = AESTowerField16b;
	type B32 = AESTowerField32b;
	type B64 = AESTowerField64b;
	type B128 = AESTowerField128b;
}

impl ProverTowerFamily for AESTowerFamily {
	type FastB128 = BinaryField128bPolyval;

	fn packed_transformation_to_fast<Top, FastTop>() -> impl Transformation<Top, FastTop>
	where
		Top: PackedTop<Self> + PackedTransformationFactory<FastTop>,
		FastTop: PackedField<Scalar = Self::FastB128>,
	{
		Top::make_packed_transformation(AES_TO_POLYVAL_TRANSFORMATION)
	}

	fn packed_transformation_from_fast<FastTop, Top>() -> impl Transformation<FastTop, Top>
	where
		FastTop: PackedTransformationFactory<Top>,
		Top: PackedField<Scalar = Self::B128>,
	{
		FastTop::make_packed_transformation(POLYVAL_TO_AES_TRANSFORMARION)
	}

	fn transform_1b_to_bytesliced<U>(
		evals: Vec<PackedType<U, Self::B1>>,
	) -> Vec<PackedType<ByteSlicedUnderlier<U, 16>, BinaryField1b>>
	where
		U: TowerUnderlier<Self> + UnderlierWithBitOps + From<u8> + Pod,
		ByteSlicedUnderlier<U, 16>: PackScalar<BinaryField1b, Packed: Pod>,
		u8: NumCast<U>,
	{
		const BYTES_COUNT: usize = 16;
		evals
			.par_chunks_exact(BYTES_COUNT)
			.map(|values| {
				let mut aes = [MaybeUninit::<PackedType<U, BinaryField1b>>::uninit(); BYTES_COUNT];
				for (x0_aes, x0) in aes.iter_mut().zip(values.iter()) {
					_ = *x0_aes.write(*x0);
				}

				let aes = unsafe {
					transmute::<
						&mut [MaybeUninit<PackedType<U, BinaryField1b>>; BYTES_COUNT],
						&mut [U; BYTES_COUNT],
					>(&mut aes)
				};

				U::transpose_bytes_to_byte_sliced::<TowerLevel16>(aes);

				let x0_bytes_sliced = bytemuck::must_cast_mut::<
					_,
					PackedType<ByteSlicedUnderlier<U, 16>, BinaryField1b>,
				>(aes);
				x0_bytes_sliced.to_owned()
			})
			.collect::<Vec<_>>()
	}

	fn transform_128b_to_bytesliced<U>(
		evals: Vec<PackedType<U, Self::B128>>,
	) -> Vec<PackedType<ByteSlicedUnderlier<U, 16>, AESTowerField128b>>
	where
		U: TowerUnderlier<Self> + UnderlierWithBitOps + From<u8> + Pod,
		ByteSlicedUnderlier<U, 16>: PackScalar<AESTowerField128b, Packed: Pod>,
		u8: NumCast<U>,
	{
		const BYTES_COUNT: usize = 16;

		evals
			.par_chunks_exact(BYTES_COUNT)
			.map(|values| {
				let mut aes =
					[MaybeUninit::<PackedType<U, AESTowerField128b>>::uninit(); BYTES_COUNT];
				for (x0_aes, x0) in aes.iter_mut().zip(values.iter()) {
					_ = *x0_aes.write(*x0);
				}

				let aes = unsafe {
					transmute::<
						&mut [MaybeUninit<PackedType<U, AESTowerField128b>>; BYTES_COUNT],
						&mut [U; BYTES_COUNT],
					>(&mut aes)
				};

				U::transpose_bytes_to_byte_sliced::<TowerLevel16>(aes);

				let x0_bytes_sliced = bytemuck::must_cast_mut::<
					_,
					PackedType<ByteSlicedUnderlier<U, 16>, AESTowerField128b>,
				>(aes);
				x0_bytes_sliced.to_owned()
			})
			.collect::<Vec<_>>()
	}

	fn transform_32b_to_bytesliced<U>(
		evals: Vec<PackedType<U, Self::B32>>,
	) -> Vec<PackedType<ByteSlicedUnderlier<U, 16>, AESTowerField32b>>
	where
		U: ProverTowerUnderlier<Self> + UnderlierWithBitOps + From<u8> + Pod,
		ByteSlicedUnderlier<U, 16>: PackScalar<AESTowerField32b, Packed: Pod>,
		u8: NumCast<U>,
		PackedType<U, BinaryField8b>: PackedTransformationFactory<PackedType<U, AESTowerField8b>>,
	{
		todo!()
	}
}

trait_set! {
	/// An underlier with associated packed types for fields in a tower.
	pub trait TowerUnderlier<Tower: TowerFamily> =
		UnderlierType
		+ PackScalar<Tower::B1>
		+ PackScalar<Tower::B8>
		+ PackScalar<Tower::B16>
		+ PackScalar<Tower::B32>
		+ PackScalar<Tower::B64>
		+ PackScalar<Tower::B128> ;

	pub trait ProverTowerUnderlier<Tower: ProverTowerFamily> =
		TowerUnderlier<Tower> + PackScalar<Tower::FastB128> + PackScalar<AESTowerField128b> +PackScalar<AESTowerField32b> + PackScalar<AESTowerField8b> + PackScalar<BinaryField8b>+ UnderlierWithBitOps + From<u8> + Pod;

	/// A packed field type that is the top packed field in a tower.
	pub trait PackedTop<Tower: TowerFamily> =
		PackedField<Scalar=Tower::B128>
		+ PackedExtension<Tower::B1>
		+ PackedExtension<Tower::B8>
		+ PackedExtension<Tower::B16>
		+ PackedExtension<Tower::B32>
		+ PackedExtension<Tower::B64>
		+ PackedExtension<Tower::B128>;
}
