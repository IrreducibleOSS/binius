// Copyright 2024-2025 Irreducible Inc.

use binius_compute::{alloc::BumpAllocator, cpu::CpuLayer};
use binius_compute_test_utils::piop::{ComputeLayerInfo, commit_prove_verify_generic};
use binius_field::{
	AESTowerField8b, AESTowerField16b, BinaryField, BinaryField8b, BinaryField16b,
	ByteSlicedAES16x128b, PackedBinaryField2x128b, PackedExtension, PackedField,
	tower::{AESTowerFamily, CanonicalTowerFamily, TowerFamily},
};
use bytemuck::zeroed_vec;

use crate::piop::CommitMeta;

enum ComputeLayerType {
	None,
	Cpu,
}

fn commit_prove_verify<T, FDomain, FEncode, P>(
	commit_meta: Vec<usize>,
	n_transparents: usize,
	log_inv_rate: usize,
	compute_layer: ComputeLayerType,
) where
	T: TowerFamily,
	FDomain: BinaryField,
	FEncode: BinaryField,
	P: PackedField<Scalar = T::B128>
		+ PackedExtension<FDomain>
		+ PackedExtension<FEncode>
		+ PackedExtension<T::B128, PackedSubfield = P>,
{
	match compute_layer {
		ComputeLayerType::None => {
			commit_prove_verify_generic::<T, FDomain, FEncode, P, CpuLayer<T>>(
				commit_meta,
				n_transparents,
				log_inv_rate,
				None,
			);
		}
		ComputeLayerType::Cpu => {
			let compute_layer = CpuLayer::<T>::default();
			let mut slice = zeroed_vec(1 << 16);
			let dev_allocator = BumpAllocator::new(&mut slice[..]);
			let mut slice = zeroed_vec(1 << 16);
			let host_allocator = BumpAllocator::new(&mut slice[..]);
			commit_prove_verify_generic::<T, FDomain, FEncode, P, CpuLayer<T>>(
				commit_meta,
				n_transparents,
				log_inv_rate,
				Some(ComputeLayerInfo {
					compute_layer: &compute_layer,
					host_allocator: &host_allocator,
					dev_allocator: &dev_allocator,
				}),
			);
		}
	};
}

#[test]
fn test_commit_meta_total_vars() {
	let commit_meta = CommitMeta::with_vars([4, 4, 6, 7]);
	assert_eq!(commit_meta.total_vars(), 8);

	let commit_meta = CommitMeta::with_vars([4, 4, 6, 6, 6, 7]);
	assert_eq!(commit_meta.total_vars(), 9);
}

#[test]
fn test_with_one_poly() {
	let commit_meta = vec![4];
	let n_transparents = 1;
	let log_inv_rate = 1;

	commit_prove_verify::<
		CanonicalTowerFamily,
		BinaryField8b,
		BinaryField16b,
		PackedBinaryField2x128b,
	>(commit_meta.clone(), n_transparents, log_inv_rate, ComputeLayerType::None);
	commit_prove_verify::<
		CanonicalTowerFamily,
		BinaryField8b,
		BinaryField16b,
		PackedBinaryField2x128b,
	>(commit_meta, n_transparents, log_inv_rate, ComputeLayerType::Cpu);
}

#[test]
fn test_without_opening_claims() {
	let commit_meta = vec![4, 4, 6, 7];
	let n_transparents = 0;
	let log_inv_rate = 1;

	commit_prove_verify::<
		CanonicalTowerFamily,
		BinaryField8b,
		BinaryField16b,
		PackedBinaryField2x128b,
	>(commit_meta.clone(), n_transparents, log_inv_rate, ComputeLayerType::None);
	commit_prove_verify::<
		CanonicalTowerFamily,
		BinaryField8b,
		BinaryField16b,
		PackedBinaryField2x128b,
	>(commit_meta, n_transparents, log_inv_rate, ComputeLayerType::Cpu);
}

#[test]
fn test_with_one_n_vars() {
	let commit_meta = vec![4, 4];
	let n_transparents = 1;
	let log_inv_rate = 1;

	commit_prove_verify::<
		CanonicalTowerFamily,
		BinaryField8b,
		BinaryField16b,
		PackedBinaryField2x128b,
	>(commit_meta.clone(), n_transparents, log_inv_rate, ComputeLayerType::None);
	commit_prove_verify::<
		CanonicalTowerFamily,
		BinaryField8b,
		BinaryField16b,
		PackedBinaryField2x128b,
	>(commit_meta, n_transparents, log_inv_rate, ComputeLayerType::Cpu);
}

#[test]
fn test_commit_prove_verify_extreme_rate() {
	let commit_meta = vec![3, 3, 5, 6];
	let n_transparents = 2;
	let log_inv_rate = 8;

	commit_prove_verify::<
		CanonicalTowerFamily,
		BinaryField8b,
		BinaryField16b,
		PackedBinaryField2x128b,
	>(commit_meta.clone(), n_transparents, log_inv_rate, ComputeLayerType::None);
	commit_prove_verify::<
		CanonicalTowerFamily,
		BinaryField8b,
		BinaryField16b,
		PackedBinaryField2x128b,
	>(commit_meta, n_transparents, log_inv_rate, ComputeLayerType::Cpu);
}

#[test]
fn test_commit_prove_verify_small() {
	let commit_meta = vec![4, 4, 6, 7];
	let n_transparents = 2;
	let log_inv_rate = 1;

	commit_prove_verify::<
		CanonicalTowerFamily,
		BinaryField8b,
		BinaryField16b,
		PackedBinaryField2x128b,
	>(commit_meta.clone(), n_transparents, log_inv_rate, ComputeLayerType::None);
	commit_prove_verify::<
		CanonicalTowerFamily,
		BinaryField8b,
		BinaryField16b,
		PackedBinaryField2x128b,
	>(commit_meta, n_transparents, log_inv_rate, ComputeLayerType::Cpu);
}

#[test]
fn test_commit_prove_verify() {
	let commit_meta = vec![6, 6, 8, 9];
	let n_transparents = 2;
	let log_inv_rate = 1;

	commit_prove_verify::<
		CanonicalTowerFamily,
		BinaryField8b,
		BinaryField16b,
		PackedBinaryField2x128b,
	>(commit_meta.clone(), n_transparents, log_inv_rate, ComputeLayerType::None);
	commit_prove_verify::<
		CanonicalTowerFamily,
		BinaryField8b,
		BinaryField16b,
		PackedBinaryField2x128b,
	>(commit_meta, n_transparents, log_inv_rate, ComputeLayerType::Cpu);
}

#[test]
fn test_commit_prove_verify_byte_sliced() {
	let commit_meta = vec![11, 12, 13, 14];
	let n_transparents = 2;
	let log_inv_rate = 1;

	commit_prove_verify::<AESTowerFamily, AESTowerField8b, AESTowerField16b, ByteSlicedAES16x128b>(
		commit_meta,
		n_transparents,
		log_inv_rate,
		ComputeLayerType::None,
	);

	// ByteSliced is not supported on custom compute layers yet
}
