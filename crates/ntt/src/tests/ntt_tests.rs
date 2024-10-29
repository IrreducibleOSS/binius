// Copyright 2024 Irreducible Inc.

use crate::{dynamic_dispatch::DynamicDispatchNTT, AdditiveNTT, SingleThreadedNTT};
use binius_field::{
	arch::{
		packed_16::{PackedBinaryField1x16b, PackedBinaryField2x8b},
		packed_32::PackedBinaryField2x16b,
		packed_64::{PackedBinaryField2x32b, PackedBinaryField4x16b},
		packed_8::PackedBinaryField1x8b,
	},
	underlier::{NumCast, WithUnderlier},
	AESTowerField8b, BinaryField, BinaryField8b, ExtensionField, PackedBinaryField16x32b,
	PackedBinaryField8x32b, PackedField, RepackedExtension,
};
use rand::{rngs::StdRng, SeedableRng};
use std::ops::Range;

/// Check that forward and inverse transformation of `ntt` on `data` is the same as forward and inverse transformation of `reference_ntt` on `data`
/// and that the result of the roundtrip is the same as the original data.
fn check_roundtrip_with_reference<P: PackedField>(
	reference_ntt: &impl AdditiveNTT<P>,
	ntt: &impl AdditiveNTT<P>,
	data: &mut [P],
	cosets: Range<u32>,
	log_batch_size: usize,
) {
	let data_copy = data.to_vec();
	let mut data_copy_2 = data.to_vec();

	for coset in cosets {
		ntt.forward_transform(data, coset, log_batch_size).unwrap();
		reference_ntt
			.forward_transform(&mut data_copy_2, coset, log_batch_size)
			.unwrap();

		assert_eq!(data, &data_copy_2);

		ntt.inverse_transform(data, coset, log_batch_size).unwrap();
		reference_ntt
			.inverse_transform(&mut data_copy_2, coset, log_batch_size)
			.unwrap();

		assert_eq!(data, &data_copy);
		assert_eq!(data, &data_copy_2);
	}
}

/// Check tht all NTTs have the same behavior.
fn check_roundtrip_all_ntts<P>(
	log_domain_size: usize,
	log_data_size: usize,
	max_log_batch: usize,
	max_log_coset: usize,
) where
	P: PackedField<Scalar: BinaryField>,
{
	let simple_ntt = SingleThreadedNTT::<P::Scalar>::new(log_domain_size)
		.unwrap()
		.into_simple_ntt();
	let single_threaded_ntt = SingleThreadedNTT::<P::Scalar>::new(log_domain_size).unwrap();
	let single_threaded_precompute_ntt = SingleThreadedNTT::<P::Scalar>::new(log_domain_size)
		.unwrap()
		.precompute_twiddles();
	let multithreaded_ntt = SingleThreadedNTT::<P::Scalar>::new(log_domain_size)
		.unwrap()
		.multithreaded();
	let multithreaded_precompute_ntt = SingleThreadedNTT::<P::Scalar>::new(log_domain_size)
		.unwrap()
		.precompute_twiddles()
		.multithreaded();
	let dynamic_dispatch_ntt = DynamicDispatchNTT::SingleThreaded(
		SingleThreadedNTT::<P::Scalar>::new(log_domain_size).unwrap(),
	);

	let mut rng = StdRng::seed_from_u64(0);
	let mut data = (0..1u128 << log_data_size)
		.map(|_| P::random(&mut rng))
		.collect::<Vec<_>>();

	let cosets = 0..1 << max_log_coset;
	for log_batch_size in 0..max_log_batch {
		check_roundtrip_with_reference(
			&simple_ntt,
			&single_threaded_ntt,
			&mut data,
			cosets.clone(),
			log_batch_size,
		);
		check_roundtrip_with_reference(
			&simple_ntt,
			&single_threaded_precompute_ntt,
			&mut data,
			cosets.clone(),
			log_batch_size,
		);
		check_roundtrip_with_reference(
			&simple_ntt,
			&multithreaded_ntt,
			&mut data,
			cosets.clone(),
			log_batch_size,
		);
		check_roundtrip_with_reference(
			&simple_ntt,
			&multithreaded_precompute_ntt,
			&mut data,
			cosets.clone(),
			log_batch_size,
		);
		check_roundtrip_with_reference(
			&simple_ntt,
			&dynamic_dispatch_ntt,
			&mut data,
			cosets.clone(),
			log_batch_size,
		);
	}
}

#[test]
fn tests_roundtrip_packed_1() {
	check_roundtrip_all_ntts::<BinaryField8b>(8, 6, 4, 2);
}

#[test]
fn tests_roundtrip_packed_2() {
	check_roundtrip_all_ntts::<PackedBinaryField2x8b>(8, 6, 4, 1);
}

#[test]
fn tests_roundtrip_packed_4() {
	check_roundtrip_all_ntts::<PackedBinaryField2x8b>(8, 6, 4, 0);
}

#[test]
fn tests_field_256_bits() {
	check_roundtrip_all_ntts::<PackedBinaryField8x32b>(12, 8, 4, 1);
}

#[test]
fn tests_field_512_bits() {
	check_roundtrip_all_ntts::<PackedBinaryField16x32b>(12, 6, 4, 0);
}

fn check_packed_extension_roundtrip_with_reference<P, PE>(
	reference_ntt: &impl AdditiveNTT<P>,
	ntt: &impl AdditiveNTT<P>,
	data: &mut [PE],
	cosets: Range<u32>,
) where
	P: PackedField,
	PE: RepackedExtension<P>,
	PE::Scalar: ExtensionField<P::Scalar>,
{
	let data_copy = data.to_vec();
	let mut data_copy_2 = data.to_vec();

	for coset in cosets {
		ntt.forward_transform_ext(data, coset).unwrap();
		reference_ntt
			.forward_transform_ext(&mut data_copy_2, coset)
			.unwrap();

		assert_eq!(data, &data_copy_2);

		ntt.inverse_transform_ext(data, coset).unwrap();
		reference_ntt
			.inverse_transform_ext(&mut data_copy_2, coset)
			.unwrap();

		assert_eq!(data, &data_copy);
		assert_eq!(data, &data_copy_2);
	}
}

fn check_packed_extension_roundtrip_all_ntts<P, PE>(
	log_domain_size: usize,
	log_data_size: usize,
	max_log_coset: usize,
) where
	P: PackedField<Scalar: BinaryField>,
	PE: RepackedExtension<P> + WithUnderlier<Underlier: NumCast<u128>>,
	PE::Scalar: ExtensionField<P::Scalar>,
{
	let simple_ntt = SingleThreadedNTT::<P::Scalar>::new(log_domain_size)
		.unwrap()
		.into_simple_ntt();
	let single_threaded_ntt = SingleThreadedNTT::<P::Scalar>::new(log_domain_size).unwrap();
	let single_threaded_precompute_ntt = SingleThreadedNTT::<P::Scalar>::new(log_domain_size)
		.unwrap()
		.precompute_twiddles();
	let multithreaded_ntt = SingleThreadedNTT::<P::Scalar>::new(log_domain_size)
		.unwrap()
		.multithreaded();
	let multithreaded_precompute_ntt = SingleThreadedNTT::<P::Scalar>::new(log_domain_size)
		.unwrap()
		.precompute_twiddles()
		.multithreaded();
	let dynamic_dispatch_ntt = DynamicDispatchNTT::SingleThreaded(
		SingleThreadedNTT::<P::Scalar>::new(log_domain_size).unwrap(),
	);

	let mut data = (0..1u128 << log_data_size)
		.map(|i| PE::from_underlier(NumCast::num_cast_from(i)))
		.collect::<Vec<_>>();

	let cosets = 0..1 << max_log_coset;

	check_packed_extension_roundtrip_with_reference(
		&simple_ntt,
		&single_threaded_ntt,
		&mut data,
		cosets.clone(),
	);
	check_packed_extension_roundtrip_with_reference(
		&simple_ntt,
		&single_threaded_precompute_ntt,
		&mut data,
		cosets.clone(),
	);
	check_packed_extension_roundtrip_with_reference(
		&simple_ntt,
		&multithreaded_ntt,
		&mut data,
		cosets.clone(),
	);
	check_packed_extension_roundtrip_with_reference(
		&simple_ntt,
		&multithreaded_precompute_ntt,
		&mut data,
		cosets.clone(),
	);
	check_packed_extension_roundtrip_with_reference(
		&simple_ntt,
		&dynamic_dispatch_ntt,
		&mut data,
		cosets.clone(),
	);
}

#[test]
fn tests_packed_extension_roundtrip_packed_single_value() {
	check_packed_extension_roundtrip_all_ntts::<PackedBinaryField1x8b, PackedBinaryField1x8b>(
		8, 6, 2,
	);
	check_packed_extension_roundtrip_all_ntts::<PackedBinaryField1x16b, PackedBinaryField1x16b>(
		9, 6, 2,
	);
}

#[test]
fn tests_packed_extension_roundtrip_packed_2_packed() {
	check_packed_extension_roundtrip_all_ntts::<PackedBinaryField2x8b, PackedBinaryField2x8b>(
		8, 6, 1,
	);
	check_packed_extension_roundtrip_all_ntts::<PackedBinaryField2x16b, PackedBinaryField2x16b>(
		9, 6, 2,
	);
}

#[test]
fn tests_packed_extension_roundtrip_packed_extension() {
	check_packed_extension_roundtrip_all_ntts::<PackedBinaryField4x16b, PackedBinaryField2x32b>(
		8, 6, 1,
	);
}

#[test]
fn tests_packed_extension_roundtrip_message_size_1() {
	check_packed_extension_roundtrip_all_ntts::<PackedBinaryField2x32b, PackedBinaryField2x32b>(
		8, 0, 1,
	);
	check_packed_extension_roundtrip_all_ntts::<PackedBinaryField4x16b, PackedBinaryField4x16b>(
		8, 0, 1,
	);
}

/// Check that NTT transformations give isomorphic results for isomorphic fields.
fn check_ntt_with_transform<P1, P2>(
	ntt_binary: &impl AdditiveNTT<P1::Scalar>,
	ntt_aes_1: &impl AdditiveNTT<P2::Scalar>,
	ntt_aes_2: &impl AdditiveNTT<P2::Scalar>,
	data_size: usize,
) where
	P1: PackedField<Scalar: BinaryField>,
	P2: PackedField<Scalar: BinaryField + From<P1::Scalar>> + From<P1>,
{
	let mut rng = StdRng::seed_from_u64(0);

	let data: Vec<_> = (0..data_size)
		.map(|_| P1::Scalar::random(&mut rng))
		.collect();
	let data_as_aes: Vec<_> = data.iter().map(|x| P2::Scalar::from(*x)).collect();

	for coset in 0..4 {
		let mut result_bin = data.clone();
		let mut result_aes = data_as_aes.clone();
		let mut result_aes_cob = data_as_aes.clone();
		ntt_binary
			.forward_transform(&mut result_bin, coset, 0)
			.unwrap();
		ntt_aes_1
			.forward_transform(&mut result_aes, coset, 0)
			.unwrap();
		ntt_aes_2
			.forward_transform(&mut result_aes_cob, coset, 0)
			.unwrap();

		let result_bin_to_aes: Vec<_> = result_bin.iter().map(|x| P2::Scalar::from(*x)).collect();

		assert_eq!(result_bin_to_aes, result_aes_cob);
		assert_ne!(result_bin_to_aes, result_aes);

		ntt_binary
			.inverse_transform(&mut result_bin, coset, 0)
			.unwrap();
		ntt_aes_1
			.inverse_transform(&mut result_aes, coset, 0)
			.unwrap();
		ntt_aes_2
			.inverse_transform(&mut result_aes_cob, coset, 0)
			.unwrap();

		let result_bin_to_aes: Vec<_> = result_bin.iter().map(|x| P2::Scalar::from(*x)).collect();

		assert_eq!(result_bin_to_aes, result_aes_cob);
	}
}

#[test]
fn test_transform_single_thread_on_the_fly() {
	check_ntt_with_transform::<BinaryField8b, AESTowerField8b>(
		&SingleThreadedNTT::<BinaryField8b>::new(8).unwrap(),
		&SingleThreadedNTT::<AESTowerField8b>::new(8).unwrap(),
		&SingleThreadedNTT::<AESTowerField8b, _>::with_domain_field::<BinaryField8b>(8).unwrap(),
		64,
	)
}

#[test]
fn test_transform_single_thread_precompute() {
	check_ntt_with_transform::<BinaryField8b, AESTowerField8b>(
		&SingleThreadedNTT::<BinaryField8b>::new(8).unwrap(),
		&SingleThreadedNTT::<AESTowerField8b>::new(8)
			.unwrap()
			.precompute_twiddles(),
		&SingleThreadedNTT::<AESTowerField8b, _>::with_domain_field::<BinaryField8b>(8)
			.unwrap()
			.precompute_twiddles(),
		64,
	)
}

#[test]
fn test_transform_multithread_on_the_fly() {
	check_ntt_with_transform::<BinaryField8b, AESTowerField8b>(
		&SingleThreadedNTT::<BinaryField8b>::new(8).unwrap(),
		&SingleThreadedNTT::<AESTowerField8b>::new(8)
			.unwrap()
			.multithreaded(),
		&SingleThreadedNTT::<AESTowerField8b, _>::with_domain_field::<BinaryField8b>(8)
			.unwrap()
			.multithreaded(),
		64,
	)
}

#[test]
fn test_transform_multithread_precompute() {
	check_ntt_with_transform::<BinaryField8b, AESTowerField8b>(
		&SingleThreadedNTT::<BinaryField8b>::new(8).unwrap(),
		&SingleThreadedNTT::<AESTowerField8b>::new(8)
			.unwrap()
			.precompute_twiddles()
			.multithreaded(),
		&SingleThreadedNTT::<AESTowerField8b, _>::with_domain_field::<BinaryField8b>(8)
			.unwrap()
			.precompute_twiddles()
			.multithreaded(),
		64,
	)
}
