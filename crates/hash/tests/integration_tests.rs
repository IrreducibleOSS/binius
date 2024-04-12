// Copyright 2024 Ulvetanna Inc.
use binius_hash::{Digest, Groestl256};
use hex_literal::hex;
// NOTE: This is meant to test current portable implementation with platform specific ones

#[test]
fn test_empty_input() {
	// Empty input test
	let hasher = Groestl256::default();
	let exptected = hex!("1a52d11d550039be16107f9c58db9ebcc417f16f736adb2502567119f0083467");
	let out: [u8; 32] = hasher.finalize().into();
	assert_eq!(out, exptected);
}

#[test]
fn test_multi_update() {
	// Testing breaking it out into chunks
	let mut hasher_1 = Groestl256::default();
	hasher_1.update("The quick brown fox jumps over the lazy dog".as_bytes());
	let expected = hex!("8c7ad62eb26a21297bc39c2d7293b4bd4d3399fa8afab29e970471739e28b301");
	let hash: [u8; 32] = hasher_1.finalize().into();
	assert_eq!(hash, expected);

	let mut hasher_2 = Groestl256::default();
	hasher_2.update("The quick brown fox jumps".as_bytes());
	hasher_2.update(" over the lazy dog".as_bytes());
	let hash: [u8; 32] = hasher_2.finalize().into();
	assert_eq!(expected, hash);
}

#[test]
fn test_simple_word() {
	let mut hasher = Groestl256::default();
	hasher.update("The quick brown fox jumps over the lazy dog.".as_bytes());
	let hash: [u8; 32] = hasher.finalize().into();
	let expected = hex!("f48290b1bcacee406a0429b993adb8fb3d065f4b09cbcdb464a631d4a0080aaf");
	assert_eq!(hash, expected);
}

#[test]
fn test_one_chunk_input() {
	// Testing inputing exactly 64 bytes
	let mut hasher = Groestl256::default();
	hasher.update("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA".as_bytes());
	let expected = hex!("fb3ba0dd1af025433fd91b0813a5c7e14a885beb988a61e9efabcf6e9fcb1073");
	let hash: [u8; 32] = hasher.finalize().into();
	assert_eq!(hash, expected);
}

#[test]
fn test_two_chunk_input() {
	// Testing 2 blocks
	let mut hasher = Groestl256::default();
	hasher.update("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA".as_bytes());
	let hash: [u8; 32] = hasher.finalize().into();
	let expected = hex!("55bbe6657e052e83b95f2f468a22fd4ed8f4dd07f966e3addb593ffaa874820c");
	assert_eq!(hash, expected);
}

#[test]
fn test_aligned_block_updates() {
	// This lets us test when you supply the first block in smaller sizes but aligns to 64
	// bytes in the update function
	let mut hasher = Groestl256::new();
	hasher.update("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA".as_bytes());
	hasher.update("AAAAAAAAAAAAAAAAAAAA".as_bytes());
	hasher.update("A".as_bytes());
	let expected = hex!("55bbe6657e052e83b95f2f468a22fd4ed8f4dd07f966e3addb593ffaa874820c");
	let hash: [u8; 32] = hasher.finalize().into();
	assert_eq!(hash, expected);
}

#[test]
fn test_force_two_chunk_pads() {
	// This lets us test when even though you have <64 bytes, padding will force you to have 2
	// blocks in the finalize function
	let mut hasher = Groestl256::new();
	hasher.update("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA".as_bytes());
	let expected = hex!("075afb879c3185a1ee2afab192621e401e62d6d158d7156ce884d40822a8f277");
	let hash: [u8; 32] = hasher.finalize().into();
	assert_eq!(hash, expected);
}
