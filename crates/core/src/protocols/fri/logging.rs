// Copyright 2025 Irreducible Inc.

use binius_field::{BinaryField, BinaryField1b, ExtensionField};

#[derive(Debug)]
#[allow(dead_code)]
pub(super) struct SortAndMergeDimensionData {
	log_elems: usize,
	element_size: usize,
}

impl SortAndMergeDimensionData {
	pub(super) fn new<F: BinaryField>(log_elems: usize) -> Self {
		let element_size = <F as ExtensionField<BinaryField1b>>::DEGREE;
		Self {
			log_elems,
			element_size,
		}
	}
}

#[derive(Debug)]
#[allow(dead_code)]
pub(super) struct RSEncodeDimensionData {
	log_elems: usize,
	element_size: usize,
	log_batch_size: usize,
}

impl RSEncodeDimensionData {
	pub(super) fn new<F: BinaryField>(log_elems: usize, log_batch_size: usize) -> Self {
		let element_size = <F as ExtensionField<BinaryField1b>>::DEGREE;
		Self {
			log_elems,
			element_size,
			log_batch_size,
		}
	}
}

#[derive(Debug)]
#[allow(dead_code)]
pub(super) struct MerkleTreeDimensionData {
	log_elems: usize,
	element_size: usize,
	batch_size: usize,
}

impl MerkleTreeDimensionData {
	pub(super) fn new<F: BinaryField>(log_elems: usize, batch_size: usize) -> Self {
		let element_size = <F as ExtensionField<BinaryField1b>>::DEGREE;
		Self {
			log_elems,
			element_size,
			batch_size,
		}
	}
}

#[derive(Debug)]
#[allow(dead_code)]
pub(super) struct FRIFoldData {
	log_len: usize,
	log_batch_size: usize,
}

impl FRIFoldData {
	pub(super) fn new(log_len: usize, log_batch_size: usize) -> Self {
		Self {
			log_len,
			log_batch_size,
		}
	}

	pub(super) fn log_len(&self) -> usize {
		self.log_len
	}
}
