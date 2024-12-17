// Copyright 2024 Irreducible, Inc

/// An index mapping positive integer IDs to optional values.
#[derive(Debug)]
pub struct SparseIndex<T> {
	entries: Vec<Option<T>>,
}

impl<T: Clone> SparseIndex<T> {
	pub fn new(id_bound: usize) -> Self {
		Self {
			entries: vec![None; id_bound],
		}
	}

	pub fn get(&self, id: usize) -> Option<&T> {
		self.entries.get(id)?.as_ref()
	}

	pub fn set(&mut self, id: usize, val: T) {
		if self.entries.len() <= id {
			self.entries.resize(id + 1, None);
		}
		self.entries[id] = Some(val);
	}
}
