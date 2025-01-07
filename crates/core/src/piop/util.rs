// Copyright 2024-2025 Irreducible Inc.

pub struct ResizeableIndex<T> {
	entries: Vec<T>,
}

impl<T: Default> ResizeableIndex<T> {
	pub fn new() -> Self {
		Self {
			entries: Vec::new(),
		}
	}

	pub fn get_mut(&mut self, id: usize) -> &mut T {
		if id >= self.entries.len() {
			self.entries.resize_with(id + 1, T::default);
		}
		&mut self.entries[id]
	}

	pub fn into_vec(self) -> Vec<T> {
		self.entries
	}
}
