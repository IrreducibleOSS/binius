// Copyright 2025 Irreducible Inc.

use std::{collections::HashMap, hash::Hash};

/// A channel used to validate a high-level M3 trace.
#[derive(Debug, Default)]
pub struct Channel<T> {
	net_multiplicities: HashMap<T, isize>,
}

impl<T: Hash + Eq> Channel<T> {
	pub fn push(&mut self, val: T) {
		match self.net_multiplicities.get_mut(&val) {
			Some(multiplicity) => {
				*multiplicity += 1;

				// Remove the key if the multiplicity is zero, to improve Debug behavior.
				if *multiplicity == 0 {
					self.net_multiplicities.remove(&val);
				}
			}
			None => {
				let _ = self.net_multiplicities.insert(val, 1);
			}
		}
	}

	pub fn pull(&mut self, val: T) {
		match self.net_multiplicities.get_mut(&val) {
			Some(multiplicity) => {
				*multiplicity -= 1;

				// Remove the key if the multiplicity is zero, to improve Debug behavior.
				if *multiplicity == 0 {
					self.net_multiplicities.remove(&val);
				}
			}
			None => {
				let _ = self.net_multiplicities.insert(val, -1);
			}
		}
	}

	pub fn is_balanced(&self) -> bool {
		self.net_multiplicities.is_empty()
	}
}
