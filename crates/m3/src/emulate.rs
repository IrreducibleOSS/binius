// Copyright 2025 Irreducible Inc.

use std::{collections::BTreeMap, fmt::Debug};

/// A channel used to validate a high-level M3 trace.
#[derive(Debug)]
pub struct Channel<T> {
	net_multiplicities: BTreeMap<T, isize>,
}

impl<T> Default for Channel<T> {
	fn default() -> Self {
		Self {
			net_multiplicities: BTreeMap::default(),
		}
	}
}

impl<T: Eq + PartialEq + Ord + PartialOrd> Channel<T> {
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

impl<T: Debug + Ord + PartialOrd> Channel<T> {
	#[track_caller]
	pub fn assert_balanced(&self) {
		if !self.is_balanced() {
			let (push, pull) = self
				.net_multiplicities
				.iter()
				.partition::<Vec<_>, _>(|(_, multiplicity)| multiplicity.is_positive());

			let mut output = String::new();
			output.push_str("Channel is not balanced: \n");
			if !push.is_empty() {
				output.push_str("  Unbalanced pushes:\n");
				for (v, balance) in push {
					output.push_str(&format!("    {balance}: {v:?}\n"));
				}
			}
			if !pull.is_empty() {
				output.push_str("  Unbalanced pulls:\n");
				for (v, balance) in pull {
					output.push_str(&format!("    {}: {v:?}\n", balance.abs()));
				}
			}

			panic!("{}", output);
		}
	}
}
