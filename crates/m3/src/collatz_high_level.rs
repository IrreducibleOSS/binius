// Copyright 2025 Irreducible Inc.

//! Example of a high-level emulation of a Collatz M3 instance.

// TODO: Move this into an examples directory

use std::{collections::HashMap, hash::Hash};

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

#[derive(Debug, Default, Clone)]
pub struct EvensEvent {
	pub val: u32,
}

impl EvensEvent {
	fn fire(&self, sequence_chan: &mut Channel<u32>) {
		assert_eq!(self.val % 2, 0);

		sequence_chan.pull(self.val);
		sequence_chan.push(self.val / 2);
	}
}

#[derive(Debug, Default, Clone)]
pub struct OddsEvent {
	pub val: u32,
}

impl OddsEvent {
	pub fn fire(&self, sequence_chan: &mut Channel<u32>) {
		assert_eq!(self.val % 2, 1);
		let next_val = self
			.val
			.checked_mul(3)
			.and_then(|val| val.checked_add(1))
			.unwrap();

		sequence_chan.pull(self.val);
		sequence_chan.push(next_val);
	}
}

#[derive(Debug, Default)]
pub struct CollatzTrace {
	pub evens: Vec<EvensEvent>,
	pub odds: Vec<OddsEvent>,
}

impl CollatzTrace {
	pub fn generate(initial_val: u32) -> Self {
		assert_ne!(initial_val, 0);

		let mut trace = CollatzTrace::default();
		let mut val = initial_val;

		while val != 1 {
			if val % 2 == 0 {
				trace.evens.push(EvensEvent { val });
				val /= 2;
			} else {
				trace.odds.push(OddsEvent { val });
				val = 3 * val + 1;
			}
		}

		trace
	}

	pub fn validate(&self, initial_val: u32) {
		let mut sequence_chan = Channel::default();

		// Boundaries
		sequence_chan.push(initial_val);
		sequence_chan.pull(1);

		// Events
		for event in &self.evens {
			event.fire(&mut sequence_chan);
		}
		for event in &self.odds {
			event.fire(&mut sequence_chan);
		}

		assert!(sequence_chan.is_balanced());
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_collatz() {
		let initial_val = 3999;
		let trace = CollatzTrace::generate(initial_val);
		trace.validate(initial_val);
	}
}
