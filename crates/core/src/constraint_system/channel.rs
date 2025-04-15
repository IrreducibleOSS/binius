// Copyright 2024-2025 Irreducible Inc.

//! A channel allows communication between tables.
//!
//! Note that the channel is unordered - meaning that rows are not
//! constrained to be in the same order when being pushed and pulled.
//!
//! The number of columns per channel must be fixed, but can be any
//! positive integer. Column order is guaranteed, and column values within
//! the same row must always stay together.
//!
//! A channel only ensures that the inputs and outputs match, using a
//! multiset check. If you want any kind of ordering, you have to
//! use polynomial constraints to additionally constraint this.
//!
//! The example below shows a channel with width=2, with multiple inputs
//! and outputs.
//! ```txt
//!                                       +-+-+
//!                                       |C|D|
//! +-+-+                           +---> +-+-+
//! |A|B|                           |     |M|N|
//! +-+-+                           |     +-+-+
//! |C|D|                           |
//! +-+-+  --+                      |     +-+-+
//! |E|F|    |                      |     |I|J|
//! +-+-+    |                      |     +-+-+
//! |G|H|    |                      |     |W|X|
//! +-+-+    |                      | +-> +-+-+
//!          |                      | |   |A|B|
//! +-+-+    +-> /¯\¯¯¯¯¯¯¯¯¯¯¯\  --+ |   +-+-+
//! |I|J|       :   :           : ----+   |K|L|
//! +-+-+  PUSH |   |  channel  |  PULL   +-+-+
//! |K|L|       :   :           : ----+
//! +-+-+    +-> \_/___________/  --+ |   +-+-+
//! |M|N|    |                      | |   |U|V|
//! +-+-+    |                      | |   +-+-+
//! |O|P|    |                      | |   |G|H|
//! +-+-+  --+                      | +-> +-+-+
//! |Q|R|                           |     |E|F|
//! +-+-+                           |     +-+-+
//! |S|T|                           |     |Q|R|
//! +-+-+                           |     +-+-+
//! |U|V|                           |
//! +-+-+                           |     +-+-+
//! |W|X|                           |     |O|P|
//! +-+-+                           +---> +-+-+
//!                                       |S|T|
//!                                       +-+-+
//! ```

use std::collections::HashMap;

use binius_field::{Field, PackedField, TowerField};
use binius_macros::{DeserializeBytes, SerializeBytes};
use binius_math::MultilinearPoly;

use super::error::{Error, VerificationError};
use crate::{oracle::OracleId, witness::MultilinearExtensionIndex};

pub type ChannelId = usize;

#[derive(Debug, Clone, Copy, SerializeBytes, DeserializeBytes, PartialEq, Eq)]
pub enum OracleOrConst<F: Field> {
	Oracle(usize),
	Const { base: F, tower_level: usize },
}
#[derive(Debug, Clone, SerializeBytes, DeserializeBytes)]
pub struct Flush<F: TowerField> {
	pub oracles: Vec<OracleOrConst<F>>,
	pub channel_id: ChannelId,
	pub direction: FlushDirection,
	pub selector: Option<OracleId>,
	pub multiplicity: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, SerializeBytes, DeserializeBytes)]
pub struct Boundary<F: TowerField> {
	pub values: Vec<F>,
	pub channel_id: ChannelId,
	pub direction: FlushDirection,
	pub multiplicity: u64,
}

impl<F: TowerField> Boundary<F> {
	pub fn convert_field<FTarget: TowerField + From<F>>(self) -> Boundary<FTarget> {
		Boundary {
			values: self.values.into_iter().map(FTarget::from).collect(),
			channel_id: self.channel_id,
			direction: self.direction,
			multiplicity: self.multiplicity,
		}
	}
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, SerializeBytes, DeserializeBytes)]
pub enum FlushDirection {
	Push,
	Pull,
}

pub fn validate_witness<F, P>(
	witness: &MultilinearExtensionIndex<P>,
	flushes: &[Flush<F>],
	boundaries: &[Boundary<F>],
	max_channel_id: ChannelId,
) -> Result<(), Error>
where
	P: PackedField<Scalar = F>,
	F: TowerField,
{
	let mut channels = vec![Channel::<F>::new(); max_channel_id + 1];

	for boundary in boundaries.iter().cloned() {
		let Boundary {
			channel_id,
			values,
			direction,
			multiplicity,
		} = boundary;
		if channel_id > max_channel_id {
			return Err(Error::ChannelIdOutOfRange {
				max: max_channel_id,
				got: channel_id,
			});
		}
		channels[channel_id].flush(direction, multiplicity, values.clone())?;
	}

	for flush in flushes {
		let &Flush {
			ref oracles,
			channel_id,
			direction,
			selector,
			multiplicity,
		} = flush;

		if channel_id > max_channel_id {
			return Err(Error::ChannelIdOutOfRange {
				max: max_channel_id,
				got: channel_id,
			});
		}

		let channel = &mut channels[channel_id];

		// We check the variables only of OracleOrConst::Oracle variant oracles being the same.
		let non_const_polys = oracles
			.iter()
			.filter_map(|&id| match id {
				OracleOrConst::Oracle(oracle_id) => Some(witness.get_multilin_poly(oracle_id)),
				_ => None,
			})
			.collect::<Result<Vec<_>, _>>()?;

		let selector_poly = selector
			.map(|selector| witness.get_multilin_poly(selector))
			.transpose()?;

		let n_vars = non_const_polys
			.first()
			.map(|poly| poly.n_vars())
			.unwrap_or(0);

		// Ensure that all the polys in a single flush have the same n_vars
		for poly in &non_const_polys {
			if poly.n_vars() != n_vars {
				return Err(Error::ChannelFlushNvarsMismatch {
					expected: n_vars,
					got: poly.n_vars(),
				});
			}
		}

		// Check selector polynomials are compatible
		if let (Some(selector), Some(selector_poly)) = (selector, &selector_poly) {
			if selector_poly.n_vars() != n_vars {
				let id = oracles
					.iter()
					.copied()
					.filter_map(|id| match id {
						OracleOrConst::Oracle(oracle_id) => Some(oracle_id),
						_ => None,
					})
					.next()
					.expect("non_const_polys is not empty");
				return Err(Error::IncompatibleFlushSelector { id, selector });
			}
		}

		for i in 0..1 << n_vars {
			let selector_off = selector_poly
				.as_ref()
				.map(|selector_poly| {
					selector_poly
						.evaluate_on_hypercube(i)
						.expect(
							"i in range 0..1 << n_vars; \
							selector_poly checked above to have n_vars variables",
						)
						.is_zero()
				})
				.unwrap_or(false);
			if selector_off {
				continue;
			}

			let values = oracles
				.iter()
				.copied()
				.map(|id| match id {
					OracleOrConst::Const { base, .. } => Ok(base),
					OracleOrConst::Oracle(oracle_id) => witness
						.get_multilin_poly(oracle_id)
						.expect("Witness error would have been caught while checking variables.")
						.evaluate_on_hypercube(i),
				})
				.collect::<Result<Vec<_>, _>>()?;
			channel.flush(direction, multiplicity, values)?;
		}
	}

	for (id, channel) in channels.iter().enumerate() {
		if !channel.is_balanced() {
			return Err((VerificationError::ChannelUnbalanced { id }).into());
		}
	}

	Ok(())
}

#[derive(Default, Debug, Clone)]
struct Channel<F: TowerField> {
	width: Option<usize>,
	multiplicities: HashMap<Vec<F>, i64>,
}

impl<F: TowerField> Channel<F> {
	fn new() -> Self {
		Self::default()
	}

	fn _print_unbalanced_values(&self) {
		for (key, val) in &self.multiplicities {
			if *val != 0 {
				println!("{key:?}: {val}");
			}
		}
	}

	fn flush(
		&mut self,
		direction: FlushDirection,
		multiplicity: u64,
		values: Vec<F>,
	) -> Result<(), Error> {
		if self.width.is_none() {
			self.width = Some(values.len());
		} else if self.width.expect("checked for None above") != values.len() {
			return Err(Error::ChannelFlushWidthMismatch {
				expected: self.width.unwrap(),
				got: values.len(),
			});
		}
		*self.multiplicities.entry(values).or_default() += (multiplicity as i64)
			* (match direction {
				FlushDirection::Pull => -1i64,
				FlushDirection::Push => 1i64,
			});
		Ok(())
	}

	fn is_balanced(&self) -> bool {
		self.multiplicities.iter().all(|(_, m)| *m == 0)
	}
}

#[cfg(test)]
mod tests {
	use binius_field::BinaryField64b;

	use super::*;

	#[test]
	fn test_flush_push_single_row() {
		let mut channel = Channel::<BinaryField64b>::new();

		// Push a single row of data
		let values = vec![BinaryField64b::from(1), BinaryField64b::from(2)];
		let result = channel.flush(FlushDirection::Push, 1, values.clone());

		assert!(result.is_ok());
		assert!(!channel.is_balanced());
		assert_eq!(channel.multiplicities.get(&values).unwrap(), &1);
	}

	#[test]
	fn test_flush_pull_single_row() {
		let mut channel = Channel::<BinaryField64b>::new();

		// Pull a single row of data
		let values = vec![BinaryField64b::from(1), BinaryField64b::from(2)];
		let result = channel.flush(FlushDirection::Pull, 1, values.clone());

		assert!(result.is_ok());
		assert!(!channel.is_balanced());
		assert_eq!(channel.multiplicities.get(&values).unwrap(), &-1);
	}

	#[test]
	fn test_flush_push_pull_single_row() {
		let mut channel = Channel::<BinaryField64b>::new();

		// Push and then pull the same row
		let values = vec![BinaryField64b::from(1), BinaryField64b::from(2)];
		channel
			.flush(FlushDirection::Push, 1, values.clone())
			.unwrap();
		let result = channel.flush(FlushDirection::Pull, 1, values.clone());

		assert!(result.is_ok());
		assert!(channel.is_balanced());
		assert_eq!(channel.multiplicities.get(&values).unwrap_or(&0), &0);
	}

	#[test]
	fn test_flush_multiplicity() {
		let mut channel = Channel::<BinaryField64b>::new();

		// Push multiple rows with a multiplicity of 2
		let values = vec![BinaryField64b::from(3), BinaryField64b::from(4)];
		channel
			.flush(FlushDirection::Push, 2, values.clone())
			.unwrap();

		// Pull the same row with a multiplicity of 1
		channel
			.flush(FlushDirection::Pull, 1, values.clone())
			.unwrap();

		// The channel should not be balanced yet
		assert!(!channel.is_balanced());
		assert_eq!(channel.multiplicities.get(&values).unwrap(), &1);

		// Pull the same row again with a multiplicity of 1
		channel
			.flush(FlushDirection::Pull, 1, values.clone())
			.unwrap();

		// Now the channel should be balanced
		assert!(channel.is_balanced());
		assert_eq!(channel.multiplicities.get(&values).unwrap_or(&0), &0);
	}

	#[test]
	fn test_flush_width_mismatch() {
		let mut channel = Channel::<BinaryField64b>::new();

		// Push a row with width 2
		let values1 = vec![BinaryField64b::from(1), BinaryField64b::from(2)];
		channel.flush(FlushDirection::Push, 1, values1).unwrap();

		// Attempt to push a row with width 3
		let values2 = vec![
			BinaryField64b::from(3),
			BinaryField64b::from(4),
			BinaryField64b::from(5),
		];
		let result = channel.flush(FlushDirection::Push, 1, values2);

		assert!(result.is_err());
		if let Err(Error::ChannelFlushWidthMismatch { expected, got }) = result {
			assert_eq!(expected, 2);
			assert_eq!(got, 3);
		} else {
			panic!("Expected ChannelFlushWidthMismatch error");
		}
	}

	#[test]
	fn test_flush_direction_effects() {
		let mut channel = Channel::<BinaryField64b>::new();

		// Push a row
		let values = vec![BinaryField64b::from(7), BinaryField64b::from(8)];
		channel
			.flush(FlushDirection::Push, 1, values.clone())
			.unwrap();

		// Pull a different row
		let values2 = vec![BinaryField64b::from(9), BinaryField64b::from(10)];
		channel
			.flush(FlushDirection::Pull, 1, values2.clone())
			.unwrap();

		// The channel should not be balanced because different rows were pushed and pulled
		assert!(!channel.is_balanced());
		assert_eq!(channel.multiplicities.get(&values).unwrap(), &1);
		assert_eq!(channel.multiplicities.get(&values2).unwrap(), &-1);
	}
}
