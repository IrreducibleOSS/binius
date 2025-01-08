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

use binius_field::{as_packed_field::PackScalar, underlier::UnderlierType, TowerField};

use super::error::{Error, VerificationError};
use crate::{oracle::OracleId, witness::MultilinearExtensionIndex};

pub type ChannelId = usize;

#[derive(Debug, Clone)]
pub struct Flush {
	pub oracles: Vec<OracleId>,
	pub channel_id: ChannelId,
	pub direction: FlushDirection,
	pub count: usize,
	pub multiplicity: u64,
}

#[derive(Debug, Clone)]
pub struct Boundary<F: TowerField> {
	pub values: Vec<F>,
	pub channel_id: ChannelId,
	pub direction: FlushDirection,
	pub multiplicity: u64,
}

#[derive(Debug, Clone, Copy)]
pub enum FlushDirection {
	Push,
	Pull,
}

pub fn validate_witness<U, F>(
	witness: &MultilinearExtensionIndex<U, F>,
	flushes: &[Flush],
	boundaries: &[Boundary<F>],
	max_channel_id: ChannelId,
) -> Result<(), Error>
where
	U: UnderlierType + PackScalar<F>,
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
		channels[channel_id].flush(&direction, multiplicity, values.clone())?;
	}

	for flush in flushes {
		let Flush {
			oracles,
			channel_id,
			direction,
			count,
			multiplicity,
		} = flush;

		if *channel_id > max_channel_id {
			return Err(Error::ChannelIdOutOfRange {
				max: max_channel_id,
				got: *channel_id,
			});
		}

		let channel = &mut channels[*channel_id];

		let polys = oracles
			.iter()
			.map(|id| witness.get_multilin_poly(*id).unwrap())
			.collect::<Vec<_>>();

		// Ensure that all the polys in a single flush have the same n_vars
		if let Some(first_poly) = polys.first() {
			let n_vars = first_poly.n_vars();
			for poly in &polys {
				if poly.n_vars() != n_vars {
					return Err(Error::ChannelFlushNvarsMismatch {
						expected: n_vars,
						got: poly.n_vars(),
					});
				}
			}

			// Check count is within range
			if *count > 1 << n_vars {
				let id = oracles.first().expect("polys is not empty");
				return Err(Error::FlushCountExceedsOracleSize {
					id: *id,
					count: *count,
				});
			}

			for i in 0..*count {
				let values = polys
					.iter()
					.map(|poly| poly.evaluate_on_hypercube(i).unwrap())
					.collect();
				channel.flush(direction, *multiplicity, values)?;
			}
		}
	}

	for (id, channel) in channels.iter().enumerate() {
		if !channel.is_balanced() {
			return Err(VerificationError::ChannelUnbalanced { id }.into());
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
		for (key, val) in self.multiplicities.iter() {
			if *val != 0 {
				println!("{key:?}: {val}");
			}
		}
	}

	fn flush(
		&mut self,
		direction: &FlushDirection,
		multiplicity: u64,
		values: Vec<F>,
	) -> Result<(), Error> {
		if self.width.is_none() {
			self.width = Some(values.len());
		} else if self.width.unwrap() != values.len() {
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
