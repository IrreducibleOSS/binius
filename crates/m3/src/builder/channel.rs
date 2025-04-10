// Copyright 2025 Irreducible Inc.

use binius_core::constraint_system::channel::{ChannelId, FlushDirection};

use super::column::ColumnIndex;
use crate::builder::{Col, B1};

/// A flushing rule within a table.
#[derive(Debug, Clone)]
pub struct Flush {
	pub column_indices: Vec<ColumnIndex>,
	pub channel_id: ChannelId,
	pub direction: FlushDirection,
	/// The number of times the values are flushed to the channel.
	pub multiplicity: u32,
	/// An optional reference to a column to select which values to flush.
	///
	/// The referenced selector column must hold 1-bit values and contain only zeros after the
	/// index that is the height of the table. If the selector is `None`, all values up to the
	/// table height are flushed.
	pub selector: Option<ColumnIndex>,
}

/// Options for a channel flush.
#[derive(Debug)]
pub struct FlushOpts {
	/// The number of times the values are flushed to the channel.
	pub multiplicity: u32,
	/// An optional reference to a column to select which values to flush.
	///
	/// The referenced selector column must hold 1-bit values and contain only zeros after the
	/// index that is the height of the table. If the selector is `None`, all values up to the
	/// table height are flushed.
	pub selector: Option<Col<B1>>,
}

impl Default for FlushOpts {
	fn default() -> Self {
		Self {
			multiplicity: 1,
			selector: None,
		}
	}
}

/// A channel.
#[derive(Debug, Clone)]
pub struct Channel {
	pub name: String,
}
