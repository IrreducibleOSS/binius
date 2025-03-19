// Copyright 2025 Irreducible Inc.

use binius_core::constraint_system::channel::{ChannelId, FlushDirection};

use super::column::ColumnIndex;

/// A flushing rule within a table.
#[derive(Debug)]
pub struct Flush {
	pub column_indices: Vec<ColumnIndex>,
	pub channel_id: ChannelId,
	pub direction: FlushDirection,
	pub selector: Option<ColumnIndex>,
}

/// A channel.
#[derive(Debug)]
pub struct Channel {
	pub name: String,
}
