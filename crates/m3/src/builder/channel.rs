use binius_core::constraint_system::channel::{ChannelId, FlushDirection};

#[derive(Debug)]
pub struct Flush {
	pub column_indices: Vec<usize>,
	pub channel_id: ChannelId,
	pub direction: FlushDirection,
}

#[derive(Debug)]
pub struct Channel {
	pub name: String,
}
