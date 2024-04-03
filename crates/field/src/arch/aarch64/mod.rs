use cfg_if::cfg_if;

cfg_if! {
	if #[cfg(all(target_feature = "neon", target_feature = "aes"))] {
		pub(super) mod m128;
		pub mod packed_aes_128;
		pub mod polyval;
	} else {
		pub use super::portable::packed_aes_128;
		pub use super::portable::polyval;
	}
}
