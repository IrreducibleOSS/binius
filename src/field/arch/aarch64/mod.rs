use cfg_if::cfg_if;

cfg_if! {
	if #[cfg(all(target_feature = "neon", target_feature = "aes"))] {
		pub mod polyval;
	} else {
		pub use super::portable::polyval;
	}
}
