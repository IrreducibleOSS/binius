// Copyright 2023 Ulvetanna Inc.

use tracing_subscriber::util::TryInitError;

pub fn init_tracing() -> Result<(), TryInitError> {
	use std::env;

	use tracing_profile::{CsvLayer, PrintTreeConfig, PrintTreeLayer};
	use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

	#[cfg(feature = "tracy")]
	{
		if crate::env::boolean_env_flag_set("PROFILE_TRACING") {
			return tracing_subscriber::registry()
				.with(tracing_tracy::TracyLayer::default())
				.with(tracing_subscriber::fmt::layer())
				.try_init();
		}
	}

	if let Ok(csv_path) = env::var("PROFILE_CSV_FILE") {
		tracing_subscriber::registry()
			.with(CsvLayer::new(csv_path))
			.with(tracing_subscriber::fmt::layer())
			.try_init()
	} else {
		tracing_subscriber::registry()
			.with(PrintTreeLayer::new(PrintTreeConfig {
				attention_above_percent: 25.0,
				relevant_above_percent: 2.5,
				hide_below_percent: 1.0,
				display_unaccounted: false,
			}))
			.with(tracing_subscriber::fmt::layer())
			.try_init()
	}
}
