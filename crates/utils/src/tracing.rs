// Copyright 2023 Ulvetanna Inc.

pub fn init_tracing() {
	use std::env;

	use tracing_profile::{CsvLayer, PrintTreeConfig, PrintTreeLayer};
	use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

	if let Ok(csv_path) = env::var("PROFILE_CSV_FILE") {
		let _ = tracing_subscriber::registry()
			.with(CsvLayer::new(csv_path))
			.with(tracing_subscriber::fmt::layer())
			.try_init();
	} else {
		let _ = tracing_subscriber::registry()
			.with(PrintTreeLayer::new(PrintTreeConfig {
				attention_above_percent: 25.0,
				relevant_above_percent: 2.5,
				hide_below_percent: 1.0,
				display_unaccounted: false,
			}))
			.with(tracing_subscriber::fmt::layer())
			.try_init();
	};
}
