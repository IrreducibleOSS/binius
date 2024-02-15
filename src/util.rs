// Copyright 2023 Ulvetanna Inc.

#[cfg(test)]
pub fn init_tracing() {
	use std::env;

	use tracing_profile::{CsvLayer, PrintTreeLayer};
	use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

	if let Ok(csv_path) = env::var("PROFILE_CSV_FILE") {
		let _ = tracing_subscriber::registry()
			.with(CsvLayer::new(csv_path))
			.with(tracing_subscriber::fmt::layer())
			.try_init();
	} else if env::var("PROFILE_PRINT_TREE").is_ok() {
		let _ = tracing_subscriber::registry()
			.with(PrintTreeLayer::new())
			.try_init();
	} else {
		let _ = tracing_subscriber::registry()
			.with(tracing_subscriber::fmt::layer())
			.try_init();
	};
}
