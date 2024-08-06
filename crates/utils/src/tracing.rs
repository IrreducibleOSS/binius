// Copyright 2023 Ulvetanna Inc.

use tracing_chrome::FlushGuard;
use tracing_subscriber::{util::TryInitError, EnvFilter};

pub fn init_tracing() -> Result<Option<FlushGuard>, TryInitError> {
	use std::env;
	use tracing_profile::{CsvLayer, PrintTreeConfig, PrintTreeLayer};
	use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

	#[cfg(feature = "tracy")]
	{
		if crate::env::boolean_env_flag_set("PROFILE_TRACING") {
			tracing_subscriber::registry()
				.with(tracing_tracy::TracyLayer::default())
				.with(tracing_subscriber::fmt::layer())
				.try_init()?;
			return Ok(None);
		}
	}

	{
		let with_ansi = env::var("COLOR_LOG").unwrap_or("true".to_string()) == "true";

		if let Ok(csv_path) = env::var("PROFILE_CSV_FILE") {
			tracing_subscriber::registry()
				.with(CsvLayer::new(csv_path))
				.with(tracing_subscriber::fmt::layer().with_ansi(with_ansi))
				.try_init()?;
			Ok(None)
		} else {
			let subscriber = tracing_subscriber::registry()
				.with(PrintTreeLayer::new(PrintTreeConfig {
					attention_above_percent: 25.0,
					relevant_above_percent: 2.5,
					hide_below_percent: 1.0,
					display_unaccounted: false,
					accumulate_events: true,
				}))
				.with(EnvFilter::from_default_env())
				.with(tracing_subscriber::fmt::layer().with_ansi(with_ansi));

			if crate::env::boolean_env_flag_set("PERFETTO_TRACING") {
				use tracing_chrome::ChromeLayerBuilder;
				let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
				let subscriber = subscriber.with(chrome_layer);
				subscriber.try_init()?;
				Ok(Some(guard))
			} else {
				subscriber.try_init()?;
				Ok(None)
			}
		}
	}
}
