// Copyright 2023-2024 Irreducible Inc.

use cfg_if::cfg_if;
use std::env;
use tracing_subscriber::{
	layer::SubscriberExt,
	registry::LookupSpan,
	util::{SubscriberInitExt, TryInitError},
};

fn with_perf_counters<S>(subscriber: S) -> impl SubscriberExt + for<'lookup> LookupSpan<'lookup>
where
	S: SubscriberExt + for<'lookup> LookupSpan<'lookup>,
{
	cfg_if! {
		if #[cfg(feature = "perf_counters")] {
			subscriber.with(
				tracing_profile::PrintPerfCountersLayer::new(vec![
					("instructions".to_string(), tracing_profile::PerfHardwareEvent::INSTRUCTIONS.into()),
					("cycles".to_string(), tracing_profile::PerfHardwareEvent::CPU_CYCLES.into()),
				])
				.unwrap(),
			)
		} else {
			subscriber
		}
	}
}

fn with_ittapi<S>(subscriber: S) -> impl SubscriberExt + for<'lookup> LookupSpan<'lookup>
where
	S: SubscriberExt + for<'lookup> LookupSpan<'lookup>,
{
	cfg_if! {
		if #[cfg(feature = "ittapi")] {
			subscriber.with(tracing_profile::IttApiLayer)
		} else {
			subscriber
		}
	}
}

#[cfg(not(feature = "perfetto"))]
pub struct TracingGuard;

#[cfg(feature = "perfetto")]
pub type TracingGuard = tracing_profile::PerfettoGuard;

fn with_perfetto<S>(
	subscriber: S,
) -> (impl SubscriberExt + for<'lookup> LookupSpan<'lookup>, TracingGuard)
where
	S: SubscriberExt + for<'lookup> LookupSpan<'lookup>,
{
	cfg_if! {
		if #[cfg(feature = "perfetto")] {
			let (layer, guard) = tracing_profile::PerfettoLayer::new_from_env().expect("failed to initialize perfetto layer");
			(subscriber.with(layer), guard)
		} else {
			(subscriber, TracingGuard{})
		}
	}
}

fn with_tracy<S>(subscriber: S) -> impl SubscriberExt + for<'lookup> LookupSpan<'lookup>
where
	S: SubscriberExt + for<'lookup> LookupSpan<'lookup>,
{
	cfg_if! {
		if #[cfg(feature = "tracy")] {
			subscriber.with(tracing_tracy::TracyLayer::default())
		} else {
			subscriber
		}
	}
}

pub fn init_tracing() -> Result<TracingGuard, TryInitError> {
	use tracing_profile::{CsvLayer, PrintTreeConfig, PrintTreeLayer};

	if let Ok(csv_path) = env::var("PROFILE_CSV_FILE") {
		let (layer, guard) = with_perfetto(with_perf_counters(with_tracy(with_ittapi(
			tracing_subscriber::registry()
				.with(CsvLayer::new(csv_path))
				.with(tracing_subscriber::fmt::layer()),
		))));
		layer.try_init()?;

		Ok(guard)
	} else {
		let (layer, guard) = with_perfetto(with_perf_counters(with_tracy(with_ittapi(
			tracing_subscriber::registry().with(PrintTreeLayer::new(PrintTreeConfig {
				attention_above_percent: 25.0,
				relevant_above_percent: 2.5,
				hide_below_percent: 1.0,
				display_unaccounted: false,
				accumulate_events: true,
			})),
		))));
		layer.try_init()?;

		Ok(guard)
	}
}
