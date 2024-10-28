// Copyright 2024 Irreducible Inc.

// Get log trace size from the environment variable.
// Panics if the environment variable is not a valid integer.
pub fn get_log_trace_size() -> Option<usize> {
	match std::env::var("BINIUS_LOG_TRACE") {
		Ok(val) => Some(
			val.parse::<usize>()
				.expect("BINIUS_LOG_TRACE must be a valid integer"),
		),
		Err(_) => None,
	}
}
