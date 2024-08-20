// Copyright 2024 Ulvetanna Inc.

#[macro_export]
macro_rules! bail {
	($err:expr) => {{
		if cfg!(feature = "bail_panic") {
			panic!("{}", $err);
		} else {
			return Err($err.into());
		}
	}};
}
