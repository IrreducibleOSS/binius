// Copyright 2024 Irreducible Inc.

/// Trace multiplication event
macro_rules! trace_multiplication {
    ($name: ty) => {
        #[cfg(feature = "trace_multiplications")]
        {
            tracing::event!(name: "mul", tracing::Level::TRACE, {lhs = stringify!($name), rhs = stringify!($name)});
        }
    };
    ($lhs: ty, $rhs: ty) => {
        #[cfg(feature = "trace_multiplications")]
        {
            tracing::event!(name: "mul", tracing::Level::TRACE, {lhs = stringify!($lhs), rhs = stringify!($rhs)});
        }
    };
}

pub(crate) use trace_multiplication;
