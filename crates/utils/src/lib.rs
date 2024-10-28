// Copyright 2024 Irreducible Inc.

//! Utility modules used in Binius.

pub mod array_2d;
pub mod checked_arithmetics;
pub mod env;
pub mod error_utils;
pub mod examples;
pub mod felts;
pub mod iter;
pub mod rayon;
pub mod serialization;
pub mod sorting;
pub mod thread_local_mut;
#[cfg(feature = "tracing")]
pub mod tracing;
