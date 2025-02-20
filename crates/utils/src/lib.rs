// Copyright 2024-2025 Irreducible Inc.

//! Utility modules used in Binius.

#![cfg_attr(not(feature = "stable_only"), feature(iter_advance_by))]

pub mod array_2d;
pub mod checked_arithmetics;
pub mod env;
pub mod error_utils;
pub mod examples;
pub mod felts;
pub mod graph;
pub mod iter;
pub mod rayon;
pub mod serialization;
pub mod sorting;
pub mod sparse_index;
pub mod thread_local_mut;

pub use bytes;
pub use serialization::{DeserializeBytes, SerializationError, SerializationMode, SerializeBytes};
