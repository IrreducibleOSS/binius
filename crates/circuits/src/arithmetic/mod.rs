// Copyright 2024 Irreducible Inc.

pub mod u32;

/// Whether to allow or disallow arithmetic overflow
#[derive(Debug, Clone, Copy)]
pub enum Flags {
	Checked,
	Unchecked,
}
