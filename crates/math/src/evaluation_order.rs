// Copyright 2025 Irreducible Inc.

/// Sumcheck evaluation order.
///
/// While one can reasonably perform sumcheck over any permutation of the variables,
/// ultimately only two evaluation orders make sense - low-to-high and high-to-low.
///
/// Each have pros and cons under little endian representation of multilinears:
///  1) Low-to-high - good locality of access (especially for univariate skip), but needs
///                   deinterleaving which prevents byteslicing; computes partial low foldings.
///  2) High-to-low - worse locality of access in univariate skip, but byteslicing friendly,
///                   has inplace multithreaded folding; computes partial high foldings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvaluationOrder {
	/// Substituting lower indexed variables first.
	LowToHigh,
	/// Substituting higher indexed variables first.
	HighToLow,
}
