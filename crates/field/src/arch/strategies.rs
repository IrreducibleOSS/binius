// Copyright 2024-2025 Irreducible Inc.

/// Packed strategy for arithmetic operations.
/// (Uses arithmetic operations with underlier and subfield to simultaneously calculate the result
/// for all packed values)
pub struct PackedStrategy;
/// This strategies uses bot operations over packed subfield and operations over sub-elements.
pub struct HybridRecursiveStrategy;
/// Pairwise recursive strategy. Calculates the result by applying recursive algorithm for each
/// packed value independently.
pub struct PairwiseRecursiveStrategy;
/// Pairwise strategy. Apply the result of the operation to each packed element independently.
pub struct PairwiseStrategy;
/// Get result of operation from the table for each sub-element
pub struct PairwiseTableStrategy;
/// Similar to `PackedStrategy`, but uses SIMD operations supported by the platform.
pub struct SimdStrategy;
/// Applicable only for multiply by alpha and square operations.
/// Reuse multiplication operation for that.
pub struct ReuseMultiplyStrategy;

/// Use operations with GFNI instructions
pub struct GfniStrategy;
/// Specialized versions of the above to resolve conflicting implementations
pub struct GfniSpecializedStrategy256b;
pub struct GfniSpecializedStrategy512b;

/// Strategy for packed canonical tower fields.
/// Performs conversion to the packed isomorphic AES field, applies the operation and
/// converts the result back to the canonical tower field.
pub struct AESIsomorphicStrategy;
