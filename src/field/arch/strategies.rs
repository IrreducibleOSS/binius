/// Packed strategy for arithmetic operations.
/// (Uses arithmetic operations with underlier and subfield to simultaneously calculate the result for all packed values)
pub struct PackedStrategy;
/// Pairwise strategy. Calculate the result by applying operation for each packed value independently.
pub struct PairwiseStrategy;
/// Similar to `PackedStrategy`, but uses SIMD operations supported by the platform.
pub struct SimdStrategy;
/// Applicable only for multiply by alpha and square operations.
/// Reuse multiplication operation for that.
pub struct ReuseMultiplyStrategy;

/// Use operations with GFNI instructions
pub struct GfniStrategy<const TO_AES_MAP: i64, const FROM_AES_MAP: i64>;
