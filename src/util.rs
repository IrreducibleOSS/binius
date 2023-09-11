pub fn log2(v: usize) -> usize {
	63 - (v as u64).leading_zeros() as usize
}
