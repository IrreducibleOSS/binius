// Copyright 2024-2025 Irreducible Inc.

use super::verify::CommitMeta;

#[test]
fn test_commit_meta_total_vars() {
	let commit_meta = CommitMeta::with_vars([4, 4, 6, 7]);
	assert_eq!(commit_meta.total_vars(), 8);

	let commit_meta = CommitMeta::with_vars([4, 4, 6, 6, 6, 7]);
	assert_eq!(commit_meta.total_vars(), 9);
}
