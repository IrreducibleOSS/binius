// Copyright 2024-2025 Irreducible Inc.

use crate::tower::TowerFamily;

/// The cryptographic extension field that the constraint system protocol is defined over.
pub type FExt<Tower> = <Tower as TowerFamily>::B128;

/// The evaluation domain used in sumcheck protocols.
///
/// This is fixed to be 8-bits, which is large enough to handle all reasonable sumcheck
/// constraint degrees, even with a moderate number of skipped rounds using the univariate skip
/// technique.
pub type FDomain<Tower> = <Tower as TowerFamily>::B8;

/// The Reed–Solomon alphabet used for FRI encoding.
///
/// This is fixed to be 32-bits, which is large enough to handle trace sizes up to 64 GiB
/// of committed data.
pub type FEncode<Tower> = <Tower as TowerFamily>::B32;
