// Copyright 2024 Irreducible Inc.

//! The multivariate sumcheck polynomial protocol, special cased when the sumcheck is part of the GPA Protocol.
//!
//! The parent GPA Module must have certain properties in order to qualify to use this sumcheck protocol.
//!     1) The GKR grand product circuit must be data-parallel
//!     2) The GKR grand product circuit must admit layer reduction sumcheck claims on multilinear composite polynomials $f(x)$
//! that have the form $f(x) = g(x) \cdot h(x)$, where $g(x)$ is a partially evaluated equality indicator
//! multilinear polynomial.
//!     3) $h(x)$ must agree with the current layer multilinear everywhere on the boolean hypercube.
//!
//! If the above conditions are met, then this GPA sumcheck module exploits this structure to yield a
//! more efficient prover algorithm than the sumcheck module.
//!

pub mod error;
pub mod prove;
pub mod verify;
