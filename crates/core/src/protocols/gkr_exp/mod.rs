// Copyright 2025 Irreducible Inc.

//! Exponentiation via GKR.
//!
//! Let's represent $A$ by its bits, i.e., for each $v \in B_\ell$ we have $A(v)= \sum_{i=0}^{n-1}
//! 2^{i} \cdot b_{i}(v)$. Then, exponentiation can be split into cases, each with its own circuit.
//!
//! 1) Exponentiation of the generator by a chunk of bit-columns:
//!
//!    $$
//!    V_0(X) = 1 - a_0(X) + a_0(X) \cdot g \\\\
//!    V_1(X) = \sum_{v \in B_\ell} \tilde{\mathbf{eq}}(v, X) \cdot V_{0}(v) \cdot \left(1 -
//!    a_{1}(v) + a_{1}(v) \cdot g^{2} \right) \\\\
//!    \ldots \\\\
//!    V_n(X) = \sum_{v \in B_\ell} \tilde{\mathbf{eq}}(v, X) \cdot V_{n-2}(v) \cdot \left(1 -
//!    a_{n-1}(v) + a_{n-1}(v) \cdot g^{2^{n-1}} \right)
//!    $$
//!
//! 2) Exponentiation of the multilinear base by a chunk of bit-columns:
//!
//!    $$
//!    W_0(X) = \sum_{v \in B_\ell} \tilde{\mathbf{eq}}(v, X) \cdot (1 - a_{n-1}(v) + a_{n-1}(v)
//!    \cdot V(v)) \\\\
//!    W_1(X) = \sum_{v \in B_\ell} \tilde{\mathbf{eq}}(v, X) \cdot (W_0(v))^2 \cdot (1 -
//!    a_{n-2}(v) + a_{n-2}(v) \cdot V(v)) \\\\
//!    \ldots \\\\
//!    W_{n-1}(X) = \sum_{v \in B_\ell} \tilde{\mathbf{eq}}(v, X) \cdot (W_{n-2}(v))^2 \cdot (1 -
//!    a_0(v) + a_0(v) \cdot V(v)).
//!    $$
//!
//! You can read more information in [Integer Multiplication in Binius](https://www.irreducible.com/posts/integer-multiplication-in-binius).

mod batch_prove;
mod batch_verify;
mod common;
mod compositions;
mod error;
mod oracles;
mod provers;
mod utils;
mod verifiers;
mod witness;

pub use batch_prove::batch_prove;
pub use batch_verify::batch_verify;
pub use common::{BaseExpReductionOutput, ExpClaim, LayerClaim};
pub use error::Error;
pub use oracles::{construct_gkr_exp_claims, get_evals_in_point_from_witnesses, make_eval_claims};
pub use witness::BaseExpWitness;

#[cfg(test)]
mod tests;
