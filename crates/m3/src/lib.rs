// Copyright 2025 Irreducible Inc.

//! A library for building Binius constraint systems and instances using M3 arithmetization.
//!
//! ## M3
//!
//! [M3], short for Multi-Multiset Matching, is a paradigm for expressing decision computations
//! with a direct and efficient translation to cryptographic proving techniques. M3 constraint
//! systems are built from _tables_ and _channels_ using binary field elements as the primitive
//! data types.
//!
//! ### Tower field types
//!
//! All M3 instances are defined over the canonical Fan-Paar tower, which is a family of binary
//! fields. This library supports constraint systems using the 1-bit, 8-bit, 16-bit, 32-bit,
//! 64-bit, and 128-bit tower field types. They are referred to as `B1`, `B8`, `B16`, `B32`, `B64`,
//! and `B128`, respectively.
//!
//! See the [Data Types] documentation for further information.
//!
//! ### Tables
//!
//! An M3 constraint system has several tables. Each row in a table corresponds to a
//! _transition event_, and each column in a table is a different field. In Binius, each column has
//! a different shape, defined by the tower field and the number of elements packed vertically into
//! a single row, which we call the vertical packing factor. Due to vertical packing, the columns
//! in a table will contain the same number of events but may have different numbers of field
//! elements.
//!
//! Tables can have a few types of constraints:
//!
//! 1. **Zero constraints**: these are constraints that require a certain arithmetic expression to
//!    evaluate to zero over each row. A zero constraint must reference only columns within the same
//!    table with the same vertical packing factor. The constraint must evaluate to zero for each
//!    vertically packed entry.
//! 2. **Non-zero column constraints**: these enforce that a column has no non-zero entries.
//!
//! ### Channels
//!
//! Channels are the mechanism for enforcing global constraints across transition events. The
//! constraint system _pushes_ and _pulls_ tuples of fields element to and from channels and
//! requires that the set of items pushes balances the set of items pulled. Items are pushed to
//! channels either from table _flushes_ as _boundary values_. Each table defines _flushing rules_,
//! which make it so that tuples collected from each row in a table are either pushed or pulled
//! from a channel. _Boundary values_ are the inputs to a statement, which are known to the
//! verifier.
//!
//! See the [M3] for documentation for further information and examples.
//!
//! [M3]: <https://www.binius.xyz/basics/arithmetization/m3>
//! [Data Types]: <https://www.binius.xyz/basics/arithmetization/types>
//!
//! ## Build API
//!
//! This library provides an opinionated interface for constructing M3 constraint systems over the
//! Binius core protocol. This means that it does not expose the full set of capabilities of the
//! Binius protocol.
//!
//! The library exposes a [`builder::ConstraintSystem`], which has mutating methods used to build
//! an instance. The M3 constraint system differs from a
//! [`binius_core::constraint_system::ConstraintSystem`] in that it does not implicitly contain
//! table heights as part of its definition. This allows for flexibility in the statements
//! supported, so that a verifier can audit a constraint system once and then accept proofs for
//! different statements than require different table sizes.
//!
//! The library intends the code building the constraint system is trusted and audited and thus may
//! panic if invoked incorrectly. Once the constraint system is built, it can be serialized and
//! later deserialized during an online proving and verification process. All functions that
//! interact with a constraint system after the build phase will return errors on invalid inputs.
//!
//! ### Gadgets
//!
//! Constraint systems are intended to be built using reusable _gadgets_. These are structures that
//! encapsulate the logic for defining constraints on a subset of table columns and the code for
//! populating the witness data in the table. Gadgets have input columns, output columns, and
//! internal columns. The inputs columns are defined externally and provided as inputs; the gadget
//! assumes their values are already populated during witness population. The gadget defines output
//! and internal columns, and exposes only the output columns to the caller.

pub mod builder;
pub mod emulate;
pub mod u32;
