// Copyright 2024-2025 Irreducible Inc.

pub mod arithmetic;
pub mod projected;
pub mod shifted;

pub use projected::projected;
pub use shifted::shifted;

pub fn arithmetic_filler<F: binius_field::Field>(
	expr: binius_math::ArithExpr<F>,
) -> crate::Filler {
	crate::Filler::new(|inputs, output| {
		// i guess all subexpressions are over F?
		// then query number of vars and expect that many inputs
		// but inputs of different tower levels are encoded differently, not all as if at tower level of F...
	})
}
