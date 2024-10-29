// Copyright 2024 Irreducible Inc.

extern crate proc_macro;
mod arith_circuit_poly;
mod composition_poly;

use proc_macro::TokenStream;
use quote::{quote, ToTokens};
use std::collections::BTreeSet;
use syn::{parse_macro_input, Data, DeriveInput, Fields};

use crate::{arith_circuit_poly::ArithCircuitPolyItem, composition_poly::CompositionPolyItem};

/// Useful for concisely creating structs that implement CompositionPoly.
/// This currently only supports creating composition polynomials of tower level 0.
///
/// ```
/// use binius_macros::composition_poly;
/// use binius_math::CompositionPoly;
/// use binius_field::{Field, BinaryField1b as F};
///
/// // Defines named struct without any fields that implements CompositionPoly
/// composition_poly!(MyComposition[x, y, z] = x + y * z);
/// assert_eq!(
///     MyComposition.evaluate(&[F::ONE, F::ONE, F::ONE]).unwrap(),
///     F::ZERO
/// );
///
/// // If you omit the name you get an anonymous instance instead, which can be used inline
/// assert_eq!(
///     composition_poly!([x, y, z] = x + y * z)
///         .evaluate(&[F::ONE, F::ONE, F::ONE]).unwrap(),
///     F::ZERO
/// );
/// ```
#[proc_macro]
pub fn composition_poly(input: TokenStream) -> TokenStream {
	parse_macro_input!(input as CompositionPolyItem)
		.into_token_stream()
		.into()
}

#[proc_macro]
pub fn arith_circuit_poly(input: TokenStream) -> TokenStream {
	parse_macro_input!(input as ArithCircuitPolyItem)
		.into_token_stream()
		.into()
}

/// Implements `pub fn iter_oracles(&self) -> impl Iterator<Item = OracleId>`.
///
/// Detects and includes fields with type `OracleId`, `[OracleId; N]`
///
/// ```
/// use binius_macros::IterOracles;
/// type OracleId = usize;
/// type BatchId = usize;
///
/// #[derive(IterOracles)]
/// struct Oracle {
///     x: OracleId,
///     y: [OracleId; 5],
///     z: [OracleId; 5*2],
///     ignored_field1: usize,
///     ignored_field2: BatchId,
///     ignored_field3: [[OracleId; 5]; 2],
/// }
/// ```
#[proc_macro_derive(IterOracles)]
pub fn iter_oracle_derive(input: TokenStream) -> TokenStream {
	let input = parse_macro_input!(input as DeriveInput);
	let Data::Struct(data) = &input.data else {
		panic!("#[derive(IterOracles)] is only defined for structs with named fields");
	};
	let Fields::Named(fields) = &data.fields else {
		panic!("#[derive(IterOracles)] is only defined for structs with named fields");
	};

	let name = &input.ident;
	let (impl_generics, ty_generics, where_clause) = &input.generics.split_for_impl();

	let oracles = fields
		.named
		.iter()
		.filter_map(|f| {
			let name = f.ident.clone();
			match &f.ty {
				syn::Type::Path(type_path) if type_path.path.is_ident("OracleId") => {
					Some(quote!(std::iter::once(self.#name)))
				}
				syn::Type::Array(array) => {
					if let syn::Type::Path(type_path) = *array.elem.clone() {
						if type_path.path.is_ident("OracleId") {
							Some(quote!(self.#name.into_iter()))
						} else {
							None
						}
					} else {
						None
					}
				}
				_ => None,
			}
		})
		.collect::<Vec<_>>();

	quote! {
		impl #impl_generics #name #ty_generics #where_clause {
			pub fn iter_oracles(&self) -> impl Iterator<Item = OracleId> {
				std::iter::empty()
					#(.chain(#oracles))*
			}
		}
	}
	.into()
}

/// Implements `pub fn iter_polys(&self) -> impl Iterator<Item = MultilinearExtension<P>>`.
///
/// Supports `Vec<P>`, `[Vec<P>; N]`. Currently doesn't filter out fields from the struct, so you can't add any other fields.
///
/// ```
/// use binius_macros::IterPolys;
/// use binius_field::PackedField;
///
/// #[derive(IterPolys)]
/// struct Witness<P: PackedField> {
///     x: Vec<P>,
///     y: [Vec<P>; 5],
///     z: [Vec<P>; 5*2],
/// }
/// ```
#[proc_macro_derive(IterPolys)]
pub fn iter_witness_derive(input: TokenStream) -> TokenStream {
	let input = parse_macro_input!(input as DeriveInput);
	let Data::Struct(data) = &input.data else {
		panic!("#[derive(IterPolys)] is only defined for structs with named fields");
	};
	let Fields::Named(fields) = &data.fields else {
		panic!("#[derive(IterPolys)] is only defined for structs with named fields");
	};

	let name = &input.ident;
	let witnesses = fields
		.named
		.iter()
		.map(|f| {
			let name = f.ident.clone();
			match &f.ty {
				syn::Type::Array(_) => quote!(self.#name.iter()),
				_ => quote!(std::iter::once(&self.#name)),
			}
		})
		.collect::<Vec<_>>();

	let packed_field_vars = generic_vars_with_trait(&input.generics, "PackedField");
	assert_eq!(packed_field_vars.len(), 1, "Only a single packed field is supported for now");
	let p = packed_field_vars.first();
	let (impl_generics, ty_generics, where_clause) = &input.generics.split_for_impl();
	quote! {
		impl #impl_generics #name #ty_generics #where_clause {
			pub fn iter_polys(&self) -> impl Iterator<Item = binius_math::MultilinearExtension<#p, &[#p]>> {
				std::iter::empty()
					#(.chain(#witnesses))*
					.map(|values| binius_math::MultilinearExtension::from_values_slice(values.as_slice()).unwrap())
			}
		}
	}
	.into()
}

/// This will accept the generics definition of a struct (relevant for derive macros),
/// and return all the generic vars that are constrained by a specific trait identifier.
/// ```
/// use binius_field::{PackedField, Field};
/// struct Example<A: PackedField, B: PackedField + Field, C: Field>(A, B, C);
/// ```
/// In the above example, when matching against the trait_name "PackedField",
/// the identifiers A and B will be returned, but not C
pub(crate) fn generic_vars_with_trait(
	vars: &syn::Generics,
	trait_name: &str,
) -> BTreeSet<syn::Ident> {
	vars.params
		.iter()
		.filter_map(|param| match param {
			syn::GenericParam::Type(type_param) => {
				let is_bounded_by_trait_name = type_param.bounds.iter().any(|bound| match bound {
					syn::TypeParamBound::Trait(trait_bound) => {
						if let Some(last_segment) = trait_bound.path.segments.last() {
							last_segment.ident == trait_name
						} else {
							false
						}
					}
					_ => false,
				});
				if is_bounded_by_trait_name {
					Some(type_param.ident.clone())
				} else {
					None
				}
			}
			syn::GenericParam::Const(_) => None,
			syn::GenericParam::Lifetime(_) => None,
		})
		.collect()
}
