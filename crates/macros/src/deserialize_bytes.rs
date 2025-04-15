// Copyright 2024-2025 Irreducible Inc.

mod parse;
mod quote;

use parse::ContainerAttributes;
pub use quote::GenericsSplit;
use syn::{DeriveInput, Generics, Meta, MetaList};

/// Parse the container attributes for DeserializeBytes.
pub fn parse_container_attributes(input: &DeriveInput) -> syn::Result<ContainerAttributes> {
	let maybe_deserialize_bytes_attr = input
		.attrs
		.iter()
		.find(|attr| attr.path().is_ident("deserialize_bytes"));
	let deserialize_bytes_attr = match maybe_deserialize_bytes_attr {
		Some(attr) => attr,
		None => return Ok(ContainerAttributes::default()),
	};
	let container_attributes_tokens = match &deserialize_bytes_attr.meta {
		Meta::List(MetaList { tokens, .. }) => tokens.clone(),
		meta => {
			return Err(syn::Error::new_spanned(
				meta,
				"expected `deserialize_bytes(eval_generics(X = Y, ...))`",
			));
		}
	};
	let container_attributes: ContainerAttributes = syn::parse2(container_attributes_tokens)?;
	Ok(container_attributes)
}

/// Splits the generics into impl generics, type generics and where clauses
/// taking into account container attributes like `eval_generics(X = Y, ...)`.
///
/// This is similar to [`syn::Generics::split_for_impl`].
pub fn split_for_impl<'gen, 'attr>(
	generics: &'gen Generics,
	container_attributes: &'attr ContainerAttributes,
) -> GenericsSplit<'gen, 'attr> {
	GenericsSplit::new(generics, &container_attributes.eval_generics)
}

#[cfg(test)]
mod tests {
	use ::quote::quote;
	use proc_macro2::TokenStream;
	use syn::DeriveInput;

	use super::*;

	#[test]
	fn test_split_for_impl() {
		struct Case {
			struct_def: TokenStream,
			expected_impl_def: TokenStream,
		}
		let cases = vec![
			Case {
				struct_def: quote! {
					#[derive(DeserializeBytes)]
					#[deserialize_bytes(eval_generics(F = BinaryField128b))]
					pub struct MultilinearOracleSet<F: TowerField> {
						oracles: Vec<MultilinearPolyOracle<F>>
					}
				},
				expected_impl_def: quote! {
					impl MultilinearOracleSet<BinaryField128b>
				},
			},
			Case {
				struct_def: quote! {
					#[derive(DeserializeBytes)]
					#[deserialize_bytes(eval_generics(F = BinaryField128b))]
					pub struct FieldPair<F: TowerField, G: TowerField + DeserializeBytes>
					where G: Debug
					{
						x: F,
						y: G,
					}
				},
				expected_impl_def: quote! {
					impl <G: TowerField + DeserializeBytes> FieldPair<BinaryField128b, G>
					where G: Debug
				},
			},
		];
		for case in cases {
			let input = syn::parse2::<DeriveInput>(case.struct_def).unwrap();
			let container_attributes = parse_container_attributes(&input).unwrap();
			let GenericsSplit {
				impl_generics,
				type_generics,
				where_clause,
			} = split_for_impl(&input.generics, &container_attributes);
			let name = input.ident;
			let impl_def = quote! {
				impl #impl_generics #name #type_generics #where_clause
			};
			assert_eq!(impl_def.to_string(), case.expected_impl_def.to_string());
		}
	}
}
