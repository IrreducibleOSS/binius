mod parse;
mod quote;

use parse::ContainerAttributes;
pub use quote::GenericsSplit;
use syn::{Attribute, Generics, Meta, MetaList};

pub fn get_generics<'attr, 'gen>(
	attrs: &'attr [Attribute],
	generics: &'gen Generics,
) -> syn::Result<GenericsSplit<'gen>> {
	let maybe_deserialize_bytes_attr = attrs
		.iter()
		.find(|attr| attr.path().is_ident("deserialize_bytes"));
	let deserialize_bytes_attr = match maybe_deserialize_bytes_attr {
		Some(attr) => attr,
		None => {
			let generics_split = GenericsSplit::new(generics, Default::default());
			return Ok(generics_split);
		}
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
	let generics_split = GenericsSplit::new(generics, container_attributes.eval_generics);
	Ok(generics_split)
}

#[cfg(test)]
mod tests {
	use ::quote::quote;
	use proc_macro2::TokenStream;
	use syn::DeriveInput;

	use super::*;

	#[test]
	fn test_get_generics() {
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
			let GenericsSplit {
				impl_generics,
				type_generics,
				where_clause,
			} = get_generics(&input.attrs, &input.generics).unwrap();
			let name = input.ident;
			let impl_def = quote! {
				impl #impl_generics #name #type_generics #where_clause
			};
			assert_eq!(impl_def.to_string(), case.expected_impl_def.to_string());
		}
	}
}
