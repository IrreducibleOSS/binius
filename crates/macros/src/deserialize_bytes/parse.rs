// Copyright 2024-2025 Irreducible Inc.

use syn::{
	Ident, TypePath,
	parse::Parse,
	punctuated::Punctuated,
	token::{Comma, Eq},
};

/// Container attributes for the `deserialize_bytes` derive macro.
///
/// ## Example
/// ```
/// use binius_macros::DeserializeBytes;
///
/// #[derive(DeserializeBytes)]
/// #[deserialize_bytes(eval_generics(X = binius_field::BinaryField128b))]
/// struct MyStruct<X>(X);
/// ```
///
/// Will have a list of eval_generics (from_generic = X, to_generic = BinaryField128)
#[derive(Debug, Clone, Default)]
pub struct ContainerAttributes {
	pub eval_generics: Vec<GenericBinding>,
}

impl Parse for ContainerAttributes {
	fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
		let attr_name: Ident = input.parse()?;
		if attr_name != "eval_generics" {
			return Err(syn::Error::new(attr_name.span(), "expected `eval_generics(X = Y, ...)`"));
		}
		let parens_content;
		syn::parenthesized!(parens_content in input);
		let eval_generics = Punctuated::<GenericBinding, Comma>::parse_terminated(&parens_content)?;

		Ok(Self {
			eval_generics: eval_generics.into_iter().collect(),
		})
	}
}

#[derive(Debug, Clone)]
pub struct GenericBinding {
	pub from_generic: Ident,
	pub to_generic: TypePath,
}

impl Parse for GenericBinding {
	fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
		let from_generic = input.parse()?;
		let _eq: Eq = input.parse()?;
		let to_generic = input.parse()?;
		Ok(Self {
			from_generic,
			to_generic,
		})
	}
}

#[cfg(test)]
mod tests {
	use proc_macro2::TokenStream;
	use quote::quote;

	use super::*;

	#[test]
	fn test_parse_container_attributes() {
		struct Case {
			input: TokenStream,
			expected: Vec<(&'static str, &'static str)>,
		}
		let cases = vec![
			Case {
				input: quote! {
					eval_generics(X = crate::module::T)
				},
				expected: vec![("X", "crate::module::T")],
			},
			Case {
				input: quote! {
					eval_generics(X = crate::module::T, Y = U)
				},
				expected: vec![("X", "crate::module::T"), ("Y", "U")],
			},
		];
		for case in cases {
			let input = case.input;
			let result = syn::parse2::<ContainerAttributes>(input).unwrap();
			assert_eq!(result.eval_generics.len(), case.expected.len());
			for (i, binding) in result.eval_generics.iter().enumerate() {
				assert_eq!(binding.from_generic.to_string(), case.expected[i].0);
				assert_eq!(display_type_path(&binding.to_generic), case.expected[i].1);
			}
		}
	}

	#[test]
	fn test_parse_generic_binding() {
		let input = quote! {
			X = crate::module::T
		};
		let result = syn::parse2::<GenericBinding>(input).unwrap();
		assert_eq!(result.from_generic.to_string(), "X");
		assert_eq!(display_type_path(&result.to_generic), "crate::module::T");
	}

	fn display_type_path(input: &TypePath) -> String {
		input
			.path
			.segments
			.iter()
			.map(|segment| segment.ident.to_string())
			.collect::<Vec<_>>()
			.join("::")
	}
}
