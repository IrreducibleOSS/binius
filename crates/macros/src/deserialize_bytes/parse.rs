use syn::{
	parse::Parse,
	punctuated::Punctuated,
	token::{Comma, Eq},
	Ident,
};

#[derive(Debug, Clone, Default)]
pub struct ContainerAttributes {
	pub eval_generics: Vec<GenericBinding>,
}

impl Parse for ContainerAttributes {
	fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
		let attr_name: Ident = input.parse()?;
		if attr_name.to_string() != "eval_generics" {
			return Err(syn::Error::new(attr_name.span(), "expected `eval_generics = \"...\"`"));
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
	pub _eq: Eq,
	pub to_generic: Ident,
}

impl Parse for GenericBinding {
	fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
		let from_generic = input.parse()?;
		let eq = input.parse()?;
		let to_generic = input.parse()?;
		Ok(Self {
			from_generic,
			_eq: eq,
			to_generic,
		})
	}
}
