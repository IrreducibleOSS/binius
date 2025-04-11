use std::collections::{HashMap, HashSet};

use quote::{quote, ToTokens};
use syn::{
	parse::Parse,
	punctuated::Punctuated,
	token::{Comma, Eq, Lt},
	Attribute, GenericParam, Generics, Ident, ImplGenerics, Meta, MetaList, PredicateType, Type,
	TypeGenerics, TypeParam, WhereClause, WherePredicate,
};

pub fn apply_container_attributes<'attr, 'gen>(
	attrs: &'attr [Attribute],
	generics: &'gen Generics,
) -> syn::Result<(
	DeserializeBytesImplGenerics<'gen>,
	DeserializeBytesTypeGenerics<'gen>,
	DeserializeBytesWhereClause<'gen>,
)> {
	let maybe_deserialize_bytes_attr = attrs
		.iter()
		.find(|attr| attr.path().is_ident("deserialize_bytes"));
	let deserialize_bytes_attr = match maybe_deserialize_bytes_attr {
		Some(attr) => attr,
		None => {
			let impl_generics = DeserializeBytesImplGenerics {
				container_attributes: Default::default(),
				generics,
			};
			let type_generics = DeserializeBytesTypeGenerics {
				container_attributes: Default::default(),
				generics,
			};
			let where_clause = DeserializeBytesWhereClause {
				container_attributes: Default::default(),
				generics,
			};

			return Ok((impl_generics, type_generics, where_clause));
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
	let impl_generics = DeserializeBytesImplGenerics {
		container_attributes: container_attributes.clone(),
		generics,
	};
	let type_generics = DeserializeBytesTypeGenerics {
		container_attributes: container_attributes.clone(),
		generics,
	};
	let where_clause = DeserializeBytesWhereClause {
		container_attributes,
		generics,
	};
	Ok((impl_generics, type_generics, where_clause))
}

#[derive(Debug, Clone)]
struct DeserializeBytesImplGenerics<'gen> {
	container_attributes: ContainerAttributes,
	generics: &'gen Generics,
}

impl<'gen> ToTokens for DeserializeBytesImplGenerics<'gen> {
	fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
		let from_param_names = self
			.container_attributes
			.eval_generics
			.iter()
			.map(|binding| binding.from_generic.to_string())
			.collect::<HashSet<_>>();
		let mut impl_generics_params = Punctuated::<GenericParam, Comma>::new();
		for param in self.generics.params.iter() {
			if let GenericParam::Type(type_param) = param {
				if from_param_names.contains(&type_param.ident.to_string()) {
					continue;
				}
			}
			impl_generics_params.push(param.clone());
		}
		tokens.extend(quote! {<#impl_generics_params>});
	}
}

#[derive(Debug, Clone)]
struct DeserializeBytesTypeGenerics<'gen> {
	container_attributes: ContainerAttributes,
	generics: &'gen Generics,
}

impl<'gen> ToTokens for DeserializeBytesTypeGenerics<'gen> {
	fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
		let mut eval_params = HashMap::new();
		for binding in self.container_attributes.eval_generics.iter() {
			let key = binding.from_generic.to_string();
			eval_params.entry(key).or_insert(&binding.to_generic);
		}
		let mut type_generics_params = Punctuated::<GenericParam, Comma>::new();
		for param in self.generics.params.iter() {
			if let GenericParam::Type(type_param) = param {
				if let Some(&to_param) = eval_params.get(&type_param.ident.to_string()) {
					let generic_param = GenericParam::Type(TypeParam {
						ident: to_param.clone(),
						attrs: Default::default(),
						bounds: Default::default(),
						colon_token: None,
						default: None,
						eq_token: None,
					});
					type_generics_params.push(generic_param);
					continue;
				}
			}
			type_generics_params.push(param.clone());
		}
		tokens.extend(quote! {<#type_generics_params>});
	}
}

#[derive(Debug, Clone)]
struct DeserializeBytesWhereClause<'gen> {
	container_attributes: ContainerAttributes,
	generics: &'gen Generics,
}

impl<'gen> ToTokens for DeserializeBytesWhereClause<'gen> {
	fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
		let where_clause = match self.generics.where_clause {
			Some(ref where_clause) => where_clause.clone(),
			None => return,
		};
		let from_param_names = self
			.container_attributes
			.eval_generics
			.iter()
			.map(|binding| binding.from_generic.to_string())
			.collect::<HashSet<_>>();

		let mut where_clause_params = Punctuated::<WherePredicate, Comma>::new();
		for predicate in where_clause.predicates.iter() {
			if let WherePredicate::Type(where_type) = predicate {
				if let Type::Path(ref bounded_ty) = where_type.bounded_ty {
					let bounded_ty_name = bounded_ty.to_token_stream().to_string();
					if from_param_names.contains(&bounded_ty_name) {
						continue;
					}
				}
			}
			where_clause_params.push(predicate.clone());
		}
		tokens.extend(quote! {where #where_clause_params});
	}
}

#[derive(Debug, Clone, Default)]
struct ContainerAttributes {
	eval_generics: Vec<GenericBinding>,
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
struct GenericBinding {
	from_generic: Ident,
	eq: Eq,
	to_generic: Ident,
}

impl Parse for GenericBinding {
	fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
		let from_generic = input.parse()?;
		let eq = input.parse()?;
		let to_generic = input.parse()?;
		Ok(Self {
			from_generic,
			eq,
			to_generic,
		})
	}
}
