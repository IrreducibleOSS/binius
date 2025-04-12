use std::collections::{HashMap, HashSet};

use quote::{quote, ToTokens};
use syn::{punctuated::Punctuated, token::Comma, GenericParam, Generics, Type, WherePredicate};

use super::parse::GenericBinding;

#[derive(Debug, Clone)]
pub struct GenericsSplit<'gen> {
	pub impl_generics: ImplGenerics<'gen>,
	pub type_generics: TypeGenerics<'gen>,
	pub where_clause: WhereClause<'gen>,
}

impl GenericsSplit<'_> {
	pub fn new<'gen>(
		generics: &'gen Generics,
		eval_generics: Vec<GenericBinding>,
	) -> GenericsSplit<'gen> {
		let impl_generics = ImplGenerics::new(generics, eval_generics.clone());
		let type_generics = TypeGenerics::new(generics, eval_generics.clone());
		let where_clause = WhereClause::new(generics, eval_generics);
		GenericsSplit {
			impl_generics,
			type_generics,
			where_clause,
		}
	}
}

#[derive(Debug, Clone)]
pub struct ImplGenerics<'gen> {
	generics: &'gen Generics,
	eval_generics: Vec<GenericBinding>,
}

impl ImplGenerics<'_> {
	pub fn new(generics: &Generics, eval_generics: Vec<GenericBinding>) -> ImplGenerics {
		ImplGenerics {
			generics,
			eval_generics,
		}
	}
}

impl<'gen> ToTokens for ImplGenerics<'gen> {
	fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
		let from_param_names = self
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

		if impl_generics_params.is_empty() {
			return;
		}

		tokens.extend(quote! {<#impl_generics_params>});
	}
}

#[derive(Debug, Clone)]
pub struct TypeGenerics<'gen> {
	generics: &'gen Generics,
	eval_generics: Vec<GenericBinding>,
}

impl TypeGenerics<'_> {
	pub fn new(generics: &Generics, eval_generics: Vec<GenericBinding>) -> TypeGenerics {
		TypeGenerics {
			generics,
			eval_generics,
		}
	}
}

impl<'gen> ToTokens for TypeGenerics<'gen> {
	fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
		let mut eval_params = HashMap::new();
		for binding in self.eval_generics.iter() {
			let key = binding.from_generic.to_string();
			eval_params.entry(key).or_insert(&binding.to_generic);
		}

		quote! {<}.to_tokens(tokens);
		for (index, param) in self.generics.params.iter().enumerate() {
			if index != 0 {
				quote! {,}.to_tokens(tokens);
			}
			if let GenericParam::Type(type_param) = param {
				match eval_params.get(&type_param.ident.to_string()) {
					Some(to_param) => {
						to_param.to_tokens(tokens);
					}
					None => type_param.ident.to_tokens(tokens),
				}
				continue;
			}
			param.to_tokens(tokens);
		}
		quote! {>}.to_tokens(tokens);
	}
}

#[derive(Debug, Clone)]
pub struct WhereClause<'gen> {
	generics: &'gen Generics,
	eval_generics: Vec<GenericBinding>,
}

impl WhereClause<'_> {
	pub fn new(generics: &Generics, eval_generics: Vec<GenericBinding>) -> WhereClause {
		WhereClause {
			generics,
			eval_generics,
		}
	}
}

impl<'gen> ToTokens for WhereClause<'gen> {
	fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
		let where_clause = match self.generics.where_clause {
			Some(ref where_clause) => where_clause.clone(),
			None => return,
		};
		let from_param_names = self
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

#[cfg(test)]
mod tests {
	use ::quote::quote;
	use proc_macro2::TokenStream;
	use syn::ItemStruct;

	use crate::deserialize_bytes::parse::ContainerAttributes;

	#[test]
	fn test_generics() {
		struct Case {
			struct_def: TokenStream,
			attrubutes_def: TokenStream,
			expected_impl_def: TokenStream,
		}
		let cases = vec![
			Case {
				struct_def: quote! {
					pub struct MultilinearOracleSet<F: TowerField> {
						oracles: Vec<MultilinearPolyOracle<F>>
					}
				},
				attrubutes_def: quote! {
					eval_generics(F = BinaryField128b)
				},
				expected_impl_def: quote! {
					impl MultilinearOracleSet<BinaryField128b>
				},
			},
			Case {
				struct_def: quote! {
					struct MyStruct<T: Field, U, V: Oracle>
						where T: Debug, U: Clone
						{
							field: T,
						}
				},
				attrubutes_def: quote! {
					eval_generics(T = B128)
				},
				expected_impl_def: quote! {
					impl<U, V: Oracle> MyStruct<B128, U, V>
						where U: Clone
				},
			},
			Case {
				struct_def: quote! {
					struct MyStruct<T: Field, U, V: Oracle>
						where T: Debug, U: Clone
						{
							field: T,
						}
				},
				attrubutes_def: quote! {
					eval_generics(T = B128, V = B256)
				},
				expected_impl_def: quote! {
					impl<U> MyStruct<B128, U, B256>
					where U: Clone
				},
			},
			Case {
				struct_def: quote! {
					struct MyStruct<T: Field, U, V: Oracle>
						where T: Debug, U: Clone
						{
							field: T,
						}
				},
				attrubutes_def: quote! {
					eval_generics(U = B128)
				},
				expected_impl_def: quote! {
					impl<T: Field, V: Oracle> MyStruct<T, B128, V>
					where T: Debug
				},
			},
		];
		for case in cases {
			let struct_def =
				syn::parse2::<ItemStruct>(case.struct_def).expect("Failed to parse struct");
			let struct_name = struct_def.ident;
			let container_attributes =
				syn::parse2::<ContainerAttributes>(case.attrubutes_def).unwrap();
			let GenericsSplit {
				impl_generics,
				type_generics,
				where_clause,
			} = GenericsSplit::new(&struct_def.generics, container_attributes.eval_generics);
			let impl_def = quote! {
				impl #impl_generics #struct_name #type_generics
					#where_clause
			};
			assert_eq!(impl_def.to_string(), case.expected_impl_def.to_string());
		}
	}
}
