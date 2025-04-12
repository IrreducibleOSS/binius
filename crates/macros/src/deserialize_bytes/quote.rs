use std::collections::{HashMap, HashSet};

use quote::{quote, ToTokens};
use syn::{
	punctuated::Punctuated,
	token::{Comma, Gt, Lt},
	GenericParam, Generics, Type, TypeParam, WherePredicate,
};

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
				if let Some(&to_param) = eval_params.get(&type_param.ident.to_string()) {
					to_param.to_tokens(tokens);
					continue;
				}
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
