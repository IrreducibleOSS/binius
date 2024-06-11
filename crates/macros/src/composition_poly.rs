// Copyright 2024 Ulvetanna Inc.

use quote::{quote, ToTokens};
use syn::{bracketed, parse::Parse, parse_quote, spanned::Spanned, Token};

#[derive(Debug)]
pub(crate) struct CompositionPolyItem {
	pub is_anonymous: bool,
	pub name: syn::Ident,
	pub vars: Vec<syn::Ident>,
	pub poly: syn::Expr,
	pub degree: usize,
}

impl ToTokens for CompositionPolyItem {
	fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
		let Self {
			is_anonymous,
			name,
			vars,
			poly,
			degree,
		} = self;
		let n_vars = vars.len();
		let i = (0..n_vars).collect::<Vec<_>>();

		let result = quote! {
			#[derive(Debug, Clone, Copy)]
			struct #name;

			impl<F: binius_field::Field> binius_core::polynomial::multivariate::CompositionPoly<F> for #name {
				fn n_vars(&self) -> usize {
					#n_vars
				}

				fn degree(&self) -> usize {
					#degree
				}

				fn evaluate<P: binius_field::PackedField<Scalar=F>>(&self, query: &[P]) -> Result<P, binius_core::polynomial::Error> {
					if query.len() != #n_vars {
						return Err(binius_core::polynomial::Error::IncorrectQuerySize { expected: #n_vars });
					}
					#( let #vars = query[#i]; )*
					Ok(#poly)
				}

				fn binary_tower_level(&self) -> usize {
					0
				}
			}
		};

		if *is_anonymous {
			// In this case we return an instance of our struct rather
			// than defining the struct within the current scope
			tokens.extend(quote! {
				{
					#result
					#name
				}
			});
		} else {
			tokens.extend(result);
		}
	}
}

impl Parse for CompositionPolyItem {
	fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
		let name = input.parse::<syn::Ident>();
		let is_anonymous = name.is_err();
		let name = name.unwrap_or(parse_quote!(UnnamedCompositionPoly));
		let vars = {
			let content;
			bracketed!(content in input);
			let vars = content.parse_terminated(syn::Ident::parse, Token![,])?;
			vars.into_iter().collect()
		};
		input.parse::<Token![=]>()?;
		let mut poly = input.parse::<syn::Expr>()?;
		let degree = poly_degree(&poly)?;
		rewrite_literals(&mut poly)?;
		Ok(Self {
			is_anonymous,
			name,
			vars,
			poly,
			degree,
		})
	}
}

/// Make sure to run this before rewrite_literals as it will rewrite Lit to Path,
/// which will mess up the degree
fn poly_degree(expr: &syn::Expr) -> Result<usize, syn::Error> {
	Ok(match expr.clone() {
		syn::Expr::Lit(_) => 0,
		syn::Expr::Path(_) => 1,
		syn::Expr::Paren(paren) => poly_degree(&paren.expr)?,
		syn::Expr::Binary(binary) => {
			let op = binary.op;
			let left = poly_degree(&binary.left)?;
			let right = poly_degree(&binary.right)?;
			match op {
				syn::BinOp::Add(_) | syn::BinOp::Sub(_) => std::cmp::max(left, right),
				syn::BinOp::Mul(_) => left + right,
				expr => {
					return Err(syn::Error::new(expr.span(), "Unsupported binop"));
				}
			}
		}
		expr => return Err(syn::Error::new(expr.span(), "Unsupported expression")),
	})
}

/// Rewrites 0 => P::zero(), 1 => P::one()
fn rewrite_literals(expr: &mut syn::Expr) -> Result<(), syn::Error> {
	match expr {
		syn::Expr::Lit(exprlit) => {
			if let syn::Lit::Int(int) = &exprlit.lit {
				*expr = match &*int.to_string() {
					"0" => parse_quote!(P::zero()),
					"1" => parse_quote!(P::one()),
					_ => return Err(syn::Error::new(expr.span(), "Unsupported integer")),
				};
			}
		}
		syn::Expr::Paren(paren) => {
			rewrite_literals(&mut paren.expr)?;
		}
		syn::Expr::Binary(binary) => {
			rewrite_literals(&mut binary.left)?;
			rewrite_literals(&mut binary.right)?;
		}
		_ => {}
	}
	Ok(())
}
