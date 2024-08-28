// Copyright 2024 Ulvetanna Inc.

use quote::{quote, ToTokens};
use syn::{bracketed, parse::Parse, parse_quote, spanned::Spanned, Token};

#[derive(Debug)]
pub(crate) struct ArithCircuitPolyItem {
	pub is_anonymous: bool,
	pub name: syn::Ident,
	pub poly: Vec<syn::Expr>,
}

impl ToTokens for ArithCircuitPolyItem {
	fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
		let Self {
			is_anonymous,
			name,
			poly,
			..
		} = self;
		let result = quote! {
			let #name = std::sync::Arc::new(binius_math::polynomial::ArithCircuitPoly::new(vec![ #( #poly ),* ]));
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

impl Parse for ArithCircuitPolyItem {
	fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
		let name = input.parse::<syn::Ident>();
		let is_anonymous = name.is_err();
		let name = name.unwrap_or(parse_quote!(UnnamedArithCircuitPoly));
		let vars: Vec<syn::Ident> = {
			let content;
			bracketed!(content in input);
			let vars = content.parse_terminated(syn::Ident::parse, Token![,])?;
			vars.into_iter().collect()
		};
		input.parse::<Token![=]>()?;
		let poly_packed = input.parse::<syn::Expr>()?;
		let mut poly = vec![];
		for i in 0usize..vars.len() {
			poly.push(parse_quote!(binius_math::polynomial::Expr::Var(#i)));
		}
		flatten_expr(&poly_packed, &vars, &mut poly)?;
		Ok(Self {
			is_anonymous,
			name,
			poly,
		})
	}
}

fn flatten_expr(
	expr: &syn::Expr,
	vars: &[syn::Ident],
	result: &mut Vec<syn::Expr>,
) -> Result<usize, syn::Error> {
	match expr.clone() {
		syn::Expr::Lit(_v) => {
			todo!()
		}
		syn::Expr::Path(p) => {
			for (i, var) in vars.iter().enumerate() {
				if p.path.is_ident(var) {
					return Ok(i);
				}
			}
			Err(syn::Error::new(expr.span(), "Unknown variable"))
		}
		syn::Expr::Paren(paren) => flatten_expr(&paren.expr, vars, result),
		syn::Expr::Binary(binary) => {
			let left = flatten_expr(&binary.left, vars, result)?;
			let right = flatten_expr(&binary.right, vars, result)?;
			match binary.op {
				syn::BinOp::Add(_) | syn::BinOp::Sub(_) => {
					result.push(parse_quote!(binius_math::polynomial::Expr::Add(#left, #right)));
					Ok(result.len() - 1)
				}
				syn::BinOp::Mul(_) => {
					result.push(parse_quote!(binius_math::polynomial::Expr::Mul(#left, #right)));
					Ok(result.len() - 1)
				}
				expr => Err(syn::Error::new(expr.span(), "Unsupported binop")),
			}
		}
		_ => {
			todo!()
		}
	}
}
