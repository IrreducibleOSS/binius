// Copyright 2024-2025 Irreducible Inc.

use quote::{quote, ToTokens};
use syn::{bracketed, parse::Parse, parse_quote, spanned::Spanned, Token};

#[derive(Debug)]
pub(crate) struct ArithExprItem(syn::Expr);

impl ToTokens for ArithExprItem {
	fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
		let Self(expr) = self;
		tokens.extend(quote!(#expr));
	}
}

impl Parse for ArithExprItem {
	fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
		let prefixed_field = input.parse::<syn::Path>().ok();
		let vars: Vec<syn::Ident> = {
			let content;
			bracketed!(content in input);
			let vars = content.parse_terminated(syn::Ident::parse, Token![,])?;
			vars.into_iter().collect()
		};
		input.parse::<Token![=]>()?;
		let mut expr = input.parse::<syn::Expr>()?;
		rewrite_expr(&mut expr, &vars, &prefixed_field)?;
		Ok(Self(expr))
	}
}

fn rewrite_expr(
	expr: &mut syn::Expr,
	vars: &[syn::Ident],
	prefixed_field: &Option<syn::Path>,
) -> Result<(), syn::Error> {
	let default_field = parse_quote!(binius_field::BinaryField1b);
	let field = prefixed_field.as_ref().unwrap_or(&default_field);
	match expr {
		syn::Expr::Path(path) => {
			let mut var_index = None;
			for (i, var) in vars.iter().enumerate() {
				if path.path.is_ident(var) {
					var_index = Some(i);
				}
			}
			if let Some(i) = var_index {
				*expr = parse_quote!(binius_math::ArithExpr::<#field>::Var(#i));
			} else {
				return Err(syn::Error::new(path.span(), "Unknown variable"));
			}
		}
		syn::Expr::Lit(exprlit) => {
			if let syn::Lit::Int(int) = &exprlit.lit {
				let value: syn::Expr = match &*int.to_string() {
					"0" => parse_quote!(binius_field::Field::ZERO),
					"1" => parse_quote!(binius_field::Field::ONE),
					_ => match prefixed_field {
						Some(field) => parse_quote!(#field::new(#int)),
						_ => return Err(syn::Error::new(expr.span(), "You need to specify an explicit field to use constants other than 0 or 1"))
					}
				};
				*expr = parse_quote!(binius_math::ArithExpr::<#field>::Const(#value));
			}
		}
		syn::Expr::Paren(paren) => {
			rewrite_expr(&mut paren.expr, vars, prefixed_field)?;
		}
		syn::Expr::Binary(binary) => {
			rewrite_expr(&mut binary.left, vars, prefixed_field)?;
			rewrite_expr(&mut binary.right, vars, prefixed_field)?;
		}
		_ => {}
	}
	Ok(())
}
