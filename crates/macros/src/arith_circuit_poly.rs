// Copyright 2024 Irreducible Inc.

use quote::{quote, ToTokens};
use syn::{bracketed, parse::Parse, parse_quote, spanned::Spanned, Token};

#[derive(Debug)]
pub(crate) struct ArithCircuitPolyItem {
	pub poly: syn::Expr,
}

impl ToTokens for ArithCircuitPolyItem {
	fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
		let Self { poly, .. } = self;
		tokens.extend(quote! {
			{
				use binius_field::Field;
				use binius_core::polynomial::Expr;

				std::sync::Arc::new(binius_core::polynomial::ArithCircuitPoly::new(#poly))
			}
		});
	}
}

impl Parse for ArithCircuitPolyItem {
	fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
		let vars: Vec<syn::Ident> = {
			let content;
			bracketed!(content in input);
			let vars = content.parse_terminated(syn::Ident::parse, Token![,])?;
			vars.into_iter().collect()
		};
		input.parse::<Token![=]>()?;
		let poly_packed = input.parse::<syn::Expr>()?;
		let poly = flatten_expr(&poly_packed, &vars)?;
		Ok(Self { poly })
	}
}

fn flatten_expr(expr: &syn::Expr, vars: &[syn::Ident]) -> Result<syn::Expr, syn::Error> {
	match expr.clone() {
		syn::Expr::Lit(exprlit) => {
			if let syn::Lit::Int(int) = &exprlit.lit {
				match &*int.to_string() {
					"0" => Ok(parse_quote!(Expr::Const(Field::ZERO))),
					"1" => Ok(parse_quote!(Expr::Const(Field::ONE))),
					_ => Err(syn::Error::new(expr.span(), "Unsupported integer")),
				}
			} else {
				Err(syn::Error::new(expr.span(), "Unsupported literal"))
			}
		}
		syn::Expr::Path(p) => {
			for (i, var) in vars.iter().enumerate() {
				if p.path.is_ident(var) {
					return Ok(parse_quote!(Expr::Var(#i)));
				}
			}
			Err(syn::Error::new(expr.span(), "Unknown variable"))
		}
		syn::Expr::Paren(paren) => flatten_expr(&paren.expr, vars),
		syn::Expr::Binary(binary) => {
			let left = flatten_expr(&binary.left, vars)?;
			let right = flatten_expr(&binary.right, vars)?;
			match binary.op {
				syn::BinOp::Add(_) | syn::BinOp::Sub(_) => Ok(parse_quote!((#left + #right))),
				syn::BinOp::Mul(_) => Ok(parse_quote!((#left * #right))),
				expr => Err(syn::Error::new(expr.span(), "Unsupported binop")),
			}
		}
		_ => {
			todo!()
		}
	}
}
