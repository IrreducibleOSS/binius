extern crate proc_macro;

use proc_macro::TokenStream;
use quote::quote;
use syn::{bracketed, parse::Parse, parse_macro_input, parse_quote, Token};

#[proc_macro]
pub fn composition_poly(input: TokenStream) -> TokenStream {
	let CompositionPolyItem {
		name,
		vars,
		mut poly,
	} = parse_macro_input!(input);

	let n_vars = vars.len();
	let i = 0..n_vars;
	let degree = poly_degree(&poly);
	rewrite_literals(&mut poly);

	quote! {
		#[derive(Debug, Clone)]
		struct #name;

		impl<F: binius_field::Field> binius_core::polynomial::multivariate::CompositionPoly<F> for #name {
			fn n_vars(&self) -> usize {
				#n_vars
			}

			fn degree(&self) -> usize {
				#degree
			}

			fn evaluate<P:PackedField<Scalar=F>>(&self, query: &[P]) -> Result<P, binius_core::polynomial::Error> {
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
	}
	.into()
}

/// Make sure to run this before rewrite_literals as it will rewrite Lit to Path,
/// which will mess up the degree
fn poly_degree(expr: &syn::Expr) -> usize {
	match expr.clone() {
		syn::Expr::Lit(_) => 0,
		syn::Expr::Path(_) => 1,
		syn::Expr::Paren(paren) => poly_degree(&paren.expr),
		syn::Expr::Binary(binary) => {
			let op = binary.op;
			let left = poly_degree(&binary.left);
			let right = poly_degree(&binary.right);
			match op {
				syn::BinOp::Add(_) | syn::BinOp::Sub(_) => std::cmp::max(left, right),
				syn::BinOp::Mul(_) => left + right,
				_ => panic!("Binary operation is not supported: {}", quote! { #op }),
			}
		}
		_ => panic!("Unsupported expression: `{}`", quote! { #expr }),
	}
}

/// Rewrites 0 => F::ZERO, 1 => F::ONE
fn rewrite_literals(expr: &mut syn::Expr) {
	match expr {
		syn::Expr::Lit(exprlit) => {
			if let syn::Lit::Int(int) = &exprlit.lit {
				match &*int.to_string() {
					"0" => {
						*expr = parse_quote!(P::zero());
					}
					"1" => {
						*expr = parse_quote!(P::one());
					}
					int => panic!("Value not supported: {int}"),
				}
			}
		}
		syn::Expr::Paren(paren) => {
			rewrite_literals(&mut paren.expr);
		}
		syn::Expr::Binary(binary) => {
			rewrite_literals(&mut binary.left);
			rewrite_literals(&mut binary.right);
		}
		_ => {}
	}
}

#[derive(Debug)]
struct CompositionPolyItem {
	name: syn::Ident,
	vars: Vec<syn::Ident>,
	poly: syn::Expr,
}

impl Parse for CompositionPolyItem {
	fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
		Ok(Self {
			name: input.parse::<syn::Ident>()?,
			vars: {
				let content;
				bracketed!(content in input);
				let vars = content.parse_terminated(syn::Ident::parse, Token![,])?;
				input.parse::<Token![=]>()?;
				vars.into_iter().collect()
			},
			poly: input.parse::<syn::Expr>()?,
		})
	}
}
