// Copyright 2024 Irreducible Inc.

use quote::{quote, ToTokens};
use syn::{bracketed, parse::Parse, parse_quote, spanned::Spanned, Token};

#[derive(Debug)]
pub(crate) struct CompositionPolyItem {
	pub is_anonymous: bool,
	pub name: syn::Ident,
	pub vars: Vec<syn::Ident>,
	pub poly_packed: syn::Expr,
	pub degree: usize,
}

impl ToTokens for CompositionPolyItem {
	fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
		let Self {
			is_anonymous,
			name,
			vars,
			poly_packed,
			degree,
		} = self;
		let n_vars = vars.len();

		let mut eval_single = poly_packed.clone();
		subst_vars(&mut eval_single, vars, &|i| parse_quote!(unsafe {*query.get_unchecked(#i)}))
			.expect("Failed to substitute vars");

		let mut eval_batch = poly_packed.clone();
		subst_vars(
			&mut eval_batch,
			vars,
			&|i| parse_quote!(unsafe {*sparse_batch_query.get_unchecked(#i).get_unchecked(row)}),
		)
		.expect("Failed to substitute vars");

		let result = quote! {
			#[derive(Debug, Clone, Copy)]
			struct #name;

			impl<P: binius_field::PackedField> binius_math::CompositionPoly<P> for #name {
				fn n_vars(&self) -> usize {
					#n_vars
				}

				fn degree(&self) -> usize {
					#degree
				}

				fn evaluate(&self, query: &[P]) -> Result<P, binius_math::Error> {
					if query.len() != #n_vars {
						return Err(binius_math::Error::IncorrectQuerySize { expected: #n_vars });
					}
					Ok(#eval_single)
				}

				fn batch_evaluate(
					&self,
					sparse_batch_query: &[&[P]],
					evals: &mut [P],
				) -> Result<(), binius_math::Error> {
					if sparse_batch_query.len() != #n_vars {
						return Err(binius_math::Error::IncorrectQuerySize { expected: #n_vars });
					}

					for col in 1..sparse_batch_query.len() {
						if sparse_batch_query[col].len() != sparse_batch_query[0].len() {
							return Err(binius_math::Error::BatchEvaluateSizeMismatch);
						}
					}

					for row in 0..sparse_batch_query[0].len() {
						evals[row] = #eval_batch;
					}

					Ok(())
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
		let mut poly_packed = input.parse::<syn::Expr>()?;
		let degree = poly_degree(&poly_packed)?;
		rewrite_literals(&mut poly_packed)?;
		Ok(Self {
			is_anonymous,
			name,
			vars,
			poly_packed,
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
					"0" => {
						parse_quote!(P::zero())
					}
					"1" => {
						parse_quote!(P::one())
					}
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

fn subst_vars(
	expr: &mut syn::Expr,
	vars: &[syn::Ident],
	f: &impl Fn(usize) -> syn::Expr,
) -> Result<(), syn::Error> {
	match expr {
		syn::Expr::Path(p) => {
			for (i, var) in vars.iter().enumerate() {
				if p.path.is_ident(var) {
					*expr = f(i);
					return Ok(());
				}
			}
			Err(syn::Error::new(p.span(), "unknown variable"))
		}
		syn::Expr::Paren(paren) => subst_vars(&mut paren.expr, vars, f),
		syn::Expr::Binary(binary) => {
			subst_vars(&mut binary.left, vars, f)?;
			subst_vars(&mut binary.right, vars, f)
		}
		_ => Ok(()),
	}
}
