// Copyright 2024-2025 Irreducible Inc.

use quote::{ToTokens, quote};
use syn::{Token, bracketed, parse::Parse, parse_quote, spanned::Spanned};

#[derive(Debug)]
pub(crate) struct CompositionPolyItem {
	pub is_anonymous: bool,
	pub name: syn::Ident,
	pub vars: Vec<syn::Ident>,
	pub poly_packed: syn::Expr,
	pub expr: syn::Expr,
	pub scalar_type: syn::Type,
	pub degree: usize,
}

impl ToTokens for CompositionPolyItem {
	fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
		let Self {
			is_anonymous,
			name,
			vars,
			poly_packed,
			expr,
			scalar_type,
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
			&|i| parse_quote!(unsafe {*batch_query.row(#i).get_unchecked(row)}),
		)
		.expect("Failed to substitute vars");

		let result = quote! {
			#[derive(Debug, Clone, Copy)]
			struct #name;

			impl<P> binius_math::CompositionPoly<P> for #name
			where
				P: binius_field::PackedField<Scalar: binius_field::ExtensionField<#scalar_type>>,
			{
				fn n_vars(&self) -> usize {
					#n_vars
				}

				fn degree(&self) -> usize {
					#degree
				}

				fn binary_tower_level(&self) -> usize {
					0
				}

				fn expression(&self) -> binius_math::ArithCircuit<P::Scalar> {
					binius_math::ArithCircuit::from(#expr).convert_field()
				}

				fn evaluate(&self, query: &[P]) -> Result<P, binius_math::Error> {
					if query.len() != #n_vars {
						return Err(binius_math::Error::IncorrectQuerySize { expected: #n_vars });
					}
					Ok(#eval_single)
				}

				fn batch_evaluate(
					&self,
					batch_query: &binius_math::RowsBatchRef<P>,
					evals: &mut [P],
				) -> Result<(), binius_math::Error> {
					if batch_query.row_len() != #n_vars {
						return Err(binius_math::Error::IncorrectQuerySize { expected: #n_vars });
					}

					for row in 0..batch_query.row(0).len() {
						evals[row] = #eval_batch;
					}

					Ok(())
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
		let name = name.unwrap_or_else(|_| parse_quote!(UnnamedCompositionPoly));
		let vars = {
			let content;
			bracketed!(content in input);
			let vars = content.parse_terminated(syn::Ident::parse, Token![,])?;
			vars.into_iter().collect::<Vec<_>>()
		};
		input.parse::<Token![=]>()?;
		let mut poly_packed = input.parse::<syn::Expr>()?;
		let mut expr = poly_packed.clone();

		let degree = poly_degree(&poly_packed)?;
		rewrite_literals(&mut poly_packed, &replace_packed_literals)?;

		subst_vars(&mut expr, &vars, &|i| parse_quote!(binius_math::ArithExpr::Var(#i)))?;
		rewrite_literals(&mut expr, &replace_expr_literals)?;

		let scalar_type = if input.is_empty() {
			parse_quote!(binius_field::BinaryField1b)
		} else {
			input.parse::<Token![,]>()?;

			input.parse()?
		};

		Ok(Self {
			is_anonymous,
			name,
			vars,
			poly_packed,
			expr,
			scalar_type,
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

/// Replace literals to P::zero() and P::one() to be used in `evaluate` and `batch_evaluate`.
fn replace_packed_literals(literal: &syn::LitInt) -> Result<syn::Expr, syn::Error> {
	Ok(match &*literal.to_string() {
		"0" => parse_quote!(P::zero()),
		"1" => parse_quote!(P::one()),
		_ => return Err(syn::Error::new(literal.span(), "Unsupported integer")),
	})
}

/// Replace literals to Expr::zero() and Expr::one() to be used in `expression` method.
fn replace_expr_literals(literal: &syn::LitInt) -> Result<syn::Expr, syn::Error> {
	Ok(match &*literal.to_string() {
		"0" => parse_quote!(binius_math::ArithExpr::zero()),
		"1" => parse_quote!(binius_math::ArithExpr::one()),
		_ => return Err(syn::Error::new(literal.span(), "Unsupported integer")),
	})
}

/// Replace literals in an expression
fn rewrite_literals(
	expr: &mut syn::Expr,
	f: &impl Fn(&syn::LitInt) -> Result<syn::Expr, syn::Error>,
) -> Result<(), syn::Error> {
	match expr {
		syn::Expr::Lit(exprlit) => {
			if let syn::Lit::Int(int) = &exprlit.lit {
				*expr = f(int)?;
			}
		}
		syn::Expr::Paren(paren) => {
			rewrite_literals(&mut paren.expr, f)?;
		}
		syn::Expr::Binary(binary) => {
			rewrite_literals(&mut binary.left, f)?;
			rewrite_literals(&mut binary.right, f)?;
		}
		_ => {}
	}
	Ok(())
}

/// Substitutes variables in an expression with a slice access
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
