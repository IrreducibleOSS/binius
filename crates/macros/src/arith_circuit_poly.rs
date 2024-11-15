// Copyright 2024 Irreducible Inc.

use quote::{quote, ToTokens};
use syn::{bracketed, parse::Parse, parse_quote, spanned::Spanned, Token};

use crate::composition_poly::CompositionPolyItem;

#[derive(Debug)]
pub(crate) struct ArithCircuitPolyItem {
	poly: syn::Expr,
	/// We create a composition poly to cache the efficient evaluation implementations
	/// for the known packed field types.
	composition_poly: CompositionPolyItem,
	field_name: syn::Ident,
}

impl ToTokens for ArithCircuitPolyItem {
	fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
		let Self {
			poly,
			composition_poly,
			field_name,
		} = self;

		let mut register_cached_impls = proc_macro2::TokenStream::new();
		let packed_extensions = get_packed_extensions(field_name);
		if packed_extensions.is_empty() {
			register_cached_impls.extend(quote! { result });
		} else {
			register_cached_impls.extend(quote! (
				let mut cached = binius_core::polynomial::CachedPoly::new(composition);

			));

			for packed_extension in get_packed_extensions(field_name) {
				register_cached_impls.extend(quote! {
					cached.register::<binius_field::#packed_extension>(composition.clone());
				});
			}

			register_cached_impls.extend(quote! {
				cached
			});
		}

		tokens.extend(quote! {
			{
				use binius_field::Field;
				use binius_core::polynomial::Expr;

				let mut result = binius_core::polynomial::ArithCircuitPoly::<binius_field::#field_name>::new(#poly);
				let composition = #composition_poly;

				#register_cached_impls
			}
		});
	}
}

impl Parse for ArithCircuitPolyItem {
	fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
		let original_tokens = input.fork();
		let vars: Vec<syn::Ident> = {
			let content;
			bracketed!(content in input);
			let vars = content.parse_terminated(syn::Ident::parse, Token![,])?;
			vars.into_iter().collect()
		};
		input.parse::<Token![=]>()?;
		let poly_packed = input.parse::<syn::Expr>()?;
		let poly = flatten_expr(&poly_packed, &vars)?;

		input.parse::<Token![,]>()?;

		let field_name = input.parse()?;
		// Here we assume that the `composition_poly` shares the expression syntax with the `arithmetic_circuit_poly`.
		let composition_poly = CompositionPolyItem::parse(&original_tokens)?;

		Ok(Self {
			poly,
			composition_poly,
			field_name,
		})
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

/// Map the field name to all packed extensions that are relevant for the field.
/// Every time when we are adding/removing a new field or packed field type, we need to update this function.
fn get_packed_extensions(ident: &syn::Ident) -> Vec<syn::Ident> {
	match ident.to_string().as_str() {
		"BinaryField1b" => vec![
			parse_quote!(PackedBinaryField1x1b),
			parse_quote!(PackedBinaryField2x1b),
			parse_quote!(PackedBinaryField4x1b),
			parse_quote!(PackedBinaryField8x1b),
			parse_quote!(PackedBinaryField16x1b),
			parse_quote!(PackedBinaryField32x1b),
			parse_quote!(PackedBinaryField64x1b),
			parse_quote!(PackedBinaryField128x1b),
			parse_quote!(PackedBinaryField256x1b),
			parse_quote!(PackedBinaryField512x1b),
			parse_quote!(PackedBinaryField1x2b),
			parse_quote!(PackedBinaryField2x2b),
			parse_quote!(PackedBinaryField4x2b),
			parse_quote!(PackedBinaryField8x2b),
			parse_quote!(PackedBinaryField16x2b),
			parse_quote!(PackedBinaryField32x2b),
			parse_quote!(PackedBinaryField64x2b),
			parse_quote!(PackedBinaryField128x2b),
			parse_quote!(PackedBinaryField256x2b),
			parse_quote!(PackedBinaryField1x4b),
			parse_quote!(PackedBinaryField2x4b),
			parse_quote!(PackedBinaryField4x4b),
			parse_quote!(PackedBinaryField8x4b),
			parse_quote!(PackedBinaryField16x4b),
			parse_quote!(PackedBinaryField32x4b),
			parse_quote!(PackedBinaryField64x4b),
			parse_quote!(PackedBinaryField128x4b),
			parse_quote!(PackedBinaryField1x8b),
			parse_quote!(PackedBinaryField2x8b),
			parse_quote!(PackedBinaryField4x8b),
			parse_quote!(PackedBinaryField8x8b),
			parse_quote!(PackedBinaryField16x8b),
			parse_quote!(PackedBinaryField32x8b),
			parse_quote!(PackedBinaryField64x8b),
			parse_quote!(PackedBinaryField1x16b),
			parse_quote!(PackedBinaryField2x16b),
			parse_quote!(PackedBinaryField4x16b),
			parse_quote!(PackedBinaryField8x16b),
			parse_quote!(PackedBinaryField16x16b),
			parse_quote!(PackedBinaryField32x16b),
			parse_quote!(PackedBinaryField1x32b),
			parse_quote!(PackedBinaryField2x32b),
			parse_quote!(PackedBinaryField4x32b),
			parse_quote!(PackedBinaryField8x32b),
			parse_quote!(PackedBinaryField16x32b),
			parse_quote!(PackedBinaryField1x64b),
			parse_quote!(PackedBinaryField2x64b),
			parse_quote!(PackedBinaryField4x64b),
			parse_quote!(PackedBinaryField8x64b),
			parse_quote!(PackedBinaryField1x128b),
			parse_quote!(PackedBinaryField2x128b),
			parse_quote!(PackedBinaryField4x128b),
			parse_quote!(PackedAESBinaryField1x8b),
			parse_quote!(PackedAESBinaryField2x8b),
			parse_quote!(PackedAESBinaryField4x8b),
			parse_quote!(PackedAESBinaryField8x8b),
			parse_quote!(PackedAESBinaryField16x8b),
			parse_quote!(PackedAESBinaryField32x8b),
			parse_quote!(PackedAESBinaryField64x8b),
			parse_quote!(PackedAESBinaryField1x16b),
			parse_quote!(PackedAESBinaryField2x16b),
			parse_quote!(PackedAESBinaryField4x16b),
			parse_quote!(PackedAESBinaryField8x16b),
			parse_quote!(PackedAESBinaryField16x16b),
			parse_quote!(PackedAESBinaryField32x16b),
			parse_quote!(PackedAESBinaryField1x32b),
			parse_quote!(PackedAESBinaryField2x32b),
			parse_quote!(PackedAESBinaryField4x32b),
			parse_quote!(PackedAESBinaryField8x32b),
			parse_quote!(PackedAESBinaryField16x32b),
			parse_quote!(PackedAESBinaryField1x64b),
			parse_quote!(PackedAESBinaryField2x64b),
			parse_quote!(PackedAESBinaryField4x64b),
			parse_quote!(PackedAESBinaryField8x64b),
			parse_quote!(PackedAESBinaryField1x128b),
			parse_quote!(PackedAESBinaryField2x128b),
			parse_quote!(PackedAESBinaryField4x128b),
			parse_quote!(PackedBinaryPolyval1x128b),
			parse_quote!(PackedBinaryPolyval2x128b),
			parse_quote!(PackedBinaryPolyval4x128b),
		],
		"BinaryField2b" => {
			vec![
				parse_quote!(PackedBinaryField1x2b),
				parse_quote!(PackedBinaryField2x2b),
				parse_quote!(PackedBinaryField4x2b),
				parse_quote!(PackedBinaryField8x2b),
				parse_quote!(PackedBinaryField16x2b),
				parse_quote!(PackedBinaryField32x2b),
				parse_quote!(PackedBinaryField64x2b),
				parse_quote!(PackedBinaryField128x2b),
				parse_quote!(PackedBinaryField256x2b),
				parse_quote!(PackedBinaryField1x4b),
				parse_quote!(PackedBinaryField2x4b),
				parse_quote!(PackedBinaryField4x4b),
				parse_quote!(PackedBinaryField8x4b),
				parse_quote!(PackedBinaryField16x4b),
				parse_quote!(PackedBinaryField32x4b),
				parse_quote!(PackedBinaryField64x4b),
				parse_quote!(PackedBinaryField128x4b),
				parse_quote!(PackedBinaryField1x8b),
				parse_quote!(PackedBinaryField2x8b),
				parse_quote!(PackedBinaryField4x8b),
				parse_quote!(PackedBinaryField8x8b),
				parse_quote!(PackedBinaryField16x8b),
				parse_quote!(PackedBinaryField32x8b),
				parse_quote!(PackedBinaryField64x8b),
				parse_quote!(PackedBinaryField1x16b),
				parse_quote!(PackedBinaryField2x16b),
				parse_quote!(PackedBinaryField4x16b),
				parse_quote!(PackedBinaryField8x16b),
				parse_quote!(PackedBinaryField16x16b),
				parse_quote!(PackedBinaryField32x16b),
				parse_quote!(PackedBinaryField1x32b),
				parse_quote!(PackedBinaryField2x32b),
				parse_quote!(PackedBinaryField4x32b),
				parse_quote!(PackedBinaryField8x32b),
				parse_quote!(PackedBinaryField16x32b),
				parse_quote!(PackedBinaryField1x64b),
				parse_quote!(PackedBinaryField2x64b),
				parse_quote!(PackedBinaryField4x64b),
				parse_quote!(PackedBinaryField8x64b),
				parse_quote!(PackedBinaryField1x128b),
				parse_quote!(PackedBinaryField2x128b),
				parse_quote!(PackedBinaryField4x128b),
			]
		}
		"BinaryField4b" => {
			vec![
				parse_quote!(PackedBinaryField1x4b),
				parse_quote!(PackedBinaryField2x4b),
				parse_quote!(PackedBinaryField4x4b),
				parse_quote!(PackedBinaryField8x4b),
				parse_quote!(PackedBinaryField16x4b),
				parse_quote!(PackedBinaryField32x4b),
				parse_quote!(PackedBinaryField64x4b),
				parse_quote!(PackedBinaryField128x4b),
				parse_quote!(PackedBinaryField1x8b),
				parse_quote!(PackedBinaryField2x8b),
				parse_quote!(PackedBinaryField4x8b),
				parse_quote!(PackedBinaryField8x8b),
				parse_quote!(PackedBinaryField16x8b),
				parse_quote!(PackedBinaryField32x8b),
				parse_quote!(PackedBinaryField64x8b),
				parse_quote!(PackedBinaryField1x16b),
				parse_quote!(PackedBinaryField2x16b),
				parse_quote!(PackedBinaryField4x16b),
				parse_quote!(PackedBinaryField8x16b),
				parse_quote!(PackedBinaryField16x16b),
				parse_quote!(PackedBinaryField32x16b),
				parse_quote!(PackedBinaryField1x32b),
				parse_quote!(PackedBinaryField2x32b),
				parse_quote!(PackedBinaryField4x32b),
				parse_quote!(PackedBinaryField8x32b),
				parse_quote!(PackedBinaryField16x32b),
				parse_quote!(PackedBinaryField1x64b),
				parse_quote!(PackedBinaryField2x64b),
				parse_quote!(PackedBinaryField4x64b),
				parse_quote!(PackedBinaryField8x64b),
				parse_quote!(PackedBinaryField1x128b),
				parse_quote!(PackedBinaryField2x128b),
				parse_quote!(PackedBinaryField4x128b),
			]
		}
		"BinaryField8b" => {
			vec![
				parse_quote!(PackedBinaryField1x8b),
				parse_quote!(PackedBinaryField2x8b),
				parse_quote!(PackedBinaryField4x8b),
				parse_quote!(PackedBinaryField8x8b),
				parse_quote!(PackedBinaryField16x8b),
				parse_quote!(PackedBinaryField32x8b),
				parse_quote!(PackedBinaryField64x8b),
				parse_quote!(PackedBinaryField1x16b),
				parse_quote!(PackedBinaryField2x16b),
				parse_quote!(PackedBinaryField4x16b),
				parse_quote!(PackedBinaryField8x16b),
				parse_quote!(PackedBinaryField16x16b),
				parse_quote!(PackedBinaryField32x16b),
				parse_quote!(PackedBinaryField1x32b),
				parse_quote!(PackedBinaryField2x32b),
				parse_quote!(PackedBinaryField4x32b),
				parse_quote!(PackedBinaryField8x32b),
				parse_quote!(PackedBinaryField16x32b),
				parse_quote!(PackedBinaryField1x64b),
				parse_quote!(PackedBinaryField2x64b),
				parse_quote!(PackedBinaryField4x64b),
				parse_quote!(PackedBinaryField8x64b),
				parse_quote!(PackedBinaryField1x128b),
				parse_quote!(PackedBinaryField2x128b),
				parse_quote!(PackedBinaryField4x128b),
			]
		}
		"BinaryField16b" => {
			vec![
				parse_quote!(PackedBinaryField1x16b),
				parse_quote!(PackedBinaryField2x16b),
				parse_quote!(PackedBinaryField4x16b),
				parse_quote!(PackedBinaryField8x16b),
				parse_quote!(PackedBinaryField16x16b),
				parse_quote!(PackedBinaryField32x16b),
				parse_quote!(PackedBinaryField1x32b),
				parse_quote!(PackedBinaryField2x32b),
				parse_quote!(PackedBinaryField4x32b),
				parse_quote!(PackedBinaryField8x32b),
				parse_quote!(PackedBinaryField16x32b),
				parse_quote!(PackedBinaryField1x64b),
				parse_quote!(PackedBinaryField2x64b),
				parse_quote!(PackedBinaryField4x64b),
				parse_quote!(PackedBinaryField8x64b),
				parse_quote!(PackedBinaryField1x128b),
				parse_quote!(PackedBinaryField2x128b),
				parse_quote!(PackedBinaryField4x128b),
			]
		}
		"BinaryField32b" => {
			vec![
				parse_quote!(PackedBinaryField1x32b),
				parse_quote!(PackedBinaryField2x32b),
				parse_quote!(PackedBinaryField4x32b),
				parse_quote!(PackedBinaryField8x32b),
				parse_quote!(PackedBinaryField16x32b),
				parse_quote!(PackedBinaryField1x64b),
				parse_quote!(PackedBinaryField2x64b),
				parse_quote!(PackedBinaryField4x64b),
				parse_quote!(PackedBinaryField8x64b),
				parse_quote!(PackedBinaryField1x128b),
				parse_quote!(PackedBinaryField2x128b),
				parse_quote!(PackedBinaryField4x128b),
			]
		}
		"BinaryField64b" => {
			vec![
				parse_quote!(PackedBinaryField1x64b),
				parse_quote!(PackedBinaryField2x64b),
				parse_quote!(PackedBinaryField4x64b),
				parse_quote!(PackedBinaryField8x64b),
				parse_quote!(PackedBinaryField1x128b),
				parse_quote!(PackedBinaryField2x128b),
				parse_quote!(PackedBinaryField4x128b),
			]
		}

		"BinaryField128b" => {
			vec![
				parse_quote!(PackedBinaryField1x128b),
				parse_quote!(PackedBinaryField2x128b),
				parse_quote!(PackedBinaryField4x128b),
			]
		}

		"AESTowerField8b" => {
			vec![
				parse_quote!(PackedAESBinaryField1x8b),
				parse_quote!(PackedAESBinaryField2x8b),
				parse_quote!(PackedAESBinaryField4x8b),
				parse_quote!(PackedAESBinaryField8x8b),
				parse_quote!(PackedAESBinaryField16x8b),
				parse_quote!(PackedAESBinaryField32x8b),
				parse_quote!(PackedAESBinaryField64x8b),
				parse_quote!(PackedAESBinaryField1x16b),
				parse_quote!(PackedAESBinaryField2x16b),
				parse_quote!(PackedAESBinaryField4x16b),
				parse_quote!(PackedAESBinaryField8x16b),
				parse_quote!(PackedAESBinaryField16x16b),
				parse_quote!(PackedAESBinaryField32x16b),
				parse_quote!(PackedAESBinaryField1x32b),
				parse_quote!(PackedAESBinaryField2x32b),
				parse_quote!(PackedAESBinaryField4x32b),
				parse_quote!(PackedAESBinaryField8x32b),
				parse_quote!(PackedAESBinaryField16x32b),
				parse_quote!(PackedAESBinaryField1x64b),
				parse_quote!(PackedAESBinaryField2x64b),
				parse_quote!(PackedAESBinaryField4x64b),
				parse_quote!(PackedAESBinaryField8x64b),
				parse_quote!(PackedAESBinaryField1x128b),
				parse_quote!(PackedAESBinaryField2x128b),
				parse_quote!(PackedAESBinaryField4x128b),
				parse_quote!(ByteSlicedAES32x128b),
			]
		}
		"AESTowerField16b" => {
			vec![
				parse_quote!(PackedAESBinaryField1x16b),
				parse_quote!(PackedAESBinaryField2x16b),
				parse_quote!(PackedAESBinaryField4x16b),
				parse_quote!(PackedAESBinaryField8x16b),
				parse_quote!(PackedAESBinaryField16x16b),
				parse_quote!(PackedAESBinaryField32x16b),
				parse_quote!(PackedAESBinaryField1x32b),
				parse_quote!(PackedAESBinaryField2x32b),
				parse_quote!(PackedAESBinaryField4x32b),
				parse_quote!(PackedAESBinaryField8x32b),
				parse_quote!(PackedAESBinaryField16x32b),
				parse_quote!(PackedAESBinaryField1x64b),
				parse_quote!(PackedAESBinaryField2x64b),
				parse_quote!(PackedAESBinaryField4x64b),
				parse_quote!(PackedAESBinaryField8x64b),
				parse_quote!(PackedAESBinaryField1x128b),
				parse_quote!(PackedAESBinaryField2x128b),
				parse_quote!(PackedAESBinaryField4x128b),
			]
		}
		"AESTowerField32b" => {
			vec![
				parse_quote!(PackedAESBinaryField1x32b),
				parse_quote!(PackedAESBinaryField2x32b),
				parse_quote!(PackedAESBinaryField4x32b),
				parse_quote!(PackedAESBinaryField8x32b),
				parse_quote!(PackedAESBinaryField16x32b),
				parse_quote!(PackedAESBinaryField1x64b),
				parse_quote!(PackedAESBinaryField2x64b),
				parse_quote!(PackedAESBinaryField4x64b),
				parse_quote!(PackedAESBinaryField8x64b),
				parse_quote!(PackedAESBinaryField1x128b),
				parse_quote!(PackedAESBinaryField2x128b),
				parse_quote!(PackedAESBinaryField4x128b),
			]
		}
		"AESTowerField64b" => {
			vec![
				parse_quote!(PackedAESBinaryField1x64b),
				parse_quote!(PackedAESBinaryField2x64b),
				parse_quote!(PackedAESBinaryField4x64b),
				parse_quote!(PackedAESBinaryField8x64b),
				parse_quote!(PackedAESBinaryField1x128b),
				parse_quote!(PackedAESBinaryField2x128b),
				parse_quote!(PackedAESBinaryField4x128b),
			]
		}
		"AESTowerField128b" => {
			vec![
				parse_quote!(PackedAESBinaryField1x128b),
				parse_quote!(PackedAESBinaryField2x128b),
				parse_quote!(PackedAESBinaryField4x128b),
			]
		}

		_ => vec![],
	}
}
