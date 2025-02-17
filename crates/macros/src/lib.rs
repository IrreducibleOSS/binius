// Copyright 2024-2025 Irreducible Inc.

extern crate proc_macro;
mod arith_circuit_poly;
mod arith_expr;
mod composition_poly;

use proc_macro::TokenStream;
use quote::{quote, ToTokens};
use syn::{parse_macro_input, parse_quote, spanned::Spanned, Data, DeriveInput, Fields, ItemImpl};

use crate::{
	arith_circuit_poly::ArithCircuitPolyItem, arith_expr::ArithExprItem,
	composition_poly::CompositionPolyItem,
};

/// Useful for concisely creating structs that implement CompositionPolyOS.
/// This currently only supports creating composition polynomials of tower level 0.
///
/// ```
/// use binius_macros::composition_poly;
/// use binius_math::CompositionPolyOS;
/// use binius_field::{Field, BinaryField1b as F};
///
/// // Defines named struct without any fields that implements CompositionPolyOS
/// composition_poly!(MyComposition[x, y, z] = x + y * z);
/// assert_eq!(
///     MyComposition.evaluate(&[F::ONE, F::ONE, F::ONE]).unwrap(),
///     F::ZERO
/// );
///
/// // If you omit the name you get an anonymous instance instead, which can be used inline
/// assert_eq!(
///     composition_poly!([x, y, z] = x + y * z)
///         .evaluate(&[F::ONE, F::ONE, F::ONE]).unwrap(),
///     F::ZERO
/// );
/// ```
#[proc_macro]
pub fn composition_poly(input: TokenStream) -> TokenStream {
	parse_macro_input!(input as CompositionPolyItem)
		.into_token_stream()
		.into()
}

/// Define polynomial expressions compactly using named positional arguments
///
/// ```
/// use binius_macros::arith_expr;
/// use binius_field::{Field, BinaryField1b, BinaryField8b};
/// use binius_math::ArithExpr as Expr;
///
/// assert_eq!(
///     arith_expr!([x, y] = x + y + 1),
///     Expr::Var(0) + Expr::Var(1) + Expr::Const(BinaryField1b::ONE)
/// );
///
/// assert_eq!(
///     arith_expr!(BinaryField8b[x] = 3*x + 15),
///     Expr::Const(BinaryField8b::new(3)) * Expr::Var(0) + Expr::Const(BinaryField8b::new(15))
/// );
/// ```
#[proc_macro]
pub fn arith_expr(input: TokenStream) -> TokenStream {
	parse_macro_input!(input as ArithExprItem)
		.into_token_stream()
		.into()
}

#[proc_macro]
pub fn arith_circuit_poly(input: TokenStream) -> TokenStream {
	parse_macro_input!(input as ArithCircuitPolyItem)
		.into_token_stream()
		.into()
}

/// Derives the trait binius_field::SerializeCanonical for a struct or enum
///
/// See the DeserializeCanonical derive macro docs for examples/tests
#[proc_macro_derive(SerializeCanonical)]
pub fn derive_serialize_canonical(input: TokenStream) -> TokenStream {
	let input: DeriveInput = parse_macro_input!(input);
	let span = input.span();
	let name = input.ident;
	let mut generics = input.generics.clone();
	generics.type_params_mut().for_each(|type_param| {
		type_param
			.bounds
			.push(parse_quote!(binius_field::SerializeCanonical))
	});
	let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();
	let body = match input.data {
		Data::Union(_) => syn::Error::new(span, "Unions are not supported").into_compile_error(),
		Data::Struct(data) => {
			let fields = field_names(data.fields, None);
			quote! {
				#(binius_field::SerializeCanonical::serialize_canonical(&self.#fields, &mut write_buf)?;)*
			}
		}
		Data::Enum(data) => {
			let variants = data
				.variants
				.into_iter()
				.enumerate()
				.map(|(i, variant)| {
					let variant_ident = &variant.ident;
					let variant_index = i as u8;
					let fields = field_names(variant.fields.clone(), Some("field_"));
					let serialize_variant = quote! {
						binius_field::SerializeCanonical::serialize_canonical(&#variant_index, &mut write_buf)?;
						#(binius_field::SerializeCanonical::serialize_canonical(#fields, &mut write_buf)?;)*
					};
					match variant.fields {
						Fields::Named(_) => quote! {
							Self::#variant_ident { #(#fields),* } => {
								#serialize_variant
							}
						},
						Fields::Unnamed(_) => quote! {
							Self::#variant_ident(#(#fields),*) => {
								#serialize_variant
							}
						},
						Fields::Unit => quote! {
							Self::#variant_ident => {
								#serialize_variant
							}
						},
					}
				})
				.collect::<Vec<_>>();

			quote! {
				match self {
					#(#variants)*
				}
			}
		}
	};
	quote! {
		impl #impl_generics binius_field::SerializeCanonical for #name #ty_generics #where_clause {
			fn serialize_canonical(&self, mut write_buf: impl binius_field::bytes::BufMut) -> Result<(), binius_field::serialization::Error> {
				#body
				Ok(())
			}
		}
	}.into()
}

/// Derives the trait binius_field::DeserializeCanonical for a struct or enum
///
/// ```
/// use binius_field::{BinaryField128b, SerializeCanonical, DeserializeCanonical};
/// use binius_macros::{SerializeCanonical, DeserializeCanonical};
///
/// #[derive(Debug, PartialEq, SerializeCanonical, DeserializeCanonical)]
/// enum MyEnum {
///     A(usize),
///     B { x: u32, y: u32 },
///     C
/// }
///
///
/// let mut buf = vec![];
/// let value = MyEnum::B { x: 42, y: 1337 };
/// MyEnum::serialize_canonical(&value, &mut buf).unwrap();
/// assert_eq!(
///     MyEnum::deserialize_canonical(buf.as_slice()).unwrap(),
///     value
/// );
///
///
/// #[derive(Debug, PartialEq, SerializeCanonical, DeserializeCanonical)]
/// struct MyStruct<F> {
///     data: Vec<F>
/// }
///
/// let mut buf = vec![];
/// let value = MyStruct {
///    data: vec![BinaryField128b::new(1234), BinaryField128b::new(5678)]
/// };
/// MyStruct::serialize_canonical(&value, &mut buf).unwrap();
/// assert_eq!(
///     MyStruct::<BinaryField128b>::deserialize_canonical(buf.as_slice()).unwrap(),
///     value
/// );
/// ```
#[proc_macro_derive(DeserializeCanonical)]
pub fn derive_deserialize_canonical(input: TokenStream) -> TokenStream {
	let input: DeriveInput = parse_macro_input!(input);
	let span = input.span();
	let name = input.ident;
	let mut generics = input.generics.clone();
	generics.type_params_mut().for_each(|type_param| {
		type_param
			.bounds
			.push(parse_quote!(binius_field::DeserializeCanonical))
	});
	let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();
	let deserialize_value = quote! {
		binius_field::DeserializeCanonical::deserialize_canonical(&mut read_buf)?
	};
	let body = match input.data {
		Data::Union(_) => syn::Error::new(span, "Unions are not supported").into_compile_error(),
		Data::Struct(data) => {
			let fields = field_names(data.fields, None);
			quote! {
				Ok(Self {
					#(#fields: #deserialize_value,)*
				})
			}
		}
		Data::Enum(data) => {
			let variants = data
				.variants
				.into_iter()
				.enumerate()
				.map(|(i, variant)| {
					let variant_ident = &variant.ident;
					let variant_index: u8 = i as u8;
					match variant.fields {
						Fields::Named(fields) => {
							let fields = fields
								.named
								.into_iter()
								.map(|field| field.ident)
								.map(|field_name| quote!(#field_name: #deserialize_value))
								.collect::<Vec<_>>();

							quote! {
								#variant_index => Self::#variant_ident { #(#fields,)* }
							}
						}
						Fields::Unnamed(fields) => {
							let fields = fields
								.unnamed
								.into_iter()
								.map(|_| quote!(#deserialize_value))
								.collect::<Vec<_>>();

							quote! {
								#variant_index => Self::#variant_ident(#(#fields,)*)
							}
						}
						Fields::Unit => quote! {
							#variant_index => Self::#variant_ident
						},
					}
				})
				.collect::<Vec<_>>();

			let name = name.to_string();
			quote! {
				let variant_index: u8 = #deserialize_value;
				Ok(match variant_index {
					#(#variants,)*
					_ => {
						return Err(binius_field::serialization::Error::UnknownEnumVariant {
							name: #name,
							index: variant_index
						})
					}
				})
			}
		}
	};
	quote! {
		impl #impl_generics binius_field::DeserializeCanonical for #name #ty_generics #where_clause {
			fn deserialize_canonical(mut read_buf: impl binius_field::bytes::Buf) -> Result<Self, binius_field::serialization::Error>
			where
				Self: Sized
			{
				#body
			}
		}
	}
	.into()
}

/// Use on an impl block for MultivariatePoly, to automatically implement erased_serialize_canonical.
///
/// Importantly, this will serialize the concrete instance, prefixed by the identifier of the data type.
///
/// This prefix can be used to figure out which concrete data type it should use for deserialization later.
#[proc_macro_attribute]
pub fn erased_serialize_canonical(_attr: TokenStream, item: TokenStream) -> TokenStream {
	let mut item_impl: ItemImpl = parse_macro_input!(item);
	let syn::Type::Path(p) = &*item_impl.self_ty else {
		return syn::Error::new(
			item_impl.span(),
			"#[erased_serialize_canonical] can only be used on an impl for a concrete type",
		)
		.into_compile_error()
		.into();
	};
	let name = p.path.segments.last().unwrap().ident.to_string();

	let method = parse_quote! {
		fn erased_serialize_canonical(
			&self,
			write_buf: &mut dyn binius_field::bytes::BufMut,
		) -> Result<(), binius_field::serialization::Error> {
			binius_field::SerializeCanonical::serialize_canonical(&#name, &mut *write_buf)?;
			binius_field::SerializeCanonical::serialize_canonical(self, &mut *write_buf)
		}
	};

	item_impl.items.push(syn::ImplItem::Fn(method));

	quote! {
		#item_impl
	}
	.into()
}

fn field_names(fields: Fields, positional_prefix: Option<&str>) -> Vec<proc_macro2::TokenStream> {
	match fields {
		Fields::Named(fields) => fields
			.named
			.into_iter()
			.map(|field| field.ident.into_token_stream())
			.collect(),
		Fields::Unnamed(fields) => fields
			.unnamed
			.into_iter()
			.enumerate()
			.map(|(i, _)| match positional_prefix {
				Some(prefix) => {
					quote::format_ident!("{}{}", prefix, syn::Index::from(i)).into_token_stream()
				}
				None => syn::Index::from(i).into_token_stream(),
			})
			.collect(),
		Fields::Unit => vec![],
	}
}
