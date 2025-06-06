// Copyright 2024-2025 Irreducible Inc.

extern crate proc_macro;
mod deserialize_bytes;

use deserialize_bytes::{GenericsSplit, parse_container_attributes, split_for_impl};
use proc_macro::TokenStream;
use quote::{ToTokens, quote};
use syn::{Data, DeriveInput, Fields, ItemImpl, parse_macro_input, parse_quote, spanned::Spanned};

/// Derives the trait binius_utils::DeserializeBytes for a struct or enum
///
/// See the DeserializeBytes derive macro docs for examples/tests
#[proc_macro_derive(SerializeBytes)]
pub fn derive_serialize_bytes(input: TokenStream) -> TokenStream {
	let input: DeriveInput = parse_macro_input!(input);
	let span = input.span();
	let name = input.ident;
	let mut generics = input.generics.clone();
	generics.type_params_mut().for_each(|type_param| {
		type_param
			.bounds
			.push(parse_quote!(binius_utils::SerializeBytes))
	});
	let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();
	let body = match input.data {
		Data::Union(_) => syn::Error::new(span, "Unions are not supported").into_compile_error(),
		Data::Struct(data) => {
			let fields = field_names(data.fields, None);
			quote! {
				#(binius_utils::SerializeBytes::serialize(&self.#fields, &mut write_buf, mode)?;)*
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
						binius_utils::SerializeBytes::serialize(&#variant_index, &mut write_buf, mode)?;
						#(binius_utils::SerializeBytes::serialize(#fields, &mut write_buf, mode)?;)*
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
		impl #impl_generics binius_utils::SerializeBytes for #name #ty_generics #where_clause {
			fn serialize(&self, mut write_buf: impl binius_utils::bytes::BufMut, mode: binius_utils::SerializationMode) -> Result<(), binius_utils::SerializationError> {
				#body
				Ok(())
			}
		}
	}.into()
}

/// Derives the trait binius_utils::DeserializeBytes for a struct or enum
///
/// ```
/// use binius_field::BinaryField128b;
/// use binius_utils::{SerializeBytes, DeserializeBytes, SerializationMode};
/// use binius_macros::{SerializeBytes, DeserializeBytes};
///
/// #[derive(Debug, PartialEq, SerializeBytes, DeserializeBytes)]
/// enum MyEnum {
///     A(usize),
///     B { x: u32, y: u32 },
///     C
/// }
///
///
/// let mut buf = vec![];
/// let value = MyEnum::B { x: 42, y: 1337 };
/// MyEnum::serialize(&value, &mut buf, SerializationMode::Native).unwrap();
/// assert_eq!(
///     MyEnum::deserialize(buf.as_slice(), SerializationMode::Native).unwrap(),
///     value
/// );
///
///
/// #[derive(Debug, PartialEq, SerializeBytes, DeserializeBytes)]
/// struct MyStruct<F> {
///     data: Vec<F>
/// }
///
/// let mut buf = vec![];
/// let value = MyStruct {
///    data: vec![BinaryField128b::new(1234), BinaryField128b::new(5678)]
/// };
/// MyStruct::serialize(&value, &mut buf, SerializationMode::CanonicalTower).unwrap();
/// assert_eq!(
///     MyStruct::<BinaryField128b>::deserialize(buf.as_slice(), SerializationMode::CanonicalTower).unwrap(),
///     value
/// );
/// ```
///
/// ## Eval generics
///
/// Sometimes it is convenient to limit the implementation of `DeserializeBytes` only to
/// specific types. For example:
///
/// ```ignore
/// impl DeserializeBytes for MultilinearOracleSet<BinaryField128b> {...}
/// ```
///
/// To do that use `eval_generics` attribute:
///
/// ```
/// use binius_field::BinaryField128b;
/// use binius_utils::{SerializeBytes, DeserializeBytes, SerializationMode};
/// use binius_macros::{SerializeBytes, DeserializeBytes};
///
///
/// #[derive(Debug, PartialEq, SerializeBytes, DeserializeBytes)]
/// #[deserialize_bytes(eval_generics(F = BinaryField128b))]
/// struct MyStruct<F> {
///     data: Vec<F>
/// }
///
/// let mut buf = vec![];
/// let value = MyStruct {
///    data: vec![BinaryField128b::new(1234), BinaryField128b::new(5678)]
/// };
/// MyStruct::serialize(&value, &mut buf, SerializationMode::CanonicalTower).unwrap();
/// assert_eq!(
///     MyStruct::<BinaryField128b>::deserialize(buf.as_slice(), SerializationMode::CanonicalTower).unwrap(),
///     value
/// );
/// ```
///
/// Additionally, `eval_generics` can be used to fix multiple params:
/// `eval_generics(F = BinaryField128b, G = binius_field::BinaryField64b)`
#[proc_macro_derive(DeserializeBytes, attributes(deserialize_bytes))]
pub fn derive_deserialize_bytes(input: TokenStream) -> TokenStream {
	let input: DeriveInput = parse_macro_input!(input);
	let span = input.span();
	let container_attributes = match parse_container_attributes(&input) {
		Ok(x) => x,
		Err(e) => return e.into_compile_error().into(),
	};
	let name = input.ident;
	let mut generics = input.generics.clone();
	generics.type_params_mut().for_each(|type_param| {
		type_param
			.bounds
			.push(parse_quote!(binius_utils::DeserializeBytes))
	});
	let GenericsSplit {
		impl_generics,
		type_generics: ty_generics,
		where_clause,
	} = split_for_impl(&generics, &container_attributes);
	let deserialize_value = quote! {
		binius_utils::DeserializeBytes::deserialize(&mut read_buf, mode)?
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
						return Err(binius_utils::SerializationError::UnknownEnumVariant {
							name: #name,
							index: variant_index
						})
					}
				})
			}
		}
	};

	quote! {
		impl #impl_generics binius_utils::DeserializeBytes for #name #ty_generics #where_clause {
			fn deserialize(mut read_buf: impl binius_utils::bytes::Buf, mode: binius_utils::SerializationMode) -> Result<Self, binius_utils::SerializationError>
			where
				Self: Sized
			{
				#body
			}
		}
	}
	.into()
}

/// Use on an impl block for MultivariatePoly, to automatically implement erased_serialize_bytes.
///
/// Importantly, this will serialize the concrete instance, prefixed by the identifier of the data
/// type.
///
/// This prefix can be used to figure out which concrete data type it should use for deserialization
/// later.
#[proc_macro_attribute]
pub fn erased_serialize_bytes(_attr: TokenStream, item: TokenStream) -> TokenStream {
	let mut item_impl: ItemImpl = parse_macro_input!(item);
	let syn::Type::Path(p) = &*item_impl.self_ty else {
		return syn::Error::new(
			item_impl.span(),
			"#[erased_serialize_bytes] can only be used on an impl for a concrete type",
		)
		.into_compile_error()
		.into();
	};
	let name = p.path.segments.last().unwrap().ident.to_string();
	item_impl.items.push(syn::ImplItem::Fn(parse_quote! {
		fn erased_serialize(
			&self,
			write_buf: &mut dyn binius_utils::bytes::BufMut,
			mode: binius_utils::SerializationMode,
		) -> Result<(), binius_utils::SerializationError> {
			binius_utils::SerializeBytes::serialize(&#name, &mut *write_buf, mode)?;
			binius_utils::SerializeBytes::serialize(self, &mut *write_buf, mode)
		}
	}));
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
