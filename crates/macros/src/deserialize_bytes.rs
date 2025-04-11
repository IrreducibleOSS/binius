mod parse;
mod quote;

use parse::ContainerAttributes;
pub use quote::GenericsSplit;
use syn::{Attribute, Generics, Meta, MetaList};

pub fn get_generics<'attr, 'gen>(
	attrs: &'attr [Attribute],
	generics: &'gen Generics,
) -> syn::Result<GenericsSplit<'gen>> {
	let maybe_deserialize_bytes_attr = attrs
		.iter()
		.find(|attr| attr.path().is_ident("deserialize_bytes"));
	let deserialize_bytes_attr = match maybe_deserialize_bytes_attr {
		Some(attr) => attr,
		None => {
			let generics_split = GenericsSplit::new(generics, Default::default());
			return Ok(generics_split);
		}
	};
	let container_attributes_tokens = match &deserialize_bytes_attr.meta {
		Meta::List(MetaList { tokens, .. }) => tokens.clone(),
		meta => {
			return Err(syn::Error::new_spanned(
				meta,
				"expected `deserialize_bytes(eval_generics(X = Y, ...))`",
			));
		}
	};
	let container_attributes: ContainerAttributes = syn::parse2(container_attributes_tokens)?;
	let generics_split = GenericsSplit::new(generics, container_attributes);
	Ok(generics_split)
}
