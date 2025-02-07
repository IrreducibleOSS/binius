// Copyright 2024-2025 Irreducible Inc.

#[derive(Clone, thiserror::Error, Debug)]
pub enum Error {
	#[error("Write buffer is full")]
	WriteBufferFull,
	#[error("Not enough data in read buffer to deserialize")]
	NotEnoughBytes,
	#[error("Unknown enum variant index {name}::{index}")]
	UnknownEnumVariant { name: &'static str, index: u8 },
	#[error("FromUtf8Error: {0}")]
	FromUtf8Error(#[from] std::string::FromUtf8Error),
}
