use thiserror::Error;

/// Error thrown when a field operation fails.
#[derive(Clone, Error, Debug)]
pub enum Error {
	#[error("the argument does not match the field extension degree")]
	ExtensionDegreeMismatch,
	#[error("index {index} is out of range 0..{max}")]
	IndexOutOfRange { index: usize, max: usize },
	/// Thrown when trying to initialize a binary field element with a value bigger than what fits
	/// in the binary field.
	#[error("value is not in the field")]
	NotInField,
}
