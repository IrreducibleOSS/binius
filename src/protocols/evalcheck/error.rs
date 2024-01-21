use crate::polynomial::Error as PolynomialError;

#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error("sumcheck polynomial degree must be greater than zero")]
	PolynomialDegreeIsZero,
	#[error("the input was not well formed")]
	ImproperInput,
	#[error("the evaluation domain does not match the expected size")]
	EvaluationDomainMismatch,
	#[error("polynomial error: {0}")]
	Polynomial(#[from] PolynomialError),
	#[error("verification failure: {0}")]
	Verification(#[from] VerificationError),
}

#[derive(Debug, thiserror::Error)]
pub enum VerificationError {
	#[error("evaluation is incorrect")]
	IncorrectEvaluation,
	#[error("subproof type or shape does not match the claim")]
	SubproofMismatch,
}
