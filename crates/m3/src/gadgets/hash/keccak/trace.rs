// Copyright 2025 Irreducible Inc.

use std::array;

use super::{
	RC, RHO, ROUNDS_PER_PERMUTATION,
	state::{StateMatrix, StateRow},
};

/// Trace of a keccak-f round.
#[derive(Clone)]
pub struct RoundTrace {
	pub state_in: StateMatrix<u64>,
	pub state_out: StateMatrix<u64>,
	pub c: StateRow<u64>,
	pub d: StateRow<u64>,
	pub a_theta: StateMatrix<u64>,
	pub b: StateMatrix<u64>,
	pub rc: u64,
}

impl RoundTrace {
	/// Run the `round`th round with the given `state_in` (`A`) and return the trace for it.
	pub fn gather(state_in: StateMatrix<u64>, round: usize) -> RoundTrace {
		// θ step
		let c = StateRow::from_fn(|x| {
			state_in[(x, 0)]
				^ state_in[(x, 1)]
				^ state_in[(x, 2)]
				^ state_in[(x, 3)]
				^ state_in[(x, 4)]
		});
		let d = StateRow::from_fn(|x| {
			// D[x] = C[x-1] xor rot(C[x+1],1)
			c[x + 4] ^ c[x + 1].rotate_left(1)
		});

		let mut a_theta = StateMatrix::default();
		for x in 0..5 {
			for y in 0..5 {
				// A[x,y] = A[x,y] xor D[x],
				a_theta[(x, y)] = state_in[(x, y)] ^ d[x];
			}
		}

		// ρ and π steps
		let mut b = StateMatrix::default();
		for x in 0..5 {
			for y in 0..5 {
				// B[y,2*x+3*y] = rot(A[x,y], r[x,y])
				b[(y, 2 * x + 3 * y)] = a_theta[(x, y)].rotate_left(RHO[x][y]);
			}
		}

		// χ step
		let rc = RC[round];
		let state_out = StateMatrix::from_fn(|(x, y)| {
			// A[x,y] = B[x,y] xor ((not B[x+1,y]) and B[x+2,y])
			let a_chi = b[(x, y)] ^ (!b[(x + 1, y)] & b[(x + 2, y)]);

			// # ι step
			if (x, y) == (0, 0) { a_chi ^ rc } else { a_chi }
		});

		RoundTrace {
			state_in,
			state_out,
			c,
			d,
			a_theta,
			b,
			rc,
		}
	}
}

/// A trace of all [`ROUNDS_PER_PERMUTATION`] keccak-f\[1600\] rounds.
pub struct PermutationTrace {
	round_traces: [RoundTrace; ROUNDS_PER_PERMUTATION],
}

impl PermutationTrace {
	/// Get the input state for the permutation.
	pub fn input(&self) -> &StateMatrix<u64> {
		&self.round_traces[0].state_in
	}

	/// Get the output state for the permutation.
	pub fn output(&self) -> &StateMatrix<u64> {
		&self.round_traces[ROUNDS_PER_PERMUTATION - 1].state_out
	}
}

impl AsRef<[RoundTrace]> for PermutationTrace {
	fn as_ref(&self) -> &[RoundTrace] {
		&self.round_traces
	}
}

impl std::ops::Index<usize> for PermutationTrace {
	type Output = RoundTrace;

	fn index(&self, index: usize) -> &Self::Output {
		&self.round_traces[index]
	}
}

/// Gather a trace for each round of keccak-f\[1600\] permutation.
pub fn keccakf_trace(state_in: StateMatrix<u64>) -> PermutationTrace {
	let mut state = state_in;
	let round_traces = array::from_fn(|round| {
		let trace = RoundTrace::gather(state.clone(), round);
		state = trace.state_out.clone();
		trace
	});
	PermutationTrace { round_traces }
}

#[cfg(test)]
mod tests {
	use super::{
		super::{test_vector, trace::StateMatrix},
		RoundTrace,
	};

	fn homebrew_keccakf(state_in: [u64; 25]) -> [u64; 25] {
		let mut state = StateMatrix::from_values(state_in);
		for round in 0..24 {
			let trace = RoundTrace::gather(state, round);
			state = trace.state_out;
		}
		state.into_inner()
	}

	#[test]
	fn test_round_trace() {
		for &[state_in, expected_out] in &test_vector::TEST_VECTOR {
			let our = homebrew_keccakf(state_in);
			assert_eq!(our, expected_out);
		}
	}
}
