// Copyright 2025 Irreducible Inc.

use std::iter;

use binius_field::TowerField;

use crate::layer::{ComputeLayer, Error, Executor};

#[derive(Debug)]
pub struct CpuExecutor;

impl Executor for CpuExecutor {}

#[derive(Debug)]
pub struct CpuLayer;

impl<F: TowerField> ComputeLayer<F> for CpuLayer {
	type Exec = CpuExecutor;

	fn inner_product(
		&self,
		a_edeg: usize,
		a_in: Self::FSlice<'_>,
		b_in: Self::FSlice<'_>,
	) -> impl Fn(&mut Self::Exec) -> Result<F, Error> {
		move |exec| todo!()
	}

	fn tensor_expand(
		&self,
		log_n: usize,
		coordinates: &[F],
	) -> impl for<'a> Fn(&mut Self::Exec, &mut Self::FSliceMut<'a>) -> Result<(), Error> {
		let coordinates = coordinates.to_vec();
		move |_exec, data| {
			assert_eq!(data.len(), 1 << (log_n + coordinates.len()));
			for (i, r_i) in coordinates.iter().enumerate() {
				let (lhs, rest) = data.split_at_mut(1 << (log_n + i));
				let (rhs, _rest) = rest.split_at_mut(1 << (log_n + i));
				for (x_i, y_i) in iter::zip(lhs, rhs) {
					let prod = *x_i * r_i;
					*x_i -= prod;
					*y_i += prod;
				}
			}
			Ok(())
		}
	}

	/// Creates an operation that depends on the concurrent execution of two inner operations.
	fn join<In1, Out1, In2, Out2>(
		&self,
		op1: impl Fn(&mut Self::Exec, In1) -> Result<Out1, Error>,
		op2: impl Fn(&mut Self::Exec, In2) -> Result<Out2, Error>,
	) -> impl Fn(&mut Self::Exec, In1, In2) -> Result<(Out1, Out2), Error> {
		move |exec, in1, in2| {
			let out1 = op1(exec, in1)?;
			let out2 = op2(exec, in2)?;
			Ok((out1, out2))
		}
	}

	/// Creates an operation that depends on the sequential execution of two inner operations.
	fn then<In1, Out1, In2, Out2>(
		&self,
		op1: impl Fn(&mut Self::Exec, In1) -> Result<Out1, Error>,
		op2: impl Fn(&mut Self::Exec, Out1, In2) -> Result<Out2, Error>,
	) -> impl Fn(&mut Self::Exec, In1, In2) -> Result<Out2, Error> {
		move |exec, in1, in2| {
			let out1 = op1(exec, in1)?;
			let out2 = op2(exec, out1, in2)?;
			Ok(out2)
		}
	}

	/// Creates an operation that depends on the concurrent execution of a sequence of operations.
	fn map<Out, I: ExactSizeIterator>(
		&self,
		map: impl Fn(&mut Self::Exec, I::Item) -> Result<Out, Error>,
	) -> impl Fn(&mut Self::Exec, I) -> Result<Vec<Out>, Error> {
		move |exec, iter| iter.map(|item| map(exec, item)).collect()
	}

	/// Executes an operation.
	///
	/// A HAL operation is an abstract function that runs with an executor reference.
	fn execute<In, Out>(
		&self,
		f: impl Fn(&mut Self::Exec, In) -> Result<Out, Error>,
		input: In,
	) -> Result<Out, Error> {
		f(&mut CpuExecutor, input)
	}
}

#[cfg(test)]
mod tests {
	use std::iter::repeat_with;

	use binius_field::{BinaryField128b, Field};
	use binius_math::tensor_prod_eq_ind;
	use rand::{prelude::StdRng, SeedableRng};

	use super::*;

	#[test]
	fn test_exec_single_tensor_expand() {
		let n_vars = 8;
		let mut rng = StdRng::seed_from_u64(0);
		let coordinates = repeat_with(|| BinaryField128b::random(&mut rng))
			.take(n_vars)
			.collect::<Vec<_>>();

		let op = CpuLayer.tensor_expand(2, &coordinates[2..]);
		let mut buffer = vec![BinaryField128b::ZERO; 1 << n_vars];
		for x_i in &mut buffer[..4] {
			*x_i = BinaryField128b::random(&mut rng);
		}
		let mut buffer_clone = buffer.clone();

		ComputeLayer::<BinaryField128b>::execute(
			&CpuLayer,
			|exec, buffer| op(exec, buffer),
			&mut buffer.as_mut(),
		)
		.unwrap();

		tensor_prod_eq_ind(2, &mut buffer_clone, &coordinates[2..]).unwrap();
		assert_eq!(buffer, buffer_clone);
	}
}
