// Copyright 2025 Irreducible Inc.

use std::{iter, marker::PhantomData};

use binius_field::{util::inner_product_unchecked, ExtensionField, TowerField};

use crate::{
	layer::{ComputeLayer, Error, Executor},
	tower::TowerFamily,
};

#[derive(Debug)]
pub struct CpuExecutor;

impl Executor for CpuExecutor {}

#[derive(Debug, Default)]
pub struct CpuLayer<T: TowerFamily>(PhantomData<T>);

impl<T: TowerFamily> ComputeLayer<T::B128> for CpuLayer<T> {
	type Exec = CpuExecutor;

	fn inner_product<'a>(
		&'a self,
		_exec: &'a mut Self::Exec,
		a_edeg: usize,
		a_in: &'a [T::B128],
		b_in: &'a [T::B128],
	) -> Result<T::B128, Error> {
		// TODO: Assertions to input errors
		assert!(a_edeg <= T::B128::TOWER_LEVEL);
		assert_eq!(a_in.len() << (T::B128::TOWER_LEVEL - a_edeg), b_in.len());

		let result = match a_edeg {
			0 => inner_product_unchecked(
				b_in.iter().cloned(),
				a_in.iter()
					.flat_map(|ext| <T::B128 as ExtensionField<T::B1>>::iter_bases(ext)),
			),
			3 => inner_product_unchecked(
				b_in.iter().cloned(),
				a_in.iter()
					.flat_map(|ext| <T::B128 as ExtensionField<T::B8>>::iter_bases(ext)),
			),
			4 => inner_product_unchecked(
				b_in.iter().cloned(),
				a_in.iter()
					.flat_map(|ext| <T::B128 as ExtensionField<T::B16>>::iter_bases(ext)),
			),
			5 => inner_product_unchecked(
				b_in.iter().cloned(),
				a_in.iter()
					.flat_map(|ext| <T::B128 as ExtensionField<T::B32>>::iter_bases(ext)),
			),
			6 => inner_product_unchecked(
				b_in.iter().cloned(),
				a_in.iter()
					.flat_map(|ext| <T::B128 as ExtensionField<T::B64>>::iter_bases(ext)),
			),
			7 => inner_product_unchecked::<T::B128, T::B128>(
				a_in.iter().cloned(),
				b_in.iter().cloned(),
			),
			_ => {
				return Err(Error::InputValidation(format!(
					"unsupported value of a_edeg: {a_edeg}"
				)))
			}
		};
		Ok(result)
	}

	fn tensor_expand<'a>(
		&self,
		_exec: &mut Self::Exec,
		log_n: usize,
		coordinates: &[T::B128],
		data: &mut &'a mut [T::B128],
	) -> Result<(), Error> {
		// TODO: Assertions to input errors
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

	/// Creates an operation that depends on the concurrent execution of two inner operations.
	fn join<Out1, Out2>(
		&self,
		exec: &mut Self::Exec,
		op1: impl Fn(&mut Self::Exec) -> Result<Out1, Error>,
		op2: impl Fn(&mut Self::Exec) -> Result<Out2, Error>,
	) -> Result<(Out1, Out2), Error> {
		let out1 = op1(exec)?;
		let out2 = op2(exec)?;
		Ok((out1, out2))
	}

	/// Creates an operation that depends on the concurrent execution of a sequence of operations.
	fn map<Out, I: ExactSizeIterator>(
		&self,
		exec: &mut Self::Exec,
		map: impl Fn(&mut Self::Exec, I::Item) -> Result<Out, Error>,
		iter: I,
	) -> Result<Vec<Out>, Error> {
		iter.map(|item| map(exec, item)).collect()
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

	use binius_field::{
		BinaryField128b, BinaryField16b, BinaryField32b, ExtensionField, Field, PackedExtension,
		PackedField, TowerField,
	};
	use binius_math::{tensor_prod_eq_ind, MultilinearExtension, MultilinearQuery};
	use bytemuck::zeroed_vec;
	use rand::{prelude::StdRng, SeedableRng};

	use super::*;
	use crate::{memory::ComputeMemory, tower::CanonicalTowerFamily};

	#[test]
	fn test_exec_single_tensor_expand() {
		let n_vars = 8;
		let mut rng = StdRng::seed_from_u64(0);
		let coordinates = repeat_with(|| <BinaryField128b as Field>::random(&mut rng))
			.take(n_vars)
			.collect::<Vec<_>>();

		let compute = <CpuLayer<CanonicalTowerFamily>>::default();
		let mut buffer = vec![BinaryField128b::ZERO; 1 << n_vars];
		for x_i in &mut buffer[..4] {
			*x_i = <BinaryField128b as Field>::random(&mut rng);
		}
		let mut buffer_clone = buffer.clone();

		//let op = compute.tensor_expand(2, &coordinates[2..]);
		compute
			.execute(
				|exec, buffer| compute.tensor_expand(exec, 2, &coordinates[2..], buffer),
				&mut buffer.as_mut(),
			)
			.unwrap();

		tensor_prod_eq_ind(2, &mut buffer_clone, &coordinates[2..]).unwrap();
		assert_eq!(buffer, buffer_clone);
	}

	#[test]
	fn test_exec_single_inner_product() {
		let n_vars = 8;

		let mut rng = StdRng::seed_from_u64(0);

		let a = repeat_with(|| <BinaryField128b as Field>::random(&mut rng))
			.take(n_vars - <BinaryField128b as ExtensionField<BinaryField16b>>::LOG_DEGREE)
			.collect::<Vec<_>>();
		let b = repeat_with(|| <BinaryField128b as Field>::random(&mut rng))
			.take(n_vars)
			.collect::<Vec<_>>();

		let compute = <CpuLayer<CanonicalTowerFamily>>::default();
		//let op = compute.inner_product(BinaryField16b::TOWER_LEVEL, &a, &b);
		let actual = compute
			.execute(
				|exec, _in| compute.inner_product(exec, BinaryField16b::TOWER_LEVEL, &a, &b),
				(),
			)
			.unwrap();

		let expected = iter::zip(
			PackedField::iter_slice(
				<BinaryField128b as PackedExtension<BinaryField16b>>::cast_bases(&a),
			),
			&b,
		)
		.map(|(a_i, &b_i)| b_i * a_i)
		.sum::<BinaryField128b>();

		assert_eq!(actual, expected);
	}

	#[test]
	fn test_exec_multiple_multilinear_evaluations() {
		type CL = CpuLayer<CanonicalTowerFamily>;

		let n_vars = 8;

		let mut rng = StdRng::seed_from_u64(0);

		let mle1 = repeat_with(|| <BinaryField128b as Field>::random(&mut rng))
			.take(1 << (n_vars - <BinaryField128b as ExtensionField<BinaryField16b>>::LOG_DEGREE))
			.collect::<Vec<_>>();
		let mle2 = repeat_with(|| <BinaryField128b as Field>::random(&mut rng))
			.take(1 << (n_vars - <BinaryField128b as ExtensionField<BinaryField32b>>::LOG_DEGREE))
			.collect::<Vec<_>>();
		let coordinates = repeat_with(|| <BinaryField128b as Field>::random(&mut rng))
			.take(n_vars)
			.collect::<Vec<_>>();

		let compute = CL::default();
		//let compute_ref = &compute;
		let op = |exec: &mut CpuExecutor, mut eq_ind_buffer: &mut [BinaryField128b]| {
			// TODO: This memory initialization is not generic
			eq_ind_buffer[0] = BinaryField128b::ONE;
			compute.tensor_expand(exec, 0, &coordinates, &mut eq_ind_buffer)?;

			let eq_ind_buffer = CL::as_const(&mut eq_ind_buffer);
			compute.join(
				exec,
				|exec| {
					compute.inner_product(exec, BinaryField16b::TOWER_LEVEL, &mle1, eq_ind_buffer)
				},
				|exec| {
					compute.inner_product(exec, BinaryField32b::TOWER_LEVEL, &mle2, eq_ind_buffer)
				},
			)
		};

		let mut eq_ind_buffer = zeroed_vec(1 << n_vars);
		let (eval1, eval2) = compute.execute(op, &mut eq_ind_buffer).unwrap();

		let query = MultilinearQuery::<BinaryField128b>::expand(&coordinates);
		let expected_eval1 = MultilinearExtension::new(
			n_vars,
			<BinaryField128b as PackedExtension<BinaryField16b>>::cast_bases(&mle1),
		)
		.unwrap()
		.evaluate(&query)
		.unwrap();
		let expected_eval2 = MultilinearExtension::new(
			n_vars,
			<BinaryField128b as PackedExtension<BinaryField32b>>::cast_bases(&mle2),
		)
		.unwrap()
		.evaluate(&query)
		.unwrap();

		assert_eq!(eq_ind_buffer, query.into_expansion());
		assert_eq!(eval1, expected_eval1);
		assert_eq!(eval2, expected_eval2);
	}
}
