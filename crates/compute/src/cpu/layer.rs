// Copyright 2025 Irreducible Inc.

use std::{iter, marker::PhantomData};

use binius_field::{util::inner_product_unchecked, ExtensionField};

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
		a_edeg: usize,
		a_in: &'a [T::B128], //Self::FSlice<'_>,
		b_in: &'a [T::B128], //Self::FSlice<'_>,
	) -> impl Fn(&mut Self::Exec) -> Result<T::B128, Error> {
		move |_exec| {
			// TODO: Check lengths
			let result = match a_edeg {
				// 1 => inner_product_unchecked::<T::B64, F>(a_in.iter().flat_map(|ext| ext.iter_bases()) )
				// 2 => inner_product_unchecked::<T::B32, F>(a_in.iter().flat_map(|ext| ext.iter_bases()) )
				4 => inner_product_unchecked(
					b_in.iter().cloned(),
					a_in.iter()
						.flat_map(|ext| <T::B128 as ExtensionField<T::B16>>::iter_bases(ext)),
				),
				7 => inner_product_unchecked::<T::B128, T::B128>(
					a_in.iter().cloned(),
					b_in.iter().cloned(),
				),
				_ => todo!(),
			};
			Ok(result)
			// let result = iter::zip(
			// 	PackedField::iter_slice(<F as PackedExtension<BinaryField16b>>::cast_bases(a_in)),
			// 	b_in,
			// )
			// .map(|(a_i, &b_i)| b_i * a_i)
			// .sum();
			// Ok(result)
		}
	}

	fn tensor_expand(
		&self,
		log_n: usize,
		coordinates: &[T::B128],
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

	use binius_field::{
		BinaryField128b, BinaryField16b, ExtensionField, Field, PackedExtension, PackedField,
		TowerField,
	};
	use binius_math::tensor_prod_eq_ind;
	use rand::{prelude::StdRng, SeedableRng};

	use super::*;
	use crate::tower::CanonicalTowerFamily;

	#[test]
	fn test_exec_single_tensor_expand() {
		let n_vars = 8;
		let mut rng = StdRng::seed_from_u64(0);
		let coordinates = repeat_with(|| <BinaryField128b as Field>::random(&mut rng))
			.take(n_vars)
			.collect::<Vec<_>>();

		let compute = <CpuLayer<CanonicalTowerFamily>>::default();
		let op = compute.tensor_expand(2, &coordinates[2..]);
		let mut buffer = vec![BinaryField128b::ZERO; 1 << n_vars];
		for x_i in &mut buffer[..4] {
			*x_i = <BinaryField128b as Field>::random(&mut rng);
		}
		let mut buffer_clone = buffer.clone();

		compute
			.execute(|exec, buffer| op(exec, buffer), &mut buffer.as_mut())
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
		let op = compute.inner_product(BinaryField16b::TOWER_LEVEL, &a, &b);
		let actual = compute.execute(|exec, _in| op(exec), ()).unwrap();

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
}
