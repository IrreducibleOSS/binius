#[cfg(test)]
mod tests {
	use proptest::{arbitrary::any, proptest};

	use crate::{
		arch::{
			packed_polyval_256::PackedBinaryPolyval2x128b,
			packed_polyval_512::PackedBinaryPolyval4x128b,
		},
		BinaryField128bPolyval, PackedField,
	};

	fn check_get_set<const WIDTH: usize, PT>(a: [u128; WIDTH], b: [u128; WIDTH])
	where
		PT: PackedField<Scalar = BinaryField128bPolyval> + From<[u128; WIDTH]>,
	{
		let mut val = PT::from(a);
		for i in 0..WIDTH {
			assert_eq!(val.get(i), BinaryField128bPolyval::from(a[i]));
			val.set(i, BinaryField128bPolyval::from(b[i]));
			assert_eq!(val.get(i), BinaryField128bPolyval::from(b[i]));
		}
	}

	fn check_mul<const WIDTH: usize, PT>(a: [u128; WIDTH], b: [u128; WIDTH])
	where
		PT: PackedField<Scalar = BinaryField128bPolyval> + From<[u128; WIDTH]>,
	{
		let rhs = PT::from(a);
		let lhs = PT::from(b);

		let result = lhs * rhs;
		for i in 0..WIDTH {
			assert_eq!(result.get(i), lhs.get(i) * rhs.get(i));
		}
	}

	proptest! {
		#[test]
		fn test_get_set_256(a in any::<[u128; 2]>(), b in any::<[u128; 2]>()) {
			check_get_set::<2, PackedBinaryPolyval2x128b>(a, b);
		}

		#[test]
		fn test_get_set_512(a in any::<[u128; 4]>(), b in any::<[u128; 4]>()) {
			check_get_set::<4, PackedBinaryPolyval4x128b>(a, b);
		}

		#[test]
		fn test_mul_256(a in any::<[u128; 2]>(), b in any::<[u128; 2]>()) {
			check_mul::<2, PackedBinaryPolyval2x128b>(a, b);
		}

		#[test]
		fn test_mul_512(a in any::<[u128; 4]>(), b in any::<[u128; 4]>()) {
			check_mul::<4, PackedBinaryPolyval4x128b>(a, b);
		}
	}
}
