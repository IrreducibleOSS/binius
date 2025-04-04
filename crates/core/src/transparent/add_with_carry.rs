// Copyright 2024-2025 Irreducible Inc.
use binius_field::{BinaryField1b, BinaryField32b, ExtensionField, TowerField};
use binius_macros::{DeserializeBytes, SerializeBytes};
use binius_utils::bail;

use crate::polynomial::{Error, MultivariatePoly};

///Given a query of size 17, of the form (cin || i || j) the multivariate polynomial returns
///a B32 element whose representation is of the form:
///(cout || cin ||(i+j added as 8 bit integers without the overflow)|| i || j )
///where cin and cout represent carry in and carry out of bitwise addition of integers  

#[derive(Debug, Copy, Clone, SerializeBytes, DeserializeBytes)]
pub struct AddWithCarry;

impl<F: TowerField + ExtensionField<BinaryField32b>> MultivariatePoly<F> for AddWithCarry {
	fn degree(&self) -> usize {
		2
	}

	fn n_vars(&self) -> usize {
		17
	}

	//Given a query of size n_vars, split it into two halves and evaluates the polynomial

	fn evaluate(&self, query: &[F]) -> Result<F, Error> {
		if query.len() != 17 {
			bail!(Error::IncorrectQuerySize { expected: 17 });
		}

		let (mut cin, a, b) = (query[0], &query[1..9], &query[9..17]);

		let mut result = cin * <BinaryField32b as ExtensionField<BinaryField1b>>::basis(1);

		for i in 0..8 {
			//Computing the sum of a and b as integers with bitwise add and carry logic
			result += (a[i] + b[i] + cin)
				* <BinaryField32b as ExtensionField<BinaryField1b>>::basis(i + 2);
			cin = (a[i] + cin) * (b[i] + cin) + cin;

			//Computing the latter part of the output, which is the concatenation of a and b
			result += a[i] * <BinaryField32b as ExtensionField<BinaryField1b>>::basis(i + 10);
			result += b[i] * <BinaryField32b as ExtensionField<BinaryField1b>>::basis(i + 18);
		}

		//Final value of cin is the carry out
		result += cin;

		Ok(result)
	}

	fn binary_tower_level(&self) -> usize {
		5
	}
}

#[cfg(test)]
mod tests {
	use binius_field::BinaryField32b;
	use rand::{thread_rng, Rng};

	use super::*;

	fn int_to_query(int: u32) -> Vec<BinaryField32b> {
		let mut query = vec![];
		for i in 0..8 {
			let bit = (int >> i) & 1;
			query.push(BinaryField32b::from(bit));
		}
		query
	}

	//We compare the result of the polynomial with the result of bitwise addition as integers.
	#[test]
	fn test_add_with_carry() {
		let mut rng = thread_rng();
		let a_int = rng.gen::<u8>();
		let b_int = rng.gen::<u8>();
		let cin_int = rng.gen::<bool>() as u8;

		//Covers the possibility of a = 255 and cin = 1
		let (mut res_int, mut c_out_int) = a_int.overflowing_add(cin_int);

		if c_out_int {
			res_int += b_int
		} else {
			(res_int, c_out_int) = res_int.overflowing_add(b_int)
		}

		let (a_int, b_int, cin_int, res_int) =
			(a_int as u32, b_int as u32, cin_int as u32, res_int as u32);
		let result_int =
			(c_out_int as u32) + (cin_int << 1) + (res_int << 2) + (a_int << 10) + (b_int << 18);
		let a = int_to_query(a_int);
		let b = int_to_query(b_int);
		let cin = vec![BinaryField32b::from(cin_int)];
		let query = [cin, a, b].concat();

		let add_with_carry = AddWithCarry;
		let result = add_with_carry.evaluate(&query).unwrap();

		assert_eq!(result_int, result.val())
	}
}
