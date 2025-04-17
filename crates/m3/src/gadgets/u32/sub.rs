// Copyright 2025 Irreducible Inc.

use binius_core::oracle::ShiftVariant;
use binius_field::{packed::set_packed_slice, Field, PackedExtension, PackedFieldIndexable};

use crate::builder::{column::Col, types::B1, witness::TableWitnessSegment, TableBuilder, B128};

/// A gadget for performing 32-bit integer subtraction on vertically-packed bit columns.
///
/// This gadget has input columns `xin` and `yin` for the two 32-bit integers to be subtracted, and
/// an output column `zout`, and it constrains that `xin - yin = zout` as integers.
#[derive(Debug)]
pub struct U32Sub {
	// Inputs
	pub xin: Col<B1, 32>,
	pub yin: Col<B1, 32>,

	// Private
	bout: Col<B1, 32>,
	bout_shl: Col<B1, 32>,
	bin: Col<B1, 32>,

	// Outputs
	/// The output column, either committed if `flags.commit_zout` is set, otherwise a linear
	/// combination derived column.
	pub zout: Col<B1, 32>,
	/// This is `Some` if `flags.expose_final_borrow` is set, otherwise it is `None`.
	pub final_borrow: Option<Col<B1>>,
	/// Flags modifying the gadget's behavior.
	pub flags: U32SubFlags,
}

/// Flags modifying the behavior of the [`U32Sub`] gadget.
#[derive(Debug, Default, Clone)]
pub struct U32SubFlags {
	// Optionally a column for a dynamic borrow in bit. This *must* be zero in all bits except the
	// 0th.
	pub borrow_in_bit: Option<Col<B1, 32>>,
	pub expose_final_borrow: bool,
	pub commit_zout: bool,
}

impl U32Sub {
	pub fn new(
		table: &mut TableBuilder,
		xin: Col<B1, 32>,
		yin: Col<B1, 32>,
		flags: U32SubFlags,
	) -> Self {
		let bout = table.add_committed("bout");
		let bout_shl = table.add_shifted("bout_shl", bout, 5, 1, ShiftVariant::LogicalLeft);

		let bin = if let Some(borrow_in_bit) = flags.borrow_in_bit {
			table.add_computed("bin", bout_shl + borrow_in_bit)
		} else {
			bout_shl
		};

		let final_borrow = flags
			.expose_final_borrow
			.then(|| table.add_selected("final_borrow", bout, 31));

		let zout = if flags.commit_zout {
			let zout = table.add_committed("zout");
			table.assert_zero("zout", xin + yin + bin - zout);
			zout
		} else {
			table.add_computed("zout", xin + yin + bin)
		};

		U32Sub {
			xin,
			yin,
			bout,
			bout_shl,
			bin,
			zout,
			final_borrow,
			flags,
		}
	}
}

impl U32Sub {
	pub fn populate<P>(&self, index: &mut TableWitnessSegment<P>) -> Result<(), anyhow::Error>
	where
		P: PackedFieldIndexable<Scalar = B128> + PackedExtension<B1>,
	{
		let xin: std::cell::RefMut<'_, [u32]> = index.get_mut_as(self.xin)?;
		let yin: std::cell::RefMut<'_, [u32]> = index.get_mut_as(self.yin)?;
		let mut bout: std::cell::RefMut<'_, [u32]> = index.get_mut_as(self.bout)?;
		let mut zout: std::cell::RefMut<'_, [u32]> = index.get_mut_as(self.zout)?;
		let mut bin: std::cell::RefMut<'_, [u32]> = index.get_mut_as(self.bin)?;
		let mut final_borrow = if let Some(final_borrow) = self.final_borrow {
			let final_borrow = index.get_mut(final_borrow)?;
			Some(final_borrow)
		} else {
			None
		};

		if let Some(borrow_in_bit) = self.flags.borrow_in_bit {
			// This is u32 assumed to be either 0 or 1.
			let borrow_in_bit = index.get_mut_as(borrow_in_bit)?;
			let mut bout_shl = index.get_mut_as(self.bout_shl)?;

			for i in 0..index.size() {
				let (x_minus_y, borrow1) = xin[i].overflowing_sub(yin[i]);
				let borrow2;
				(zout[i], borrow2) = x_minus_y.overflowing_sub(borrow_in_bit[i]);
				let borrow = borrow1 | borrow2;

				bin[i] = xin[i] ^ yin[i] ^ zout[i];
				bout[i] = (borrow as u32) << 31 | bin[i] >> 1;
				bout_shl[i] = bout[i] << 1;

				if let Some(ref mut final_borrow) = final_borrow {
					set_packed_slice(
						&mut *final_borrow,
						i,
						if borrow { B1::ONE } else { B1::ZERO },
					);
				}
			}
		} else {
			// When the borrow in bit is fixed to zero, we can simplify the logic.
			for i in 0..index.size() {
				let borrow;
				(zout[i], borrow) = xin[i].overflowing_sub(yin[i]);
				bin[i] = xin[i] ^ yin[i] ^ zout[i];
				bout[i] = (borrow as u32) << 31 | bin[i] >> 1;

				if let Some(ref mut final_borrow) = final_borrow {
					set_packed_slice(
						&mut *final_borrow,
						i,
						if borrow { B1::ONE } else { B1::ZERO },
					);
				}
			}
		}

		Ok(())
	}
}

#[cfg(test)]
mod tests {
	use binius_field::{
		arch::OptimalUnderlier128b, as_packed_field::PackedType, packed::get_packed_slice,
	};
	use bumpalo::Bump;
	use rand::{prelude::StdRng, Rng as _, SeedableRng};

	use super::*;
	use crate::builder::{ConstraintSystem, Statement, WitnessIndex};

	#[test]
	fn prop_test_no_borrow() {
		const N_ITER: usize = 1 << 14;

		let mut rng = StdRng::seed_from_u64(0);
		let test_vector: Vec<(u32, u32, u32, u32, bool)> = (0..N_ITER)
			.map(|_| {
				let x: u32 = rng.gen();
				let y: u32 = rng.gen();
				let z: u32 = x.wrapping_sub(y);
				// (x, y, borrow_in, zout, final_borrow)
				(x, y, 0x00000000, z, false)
			})
			.collect();

		TestPlan {
			dyn_borrow_in: false,
			expose_final_borrow: false,
			commit_zout: true,
			test_vector,
		}
		.execute();
	}

	#[test]
	fn prop_test_with_borrow() {
		const N_ITER: usize = 1 << 14;

		let mut rng = StdRng::seed_from_u64(0);
		let test_vector: Vec<(u32, u32, u32, u32, bool)> = (0..N_ITER)
			.map(|_| {
				let x: u32 = rng.gen();
				let y: u32 = rng.gen();
				let borrow_in = rng.gen::<bool>() as u32;
				let (x_minus_y, borrow1) = x.overflowing_sub(y);
				let (z, borrow2) = x_minus_y.overflowing_sub(borrow_in);
				let final_borrow = borrow1 | borrow2;
				(x, y, borrow_in, z, final_borrow)
			})
			.collect();

		TestPlan {
			dyn_borrow_in: true,
			expose_final_borrow: true,
			commit_zout: true,
			test_vector,
		}
		.execute();
	}

	#[test]
	fn test_borrow() {
		// (x, y, borrow_in, zout, final_borrow)
		let test_vector = [
			(0x00000000, 0x00000001, 0x00000000, 0xFFFFFFFF, true), // 0 - 1 = max_u32 (underflow)
			(0xFFFFFFFF, 0x00000001, 0x00000000, 0xFFFFFFFE, false), // max - 1 = max - 1
			(0x80000000, 0x00000001, 0x00000000, 0x7FFFFFFF, false), // Sign bit transition
			(0x00000005, 0x00000005, 0x00000001, 0xFFFFFFFF, true), // 5 - 5 - 1 = -1 (borrow_in causes underflow)
		];
		TestPlan {
			dyn_borrow_in: true,
			expose_final_borrow: true,
			commit_zout: true,
			test_vector: test_vector.to_vec(),
		}
		.execute();
	}

	struct TestPlan {
		dyn_borrow_in: bool,
		expose_final_borrow: bool,
		commit_zout: bool,
		/// (x, y, borrow_in, zout, final_borrow)
		test_vector: Vec<(u32, u32, u32, u32, bool)>,
	}

	impl TestPlan {
		fn execute(self) {
			let mut cs = ConstraintSystem::new();
			let mut table = cs.add_table("u32_sub");

			let xin = table.add_committed::<B1, 32>("xin");
			let yin = table.add_committed::<B1, 32>("yin");

			let borrow_in = self
				.dyn_borrow_in
				.then_some(table.add_committed::<B1, 32>("borrow_in"));

			let flags = U32SubFlags {
				borrow_in_bit: borrow_in,
				expose_final_borrow: self.expose_final_borrow,
				commit_zout: self.commit_zout,
			};
			let subber = U32Sub::new(&mut table, xin, yin, flags);
			assert!(subber.final_borrow.is_some() == self.expose_final_borrow);

			let table_id = table.id();
			let statement = Statement {
				boundaries: vec![],
				table_sizes: vec![self.test_vector.len()],
			};
			let allocator = Bump::new();
			let mut witness =
				WitnessIndex::<PackedType<OptimalUnderlier128b, B128>>::new(&cs, &allocator);

			let table_witness = witness
				.init_table(table_id, self.test_vector.len())
				.unwrap();
			let mut segment = table_witness.full_segment();

			{
				let mut xin_bits = segment.get_mut_as::<u32, _, 32>(subber.xin).unwrap();
				let mut yin_bits = segment.get_mut_as::<u32, _, 32>(subber.yin).unwrap();
				let mut borrow_in_bits =
					borrow_in.map(|borrow_in| segment.get_mut_as::<u32, _, 32>(borrow_in).unwrap());
				for (i, (x, y, borrow_in, _, _)) in self.test_vector.iter().enumerate() {
					xin_bits[i] = *x;
					yin_bits[i] = *y;
					if let Some(ref mut borrow_in_bits) = borrow_in_bits {
						borrow_in_bits[i] = *borrow_in;
					}
				}
			}

			// Populate the gadget
			subber.populate(&mut segment).unwrap();

			{
				// Verify results
				let zout_bits = segment.get_as::<u32, _, 32>(subber.zout).unwrap();
				let final_borrow = subber
					.final_borrow
					.map(|final_borrow| segment.get(final_borrow).unwrap());
				for (i, (_, _, _, zout, expected_borrow)) in self.test_vector.iter().enumerate() {
					dbg!(i);
					dbg!(self.test_vector[i]);
					assert_eq!(zout_bits[i], *zout);

					if let Some(ref final_borrow) = final_borrow {
						assert_eq!(get_packed_slice(final_borrow, i), B1::from(*expected_borrow));
					}
				}
			}

			// Validate constraint system
			let ccs = cs.compile(&statement).unwrap();
			let witness = witness.into_multilinear_extension_index();

			binius_core::constraint_system::validate::validate_witness(&ccs, &[], &witness)
				.unwrap();
		}
	}
}
