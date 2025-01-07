// Copyright 2024-2025 Irreducible Inc.

use binius_field::*;
use binius_macros::arith_circuit_poly;
use binius_math::CompositionPolyOS;
use paste::paste;
use rand::{rngs::StdRng, SeedableRng};

const BATCH_SIZE: usize = 32;

/// Helper macro to generate test functions for arithmetic circuit polynomial evaluation.
/// We want to test that macro compiles for all field types that's why we need it
/// to "beat the combinatorics with combinatorics".
macro_rules! test_arithmetic_poly {
    (@inner $packed:ty, $field:ty) => {
        paste! {
            #[allow(non_snake_case)]
            #[test]
            fn [<evaluate _ $field _ $packed>] () {
                let circuit = arith_circuit_poly!([x0, x1] = x0 + x0 * x1 + 1, $field);

                let mut rng = StdRng::seed_from_u64(0);
                let query = vec![
                    std::array::from_fn::<_, BATCH_SIZE, _>(|_| <$packed>::random(&mut rng)),
                    std::array::from_fn::<_, BATCH_SIZE, _>(|_| <$packed>::random(&mut rng)),
                ];
                let query_data = query.iter().map(|q| &q[..]).collect::<Vec<_>>();
                let expected = std::array::from_fn::<_, BATCH_SIZE, _>(|i| {
                    let x0 = query[0][i];
                    let x1 = query[1][i];
                    x0 + x0 * x1 + $packed::one()
                });

                for i in 0..BATCH_SIZE {
                    let query = [
                        query_data[0][i],
                        query_data[1][i],
                    ];
                    assert_eq!(circuit.evaluate(&query).unwrap(), expected[i]);
                }

                let mut batch_result = vec![<$packed>::zero(); BATCH_SIZE];
                circuit.batch_evaluate(&query_data, &mut batch_result).unwrap();
                assert_eq!(&batch_result, &expected);
            }
        }
    };
    ($packed_type:ty, $head:ty, $($tail:ty,)*) => {
        test_arithmetic_poly!(@inner $packed_type, $head);
        test_arithmetic_poly!($packed_type, $($tail,)*);
    };
    ($packed_type:ty, ) => {};
}

test_arithmetic_poly!(
	PackedBinaryField1x128b,
	BinaryField1b,
	BinaryField2b,
	BinaryField4b,
	BinaryField8b,
	BinaryField16b,
	BinaryField32b,
	BinaryField64b,
	BinaryField128b,
);

test_arithmetic_poly!(
	PackedAESBinaryField1x128b,
	AESTowerField8b,
	AESTowerField16b,
	AESTowerField32b,
	AESTowerField64b,
	AESTowerField128b,
);

test_arithmetic_poly!(PackedBinaryPolyval1x128b, BinaryField1b, BinaryField128bPolyval,);
