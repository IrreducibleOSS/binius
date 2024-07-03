# Performance numbers of SNARKS

All performance numbers were collected from GCP's `c3-standard-44` machines, which has 22 cores of Intel Sapphire Rapids
generation. They are compiled and run in release mode with the environment variable `RUSTFLAGS="-C target-cpu=native"`.
Note that the proving time includes the witness trace generation as well.

| SNARKS                           | Project        | Number of Permutations | Proving time (s) | Verification time (ms) | Hashed Data (MB) | MB / Proving time(s) | Mb / Verification time(s) |
|----------------------------------|----------------|------------------------|------------------|------------------------|------------------|----------------------|---------------------------|
| Keccakf                          | Binius         | 2^13 (logsize 24)      | 3.598            | 103                    | 1.1              | 0.305                | 10.67                     |
| Keccakf (Keccak Merkle trees)    | Plonky3 / BB31 | 1365                   | 11.84            | 235                    | 0.186            | 0.015                | 0.79                      |
| Keccakf (Poseidon2 Merkle trees) | Plonky3 / BB31 | 1365                   | 21.85            | 237                    | 0.186            | 0.008                | 0.78                      |
| Groestl                          | Binius         | 2^14 (logsize 19)      | 5.67             | 348                    | 1.049            | 0.185                | 3.01                      |
| Vision32b                        | Binius         | 2^14 (logsize 17)      | 2.51             | 65                     | 1.049            | 0.417                | 16.13                     |
| SHA-256                          | Binius         | 2^14 (logsize 19)      | 7.49             | 816                    | 1.049            | 0.14                 | 1.28                      |

# Appendix

The above table is generated from the following runs of each of the SNARKS

### Keccakf

with logsize 24

```
2024-07-03T16:38:45.210786Z  INFO keccakf: Size of hashable Keccak-256 data: 1.1 MB
2024-07-03T16:38:45.210810Z  INFO keccakf: Size of tensorpcs: 5.9 MB
generate_trace [ 438.23ms | 100.00% ]

prove [ 3.16s | 100.00% ]
├── tensor_pcs::commit [ 96.33ms | 3.05% ]
├── zerocheck::execute_round [ 544.05ms | 17.22% ] { index = 1 }
├── zerocheck::execute_round [ 347.29ms | 10.99% ] { index = 2 }
├── zerocheck::execute_round [ 232.69ms | 7.37% ] { index = 3 }
├── zerocheck::execute_round [ 156.04ms | 4.94% ] { index = 4 }
├── zerocheck::execute_round [ 117.05ms | 3.71% ] { index = 5 }
├── zerocheck::execute_round [ 177.41ms | 5.62% ] { index = 6 }
├── zerocheck::execute_round [ 88.38ms | 2.80% ] (17 calls)
├── [...] [ 1.18µs | 0.00% ]
├── EvalcheckProverState::prove [ 421.71ms | 13.35% ]
├── test_utils::prove_bivariate_sumchecks_with_switchover [ 36.45ms | 1.15% ]
│  ├── sumcheck::execute_round [ 35.54ms | 1.13% ] (448 calls)
│  └── [...] [ 35.74µs | 0.00% ] (53 calls)
├── EvalcheckProverState::prove [ 180.21ms | 5.71% ] (53 calls)
├── [...] [ 3.99µs | 0.00% ]
├── test_utils::make_non_same_query_pcs_sumchecks [ 367.74ms | 11.64% ]
├── test_utils::prove_bivariate_sumchecks_with_switchover [ 140.30ms | 4.44% ]
│  ├── sumcheck::execute_round [ 138.26ms | 4.38% ] (1308 calls)
│  └── [...] [ 58.48µs | 0.00% ] (118 calls)
├── [...] [ 31.38ms | 0.99% ] (119 calls)
└── tensor_pcs::prove_evaluation [ 160.01ms | 5.07% ]

verify [ 103.22ms | 100.00% ]
├── EvalcheckVerifierState::verify [ 1.37ms | 1.33% ] (54 calls)
├── [...] [ 486.20µs | 0.47% ] (121 calls)
└── tensor_pcs::verify_evaluation [ 97.22ms | 94.18% ]
```

### Groestl

with logsize 19

```
generate_trace [ 2.03s | 100.00% ] { log_size = 19 }

2024-07-03T16:37:43.537320Z  INFO groestl: Size of hashable Groestl256 data: 1048.6 KB
2024-07-03T16:37:43.537338Z  INFO groestl: Size of PCS opening proof: 5.0 MB
prove [ 3.64s | 100.00% ]
├── tensor_pcs::commit [ 246.55ms | 6.77% ] { index = 1 }
├── tensor_pcs::commit [ 385.73ms | 10.59% ]
├── zerocheck::execute_round [ 565.36ms | 15.52% ] { index = 1 }
├── zerocheck::execute_round [ 455.69ms | 12.51% ] { index = 2 }
├── zerocheck::execute_round [ 331.61ms | 9.10% ] { index = 3 }
├── zerocheck::execute_round [ 249.79ms | 6.86% ] { index = 4 }
├── zerocheck::execute_round [ 223.35ms | 6.13% ] { index = 5 }
├── zerocheck::execute_round [ 215.20ms | 5.91% ] (13 calls)
├── [...] [ 990.00ns | 0.00% ]
├── EvalcheckProverState::prove [ 389.35ms | 10.69% ]
├── [...] [ 7.61ms | 0.21% ] (65 calls)
├── test_utils::make_non_same_query_pcs_sumchecks [ 166.19ms | 4.56% ]
├── [...] [ 29.00ms | 0.80% ] (257 calls)
├── tensor_pcs::prove_evaluation [ 238.70ms | 6.55% ] { index = 1 }
└── tensor_pcs::prove_evaluation [ 87.54ms | 2.40% ]

verify [ 348.48ms | 100.00% ]
├── tensor_pcs::verify_evaluation [ 283.84ms | 81.45% ] { index = 1 }
└── tensor_pcs::verify_evaluation [ 50.14ms | 14.39% ]
```

### Vision

with logsize 17

``` 
2024-07-03T16:36:36.501119Z  INFO vision32b: Size of hashable vision32b data: 1048.6 KB
2024-07-03T16:36:36.501144Z  INFO vision32b: Size of tensorpcs: 1.8 MB
generate_trace [ 778.23ms | 100.00% ] { log_size = 17 }

prove [ 1.73s | 100.00% ]
├── tensor_pcs::commit [ 604.17ms | 34.83% ]
├── zerocheck::execute_round [ 143.24ms | 8.26% ] { index = 1 }
├── zerocheck::execute_round [ 130.82ms | 7.54% ] { index = 2 }
├── zerocheck::execute_round [ 129.82ms | 7.48% ] { index = 3 }
├── zerocheck::execute_round [ 66.37ms | 3.83% ] { index = 4 }
├── zerocheck::execute_round [ 94.99ms | 5.48% ] (12 calls)
├── [...] [ 1.31µs | 0.00% ]
├── EvalcheckProverState::prove [ 371.79ms | 21.43% ]
├── [...] [ 2.14ms | 0.12% ] (25 calls)
├── test_utils::make_non_same_query_pcs_sumchecks [ 47.84ms | 2.76% ]
├── test_utils::prove_bivariate_sumchecks_with_switchover [ 21.19ms | 1.22% ]
│  ├── sumcheck::execute_round [ 19.45ms | 1.12% ] (935 calls)
│  └── [...] [ 140.62µs | 0.01% ] (311 calls)
├── [...] [ 732.77µs | 0.04% ] (312 calls)
└── tensor_pcs::prove_evaluation [ 107.25ms | 6.18% ]

verify [ 65.19ms | 100.00% ]
├── EvalcheckVerifierState::verify [ 9.25ms | 14.19% ] { index = 1 }
├── [...] [ 734.92µs | 1.13% ] (337 calls)
└── tensor_pcs::verify_evaluation [ 53.81ms | 82.54% ]
```

### SHA-256

with logsize 19

```
2024-07-03T16:40:39.963292Z  INFO sha256: Size of hashable SHA-256 data: 1048.6 KB
2024-07-03T16:40:39.963317Z  INFO sha256: Size of PCS proof: 6.4 MB
generate_trace [ 490.13ms | 100.00% ]

prove [ 7.00s | 100.00% ]
├── tensor_pcs::commit [ 664.50ms | 9.49% ]
├── zerocheck::prove [ 3.63s | 51.83% ]
│  ├── zerocheck::execute_round [ 1.61s | 22.93% ] { index = 1 }
│  ├── zerocheck::execute_round [ 806.52ms | 11.52% ] { index = 2 }
│  ├── zerocheck::execute_round [ 427.96ms | 6.11% ] { index = 3 }
│  ├── zerocheck::execute_round [ 227.89ms | 3.25% ] { index = 4 }
│  ├── zerocheck::execute_round [ 257.16ms | 3.67% ] { index = 6 }
│  ├── zerocheck::execute_round [ 299.96ms | 4.28% ] (13 calls)
│  └── [...] [ 1.10µs | 0.00% ]
├── EvalcheckProverState::prove [ 644.76ms | 9.21% ]
├── test_utils::prove_bivariate_sumchecks_with_switchover [ 241.85ms | 3.45% ]
│  ├── sumcheck::execute_round [ 228.65ms | 3.27% ] (9719 calls)
│  └── [...] [ 878.40µs | 0.01% ] (1943 calls)
├── [...] [ 16.12ms | 0.23% ] (1944 calls)
├── test_utils::make_non_same_query_pcs_sumchecks [ 804.40ms | 11.49% ]
├── test_utils::prove_bivariate_sumchecks_with_switchover [ 276.31ms | 3.95% ]
│  ├── sumcheck::execute_round [ 259.12ms | 3.70% ] (10704 calls)
│  └── [...] [ 980.43µs | 0.01% ] (2140 calls)
├── [...] [ 16.64ms | 0.24% ] (2141 calls)
└── tensor_pcs::prove_evaluation [ 679.42ms | 9.70% ]

verify [ 815.76ms | 100.00% ]
├── EvalcheckVerifierState::verify [ 26.88ms | 3.30% ] { index = 1 }
├── EvalcheckVerifierState::verify [ 9.14ms | 1.12% ] (1943 calls)
├── [...] [ 6.96ms | 0.85% ] (2143 calls)
└── tensor_pcs::verify_evaluation [ 724.27ms | 88.78% ]
```

### Keccak over BabyBear31 with Keccak Merkle trees

```
INFO     generate Keccak trace [ 1.55s | 100.00% ]
INFO     prove [ 10.3s | 0.03% / 100.00% ]
INFO     ┝━ infer log of constraint degree [ 6.06ms | 0.06% ]
INFO     ┝━ commit to trace data [ 4.63s | 30.73% / 45.05% ]
INFO     │  ┕━ coset_lde_batch [ 1.47s | 14.32% ] dims: 2633x32768
INFO     ┝━ compute quotient polynomial [ 4.37s | 42.51% ]
INFO     ┝━ commit to quotient poly chunks [ 71.5ms | 0.58% / 0.70% ]
INFO     │  ┝━ coset_lde_batch [ 6.03ms | 0.06% ] dims: 4x32768
INFO     │  ┕━ coset_lde_batch [ 6.10ms | 0.06% ] dims: 4x32768
INFO     ┝━ compute_inverse_denominators [ 11.7ms | 0.11% ]
INFO     ┝━ reduce matrix quotient [ 513ms | 0.00% / 4.99% ] dims: 2633x65536
INFO     │  ┝━ compute opened values with Lagrange interpolation [ 195ms | 1.90% ]
INFO     │  ┕━ reduce rows [ 317ms | 3.09% ]
INFO     ┝━ reduce matrix quotient [ 512ms | 0.00% / 4.99% ] dims: 2633x65536
INFO     │  ┝━ compute opened values with Lagrange interpolation [ 195ms | 1.90% ]
INFO     │  ┕━ reduce rows [ 317ms | 3.09% ]
INFO     ┝━ reduce matrix quotient [ 12.0ms | 0.00% / 0.12% ] dims: 4x65536
INFO     │  ┝━ compute opened values with Lagrange interpolation [ 3.49ms | 0.03% ]
INFO     │  ┕━ reduce rows [ 8.48ms | 0.08% ]
INFO     ┝━ reduce matrix quotient [ 12.0ms | 0.00% / 0.12% ] dims: 4x65536
INFO     │  ┝━ compute opened values with Lagrange interpolation [ 3.47ms | 0.03% ]
INFO     │  ┕━ reduce rows [ 8.49ms | 0.08% ]
INFO     ┕━ FRI prover [ 136ms | 0.00% / 1.32% ]
INFO        ┝━ commit phase [ 67.9ms | 0.66% ]
INFO        ┝━ grind for proof-of-work witness [ 66.9ms | 0.65% ]
INFO        ┕━ query phase [ 1.21ms | 0.01% ]
INFO     verify [ 237ms | 98.46% / 100.00% ]
INFO     ┕━ infer log of constraint degree [ 3.64ms | 1.54% ]
```

### Keccak over BabyBear31 with Poseidon2 Merkle trees

```
INFO     generate Keccak trace [ 1.55s | 100.00% ]
INFO     prove [ 20.3s | 0.02% / 100.00% ]
INFO     ┝━ infer log of constraint degree [ 6.13ms | 0.03% ]
INFO     ┝━ commit to trace data [ 14.6s | 65.01% / 72.28% ]
INFO     │  ┕━ coset_lde_batch [ 1.47s | 7.27% ] dims: 2633x32768
INFO     ┝━ compute quotient polynomial [ 4.34s | 21.40% ]
INFO     ┝━ commit to quotient poly chunks [ 93.5ms | 0.40% / 0.46% ]
INFO     │  ┝━ coset_lde_batch [ 5.97ms | 0.03% ] dims: 4x32768
INFO     │  ┕━ coset_lde_batch [ 6.12ms | 0.03% ] dims: 4x32768
INFO     ┝━ compute_inverse_denominators [ 11.6ms | 0.06% ]
INFO     ┝━ reduce matrix quotient [ 507ms | 0.00% / 2.50% ] dims: 2633x65536
INFO     │  ┝━ compute opened values with Lagrange interpolation [ 195ms | 0.96% ]
INFO     │  ┕━ reduce rows [ 312ms | 1.54% ]
INFO     ┝━ reduce matrix quotient [ 509ms | 0.00% / 2.51% ] dims: 2633x65536
INFO     │  ┝━ compute opened values with Lagrange interpolation [ 195ms | 0.96% ]
INFO     │  ┕━ reduce rows [ 314ms | 1.55% ]
INFO     ┝━ reduce matrix quotient [ 11.7ms | 0.00% / 0.06% ] dims: 4x65536
INFO     │  ┝━ compute opened values with Lagrange interpolation [ 3.49ms | 0.02% ]
INFO     │  ┕━ reduce rows [ 8.22ms | 0.04% ]
INFO     ┝━ reduce matrix quotient [ 11.7ms | 0.00% / 0.06% ] dims: 4x65536
INFO     │  ┝━ compute opened values with Lagrange interpolation [ 3.48ms | 0.02% ]
INFO     │  ┕━ reduce rows [ 8.22ms | 0.04% ]
INFO     ┕━ FRI prover [ 127ms | 0.00% / 0.63% ]
INFO        ┝━ commit phase [ 90.9ms | 0.45% ]
INFO        ┝━ grind for proof-of-work witness [ 34.7ms | 0.17% ]
INFO        ┕━ query phase [ 1.21ms | 0.01% ]
INFO     verify [ 281ms | 98.73% / 100.00% ]
INFO     ┕━ infer log of constraint degree [ 3.58ms | 1.27% ]
```