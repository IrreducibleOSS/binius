# Performance numbers of SNARKS

All performance numbers were collected from GCP's `c3-standard-44` machines, which has 22 cores of Intel Sapphire Rapids
generation. They are compiled and run in release mode with the environment variable `RUSTFLAGS="-C target-cpu=native"`.
Plonky3 examples are also run with the parallel feature `--features parallel`. Note that the proving time includes the
witness trace generation as well. For all the keccakf functions, the hashed data is actually the rate of SHA3-256(136
bytes) per permutation

| SNARKS                           | Project        | Number of Permutations | Proving time (s) | Verification time (ms) | Hashed Data (MB) | MB / Proving time(s) |
|----------------------------------|----------------|------------------------|------------------|------------------------|------------------|----------------------|
| Keccakf                          | Binius         | 2^13                   | 3.91             | 206                    | 1.1              | 0.281                |
| Keccakf (SHA256 Merkle trees)    | Plonky3 / BB31 | 2^13                   | 4.19             | 216                    | 1.1              | 0.262                |
| Keccakf (Poseidon2 Merkle trees) | Plonky3 / BB31 | 2^13                   | 5.38             | 260                    | 1.1              | 0.204                |
| Groestl (*)                      | Binius         | 2^14                   | 1.45             | 116                    | 1.049            | 0.721                |

(*) Our Current Groestl SNARK only has the P permutation as opposed to the compression function which has both P and Q
permutations, the number of permutations for groestl in the table assumes that both P and Q permutations would take the
same amount of proving and verification time.

# Appendix

The above table is generated from the following runs of each of the SNARKS

### Keccakf

```
Verifying 8192 Keccak-f permutations
generating trace [ 267.78ms | 100.00% ]

constraint_system::prove [ 3.64s | 100.00% ]
Events:
  event crates/core/src/constraint_system/prove.rs:74 { message: using computation backend: CpuBackend, arch: x86_64, rayon_threads: 20 }: 1

└── prove_with_pcs [ 3.64s | 99.90% ]
   ├── FRIPCS::commit [ 98.46ms | 2.70% ]
   │  └── commit_interleaved [ 98.45ms | 2.70% ]
   │     └── commit_interleaved_with [ 98.45ms | 2.70% ]
   │        └── commit_iterated [ 45.38ms | 1.25% ]
   │           └── BinaryMerkleTree::build [ 45.38ms | 1.25% ]
   ├── [...] [ 32.82µs | 0.00% ] (127 calls)
   ├── batch_prove_zerocheck_univariate_round [ 965.22ms | 26.49% ]
   │  ├── execute_univariate_round [ 276.92ms | 7.60% ] (119 calls)
   │  └── fold_univariate_round [ 679.09ms | 18.64% ] (119 calls)
   ├── sumcheck::batch_prove [ 254.60ms | 6.99% ]
   │  ├── ZerocheckProver::fold [ 55.10ms | 1.51% ] (119 calls)
   │  └── [...] [ 176.72ms | 4.85% ] (2737 calls)
   ├── [...] [ 314.26ms | 8.62% ] (240 calls)
   ├── sumcheck::batch_prove [ 83.04ms | 2.28% ]
   │  └── sumcheck::batch_prove [ 83.04ms | 2.28% ]
   ├── greedy_evalcheck::prove [ 1.29s | 35.30% ]
   │  ├── EvalcheckProver::prove [ 258.85ms | 7.10% ]
   │  │  └── EvalcheckProverState::prove_multilinear [ 257.62ms | 7.07% ] (1223 calls)
   │  ├── [...] [ 14.79ms | 0.41% ] (2 calls)
   │  ├── EvalcheckProver::prove [ 391.58ms | 10.75% ]
   │  │  └── EvalcheckProverState::prove_multilinear [ 390.61ms | 10.72% ] (1199 calls)
   │  ├── [...] [ 13.88ms | 0.38% ] (2 calls)
   │  ├── EvalcheckProver::prove [ 97.35ms | 2.67% ]
   │  │  └── EvalcheckProverState::prove_multilinear [ 96.55ms | 2.65% ] (1151 calls)
   │  ├── [...] [ 25.41µs | 0.00% ]
   │  ├── make_non_same_query_pcs_sumchecks [ 429.16ms | 11.78% ]
   │  ├── [...] [ 756.67µs | 0.02% ]
   │  ├── sumcheck::batch_prove [ 64.33ms | 1.77% ]
   │  │  └── sumcheck::batch_prove [ 64.33ms | 1.77% ]
   │  └── [...] [ 7.17ms | 0.20% ] (2 calls)
   └── fri_pcs::prove_evaluation [ 422.00ms | 11.58% ]
      ├── CpuBackend::evaluate_partial_high [ 207.85ms | 5.70% ]
      │  └── MultilinearExtension::evaluate_partial_high [ 207.85ms | 5.70% ]
      └── [...] [ 145.70ms | 4.00% ] (308 calls)

constraint_system::verify [ 206.71ms | 100.00% ]
├── batch_verify_zerocheck_univariate_round [ 5.37ms | 2.60% ]
├── [...] [ 355.86µs | 0.17% ]
├── EvalcheckVerifierState::verify [ 2.14ms | 1.04% ]
├── [...] [ 323.63µs | 0.16% ]
├── EvalcheckVerifierState::verify [ 7.30ms | 3.53% ]
├── [...] [ 312.05µs | 0.15% ]
├── EvalcheckVerifierState::verify [ 4.87ms | 2.35% ]
├── [...] [ 964.61µs | 0.47% ] (2 calls)
├── EvalcheckVerifierState::verify [ 3.16ms | 1.53% ]
├── [...] [ 20.66µs | 0.01% ]
└── fri::FRIVerifier::verify_query [ 5.22ms | 2.52% ] (240 calls)
```

### Groestl

```
Verifying 16384 Grøstl-256 P permutations
generating trace [ 174.54ms | 100.00% ]

constraint_system::prove [ 1.28s | 100.00% ]
Events:
  event crates/core/src/constraint_system/prove.rs:74 { message: using computation backend: CpuBackend, arch: x86_64, rayon_threads: 20 }: 1

└── prove_with_pcs [ 1.28s | 99.92% ]
   ├── batch_prove_zerocheck_univariate_round [ 478.20ms | 37.46% ]
   │  ├── execute_univariate_round [ 352.93ms | 27.65% ] (215 calls)
   │  └── fold_univariate_round [ 121.58ms | 9.52% ] (215 calls)
   ├── sumcheck::batch_prove [ 170.20ms | 13.33% ]
   │  ├── ZerocheckProver::execute [ 24.27ms | 1.90% ] (215 calls)
   │  ├── ZerocheckProver::fold [ 23.73ms | 1.86% ] (215 calls)
   │  ├── ZerocheckProver::execute [ 16.39ms | 1.28% ] (215 calls)
   │  ├── [...] [ 20.43ms | 1.60% ] (430 calls)
   │  ├── ZerocheckProver::fold [ 16.03ms | 1.26% ] (215 calls)
   │  ├── [...] [ 4.19ms | 0.33% ] (215 calls)
   │  ├── ZerocheckProver::fold [ 17.54ms | 1.37% ] (215 calls)
   │  └── [...] [ 43.36ms | 3.40% ] (1935 calls)
   ├── [...] [ 26.80ms | 2.10% ] (432 calls)
   ├── sumcheck::batch_prove [ 88.51ms | 6.93% ]
   │  └── sumcheck::batch_prove [ 88.51ms | 6.93% ]
   ├── greedy_evalcheck::prove [ 206.94ms | 16.21% ]
   │  ├── EvalcheckProver::prove [ 204.07ms | 15.99% ]
   │  │  └── EvalcheckProverState::prove_multilinear [ 202.59ms | 15.87% ] (2055 calls)
   │  └── [...] [ 40.16µs | 0.00% ] (4 calls)
   ├── fri_pcs::prove_evaluation [ 61.26ms | 4.80% ] { index = 1 }
   └── fri_pcs::prove_evaluation [ 107.29ms | 8.40% ]
      ├── CpuBackend::evaluate_partial_high [ 49.90ms | 3.91% ]
      │  └── MultilinearExtension::evaluate_partial_high [ 49.90ms | 3.91% ]
      └── [...] [ 37.31ms | 2.92% ] (302 calls)

constraint_system::verify [ 116.39ms | 100.00% ]
├── batch_verify_zerocheck_univariate_round [ 2.87ms | 2.47% ]
├── [...] [ 1.10ms | 0.95% ]
├── EvalcheckVerifierState::verify [ 20.15ms | 17.31% ]
├── [...] [ 28.41µs | 0.02% ] (4 calls)
└── fri::FRIVerifier::verify_query [ 5.54ms | 4.76% ] (481 calls)
```

### Vision

```
Verifying 16384 Vision-32b permutations
generating trace [ 599.12ms | 100.00% ]

constraint_system::prove [ 4.03s | 100.00% ]
Events:
  event crates/core/src/constraint_system/prove.rs:74 { message: using computation backend: CpuBackend, arch: x86_64, rayon_threads: 20 }: 1

└── prove_with_pcs [ 4.03s | 99.97% ]
   ├── FRIPCS::commit [ 97.38ms | 2.42% ]
   │  └── commit_interleaved [ 97.38ms | 2.42% ]
   │     └── commit_interleaved_with [ 97.37ms | 2.42% ]
   │        └── commit_iterated [ 44.26ms | 1.10% ]
   │           └── BinaryMerkleTree::build [ 44.26ms | 1.10% ]
   ├── [...] [ 81.51µs | 0.00% ] (391 calls)
   ├── batch_prove_zerocheck_univariate_round [ 2.52s | 62.49% ]
   │  ├── execute_univariate_round [ 2.16s | 53.54% ] (383 calls)
   │  └── fold_univariate_round [ 348.31ms | 8.64% ] (383 calls)
   ├── sumcheck::batch_prove [ 438.85ms | 10.89% ]
   │  ├── ZerocheckProver::execute [ 65.25ms | 1.62% ] (383 calls)
   │  ├── ZerocheckProver::fold [ 50.87ms | 1.26% ] (383 calls)
   │  ├── ZerocheckProver::execute [ 49.38ms | 1.23% ] (383 calls)
   │  ├── [...] [ 107.33ms | 2.66% ] (1532 calls)
   │  ├── ZerocheckProver::fold [ 40.39ms | 1.00% ] (383 calls)
   │  └── [...] [ 118.69ms | 2.95% ] (3447 calls)
   ├── [...] [ 58.54ms | 1.45% ] (768 calls)
   ├── sumcheck::batch_prove [ 160.77ms | 3.99% ]
   │  └── sumcheck::batch_prove [ 160.77ms | 3.99% ]
   ├── greedy_evalcheck::prove [ 102.99ms | 2.56% ]
   │  ├── EvalcheckProver::prove [ 101.25ms | 2.51% ]
   │  │  └── EvalcheckProverState::prove_multilinear [ 100.16ms | 2.49% ] (1175 calls)
   │  └── [...] [ 41.45µs | 0.00% ] (4 calls)
   └── fri_pcs::prove_evaluation [ 456.29ms | 11.32% ]
      ├── CpuBackend::evaluate_partial_high [ 279.55ms | 6.94% ]
      │  └── MultilinearExtension::evaluate_partial_high [ 279.55ms | 6.94% ]
      └── [...] [ 136.52ms | 3.39% ] (308 calls)

constraint_system::verify [ 162.18ms | 100.00% ]
├── batch_verify_zerocheck_univariate_round [ 5.45ms | 3.36% ]
├── batch_verify_with_start [ 2.39ms | 1.47% ]
├── EvalcheckVerifierState::verify [ 4.90ms | 3.02% ]
├── [...] [ 26.89µs | 0.02% ] (4 calls)
└── fri::FRIVerifier::verify_query [ 5.16ms | 3.18% ] (240 calls)

```

### SHA-256

```

Verifying 16384 sha256 compressions
generating trace [ 229.22ms | 100.00% ]

constraint_system::prove [ 4.28s | 100.00% ]
Events:
  event crates/core/src/constraint_system/prove.rs:74 { message: using computation backend: CpuBackend, arch: x86_64, rayon_threads: 44 }: 1

└── prove_with_pcs [ 4.27s | 99.83% ]
   ├── FRIPCS::commit [ 204.05ms | 4.77% ]
   │  └── commit_interleaved [ 204.04ms | 4.77% ]
   │     └── commit_interleaved_with [ 204.04ms | 4.77% ]
   │        ├── allocate codeword [ 45.34ms | 1.06% ]
   │        ├── encode_ext_batch_inplace [ 77.74ms | 1.82% ]
   │        └── commit_iterated [ 69.57ms | 1.63% ]
   │           └── BinaryMerkleTree::build [ 69.57ms | 1.63% ]
   ├── [...] [ 19.09µs | 0.00% ] (9 calls)
   ├── batch_prove_zerocheck_univariate_round [ 1.37s | 32.00% ]
   │  ├── execute_univariate_round [ 1.06s | 24.77% ]
   │  │  └── zerocheck_univariate_evals [ 1.05s | 24.52% ]
   │  │     └── extrapolate_round_evals [ 167.91ms | 3.92% ]
   │  └── fold_univariate_round [ 309.27ms | 7.23% ]
   ├── sumcheck::batch_prove [ 155.98ms | 3.64% ]
   ├── reduce_to_skipped_projection [ 263.79ms | 6.16% ]
   ├── [...] [ 35.96ms | 0.84% ] (2 calls)
   ├── greedy_evalcheck::prove [ 1.52s | 35.49% ]
   │  ├── EvalcheckProver::prove [ 1.11s | 25.84% ]
   │  │  └── EvalcheckProverState::prove_multilinear [ 1.10s | 25.79% ] (2239 calls)
   │  ├── [...] [ 37.24ms | 0.87% ] (4 calls)
   │  ├── make_non_same_query_pcs_sumchecks [ 299.73ms | 7.00% ]
   │  ├── [...] [ 783.31µs | 0.02% ]
   │  ├── sumcheck::batch_prove [ 60.30ms | 1.41% ]
   │  │  └── sumcheck::batch_prove [ 60.26ms | 1.41% ]
   │  └── [...] [ 8.58ms | 0.20% ] (2 calls)
   └── fri_pcs::prove_evaluation [ 647.25ms | 15.12% ]
      ├── CpuBackend::evaluate_partial_high [ 267.60ms | 6.25% ]
      │  └── MultilinearExtension::evaluate_partial_high [ 267.60ms | 6.25% ]
      ├── [...] [ 13.41ms | 0.31% ] (3 calls)
      ├── RegularSumcheckProver::fold [ 57.32ms | 1.34% ]
      │  └── ProverState::fold [ 57.31ms | 1.34% ]
      ├── [...] [ 30.23ms | 0.71% ] (7 calls)
      ├── fri::FRIFolder::execute_fold_round [ 47.19ms | 1.10% ]
      ├── [...] [ 43.49ms | 1.02% ] (58 calls)
      ├── fri::FRIFolder::finalize [ 70.36ms | 1.64% ]
      └── [...] [ 516.51µs | 0.01% ] (240 calls)

constraint_system::verify [ 39.78ms | 100.00% ]
├── batch_verify_zerocheck_univariate_round [ 5.33ms | 13.40% ]
├── batch_verify_with_start [ 603.74µs | 1.52% ]
├── EvalcheckVerifierState::verify [ 1.87ms | 4.69% ]
├── batch_verify_with_start [ 668.50µs | 1.68% ]
├── EvalcheckVerifierState::verify [ 4.23ms | 10.62% ]
├── [...] [ 25.93µs | 0.07% ]
├── batch_verify_with_start [ 1.13ms | 2.84% ]
├── EvalcheckVerifierState::verify [ 3.68ms | 9.26% ]
├── [...] [ 35.94µs | 0.09% ]
└── fri::FRIVerifier::verify_query [ 5.54ms | 13.92% ] (240 calls)

```

### Keccak over BabyBear31 with SHA256 Merkle trees

```

Proving 8192 NUM_HASHES
INFO     generate Keccak trace [ 257ms | 100.00% ]
INFO     prove [ 3.93s | 1.35% / 100.00% ]
INFO     ┝━ commit to trace data [ 3.08s | 10.91% / 78.35% ]
INFO     │  ┕━ coset_lde_batch [ 2.65s | 67.44% ] dims: 2633x262144 | added_bits: 1
INFO     ┝━ compute quotient polynomial [ 369ms | 9.39% ]
INFO     ┝━ commit to quotient poly chunks [ 35.5ms | 0.37% / 0.90% ]
INFO     │  ┝━ coset_lde_batch [ 11.4ms | 0.29% ] dims: 4x262144 | added_bits: 1
INFO     │  ┕━ coset_lde_batch [ 9.62ms | 0.24% ] dims: 4x262144 | added_bits: 1
INFO     ┕━ open [ 393ms | 0.02% / 10.00% ]
INFO        ┝━ compute_inverse_denominators [ 14.7ms | 0.38% ]
INFO        ┝━ reduce matrix quotient [ 167ms | 0.00% / 4.24% ] dims: 2633x524288
INFO        │  ┝━ compute opened values with Lagrange interpolation [ 86.1ms | 2.19% ]
INFO        │  ┕━ reduce rows [ 80.3ms | 2.05% ]
INFO        ┝━ reduce matrix quotient [ 140ms | 0.00% / 3.57% ] dims: 2633x524288
INFO        │  ┝━ compute opened values with Lagrange interpolation [ 59.8ms | 1.52% ]
INFO        │  ┕━ reduce rows [ 80.3ms | 2.04% ]
INFO        ┝━ reduce matrix quotient [ 8.26ms | 0.00% / 0.21% ] dims: 4x524288
INFO        │  ┝━ compute opened values with Lagrange interpolation [ 6.02ms | 0.15% ]
INFO        │  ┕━ reduce rows [ 2.23ms | 0.06% ]
INFO        ┝━ reduce matrix quotient [ 6.64ms | 0.00% / 0.17% ] dims: 4x524288
INFO        │  ┝━ compute opened values with Lagrange interpolation [ 4.49ms | 0.11% ]
INFO        │  ┕━ reduce rows [ 2.14ms | 0.05% ]
INFO        ┕━ FRI prover [ 55.7ms | 0.00% / 1.42% ]
INFO           ┝━ commit phase [ 52.5ms | 1.34% ]
INFO           ┝━ grind for proof-of-work witness [ 1.76ms | 0.04% ]
INFO           ┕━ query phase [ 1.43ms | 0.04% ]
INFO     verify [ 216ms | 97.70% / 100.00% ]
INFO     ┕━ infer log of constraint degree [ 4.98ms | 2.30% ]
```

### Keccak over BabyBear31 with Poseidon2 Merkle trees

```

Proving 8192 NUM_HASHES
INFO     generate Keccak trace [ 256ms | 100.00% ]
INFO     prove [ 5.12s | 1.04% / 100.00% ]
INFO     ┝━ commit to trace data [ 4.23s | 30.81% / 82.58% ]
INFO     │  ┕━ coset_lde_batch [ 2.65s | 51.77% ] dims: 2633x262144 | added_bits: 1
INFO     ┝━ compute quotient polynomial [ 374ms | 7.30% ]
INFO     ┝━ commit to quotient poly chunks [ 46.9ms | 0.51% / 0.92% ]
INFO     │  ┝━ coset_lde_batch [ 11.7ms | 0.23% ] dims: 4x262144 | added_bits: 1
INFO     │  ┕━ coset_lde_batch [ 9.32ms | 0.18% ] dims: 4x262144 | added_bits: 1
INFO     ┕━ open [ 417ms | 0.01% / 8.15% ]
INFO        ┝━ compute_inverse_denominators [ 14.9ms | 0.29% ]
INFO        ┝━ reduce matrix quotient [ 172ms | 0.00% / 3.36% ] dims: 2633x524288
INFO        │  ┝━ compute opened values with Lagrange interpolation [ 89.8ms | 1.76% ]
INFO        │  ┕━ reduce rows [ 81.8ms | 1.60% ]
INFO        ┝━ reduce matrix quotient [ 141ms | 0.00% / 2.76% ] dims: 2633x524288
INFO        │  ┝━ compute opened values with Lagrange interpolation [ 59.7ms | 1.17% ]
INFO        │  ┕━ reduce rows [ 81.6ms | 1.59% ]
INFO        ┝━ reduce matrix quotient [ 8.04ms | 0.00% / 0.16% ] dims: 4x524288
INFO        │  ┝━ compute opened values with Lagrange interpolation [ 5.94ms | 0.12% ]
INFO        │  ┕━ reduce rows [ 2.09ms | 0.04% ]
INFO        ┝━ reduce matrix quotient [ 6.25ms | 0.00% / 0.12% ] dims: 4x524288
INFO        │  ┝━ compute opened values with Lagrange interpolation [ 4.25ms | 0.08% ]
INFO        │  ┕━ reduce rows [ 2.00ms | 0.04% ]
INFO        ┕━ FRI prover [ 74.2ms | 0.00% / 1.45% ]
INFO           ┝━ commit phase [ 59.7ms | 1.17% ]
INFO           ┝━ grind for proof-of-work witness [ 13.1ms | 0.26% ]
INFO           ┕━ query phase [ 1.42ms | 0.03% ]
INFO     verify [ 260ms | 97.97% / 100.00% ]
INFO     ┕━ infer log of constraint degree [ 5.26ms | 2.03% ]
```
