![Binius logo](doc/Logo.png "Binius logo")

# Binius

Binius is a Rust library implementing a cryptographic *succinct non-interactive argument of knowledge* (SNARK) over towers of binary fields. The techniques are described formally in the papers *[Succinct Arguments over Towers of Binary Fields](https://eprint.iacr.org/2023/1784)* and *[Polylogarithmic Proofs for Multilinears over Binary Towers](https://eprint.iacr.org/2024/504)*.

Comprehensive documentation is available at [binius.xyz](https://www.binius.xyz/).

:bangbang: **_Update (2025-09-09)_: This codebase is archived and officially succeeded by [Binius64](https://github.com/IrreducibleOSS/binius64). Read Irreducible's [announcement blog post](https://www.irreducible.com/posts/announcing-binius64) for Binius64 for more context.** :bangbang:

## Usage

At this stage, the primary interfaces are the unit tests and benchmarks. The benchmarks use the [criterion](https://docs.rs/criterion/0.3.4/criterion/) library.

To run the benchmarks, use the command `cargo bench`. To run the unit tests, use the command `cargo test`.

Binius implements optimizations for certain target architectures. To enable these, export the environment variable

```bash
RUSTFLAGS="-C target-cpu=native"
```

Binius has notable optimizations on Intel processors featuring the [Galois Field New Instructions](https://networkbuilders.intel.com/solutionslibrary/galois-field-new-instructions-gfni-technology-guide) (GFNI) instruction set extension. To determine if your processor supports this feature, run

```bash
rustc --print cfg -C target-cpu=native | grep gfni
```

If the output of the command above is empty, the processor does not support these instructions.

When including binius as a dependency, it is recommended to add the following lines to your `Cargo.toml` file to have optimizations across crates

```toml
[profile.release]
lto = "fat"
```

### Examples

There are examples of simple commit-and-prove SNARKs in the `examples` directory. For example, you may run

```bash
cargo run --release --example sha256_circuit
```

By default, all the examples are run on relatively small instance sizes. The examples accept command line arguments to specify the size of the data.

### API Documentation

Rust API documentation is hosted at <https://docs.binius.xyz/>. The generated HTML pages include [KaTeX](https://katex.org/) so that LaTeX in Rust docs is rendered correctly.

To generate the documentation locally, run

```bash
cargo doc --no-deps
```

## Support

This project is under active development. The developers will make breaking changes at will. Any modules that are stabilized will be explicitly documented as such.

We use GitHub's issue system for tracking bugs, features, and other development tasks.

This codebase certainly contains bugs at this point in its development. *We discourage the use of this library for security-critical applications until future notice.* Any bugs, including those affecting the security of the system, may be filed publicly as an issue.

## Authors

Binius is developed by [Irreducible](https://www.irreducible.com).

## License

Copyright 2023-2025 Irreducible Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
