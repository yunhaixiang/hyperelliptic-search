# Hyperelliptic Sparse L-Polynomial Search

This project uses Axplorer to search for high-genus hyperelliptic curves over small odd prime fields with sparse L-polynomials.

For a genus `g` curve, write the L-polynomial coefficients as `a_1, ..., a_g, ...`. The project target is to make the coefficients among `a_1, ..., a_{g-1}` vanish. The sparsity of a curve is the number of nonzero coefficients among `a_1, ..., a_{g-1}`; lower sparsity means a sparser L-polynomial.

## Current Axplorer Environment

The project-specific Axplorer environment is registered as:

```bash
--env_name hyperelliptic
```

The current environment is intentionally a scaffold. It defines the Axplorer object representation and tokenizer, but it does not implement the mathematical scoring or local search backend yet. Those parts are expected to be supplied later by C++ code for speed.

Implemented now:

- coefficient-vector representation for hyperelliptic polynomial models
- coefficient tokenization over `F_p`
- environment registration in `src/envs/__init__.py`
- command-line parameters for genus, prime, degree model, and monicity

Deferred intentionally:

- squarefreeness and curve validity checks
- L-polynomial computation
- sparsity scoring
- local search and repair
- C++ backend integration

## Python Data Generation

The `data_gen/` directory contains a Python data-generation implementation. Its planned role is to enumerate small-genus examples, organize isomorphism classes, compute invariants, and produce datasets for later sparsity analysis and PCA clustering.

Current status: basic prime fields, finite-field polynomials, hyperelliptic model validation, point counting over extensions, Hasse-Witt filtering, SQLite orbit lookup for isomorphism-class matching, L-polynomial coefficient computation, and sparsity-limited early stopping are implemented.

The implementation is pure Python for now, with the option to port speed-critical pieces to C++ later.

## Relevant Files

- `src/envs/hyperelliptic.py` contains the project-specific Axplorer scaffold.
- `src/envs/__init__.py` registers the environment name.
- `data_gen/` contains the Python data-generation implementation.
- `README-Axplorer.md` is the upstream Axplorer README and should be treated as Axplorer reference material.

## Example Shape

The intended training entry point will look like:

```bash
python train.py \
    --env_name hyperelliptic \
    --N 10 \
    --p 3 \
    --degree_model odd \
    --monic true \
    --encoding_tokens coefficients
```

This command is not expected to complete a real search until the scoring and local-search backend is connected.
