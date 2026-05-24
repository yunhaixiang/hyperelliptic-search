# Hyperelliptic Sparse L-Polynomial Search

This project uses Axplorer to search for high-genus hyperelliptic curves over small odd prime fields with sparse L-polynomials.

For a genus `g` curve, write the L-polynomial coefficients as `a_1, ..., a_g, ...`. The project target is to make the coefficients among `a_1, ..., a_{g-1}` vanish. The sparsity of a curve is the number of nonzero coefficients among `a_1, ..., a_{g-1}`; lower sparsity means a sparser L-polynomial.

## Current Axplorer Environment

The project-specific Axplorer environment is registered as:

```bash
--env_name hyperelliptic
```

The current environment implements the first genus-2 trinomial search target. It works with monic odd-degree genus-2 curves, by default in depressed form

```text
y^2 = x^5 + a3*x^3 + a2*x^2 + a1*x + a0
```

and searches for curves whose genus-2 L-polynomial has vanishing first coefficient. For genus 2,

```text
L(T) = 1 + c1*T + c2*T^2 + p*c1*T^3 + p^2*T^4,
```

so `c1 = 0` gives a trinomial L-polynomial.

Implemented now:

- fixed-base coefficient digit tokenization, so the vocabulary size does not grow with `p`
- per-example prime tokenization, using a leading `P` block before the coefficient blocks
- deterministic primality checking for `--p`
- squarefreeness checks for the quintic model
- Sage `count_points(1)` scoring of `c1`
- clean score: invalid curves score `-1`, exact `c1 = 0` hits score `10`, misses score `1 - |c1|/(4*sqrt(p))`
- configurable coefficient-mutation local search minimizing `|c1|`
- environment registration in `src/envs/__init__.py`
- command-line parameters for genus, prime, coefficient base/digits, depressed form, and local search

Deferred:

- canonicalization by isomorphism class
- full L-polynomial computation during training; compute `c2` afterward for exported trinomial rows

## Example Shape

For a first smoke run:

```bash
python train.py \
    --env_name hyperelliptic \
    --N 2 \
    --p 1000003 \
    --degree_model odd \
    --monic true \
    --depressed true \
    --encoding_tokens coefficient_digits \
    --coefficient_base 256 \
    --max_len 32 \
    --gensize 50000 \
    --pop_size 20000 \
    --num_samples_from_model 50000 \
    --max_steps 3000 \
    --temperature 0.9 \
    --temp_span 4 \
    --inc_temp 0.05 \
    --local_search_steps 4 \
    --local_search_batch_size 32 \
    --score_batch_size 32 \
    --sage_python /Applications/SageMath-10-6.app/Contents/MacOS/Python \
    --sage_dot_dir /private/tmp/sage-dot-cache
```

Scoring and local search use a persistent Sage worker. The main Axplorer process still uses the normal Python environment for PyTorch.

Optional initial data can be loaded from a SQLite `curves` table with columns
`a0`, `a1`, `a2`, and `a3`, plus either a `p`/`prime` column or a
`run_metadata` row named `prime`:

```bash
--initial_data_sqlite exports/g2_trinomial_p1000003_batch_trinomial.sqlite \
--initial_data_max_rows 1000
```

## Python Data Generation

The `data_gen/` directory contains a Python data-generation implementation. Its planned role is to enumerate small-genus examples, organize isomorphism classes, compute invariants, and produce datasets for later sparsity analysis and PCA clustering.

Current status: basic prime fields, finite-field polynomials, hyperelliptic model validation, point counting over extensions, Hasse-Witt sparsity filtering, SQLite orbit lookup for isomorphism-class matching, L-polynomial coefficient computation, and sparsity-limited early stopping are implemented. When a sparsity bound is set, Hasse-Witt is always used before canonicalization and exact point counts.

The implementation is pure Python for now, with the option to port speed-critical pieces to C++ later.

## Relevant Files

- `src/envs/hyperelliptic.py` contains the project-specific Axplorer environment.
- `src/envs/__init__.py` registers the environment name.
- `data_gen/` contains the Python data-generation implementation.
- `README-Axplorer.md` is the upstream Axplorer README and should be treated as Axplorer reference material.
