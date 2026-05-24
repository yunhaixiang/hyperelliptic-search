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
- deterministic primality checking for `--p`
- squarefreeness checks for the quintic model
- fast NumPy Legendre-table scoring of `c1 = sum_x chi(f(x))`
- clean score: invalid curves score `-1`, exact `c1 = 0` hits score `10`, misses score `1 - |c1|/(4*sqrt(p))`
- configurable coefficient-mutation local search minimizing `|c1|`
- environment registration in `src/envs/__init__.py`
- command-line parameters for genus, prime, coefficient base/digits, depressed form, and local search

Deferred:

- canonicalization by isomorphism class
- full L-polynomial computation beyond `c1`
- C++/FLINT batch scoring for primes too large for a Legendre table

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
    --score_batch_size 8 \
    --score_x_chunk_size 32768
```

The NumPy scorer precomputes a Legendre table of size `p`, capped by `--max_legendre_table_p` and defaulting to `10000000`.

## Python Data Generation

The `data_gen/` directory contains a Python data-generation implementation. Its planned role is to enumerate small-genus examples, organize isomorphism classes, compute invariants, and produce datasets for later sparsity analysis and PCA clustering.

Current status: basic prime fields, finite-field polynomials, hyperelliptic model validation, point counting over extensions, Hasse-Witt sparsity filtering, SQLite orbit lookup for isomorphism-class matching, L-polynomial coefficient computation, and sparsity-limited early stopping are implemented. When a sparsity bound is set, Hasse-Witt is always used before canonicalization and exact point counts.

The implementation is pure Python for now, with the option to port speed-critical pieces to C++ later.

## Relevant Files

- `src/envs/hyperelliptic.py` contains the project-specific Axplorer environment.
- `src/envs/__init__.py` registers the environment name.
- `data_gen/` contains the Python data-generation implementation.
- `README-Axplorer.md` is the upstream Axplorer README and should be treated as Axplorer reference material.
