# Hyperelliptic Data Generation

This directory has a Python data-generation implementation.

Current Python status: basic prime fields, finite-field polynomials, hyperelliptic model validation, point counting over extensions, L-polynomial coefficient computation, and sparsity-limited early stopping are implemented.

The Python implementation is in `hyperelliptic.py`.

Implemented basic Python structures:

- `PrimeField` stores an odd prime characteristic and normalizes integer representatives.
- `Polynomial` stores coefficients over a `PrimeField` in low-to-high degree order and provides derivative, remainder, gcd, monic normalization, and squarefreeness checks.
- `FiniteExtension` builds a simple polynomial-basis model of `F_{p^r}` using a monic irreducible modulus.
- `HyperellipticCurve` stores a model `y^2 = f(x)`, validates squarefreeness, counts points over extensions, and computes `a_1, ..., a_g`.

The sparsity-limited method returns `None` as soon as the sparsity among `a_1, ..., a_{g-1}` exceeds the requested limit:

```python
curve.l_polynomial_coefficients_with_sparsity_limit(max_sparsity=1)
```

## Python Tests

From the repository root:

```bash
python3 -m unittest data_gen.tests.test_hyperelliptic
```
