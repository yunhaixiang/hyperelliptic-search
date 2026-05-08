# Hyperelliptic Data Generation

This directory has a Python data-generation implementation.

Current Python status: basic prime fields, finite-field polynomials, hyperelliptic model validation, point counting over extensions, Hasse-Witt sparsity filtering, L-polynomial coefficient computation, and sparsity-limited early stopping are implemented.

The Python implementation is in `hyperelliptic.py`.

Implemented basic Python structures:

- `PrimeField` stores an odd prime characteristic and normalizes integer representatives.
- `Polynomial` stores coefficients over a `PrimeField` in low-to-high degree order and provides derivative, remainder, gcd, monic normalization, and squarefreeness checks.
- `FiniteExtension` builds a simple polynomial-basis model of `F_{p^r}` using a monic irreducible modulus.
- `PointCountingContext` caches finite extensions, field elements, quadratic residues, and powers of `x` for reuse across many curves.
- `EnumerationContext` owns enumeration-level caches, uses rational-branch-count plus SQLite orbit lookup for isomorphism matching, canonicalizes binary forms under `PGL_2(F_p)` up to square scalar on orbit-cache misses, tracks seen isomorphism classes, and caches mod-`p` and exact L-polynomial results by canonical key.
- `HyperellipticCurve` stores a model `y^2 = f(x)`, validates squarefreeness, computes Hasse-Witt data, counts points over extensions, and computes `a_1, ..., a_g`.

The sparsity-limited method returns `None` as soon as the sparsity among `a_1, ..., a_{g-1}` exceeds the requested limit:

```python
curve.l_polynomial_coefficients_with_sparsity_limit(max_sparsity=1)
```

The Hasse-Witt filter gives a fast safe rejection test modulo `p`:

```python
curve.hasse_witt_matrix()
curve.l_polynomial_coefficients_mod_p()
curve.passes_hasse_witt_sparsity_filter(max_sparsity=1)
```

When a sparsity bound is supplied, enumeration always runs this Hasse-Witt mod-`p` test before canonicalization. Mod-`p` sparsity failures are counted as `rejected_hasse_witt_uncanonicalized`, are not inserted into the canonical-class tables, and never reach exact extension point counts.

For enumeration over a fixed field and degree, share one point-counting context:

```python
field = PrimeField(5)
context = PointCountingContext(field, polynomial_degree=5)
curve = HyperellipticCurve(Polynomial(field, [1, 1, 0, 0, 0, 1]), point_counting_context=context)
```

For full enumeration, use `EnumerationContext` instead:

```python
context = EnumerationContext(prime=5, genus=2)
polynomial = context.polynomial([1, 1, 0, 0, 0, 1])

if context.is_new_isomorphism_class(polynomial):
    mod_p = context.l_polynomial_coefficients_mod_p(polynomial)
    exact = context.l_polynomial_coefficients_with_sparsity_limit(polynomial, max_sparsity=1)
```

To write enumeration output to SQLite while running, pass `sqlite_path` and stream coefficient vectors through the output helper:

```python
context = EnumerationContext(prime=5, genus=2, sqlite_path="curves.sqlite")
stats = context.process_polynomials_for_output(coefficient_vectors, max_sparsity=1)
timing = context.timing_summary()
context.close_sqlite()
```

Reusing the same `sqlite_path` resumes from previous canonical-class rows by loading `curve_cache` into the in-memory indexes at startup:

```python
context = EnumerationContext(prime=5, genus=2, sqlite_path="curves.sqlite")
```

There is also a command-line runner:

```bash
python3 -m data_gen.hyperelliptic --p 5 --genus 2 --max-sparsity 2
```

Omit `--max-sparsity` to compute without a sparsity restriction:

```bash
python3 -m data_gen.hyperelliptic --p 5 --genus 2
```

For high-genus sparse search, use a sparsity bound; Hasse-Witt filtering is automatic:

```bash
python3 -m data_gen.hyperelliptic --p 5 --genus 20 --max-sparsity 1
```

By default, the command uses SQLite BLOB orbit lookup and stores orbit keys for complete canonical-class enumeration. If `--max-sparsity` is set, Hasse-Witt runs before canonicalization; failures are counted as `rejected_hasse_witt_uncanonicalized` and are not inserted into the canonical-class tables. Hasse-Witt survivors use SQLite orbit lookup before full canonicalization.

The current enumeration modes always enumerate both degree families: degree `2g+1` models first, then degree `2g+2` models. The runner uses a fixed leading-coefficient normalization: degree `2g+1` models are enumerated monic, and degree `2g+2` models are enumerated with leading coefficient `1` and the smallest nonsquare in `F_p`. Since odd models are included, squarefree degree `2g+2` models with an `F_p`-rational branch point are skipped as `covered_by_odd_model`; a `PGL_2(F_p)` transform can move that branch point to infinity, giving a degree `2g+1` representative.

The default enumeration mode is `--enumeration-mode lexicographic`. `--enumeration-mode support` enumerates by increasing support size of `f(x)`, counting the fixed leading coefficient as part of the support. `--enumeration-mode lexicoskipping` follows the same normalized lexicographic order, but after a configurable drought without a new canonical isomorphism class, it skips ahead by an adaptive jump size. The drought starts immediately, so long singular or Hasse-Witt rejection blocks can be skipped from the beginning of a run. After each skip, it processes a probe window before allowing another skip; if the probe finds a new canonical class, the skip size resets, otherwise the next skip grows. If omitted, lexicoskipping parameters are estimated from `p` and `g`: `--lexicoskip-drought` uses `max(2000, min(20000, 4*p^min(g,5)))`, `--lexicoskip-initial-skip` matches the drought estimate, `--lexicoskip-max-skip` is `20` times the initial skip, and `--lexicoskip-probe-window` uses `max(500, min(5000, 4*p^min(g,4)))`.

The runner prints progress as:

```text
progress: processed/total
skipped: S
sparse_presentations: N
sparse_isomorphism_classes: M
canonicalized_isomorphism_classes: K
-
```

The progress line always reports `canonicalized_isomorphism_classes`, since Hasse-Witt failures are not canonicalized in bounded sparse runs.

The SQLite output uses:

- `orbit_cache`: BLOB lookup table for `(rational_branch_count, ground_point_count, hasse_witt_lpoly_mod_p, orbit_key) -> canonical_key` in bounded sparse runs. Older databases without the Hasse-Witt column are still readable.
- `curve_cache`: per-canonical-key computation results for the output file's sparsity bound, including rational branch count, L-polynomial data, and rejection status.
- `sparse_curves`: sparse survivors with their exact `a_1, ..., a_g` coefficients.
- `enumeration_summary`: one-row run summary with `prime`, `genus`, `max_sparsity`, whether bounded Hasse-Witt filtering was enabled, enumeration settings, progress counts, and timing fields.

In `curve_cache` and `sparse_curves`, `coefficients` are readable JSON integer lists. In `sparse_curves`, the output `lpoly` is also readable JSON. Internal cache keys and intermediate L-polynomial fields are stored as compact BLOBs.

With `--max-sparsity`, Hasse-Witt survivors use SQLite orbit lookup to skip repeated canonicalization when a presentation is already in a stored `PGL_2` orbit. The difference is that Hasse-Witt failures are not canonicalized or stored.

## Python Tests

From the repository root:

```bash
python3 -m unittest data_gen.tests.test_hyperelliptic
```
