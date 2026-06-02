#!/usr/bin/env sage -python
"""Precompute B_{n,p} for hyperelliptic2 necklace tokenization.

Run with Sage Python:

    sage -python tools/precompute_hyperelliptic2_normal_basis.sage.py

The output is src/envs/hyperelliptic2_normal_basis.json. Runtime tokenization
loads this file and refuses to compute missing B_{n,p} values.
"""

import argparse
import json
import os
from math import gcd

from sage.all import GF, PolynomialRing, cyclotomic_polynomial, is_prime, matrix, primitive_root


DEFAULT_PRIMES = (3, 5, 7, 11)
DEFAULT_MAX_DEGREE = 1999


def coeffs_low_to_high(poly, p):
    degree = poly.degree()
    return [int(poly[i]) % p for i in range(degree + 1)]


def multiplicative_order(a, modulus):
    a %= modulus
    if gcd(a, modulus) != 1:
        return None
    value = 1
    for exponent in range(1, modulus + 1):
        value = (value * a) % modulus
        if value == 1:
            return exponent
    return None


def is_normal_polynomial(poly, p, n):
    field = GF(p)
    K = GF(p**n, "a", modulus=poly)
    a = K.gen()
    rows = []
    for j in range(n):
        rows.append([(a ** (p**i)).polynomial()[j] for i in range(n)])
    return matrix(field, rows).rank() == n


def type_i_candidate(p, n, R):
    r = n + 1
    if not is_prime(r) or multiplicative_order(p, r) != n:
        return None
    x = R.gen()
    poly = sum(x**i for i in range(n + 1))
    return poly if poly.is_irreducible() and is_normal_polynomial(poly, p, n) else None


def gaussian_period_candidate(p, n, R, k):
    r = k * n + 1
    if not is_prime(r) or gcd(p, r) != 1 or multiplicative_order(p, r) != n:
        return None

    field = GF(p)
    z_ring = PolynomialRing(field, "z")
    z = z_ring.gen()
    cyclo = z_ring(cyclotomic_polynomial(r))
    factors = [factor for factor, _ in cyclo.factor() if factor.degree() == n]
    if not factors:
        return None

    g = int(primitive_root(r))
    # Try all degree-n factors and keep the lexicographically first resulting
    # normal polynomial. This keeps the precompute deterministic.
    candidates = []
    for factor in factors:
        K = GF(p**n, f"zeta_{p}_{n}_{k}", modulus=factor)
        zeta = K.gen()
        beta = K.zero()
        for j in range(k):
            beta += zeta ** pow(g, j * n, r)
        poly = R(beta.minpoly())
        if poly.degree() == n and poly.is_irreducible() and is_normal_polynomial(poly, p, n):
            candidates.append(poly.monic())
    return min(candidates, key=lambda poly: coeffs_low_to_high(poly, p)) if candidates else None


def type_ii_candidate(p, n, R):
    r = 2 * n + 1
    if not is_prime(r) or gcd(p, r) != 1 or multiplicative_order(p, r) != 2 * n:
        return None

    field = GF(p)
    z_ring = PolynomialRing(field, "z")
    z = z_ring.gen()
    cyclo = z_ring(cyclotomic_polynomial(r))
    candidates = []
    for factor, _ in cyclo.factor():
        if factor.degree() != 2 * n:
            continue
        K = GF(p ** (2 * n), f"zeta_type2_{p}_{n}", modulus=factor)
        zeta = K.gen()
        beta = zeta + zeta**(-1)
        poly = R(beta.minimal_polynomial())
        if poly.degree() == n and poly.is_irreducible() and is_normal_polynomial(poly, p, n):
            candidates.append(poly.monic())
    return min(candidates, key=lambda poly: coeffs_low_to_high(poly, p)) if candidates else None


def gaussian_candidate(p, n, R, max_type):
    # Type I and II have already been tried. Prefer the smallest Gaussian
    # period type k that works; smaller k usually gives cheaper arithmetic.
    for k in range(3, max_type + 1):
        candidate = gaussian_period_candidate(p, n, R, k)
        if candidate is not None:
            return candidate, f"gaussian_period_type_{k}"
    return None, None


def fallback_candidate(p, n, R, samples):
    # Keep fallback cheap and resumable: after Type I/II/Gaussian attempts fail,
    # accept the first normal irreducible found instead of optimizing sparsity.
    first = R.irreducible_element(n).monic()
    if is_normal_polynomial(first, p, n):
        return first

    for _ in range(samples):
        poly = R.irreducible_element(n, algorithm="random").monic()
        if is_normal_polynomial(poly, p, n):
            return poly

    raise RuntimeError(f"could not find normal polynomial for p={p}, n={n}")


def choose_polynomial(p, n, gaussian_max_type, fallback_samples):
    field = GF(p)
    R = PolynomialRing(field, "x")

    candidate = type_i_candidate(p, n, R)
    if candidate is not None:
        return {"kind": "type_i", "polynomial": coeffs_low_to_high(candidate, p)}

    candidate = type_ii_candidate(p, n, R)
    if candidate is not None:
        return {"kind": "type_ii", "polynomial": coeffs_low_to_high(candidate, p)}

    candidate, kind = gaussian_candidate(p, n, R, gaussian_max_type)
    if candidate is not None:
        return {"kind": kind, "polynomial": coeffs_low_to_high(candidate, p)}

    candidate = fallback_candidate(p, n, R, fallback_samples)
    return {"kind": "fallback_sparse_normal", "polynomial": coeffs_low_to_high(candidate, p)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="src/envs/hyperelliptic2_normal_basis.json")
    parser.add_argument("--max-degree", type=int, default=DEFAULT_MAX_DEGREE)
    parser.add_argument("--primes", default=",".join(str(p) for p in DEFAULT_PRIMES))
    parser.add_argument("--gaussian-max-type", type=int, default=64)
    parser.add_argument("--fallback-samples", type=int, default=64)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    primes = [int(p.strip()) for p in args.primes.split(",") if p.strip()]
    if any(p not in DEFAULT_PRIMES for p in primes):
        raise ValueError("hyperelliptic2 only supports precomputing primes 3, 5, 7, 11")
    if args.max_degree >= 2000:
        raise ValueError("--max-degree must be < 2000")

    if args.resume and os.path.exists(args.output):
        with open(args.output, "r") as f:
            data = json.load(f)
    else:
        data = {"version": 1, "max_degree": args.max_degree, "primes": {}}

    data["max_degree"] = max(int(data.get("max_degree", 0)), args.max_degree)
    for p in primes:
        rows = data["primes"].setdefault(str(p), {})
        for n in range(1, args.max_degree + 1):
            if str(n) in rows:
                continue
            record = choose_polynomial(p, n, args.gaussian_max_type, args.fallback_samples)
            rows[str(n)] = record
            print(f"p={p} n={n} {record['kind']}", flush=True)
            with open(args.output, "w") as f:
                json.dump(data, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
