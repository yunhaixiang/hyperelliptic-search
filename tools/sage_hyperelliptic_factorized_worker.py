#!/usr/bin/env python3
"""Persistent Sage worker for factorized hyperelliptic curve algebra."""

import json
import random
import sys

from sage.all import GF, PolynomialRing


def ring(p):
    field = GF(int(p))
    poly_ring = PolynomialRing(field, "x")
    return field, poly_ring, poly_ring.gen()


def polynomial_from_coeffs(coeffs, p):
    field, poly_ring, x = ring(p)
    return sum(field(int(c) % p) * (x**i) for i, c in enumerate(coeffs))


def factor_polynomial(coeffs, p):
    coeffs = [int(c) % p for c in coeffs]
    if len(coeffs) <= 1 or coeffs[-1] == 0:
        return None
    poly = polynomial_from_coeffs(coeffs, p)
    if not poly.is_squarefree():
        return None
    leading_coefficient = int(poly.leading_coefficient()) % p
    monic_poly = poly.monic()
    factors = []
    for factor, multiplicity in monic_poly.factor():
        if multiplicity != 1:
            return None
        factors.append([int(factor[i]) % p for i in range(factor.degree() + 1)])
    return {"leading_coefficient": leading_coefficient, "factors": sorted(factors)}


def factors_to_polynomial(factors, p, leading_coefficient=1):
    field, poly_ring, _ = ring(p)
    poly = poly_ring(field(int(leading_coefficient) % p))
    seen = set()
    for coeffs in factors:
        coeffs = tuple(int(c) % p for c in coeffs)
        if len(coeffs) < 2 or coeffs[-1] != 1:
            raise ValueError("factor is not monic")
        if coeffs in seen:
            raise ValueError("repeated factor")
        seen.add(coeffs)
        factor = poly_ring(list(coeffs))
        if not factor.is_irreducible():
            raise ValueError("factor is not irreducible")
        poly *= factor
    return [int(poly[i]) % p for i in range(poly.degree() + 1)]


def normalize_factor_blocks(blocks, p, leading_coefficient=1):
    field, poly_ring, _ = ring(p)
    product = poly_ring(field(int(leading_coefficient) % p))
    seen = set()
    true_factors = []

    for coeffs in blocks:
        coeffs = [int(c) % p for c in coeffs]
        if len(coeffs) < 2 or coeffs[-1] != 1:
            raise ValueError("block is not monic")
        block = poly_ring(coeffs)
        product *= block
        for factor, multiplicity in block.factor():
            if multiplicity != 1:
                raise ValueError("block is not squarefree")
            factor_coeffs = tuple(int(factor[i]) % p for i in range(factor.degree() + 1))
            if factor_coeffs in seen:
                raise ValueError("repeated irreducible subfactor")
            seen.add(factor_coeffs)
            true_factors.append(list(factor_coeffs))

    return {
        "coefficients": [int(product[i]) % p for i in range(product.degree() + 1)],
        "factors": sorted(true_factors),
        "factor_degrees": sorted(len(factor) - 1 for factor in true_factors),
    }


def random_irreducible_factor(degree, p):
    _, poly_ring, _ = ring(p)
    factor = poly_ring.irreducible_element(int(degree), algorithm="random").monic()
    return [int(factor[i]) % p for i in range(factor.degree() + 1)]


def random_squarefree_monic_polynomial(degree, p):
    field, _, x = ring(p)
    while True:
        coeffs = [random.randrange(p) for _ in range(int(degree))] + [1]
        poly = sum(field(c) * (x**i) for i, c in enumerate(coeffs))
        if poly.is_squarefree():
            return coeffs


def handle(request):
    op = request["op"]
    p = int(request["p"])
    if op == "factor_polynomial":
        factorization = factor_polynomial(request["coefficients"], p)
        if factorization is None:
            return {"leading_coefficient": None, "factors": None}
        return factorization
    if op == "factors_to_polynomial":
        return {
            "coefficients": factors_to_polynomial(
                request["factors"],
                p,
                request.get("leading_coefficient", 1),
            )
        }
    if op == "factors_to_polynomial_batch":
        rows = []
        errors = []
        leading_coefficients = request.get("leading_coefficients")
        if leading_coefficients is None:
            leading_coefficients = [1] * len(request["factors_batch"])
        for factors, leading_coefficient in zip(request["factors_batch"], leading_coefficients):
            try:
                rows.append(factors_to_polynomial(factors, p, leading_coefficient))
                errors.append(None)
            except Exception as exc:
                rows.append(None)
                errors.append(f"{type(exc).__name__}: {exc}")
        return {"coefficients": rows, "errors": errors}
    if op == "normalize_factor_blocks":
        return normalize_factor_blocks(
            request["blocks"],
            p,
            request.get("leading_coefficient", 1),
        )
    if op == "normalize_factor_blocks_batch":
        rows = []
        errors = []
        leading_coefficients = request.get("leading_coefficients")
        if leading_coefficients is None:
            leading_coefficients = [1] * len(request["blocks_batch"])
        for blocks, leading_coefficient in zip(request["blocks_batch"], leading_coefficients):
            try:
                rows.append(normalize_factor_blocks(blocks, p, leading_coefficient))
                errors.append(None)
            except Exception as exc:
                rows.append(None)
                errors.append(f"{type(exc).__name__}: {exc}")
        return {"rows": rows, "errors": errors}
    if op == "random_irreducible_factor":
        return {"factor": random_irreducible_factor(int(request["degree"]), p)}
    if op == "random_irreducible_factor_batch":
        return {"factors": [random_irreducible_factor(int(degree), p) for degree in request["degrees"]]}
    if op == "random_squarefree_monic_polynomial":
        return {"coefficients": random_squarefree_monic_polynomial(int(request["degree"]), p)}
    raise ValueError(f"unknown op: {op}")


def main():
    random.seed(0)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            response = handle(json.loads(line))
        except Exception as exc:
            response = {"error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(response), flush=True)


if __name__ == "__main__":
    main()
