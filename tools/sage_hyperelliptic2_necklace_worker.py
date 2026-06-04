#!/usr/bin/env python3
"""Persistent Sage worker for hyperelliptic2 necklace algebra."""

import json
import os
import random
import sys

from sage.all import GF, PolynomialRing, matrix


TABLE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "src",
    "envs",
    "hyperelliptic2_normal_basis.json",
)

with open(TABLE_PATH, "r") as f:
    NORMAL_BASIS_TABLE = json.load(f)

_CONTEXT = {}


def normal_basis_polynomial(p, degree):
    record = NORMAL_BASIS_TABLE["primes"][str(p)][str(degree)]
    return [int(c) % p for c in record["polynomial"]]


def context(p, degree):
    key = (p, degree)
    ctx = _CONTEXT.get(key)
    if ctx is not None:
        return ctx

    field = GF(p)
    base_ring = PolynomialRing(field, "x")
    x = base_ring.gen()
    modulus_coeffs = normal_basis_polynomial(p, degree)
    modulus = base_ring(modulus_coeffs)
    ext = GF(p**degree, f"a_{p}_{degree}", modulus=modulus)
    a = ext.gen()

    columns = []
    normal_basis = []
    for i in range(degree):
        element = a ** (p**i)
        normal_basis.append(element)
        poly = element.polynomial()
        columns.append([field(poly[j]) if j <= poly.degree() else field(0) for j in range(degree)])
    basis_matrix = matrix(field, columns).transpose()
    inverse_basis_matrix = basis_matrix.inverse()
    ctx = {
        "field": field,
        "base_ring": base_ring,
        "x": x,
        "ext": ext,
        "a": a,
        "normal_basis": normal_basis,
        "inverse_basis_matrix": inverse_basis_matrix,
    }
    _CONTEXT[key] = ctx
    return ctx


def is_aperiodic(necklace):
    return all(necklace[i:] + necklace[:i] != necklace for i in range(1, len(necklace)))


def element_to_normal_coords(element, p, degree):
    ctx = context(p, degree)
    field = ctx["field"]
    poly = element.polynomial()
    vector = matrix(field, degree, 1, [field(poly[j]) if j <= poly.degree() else field(0) for j in range(degree)])
    coords = ctx["inverse_basis_matrix"] * vector
    return [int(coords[i, 0]) % p for i in range(degree)]


def element_from_normal_coords(coords, p, degree):
    ctx = context(p, degree)
    ext = ctx["ext"]
    out = ext(0)
    for coeff, basis_element in zip(coords, ctx["normal_basis"]):
        out += ext(int(coeff)) * basis_element
    return out


def factor_to_necklace(factor_coeffs, p):
    degree = len(factor_coeffs) - 1
    ctx = context(p, degree)
    ext = ctx["ext"]
    poly_ring = PolynomialRing(ext, "t")
    t = poly_ring.gen()
    factor = sum(ext(int(c)) * (t**i) for i, c in enumerate(factor_coeffs))
    roots = factor.roots(multiplicities=False)
    if not roots:
        raise ValueError("factor has no root in selected normal-basis field")
    root = roots[0]
    return [element_to_normal_coords(root ** (p**i), p, degree) for i in range(degree)]


def polynomial_to_necklaces(coeffs, p):
    coeffs = [int(c) % p for c in coeffs]
    if len(coeffs) <= 1 or coeffs[-1] == 0:
        return None
    field = GF(p)
    ring = PolynomialRing(field, "x")
    x = ring.gen()
    poly = sum(field(c) * (x**i) for i, c in enumerate(coeffs))
    if not poly.is_squarefree():
        return None
    poly = poly.monic()
    necklaces = []
    for factor, multiplicity in poly.factor():
        if multiplicity != 1:
            return None
        factor_coeffs = [int(factor[i]) % p for i in range(factor.degree() + 1)]
        necklace = factor_to_necklace(factor_coeffs, p)
        if not is_aperiodic(necklace):
            return None
        necklaces.append(necklace)
    return sorted(necklaces)


def necklace_to_factor(necklace, p):
    degree = len(necklace)
    element = element_from_normal_coords(necklace[0], p, degree)
    minpoly = element.minimal_polynomial()
    return [int(minpoly[i]) % p for i in range(minpoly.degree() + 1)]


def necklaces_to_polynomial(necklaces, p):
    field = GF(p)
    ring = PolynomialRing(field, "x")
    poly = ring(1)
    for necklace in necklaces:
        if not is_aperiodic(necklace):
            raise ValueError("not an aperiodic necklace")
        factor = ring(necklace_to_factor(necklace, p))
        poly *= factor
    return [int(poly[i]) % p for i in range(poly.degree() + 1)]


def random_necklace(degree, p):
    field = GF(p)
    ring = PolynomialRing(field, "x")
    factor = ring.irreducible_element(int(degree), algorithm="random").monic()
    coeffs = [int(factor[i]) % p for i in range(factor.degree() + 1)]
    return factor_to_necklace(coeffs, p)


def random_squarefree_monic_polynomial(degree, p):
    field = GF(p)
    ring = PolynomialRing(field, "x")
    x = ring.gen()
    while True:
        coeffs = [random.randrange(p) for _ in range(degree)] + [1]
        poly = sum(field(c) * (x**i) for i, c in enumerate(coeffs))
        if poly.is_squarefree():
            return coeffs


def handle(request):
    op = request["op"]
    p = int(request["p"])
    if op == "polynomial_to_necklaces":
        return {"necklaces": polynomial_to_necklaces(request["coefficients"], p)}
    if op == "necklaces_to_polynomial":
        return {"coefficients": necklaces_to_polynomial(request["necklaces"], p)}
    if op == "necklaces_to_polynomial_batch":
        rows = []
        errors = []
        for necklaces in request["necklaces_batch"]:
            try:
                rows.append(necklaces_to_polynomial(necklaces, p))
                errors.append(None)
            except Exception as exc:
                rows.append(None)
                errors.append(f"{type(exc).__name__}: {exc}")
        return {"coefficients": rows, "errors": errors}
    if op == "necklaces_to_factors":
        return {"factors": [necklace_to_factor(necklace, p) for necklace in request["necklaces"]]}
    if op == "necklaces_to_factors_batch":
        rows = []
        errors = []
        for necklaces in request["necklaces_batch"]:
            try:
                rows.append([necklace_to_factor(necklace, p) for necklace in necklaces])
                errors.append(None)
            except Exception as exc:
                rows.append(None)
                errors.append(f"{type(exc).__name__}: {exc}")
        return {"factors": rows, "errors": errors}
    if op == "random_necklace":
        return {"necklace": random_necklace(int(request["degree"]), p)}
    if op == "random_necklace_batch":
        return {"necklaces": [random_necklace(int(degree), p) for degree in request["degrees"]]}
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
