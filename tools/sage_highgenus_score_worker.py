#!/usr/bin/env python3
"""Persistent Sage worker for high-genus hyperelliptic exact scoring."""

import json
import sys

from sage.all import GF, HyperellipticCurve, PolynomialRing


_CACHE = {}


def context(p):
    ctx = _CACHE.get(p)
    if ctx is not None:
        return ctx
    field = GF(p)
    ring = PolynomialRing(field, "x")
    ctx = (field, ring, ring.gen())
    _CACHE[p] = ctx
    return ctx


def coeff(poly, degree):
    try:
        return int(poly[degree])
    except Exception:
        return int(poly.coefficient({poly.parent().gen(): degree}))


def score_from_sparsity(genus, sparsity):
    if sparsity == 0:
        return 1000.0 * genus + 100.0
    if sparsity == 1:
        return 1000.0 * genus
    return -float(sparsity)


def score_row(row, p, genus):
    field, _, x = context(p)
    try:
        f = x ** (2 * genus + 1)
        for degree, value in enumerate(row):
            f += field(int(value)) * (x**degree)
        curve = HyperellipticCurve(f)
        frob = curve.frobenius_polynomial()
        target_coeffs = [coeff(frob, 2 * genus - i) for i in range(1, genus)]
        sparsity = sum(1 for c in target_coeffs if c != 0)
        return {
            "score": score_from_sparsity(genus, sparsity),
            "valid": True,
            "sparsity": sparsity,
            "target_coeffs": target_coeffs,
        }
    except Exception as exc:
        return {
            "score": -1.0,
            "valid": False,
            "sparsity": None,
            "target_coeffs": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def handle(request):
    p = int(request["p"])
    genus = int(request["genus"])
    data = request.get("data", [])
    rows = [score_row(row, p, genus) for row in data]
    return {
        "scores": [row["score"] for row in rows],
        "valid": [row["valid"] for row in rows],
        "sparsities": [row["sparsity"] for row in rows],
        "target_coeffs": [row["target_coeffs"] for row in rows],
    }


def main():
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
