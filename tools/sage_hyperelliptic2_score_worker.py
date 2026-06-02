#!/usr/bin/env python3
"""Persistent Sage worker for hyperelliptic2 exact trinomial-L scoring."""

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


def genus_from_degree(degree):
    degree = int(degree)
    if degree < 3:
        return None
    if degree % 2 == 1:
        return (degree - 1) // 2
    return (degree - 2) // 2


def score_row(row, p):
    field, _, x = context(p)
    try:
        coeffs = [field(int(value)) for value in row]
        if len(coeffs) < 4 or coeffs[-1] == 0:
            return invalid()
        f = sum(value * (x**degree) for degree, value in enumerate(coeffs))
        if not f.is_squarefree():
            return invalid()

        genus = genus_from_degree(f.degree())
        if genus is None or genus < 1 or f.degree() not in (2 * genus + 1, 2 * genus + 2):
            return invalid()

        curve = HyperellipticCurve(f)
        frob = curve.frobenius_polynomial()
        lpoly = [int(coeff(frob, degree)) for degree in range(2 * genus + 1)]
        # Trinomial means only constant, middle, and top terms are nonzero.
        target_coeffs = [lpoly[i] for i in range(1, 2 * genus) if i != genus]
        if any(value != 0 for value in target_coeffs):
            return invalid(lpoly=lpoly, genus=genus, target_coeffs=target_coeffs)

        middle = int(lpoly[genus])
        return {
            "score": float(genus),
            "valid": True,
            "genus": genus,
            "lpoly": lpoly,
            "middle": middle,
            "target_coeffs": target_coeffs,
        }
    except Exception:
        return invalid()


def invalid(lpoly=None, genus=None, target_coeffs=None):
    return {
        "score": -1.0,
        "valid": False,
        "genus": genus,
        "lpoly": lpoly,
        "middle": None,
        "target_coeffs": target_coeffs,
    }


def handle(request):
    p = int(request["p"])
    data = request.get("data", [])
    rows = [score_row(row, p) for row in data]
    return {
        "scores": [row["score"] for row in rows],
        "valid": [row["valid"] for row in rows],
        "genera": [row["genus"] for row in rows],
        "lpolys": [row["lpoly"] for row in rows],
        "middles": [row["middle"] for row in rows],
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
