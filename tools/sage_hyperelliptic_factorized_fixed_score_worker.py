#!/usr/bin/env python3
"""Persistent Sage worker for fixed-genus factorized hyperelliptic scoring."""

import json
import math
import os
import sys

from sage.all import GF, HyperellipticCurve, Matrix, PolynomialRing

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.envs.hyperelliptic2_mod2 import is_mod2_allowed_factor_degrees


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


def hasse_witt_target_sparsity(poly, p, genus):
    powered = poly ** ((p - 1) // 2)
    field = poly.parent().base_ring()
    matrix_rows = []
    for i in range(1, genus + 1):
        matrix_rows.append([
            field(powered[p * i - j]) if 0 <= p * i - j <= powered.degree() else field(0)
            for j in range(1, genus + 1)
        ])
    matrix = Matrix(field, matrix_rows)
    z = PolynomialRing(field, "z").gen()
    coeffs = matrix.charpoly(z).list()
    target_coeffs = [coeffs[genus - i] for i in range(1, genus)]
    return sum(1 for value in target_coeffs if value != 0)


def invalid(lpoly=None, genus=None, target_coeffs=None):
    return {
        "score": -1.0,
        "valid": False,
        "genus": genus,
        "lpoly": lpoly,
        "middle": None,
        "target_coeffs": target_coeffs,
    }


def coefficient_bound(p, genus, index):
    return math.comb(2 * genus, index) * (float(p) ** (0.5 * index))


def lpoly_closeness_score(lpoly, p, genus):
    target_coeffs = [int(lpoly[index]) for index in range(1, genus)]
    if not target_coeffs:
        return 0.0, target_coeffs
    zero_count = sum(1 for value in target_coeffs if value == 0)
    nonzero_terms = []
    for index, value in enumerate(target_coeffs, start=1):
        if value == 0:
            continue
        bound = coefficient_bound(p, genus, index)
        normalized = abs(float(value)) / bound if bound > 0 else float(value != 0)
        nonzero_terms.append(max(0.0, 1.0 - min(1.0, normalized)))
    tie_break = sum(nonzero_terms) / len(nonzero_terms) if nonzero_terms else 0.0
    return float(zero_count + tie_break), target_coeffs


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

        factor_degrees = [factor.degree() for factor, multiplicity in f.factor() for _ in range(multiplicity)]
        if not is_mod2_allowed_factor_degrees(factor_degrees, genus, f.degree() == 2 * genus + 1):
            return invalid(genus=genus)

        if hasse_witt_target_sparsity(f, p, genus) > 0:
            return invalid(genus=genus)

        curve = HyperellipticCurve(f)
        frob = curve.frobenius_polynomial()
        lpoly = [int(coeff(frob, degree)) for degree in range(2 * genus + 1)]
        score, target_coeffs = lpoly_closeness_score(lpoly, p, genus)
        middle = int(lpoly[genus])
        return {
            "score": score,
            "valid": score >= 0.0,
            "genus": genus,
            "lpoly": lpoly,
            "middle": middle,
            "target_coeffs": target_coeffs,
        }
    except Exception:
        return invalid()


def handle(request):
    p = int(request["p"])
    rows = [score_row(row, p) for row in request.get("data", [])]
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
