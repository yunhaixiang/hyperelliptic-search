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


def hasse_witt_target_coeffs(poly, p, genus):
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
    return [int(coeffs[genus - i]) % p for i in range(1, genus)]


def invalid(lpoly=None, genus=None, target_coeffs=None, hw_target_coeffs=None, hw_zero_count=None):
    return {
        "score": -1.0,
        "valid": False,
        "genus": genus,
        "lpoly": lpoly,
        "middle": None,
        "target_coeffs": target_coeffs,
        "hw_target_coeffs": hw_target_coeffs,
        "hw_zero_count": hw_zero_count,
        "lpoly_zero_count": None,
    }


def coefficient_bound(p, genus, index):
    return math.comb(2 * genus, index) * (float(p) ** (0.5 * index))


def archimedean_tie_break(target_coeffs, p, genus):
    normalized_sizes = []
    for index, value in enumerate(target_coeffs, start=1):
        if value == 0:
            continue
        bound = coefficient_bound(p, genus, index)
        normalized = abs(float(value)) / bound if bound > 0 else 1.0
        normalized_sizes.append(min(1.0, normalized))
    if not normalized_sizes:
        return 0.0
    return max(0.0, min(1.0, 1.0 - (sum(normalized_sizes) / len(normalized_sizes))))


def hasse_witt_tie_break(lpoly_zero_count, hw_zero_count, target_len):
    max_extra_modp_zeros = target_len - lpoly_zero_count
    if max_extra_modp_zeros <= 0:
        return 0.0
    return (int(hw_zero_count) - lpoly_zero_count) / max_extra_modp_zeros


def lpoly_sparsity_score(lpoly, hw_zero_count, genus, p, tie_break_mode):
    target_coeffs = [int(lpoly[index]) for index in range(1, genus)]
    if not target_coeffs:
        return 0.0, target_coeffs, 0
    target_len = len(target_coeffs)
    lpoly_zero_count = sum(1 for value in target_coeffs if value == 0)
    if lpoly_zero_count == target_len:
        return float(target_len), target_coeffs, lpoly_zero_count
    if tie_break_mode in {"average", "archimedean"}:
        tie_break = archimedean_tie_break(target_coeffs, p, genus)
    else:
        tie_break = hasse_witt_tie_break(lpoly_zero_count, hw_zero_count, target_len)
    tie_break = max(0.0, min(1.0, float(tie_break)))
    score = min(float(lpoly_zero_count + tie_break), float(target_len) - 1e-6)
    return score, target_coeffs, lpoly_zero_count


def precheck_row(row, p, sparsity_reject_threshold):
    field, _, x = context(p)
    try:
        coeffs = [field(int(value)) for value in row]
        if len(coeffs) < 4 or coeffs[-1] == 0:
            return {"valid": False, "row": invalid()}
        f = sum(value * (x**degree) for degree, value in enumerate(coeffs))
        if not f.is_squarefree():
            return {"valid": False, "row": invalid()}

        genus = genus_from_degree(f.degree())
        if genus is None or genus < 1 or f.degree() not in (2 * genus + 1, 2 * genus + 2):
            return {"valid": False, "row": invalid()}

        factor_degrees = [factor.degree() for factor, multiplicity in f.factor() for _ in range(multiplicity)]
        if not is_mod2_allowed_factor_degrees(factor_degrees, genus, f.degree() == 2 * genus + 1):
            return {"valid": False, "row": invalid(genus=genus)}

        hw_target_coeffs = hasse_witt_target_coeffs(f, p, genus)
        hw_zero_count = sum(1 for value in hw_target_coeffs if value == 0)
        return {
            "valid": True,
            "poly": f,
            "genus": genus,
            "hw_target_coeffs": hw_target_coeffs,
            "hw_zero_count": hw_zero_count,
        }
    except Exception:
        return {"valid": False, "row": invalid()}


def score_prechecked(prechecked, p, tie_break_mode, sparsity_reject_threshold):
    if not prechecked["valid"]:
        return prechecked["row"]
    genus = int(prechecked["genus"])
    hw_zero_count = int(prechecked["hw_zero_count"])
    hw_target_coeffs = prechecked["hw_target_coeffs"]
    try:
        curve = HyperellipticCurve(prechecked["poly"])
        frob = curve.frobenius_polynomial()
        lpoly = [int(coeff(frob, degree)) for degree in range(2 * genus + 1)]
        target_coeffs = [int(lpoly[index]) for index in range(1, genus)]
        lpoly_zero_count = sum(1 for value in target_coeffs if value == 0)
        score, target_coeffs, lpoly_zero_count = lpoly_sparsity_score(
            lpoly,
            hw_zero_count,
            genus,
            p,
            tie_break_mode,
        )
        middle = int(lpoly[genus])
        return {
            "score": score,
            "valid": score >= 0.0,
            "genus": genus,
            "lpoly": lpoly,
            "middle": middle,
            "target_coeffs": target_coeffs,
            "hw_target_coeffs": hw_target_coeffs,
            "hw_zero_count": hw_zero_count,
            "lpoly_zero_count": lpoly_zero_count,
        }
    except Exception:
        return invalid(genus=genus, hw_target_coeffs=hw_target_coeffs, hw_zero_count=hw_zero_count)


def handle(request):
    p = int(request["p"])
    tie_break_mode = str(request.get("score_tiebreak_mode", "hasse_witt")).lower()
    if tie_break_mode not in {"hasse_witt", "average", "archimedean"}:
        raise ValueError(f"unknown score_tiebreak_mode: {tie_break_mode}")
    sparsity_reject_threshold = int(request.get("sparsity_reject_threshold", -1))
    data = request.get("data", [])
    prechecked = [precheck_row(row, p, sparsity_reject_threshold) for row in data]
    rows = [
        score_prechecked(row, p, tie_break_mode, sparsity_reject_threshold)
        for row in prechecked
    ]
    return {
        "scores": [row["score"] for row in rows],
        "valid": [row["valid"] for row in rows],
        "genera": [row["genus"] for row in rows],
        "lpolys": [row["lpoly"] for row in rows],
        "middles": [row["middle"] for row in rows],
        "target_coeffs": [row["target_coeffs"] for row in rows],
        "hw_target_coeffs": [row["hw_target_coeffs"] for row in rows],
        "hw_zero_counts": [row["hw_zero_count"] for row in rows],
        "lpoly_zero_counts": [row["lpoly_zero_count"] for row in rows],
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
