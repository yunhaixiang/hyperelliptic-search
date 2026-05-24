#!/usr/bin/env python3
"""Persistent Sage worker for genus-2 hyperelliptic scoring.

Protocol: read one JSON object per line from stdin and write one JSON object per
line to stdout.

Request:
    {
      "p": 1000003,
      "depressed": true,
      "data": [[a0, a1, a2, a3], ...]
    }

Response:
    {
      "scores": [...],
      "c1s": [...],
      "c2s": [...],
      "valid": [...],
      "lpolys": [[1,c1,c2,p*c1,p^2], ...]
    }
"""

import json
import math
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


def score_from_c1(c1, p):
    if c1 == 0:
        return 10.0
    return 1.0 - abs(c1) / (4.0 * math.sqrt(p))


def sage_count_points_1(curve):
    count = curve.count_points(1)
    if isinstance(count, (list, tuple)):
        return int(count[0])
    return int(count)


def score_row(row, p, depressed, mode):
    field, _, x = context(p)
    try:
        if depressed:
            a0, a1, a2, a3 = [field(int(v)) for v in row]
            f = x**5 + a3 * x**3 + a2 * x**2 + a1 * x + a0
        else:
            a0, a1, a2, a3, a4 = [field(int(v)) for v in row]
            f = x**5 + a4 * x**4 + a3 * x**3 + a2 * x**2 + a1 * x + a0
        curve = HyperellipticCurve(f)
        if mode == "c1":
            c1 = sage_count_points_1(curve) - p - 1
            return score_from_c1(c1, p), c1, None, True, None
        frob = curve.frobenius_polynomial()
        c1 = coeff(frob, 3)
        c2 = coeff(frob, 2)
        lpoly = [1, c1, c2, p * c1, p * p]
        return score_from_c1(c1, p), c1, c2, True, lpoly
    except Exception:
        return -1.0, None, None, False, None


def handle(request):
    p = int(request["p"])
    depressed = bool(request.get("depressed", True))
    mode = request.get("mode", "frob")
    data = request.get("data", [])
    scores = []
    c1s = []
    c2s = []
    valid = []
    lpolys = []
    for row in data:
        score, c1, c2, is_valid, lpoly = score_row(row, p, depressed, mode)
        scores.append(score)
        c1s.append(c1)
        c2s.append(c2)
        valid.append(is_valid)
        lpolys.append(lpoly)
    return {"scores": scores, "c1s": c1s, "c2s": c2s, "valid": valid, "lpolys": lpolys}


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
