#!/usr/bin/env python3
"""
Fill genus-2 L-polynomial middle coefficients in a trinomial SQLite export.

Run with Sage's Python, not ordinary Python:

    sage -python tools/add_lpoly_middle_with_sage.py exports/g2_trinomial_p1000003_batch_trinomial.sqlite

The input database must have a `curves` table with columns `a0`, `a1`, `a2`,
and `a3`, representing

    y^2 = x^5 + a3*x^3 + a2*x^2 + a1*x + a0

over F_p. The script adds/fills:

    lpoly_c1
    lpoly_middle_c2
    lpoly_json
"""

import json
import sqlite3
import sys
from pathlib import Path


try:
    from sage.all import GF, HyperellipticCurve, PolynomialRing
except Exception as exc:  # pragma: no cover - this is an environment check.
    raise SystemExit(
        "This script must be run with Sage's Python, for example:\n"
        "  sage -python tools/add_lpoly_middle_with_sage.py exports/g2_trinomial_p1000003_batch_trinomial.sqlite\n"
        f"Import error: {exc}"
    )


def ensure_column(conn, table, column, decl):
    columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    if column not in columns:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")


def polynomial_coeff(poly, degree):
    try:
        return int(poly[degree])
    except Exception:
        return int(poly.coefficient({poly.parent().gen(): degree}))


def lpoly_from_curve(curve, p, known_c1):
    # Preferred: zeta numerator is L(T) = 1 + c1*T + c2*T^2 + p*c1*T^3 + p^2*T^4.
    try:
        zeta = curve.zeta_function()
        numerator = zeta.numerator() if callable(getattr(zeta, "numerator", None)) else zeta.numerator
        c1 = polynomial_coeff(numerator, 1)
        c2 = polynomial_coeff(numerator, 2)
        return c1, c2, [1, c1, c2, p * c1, p * p]
    except Exception:
        pass

    # Common Sage method: characteristic polynomial of Frobenius is
    # X^4 + c1*X^3 + c2*X^2 + p*c1*X + p^2.
    try:
        frob = curve.frobenius_polynomial()
        c1 = polynomial_coeff(frob, 3)
        c2 = polynomial_coeff(frob, 2)
        return c1, c2, [1, c1, c2, p * c1, p * p]
    except Exception:
        pass

    # Last resort if Sage provides optimized point counts for this curve.
    counts = curve.count_points(2)
    if isinstance(counts, (list, tuple)):
        n2 = int(counts[1])
    else:
        n2 = int(counts)
    s2 = p * p + 1 - n2
    c2_num = known_c1 * known_c1 - s2
    if c2_num % 2 != 0:
        raise ValueError(f"nonintegral c2 from point count: c1={known_c1}, N2={n2}")
    c2 = c2_num // 2
    return known_c1, c2, [1, known_c1, c2, p * known_c1, p * p]


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: sage -python tools/add_lpoly_middle_with_sage.py PATH_TO_TRINOMIAL_SQLITE")

    db_path = Path(sys.argv[1])
    if not db_path.exists():
        raise SystemExit(f"missing sqlite file: {db_path}")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    ensure_column(conn, "curves", "lpoly_c1", "INTEGER")
    ensure_column(conn, "curves", "lpoly_middle_c2", "INTEGER")
    ensure_column(conn, "curves", "lpoly_json", "TEXT")

    metadata = dict(conn.execute("SELECT key, value FROM run_metadata"))
    p = int(metadata.get("prime", "1000003"))
    field = GF(p)
    ring = PolynomialRing(field, "x")
    x = ring.gen()

    rows = list(conn.execute("SELECT id, a0, a1, a2, a3, c1 FROM curves ORDER BY id"))
    total = len(rows)
    for idx, row in enumerate(rows, start=1):
        f = x**5 + field(row["a3"]) * x**3 + field(row["a2"]) * x**2 + field(row["a1"]) * x + field(row["a0"])
        curve = HyperellipticCurve(f)
        known_c1 = int(row["c1"])
        c1, c2, lpoly = lpoly_from_curve(curve, p, known_c1)
        if c1 != known_c1:
            raise ValueError(f"row {row['id']}: Sage c1={c1} does not match stored c1={known_c1}")
        conn.execute(
            "UPDATE curves SET lpoly_c1 = ?, lpoly_middle_c2 = ?, lpoly_json = ? WHERE id = ?",
            (c1, c2, json.dumps(lpoly), int(row["id"])),
        )
        if idx == 1 or idx % 10 == 0 or idx == total:
            conn.commit()
            print(f"updated {idx} / {total}")

    conn.execute(
        "INSERT OR REPLACE INTO run_metadata (key, value) VALUES (?, ?)",
        ("lpoly_middle_c2_status", "computed_with_sage"),
    )
    conn.commit()
    conn.close()
    print(f"updated {db_path}")


if __name__ == "__main__":
    main()
