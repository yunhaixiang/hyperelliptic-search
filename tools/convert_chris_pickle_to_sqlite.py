#!/usr/bin/env python3
"""Convert Chris's Sage pickle into the SQLite seed format used by training.

Run with Sage Python, for example:
DOT_SAGE=/private/tmp/sage-dot-cache /Applications/SageMath-10-6.app/Contents/MacOS/Python \
  tools/convert_chris_pickle_to_sqlite.py \
  data/chris/genus-2--primes-3-97.pkl \
  data/chris/genus-2--primes-3-97.sqlite
"""

import argparse
import pickle
import sqlite3
from pathlib import Path

from sage.all import GF, HyperellipticCurve, PolynomialRing


def coeffs_from_tail_stub(stub, tail, tail_len, p):
    coeffs = []
    value = int(tail)
    for _ in range(tail_len):
        coeffs.append(value % p)
        value //= p
    coeffs.extend(int(c) % p for c in stub)
    return coeffs


def convert(input_path, output_path):
    with Path(input_path).open("rb") as handle:
        by_prime = pickle.load(handle)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(output_path)
    conn.execute("DROP TABLE IF EXISTS curves")
    conn.execute("DROP TABLE IF EXISTS run_metadata")
    conn.execute(
        """
        CREATE TABLE curves (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            p INTEGER NOT NULL,
            a0 INTEGER NOT NULL,
            a1 INTEGER NOT NULL,
            a2 INTEGER NOT NULL,
            a3 INTEGER NOT NULL,
            source_prime INTEGER NOT NULL,
            source_stub TEXT NOT NULL,
            source_tail INTEGER NOT NULL,
            score REAL DEFAULT 10.0,
            c1 INTEGER DEFAULT 0
        )
        """
    )
    conn.execute("CREATE TABLE run_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.execute("INSERT INTO run_metadata (key, value) VALUES (?, ?)", ("source", str(input_path)))
    conn.execute("INSERT INTO run_metadata (key, value) VALUES (?, ?)", ("format", "chris_genus2_pickle_odd_monic_depressed"))

    seen = set()
    total = 0
    skipped = 0
    invalid = 0
    nonzero_c1 = 0
    for p, rows in sorted(by_prime.items()):
        p = int(p)
        field = GF(p)
        ring = PolynomialRing(field, "x")
        x = ring.gen()
        for row in rows:
            if len(row) != 3:
                skipped += 1
                continue
            odd_case, stub, tail = row
            if not bool(odd_case):
                skipped += 1
                continue

            stub = [int(c) for c in stub]
            tail_len = 6 - len(stub)
            coeffs = coeffs_from_tail_stub(stub, tail, tail_len, p)

            # Training currently models only y^2 = x^5 + a3*x^3 + a2*x^2 + a1*x + a0.
            if len(coeffs) != 6 or coeffs[5] % p != 1 or coeffs[4] % p != 0:
                skipped += 1
                continue

            a0, a1, a2, a3 = [int(c) % p for c in coeffs[:4]]
            key = (p, a0, a1, a2, a3)
            if key in seen:
                skipped += 1
                continue
            try:
                curve = HyperellipticCurve(x**5 + field(a3) * x**3 + field(a2) * x**2 + field(a1) * x + field(a0))
                count = curve.count_points(1)
                if isinstance(count, (list, tuple)):
                    count = count[0]
                c1 = int(count) - p - 1
            except Exception:
                invalid += 1
                continue
            if c1 != 0:
                nonzero_c1 += 1
                continue
            seen.add(key)
            conn.execute(
                """
                INSERT INTO curves
                    (p, a0, a1, a2, a3, source_prime, source_stub, source_tail, score, c1)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 10.0, 0)
                """,
                (p, a0, a1, a2, a3, p, repr(stub), int(tail)),
            )
            total += 1

    conn.commit()
    conn.close()
    return total, skipped, invalid, nonzero_c1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_pickle")
    parser.add_argument("output_sqlite")
    args = parser.parse_args()
    total, skipped, invalid, nonzero_c1 = convert(args.input_pickle, args.output_sqlite)
    print(f"wrote {total} odd monic depressed curves to {args.output_sqlite}")
    print(f"skipped {skipped} incompatible/non-odd rows")
    print(f"discarded {invalid} Sage-invalid rows")
    print(f"discarded {nonzero_c1} rows with nonzero c1")


if __name__ == "__main__":
    main()
