#!/usr/bin/env python3
"""Convert Chris's old genus-2 pickle into Axplorer seed SQLite.

The old pickle maps p -> a list of integers

    abcd = a0 + a1*p + a2*p^2 + a3*p^3

for curves y^2 = x^5 + a3*x^3 + a2*x^2 + a1*x + a0 with c1 = 0.
"""

import argparse
import pickle
import sqlite3
from pathlib import Path


def decode_abcd(abcd, p):
    value = int(abcd)
    coeffs = []
    for _ in range(4):
        coeffs.append(value % p)
        value //= p
    return coeffs


def convert(input_path, output_path, max_rows=0):
    with Path(input_path).open("rb") as handle:
        by_prime = pickle.load(handle)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(output_path)
    conn.execute("PRAGMA journal_mode = OFF")
    conn.execute("PRAGMA synchronous = OFF")
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
            source_abcd INTEGER NOT NULL,
            score REAL DEFAULT 10.0,
            c1 INTEGER DEFAULT 0
        )
        """
    )
    conn.execute("CREATE TABLE run_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.execute("INSERT INTO run_metadata (key, value) VALUES (?, ?)", ("source", str(input_path)))
    conn.execute("INSERT INTO run_metadata (key, value) VALUES (?, ?)", ("format", "chris_old_abcd"))

    total = 0
    for p in sorted(by_prime):
        rows = by_prime[p]
        p = int(p)
        batch = []
        for abcd in rows:
            a0, a1, a2, a3 = decode_abcd(abcd, p)
            batch.append((p, a0, a1, a2, a3, int(abcd), 10.0, 0))
            total += 1
            if len(batch) >= 10000:
                conn.executemany(
                    "INSERT INTO curves (p, a0, a1, a2, a3, source_abcd, score, c1) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    batch,
                )
                batch.clear()
            if max_rows and total >= max_rows:
                break
        if batch:
            conn.executemany(
                "INSERT INTO curves (p, a0, a1, a2, a3, source_abcd, score, c1) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                batch,
            )
        conn.commit()
        print(f"p={p}: converted through total={total}", flush=True)
        if max_rows and total >= max_rows:
            break

    conn.execute("CREATE INDEX idx_curves_score ON curves(score)")
    conn.execute("CREATE INDEX idx_curves_p ON curves(p)")
    conn.commit()
    conn.close()
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_pickle")
    parser.add_argument("output_sqlite")
    parser.add_argument("--max_rows", type=int, default=0, help="Optional cap for a test conversion")
    args = parser.parse_args()
    total = convert(args.input_pickle, args.output_sqlite, max_rows=args.max_rows)
    print(f"wrote {total} curves to {args.output_sqlite}")


if __name__ == "__main__":
    main()
