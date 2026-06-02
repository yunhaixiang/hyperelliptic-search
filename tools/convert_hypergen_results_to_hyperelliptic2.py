#!/usr/bin/env python3
"""Convert hypergen all_results SQLite rows into hyperelliptic2 seed pickles."""

import argparse
import ast
import os
import pickle
import random
import sqlite3
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.envs.hyperelliptic2 import Hyperelliptic2DataPoint


SUPPORTED_PRIMES = (3, 5, 7, 11)


def parse_coefficients(value):
    coeffs = ast.literal_eval(value)
    if not isinstance(coeffs, list) or not coeffs:
        raise ValueError(f"invalid coefficients: {value!r}")
    return [int(c) for c in coeffs]


def genus_from_degree(degree):
    if degree < 3:
        return None
    if degree % 2:
        return (degree - 1) // 2
    return (degree - 2) // 2


def iter_rows(conn, primes, max_genus, table, include_presentations):
    coefficient_column = "coefficients" if include_presentations else "representative_coefficients"
    where = [
        f"{coefficient_column} is not null",
        "prime in (%s)" % ",".join("?" for _ in primes),
        "sparsity = 0",
    ]
    params = list(primes)
    if max_genus > 0:
        where.append("genus <= ?")
        params.append(max_genus)
    sql = f"""
        select prime, genus, sparsity, lpoly, {coefficient_column}
        from {table}
        where {" and ".join(where)}
        order by genus desc, prime asc, id asc
    """
    yield from conn.execute(sql, params)


def make_datapoint(row):
    p = int(row["prime"])
    coeffs = parse_coefficients(row[4])
    degree = len(coeffs) - 1
    genus = genus_from_degree(degree)
    if genus is None:
        return None

    d = Hyperelliptic2DataPoint(N=genus)
    d.p = p
    d.data = np.asarray(coeffs, dtype=np.int64) % p
    d.degree = degree
    d.genus = genus
    d.N = genus
    d.calc_features()
    return d


def convert(args):
    primes = tuple(int(p.strip()) for p in args.primes.split(",") if p.strip())
    unsupported = sorted(set(primes) - set(SUPPORTED_PRIMES))
    if unsupported:
        raise ValueError(f"unsupported primes for hyperelliptic2: {unsupported}")

    Hyperelliptic2DataPoint.SCORE_BATCH_SIZE = args.score_batch_size
    os.makedirs(args.output_dir, exist_ok=True)

    conn = sqlite3.connect(args.input_sqlite)
    conn.row_factory = sqlite3.Row
    table = "orbit_presentations" if args.include_presentations else "canonical_classes"

    candidates = []
    skipped = 0
    for row in iter_rows(conn, primes, args.max_genus, table, args.include_presentations):
        try:
            d = make_datapoint(row)
        except Exception:
            d = None
        if d is None:
            skipped += 1
            continue
        candidates.append(d)
        if args.max_rows and len(candidates) >= args.max_rows:
            break

    rescored = []
    for start in range(0, len(candidates), args.score_batch_size):
        chunk = candidates[start : start + args.score_batch_size]
        rows, _ = Hyperelliptic2DataPoint._score_arrays([d.data for d in chunk], [d.p for d in chunk])
        for d, score_row in zip(chunk, rows):
            d._apply_score(score_row)
            d.calc_features()
            if d.score >= 0:
                rescored.append(d)

    rescored.sort(key=lambda d: (d.score, d.genus, len(d.data)), reverse=True)
    if args.max_valid_rows:
        rescored = rescored[: args.max_valid_rows]

    random.Random(args.seed).shuffle(rescored)
    n_test = min(args.test_size, len(rescored))
    test = rescored[:n_test]
    train = rescored[n_test:]

    train_path = os.path.join(args.output_dir, "train_data.pkl")
    test_path = os.path.join(args.output_dir, "test_data.pkl")
    metadata_path = os.path.join(args.output_dir, "metadata.txt")
    with open(train_path, "wb") as f:
        pickle.dump(train, f)
    with open(test_path, "wb") as f:
        pickle.dump(test, f)
    with open(metadata_path, "w") as f:
        f.write(f"input_sqlite={args.input_sqlite}\n")
        f.write(f"table={table}\n")
        f.write(f"primes={','.join(map(str, primes))}\n")
        f.write(f"max_genus={args.max_genus}\n")
        f.write(f"candidates={len(candidates)}\n")
        f.write(f"valid={len(rescored)}\n")
        f.write(f"skipped={skipped}\n")
        f.write(f"train={len(train)}\n")
        f.write(f"test={len(test)}\n")

    return train_path, test_path, len(candidates), len(rescored), skipped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-sqlite", default="data/hypergen/all_results.sqlite")
    parser.add_argument("--output-dir", default="data/hyperelliptic2/hypergen_seed")
    parser.add_argument("--primes", default="3,5,7,11")
    parser.add_argument("--max-genus", type=int, default=0, help="0 means no max beyond input rows")
    parser.add_argument("--max-rows", type=int, default=0, help="0 means all candidate rows")
    parser.add_argument("--max-valid-rows", type=int, default=0, help="0 means keep all valid rows")
    parser.add_argument("--test-size", type=int, default=100)
    parser.add_argument("--score-batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include-presentations", action="store_true", help="Use orbit_presentations instead of canonical_classes")
    args = parser.parse_args()

    train_path, test_path, candidates, valid, skipped = convert(args)
    print(f"read {candidates} candidates; valid after current hyperelliptic2 scoring: {valid}; skipped: {skipped}")
    print(f"wrote {train_path}")
    print(f"wrote {test_path}")


if __name__ == "__main__":
    main()
