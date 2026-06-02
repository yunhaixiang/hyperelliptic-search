#!/usr/bin/env python3
"""Export Axplorer train/test checkpoint pickles to SQLite."""

import argparse
import json
import pickle
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def datapoint_rows(data, split):
    for d in data:
        coeffs = [int(v) for v in d.data.tolist()]
        row = {
            "split": split,
            "p": int(getattr(d, "p", 0)),
            "a0": coeffs[0],
            "a1": coeffs[1],
            "a2": coeffs[2],
            "a3": coeffs[3],
            "score": float(getattr(d, "score", -1)),
            "c1": getattr(d, "c1", None),
            "c2": getattr(d, "c2", None),
            "lpoly_json": json.dumps(getattr(d, "lpoly", None)) if getattr(d, "lpoly", None) is not None else None,
            "features": getattr(d, "features", ""),
        }
        yield row


def write_sqlite(rows, output_path, metadata):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(output_path)
    conn.execute("DROP TABLE IF EXISTS curves")
    conn.execute("DROP TABLE IF EXISTS run_metadata")
    conn.execute(
        """
        CREATE TABLE curves (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            split TEXT NOT NULL,
            p INTEGER NOT NULL,
            a0 INTEGER NOT NULL,
            a1 INTEGER NOT NULL,
            a2 INTEGER NOT NULL,
            a3 INTEGER NOT NULL,
            score REAL NOT NULL,
            c1 INTEGER,
            c2 INTEGER,
            lpoly_json TEXT,
            features TEXT
        )
        """
    )
    conn.execute("CREATE TABLE run_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    for key, value in metadata.items():
        conn.execute("INSERT INTO run_metadata (key, value) VALUES (?, ?)", (key, str(value)))
    conn.executemany(
        """
        INSERT INTO curves
            (split, p, a0, a1, a2, a3, score, c1, c2, lpoly_json, features)
        VALUES
            (:split, :p, :a0, :a1, :a2, :a3, :score, :c1, :c2, :lpoly_json, :features)
        """,
        rows,
    )
    conn.execute("CREATE INDEX idx_curves_score ON curves(score)")
    conn.execute("CREATE INDEX idx_curves_c1 ON curves(c1)")
    conn.execute("CREATE INDEX idx_curves_p ON curves(p)")
    conn.commit()
    total = conn.execute("SELECT COUNT(*) FROM curves").fetchone()[0]
    exact = conn.execute("SELECT COUNT(*) FROM curves WHERE c1 = 0").fetchone()[0]
    conn.close()
    return total, exact


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_dir")
    parser.add_argument("output_sqlite")
    parser.add_argument("--trinomial_output_sqlite", default="")
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    train = pickle.load((checkpoint_dir / "train_data.pkl").open("rb"))
    test = pickle.load((checkpoint_dir / "test_data.pkl").open("rb"))
    rows = list(datapoint_rows(train, "train")) + list(datapoint_rows(test, "test"))
    metadata = {
        "checkpoint_dir": str(checkpoint_dir),
        "train_rows": len(train),
        "test_rows": len(test),
    }
    total, exact = write_sqlite(rows, args.output_sqlite, metadata)
    print(f"wrote {total} rows ({exact} with c1=0) to {args.output_sqlite}")

    if args.trinomial_output_sqlite:
        tri_rows = [row for row in rows if row["c1"] == 0]
        tri_total, tri_exact = write_sqlite(tri_rows, args.trinomial_output_sqlite, metadata)
        print(f"wrote {tri_total} rows ({tri_exact} with c1=0) to {args.trinomial_output_sqlite}")


if __name__ == "__main__":
    main()
