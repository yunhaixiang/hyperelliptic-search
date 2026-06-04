#!/usr/bin/env python3
"""Convert hyperelliptic2 coefficient seed pickles to factorized seed pickles."""

import argparse
import os
import pickle
import sys
from collections import Counter

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.envs.environment import BaseEnvironment
from src.envs.hyperelliptic_factorized import FactorizedHyperellipticDataPoint, FactorizedHyperellipticTokenizer


def convert_split(input_path, output_path, p, max_genus, max_rows):
    source = pickle.load(open(input_path, "rb"))
    if max_rows > 0:
        source = source[:max_rows]

    FactorizedHyperellipticDataPoint.PRIME = p
    FactorizedHyperellipticDataPoint.MAX_GENUS = max_genus
    tokenizer = FactorizedHyperellipticTokenizer(
        FactorizedHyperellipticDataPoint,
        max_genus=max_genus,
        p=p,
        extra_symbols=BaseEnvironment.SPECIAL_SYMBOLS,
    )

    out = []
    skipped = 0
    for item in source:
        coeffs = getattr(item, "data", None)
        item_p = int(getattr(item, "p", p))
        if coeffs is None or item_p != p:
            skipped += 1
            continue
        d = FactorizedHyperellipticDataPoint._from_coefficients(np.asarray(coeffs, dtype=np.int64).tolist(), p)
        if d is None:
            skipped += 1
            continue
        d.score = float(getattr(item, "score", -1.0))
        d.lpoly = getattr(item, "lpoly", None)
        d.middle = getattr(item, "middle", None)
        d.target_coeffs = getattr(item, "target_coeffs", None)
        tokens = tokenizer.encode(d)
        d._encoded_token_cache_key = tokenizer.cache_key_for_datapoint(d, None)
        d._encoded_token_cache_value = tokens
        out.append(d)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(out, f)
    return len(source), len(out), skipped, Counter(int(d.genus) for d in out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="training_data/hyperelliptic2_pgl2_highgenus_g100")
    parser.add_argument("--output-dir", default="training_data/hyperelliptic_factorized_pgl2_highgenus_g100")
    parser.add_argument("--p", type=int, default=3)
    parser.add_argument("--max-genus", type=int, default=24)
    parser.add_argument("--max-rows", type=int, default=0)
    args = parser.parse_args()

    summaries = {}
    for split in ("train", "test"):
        input_path = os.path.join(args.input_dir, f"{split}_data.pkl")
        output_path = os.path.join(args.output_dir, f"{split}_data.pkl")
        read, written, skipped, distribution = convert_split(
            input_path,
            output_path,
            args.p,
            args.max_genus,
            args.max_rows,
        )
        summaries[split] = (read, written, skipped, distribution)

    metadata_path = os.path.join(args.output_dir, "metadata.txt")
    with open(metadata_path, "w") as f:
        f.write("format=hyperelliptic_factorized_seed_v1\n")
        f.write(f"source={args.input_dir}\n")
        f.write(f"p={args.p}\n")
        f.write(f"max_genus={args.max_genus}\n")
        for split, (read, written, skipped, distribution) in summaries.items():
            f.write(f"{split}_read={read}\n")
            f.write(f"{split}_written={written}\n")
            f.write(f"{split}_skipped={skipped}\n")
            f.write(f"{split}_genus_distribution={dict(sorted(distribution.items()))}\n")

    readme_path = os.path.join(args.output_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write("# hyperelliptic_factorized seed data\n\n")
        f.write("Factorized-form seed data converted from the hyperelliptic2 PGL2 high-genus seed set.\n")
        f.write("Each datapoint stores f(x) as leading_coefficient times a monic irreducible factor multiset, plus cached tokens for `hyperelliptic_factorized`.\n")

    print(f"wrote {args.output_dir}")
    for split, (read, written, skipped, distribution) in summaries.items():
        print(split, "read", read, "written", written, "skipped", skipped, "dist", dict(sorted(distribution.items())))


if __name__ == "__main__":
    main()
