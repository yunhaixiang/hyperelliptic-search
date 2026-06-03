#!/usr/bin/env python3
"""Create a genus-rebalanced hyperelliptic2 pickle dataset."""

import argparse
from collections import Counter, defaultdict
import os
import pickle
import random
import sys


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def parse_caps(values):
    caps = {}
    for value in values:
        genus_text, cap_text = value.split(":", 1)
        caps[int(genus_text)] = int(cap_text)
    return caps


def load_points(paths):
    points = []
    for path in paths:
        with open(path, "rb") as f:
            points.extend(pickle.load(f))
    return points


def point_prime(point):
    return int(getattr(point, "p", 0))


def point_genus(point):
    genus = getattr(point, "genus", None)
    if genus is not None:
        return int(genus)
    score = getattr(point, "score", None)
    if score is not None and score >= 0:
        return int(score)
    return int(getattr(point, "N"))


def distribution(points):
    counts = Counter()
    for point in points:
        counts[(point_prime(point), point_genus(point))] += 1
    return counts


def rebalance(points, args):
    rng = random.Random(args.seed)
    caps = parse_caps(args.cap)
    by_prime_genus = defaultdict(list)
    for point in points:
        p = point_prime(point)
        g = point_genus(point)
        if args.primes and p not in args.primes:
            continue
        if args.max_genus and g > args.max_genus:
            continue
        if g < args.keep_all_from:
            by_prime_genus[(p, g)].append(point)
        else:
            by_prime_genus[(p, g)].append(point)

    selected = []
    for (p, g), group in sorted(by_prime_genus.items()):
        rng.shuffle(group)
        if g >= args.keep_all_from:
            selected.extend(group)
        else:
            cap = caps.get(g, args.default_low_cap)
            if cap <= 0:
                selected.extend(group)
            else:
                selected.extend(group[:cap])

    selected.sort(key=lambda point: (point_genus(point), point_prime(point)), reverse=True)
    if args.max_rows:
        selected = selected[: args.max_rows]
    rng.shuffle(selected)
    return selected


def write_metadata(path, args, source_count, selected, train, test, before, after):
    with open(path, "w") as f:
        f.write("source_files=%s\n" % ",".join(args.input))
        f.write("max_genus=%d\n" % args.max_genus)
        f.write("keep_all_from=%d\n" % args.keep_all_from)
        f.write("default_low_cap=%d\n" % args.default_low_cap)
        f.write("caps=%s\n" % ",".join(args.cap))
        f.write("source_count=%d\n" % source_count)
        f.write("selected=%d\n" % len(selected))
        f.write("train=%d\n" % len(train))
        f.write("test=%d\n" % len(test))
        f.write("before_distribution_by_prime_genus:\n")
        for (p, g), count in sorted(before.items()):
            f.write(f"  p={p} g={g}: {count}\n")
        f.write("after_distribution_by_prime_genus:\n")
        for (p, g), count in sorted(after.items()):
            f.write(f"  p={p} g={g}: {count}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--primes", type=int, nargs="*", default=[])
    parser.add_argument("--max-genus", type=int, default=100)
    parser.add_argument("--keep-all-from", type=int, default=5)
    parser.add_argument("--default-low-cap", type=int, default=100)
    parser.add_argument("--cap", action="append", default=[])
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    points = load_points(args.input)
    before = distribution(points)
    selected = rebalance(points, args)
    after = distribution(selected)

    n_test = min(args.test_size, len(selected))
    test = selected[:n_test]
    train = selected[n_test:]

    train_path = os.path.join(args.output_dir, "train_data.pkl")
    test_path = os.path.join(args.output_dir, "test_data.pkl")
    metadata_path = os.path.join(args.output_dir, "metadata.txt")
    with open(train_path, "wb") as f:
        pickle.dump(train, f)
    with open(test_path, "wb") as f:
        pickle.dump(test, f)
    write_metadata(metadata_path, args, len(points), selected, train, test, before, after)

    print(f"loaded {len(points)} points; selected {len(selected)}")
    print(f"wrote {train_path}")
    print(f"wrote {test_path}")
    print(f"wrote {metadata_path}")
    print("after genus distribution:")
    for genus, count in sorted(Counter(point_genus(p) for p in selected).items()):
        print(f"  g={genus}: {count}")


if __name__ == "__main__":
    main()
