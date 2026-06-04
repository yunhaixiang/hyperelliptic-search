#!/usr/bin/env python3
"""Train an XGBoost classifier on trinomial vs non-trinomial necklace data."""

import argparse
import json
import os
import pickle
import random
import sys
import time
from collections import Counter, defaultdict

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.envs.environment import BaseEnvironment
from src.envs.hyperelliptic2 import (
    Hyperelliptic2DataPoint,
    Hyperelliptic2Tokenizer,
    _is_aperiodic_necklace,
    _sage_necklace_request,
)


def tokenizer(max_genus, p):
    return Hyperelliptic2Tokenizer(
        dataclass=Hyperelliptic2DataPoint,
        max_genus=max_genus,
        p=p,
        extra_symbols=BaseEnvironment.SPECIAL_SYMBOLS,
    )


def cached_token(point):
    value = getattr(point, "_encoded_token_cache_value", None)
    if value is None:
        return None
    return [int(token) for token in value]


def parse_necklaces(tokens, tok):
    pos = 0
    if tok.itos.get(tokens[pos]) != "BOS":
        raise ValueError("missing BOS")
    pos += 1
    if tok.itos.get(tokens[pos]) != "SEP":
        raise ValueError("missing leading SEP")
    pos += 1
    necklaces = []
    while pos < len(tokens):
        token = tok.itos.get(tokens[pos])
        if token == "EOS":
            break
        if not isinstance(token, str) or not token.startswith("D"):
            raise ValueError("missing degree token")
        degree = int(token[1:])
        pos += 1
        raw = tokens[pos : pos + degree * degree]
        if len(raw) != degree * degree:
            raise ValueError("truncated necklace")
        necklace = tuple(tuple(raw[i : i + degree]) for i in range(0, len(raw), degree))
        necklaces.append(necklace)
        pos += degree * degree
        separator = tok.itos.get(tokens[pos]) if pos < len(tokens) else None
        if separator == "SEP":
            pos += 1
        elif separator == "EOS":
            break
        else:
            raise ValueError("missing separator")
    return tuple(necklaces)


def necklaces_to_tokens(necklaces, tok):
    out = [tok.stoi["BOS"], tok.stoi["SEP"]]
    for idx, necklace in enumerate(necklaces):
        degree = len(necklace)
        out.append(tok.stoi[f"D{degree}"])
        for vector in necklace:
            out.extend(int(c) for c in vector)
        if idx + 1 < len(necklaces):
            out.append(tok.stoi["SEP"])
    out.append(tok.stoi["EOS"])
    return out


def mutate_necklaces(necklaces, p, rng):
    mutated = [list(list(vector) for vector in necklace) for necklace in necklaces]
    editable = [
        (factor_idx, vector_idx, coord_idx)
        for factor_idx, necklace in enumerate(mutated)
        for vector_idx, vector in enumerate(necklace)
        for coord_idx, _ in enumerate(vector)
    ]
    factor_idx, vector_idx, coord_idx = rng.choice(editable)
    old_value = mutated[factor_idx][vector_idx][coord_idx]
    choices = [value for value in range(p) if value != old_value]
    mutated[factor_idx][vector_idx][coord_idx] = rng.choice(choices)
    out = tuple(
        tuple(tuple(int(c) for c in vector) for vector in necklace)
        for necklace in mutated
    )
    if not all(_is_aperiodic_necklace(necklace) for necklace in out):
        return None
    return tuple(sorted(out))


def factor_degrees_from_necklaces(necklaces):
    return [len(necklace) for necklace in necklaces]


def necklace_feature_names(max_factor_degree, p):
    names = [
        "genus",
        "degree",
        "odd_model",
        "num_factors",
        "min_factor_degree",
        "max_factor_degree",
        "mean_factor_degree",
        "degree_variance",
        "num_linear_factors",
        "coord_count",
    ]
    names.extend(f"factor_degree_{degree}" for degree in range(1, max_factor_degree + 1))
    names.extend(f"coord_value_{value}" for value in range(p))
    names.extend(f"coord_transition_{a}_{b}" for a in range(p) for b in range(p))
    names.extend(f"coord_position_mod_{value}" for value in range(p))
    return names


def factorized_feature_names(max_factor_degree, p):
    names = [
        "genus",
        "degree",
        "odd_model",
        "num_factors",
        "min_factor_degree",
        "max_factor_degree",
        "mean_factor_degree",
        "degree_variance",
        "num_linear_factors",
    ]
    names.extend(f"factor_degree_{degree}" for degree in range(1, max_factor_degree + 1))
    for degree in range(1, max_factor_degree + 1):
        for pos in range(degree):
            for value in range(p):
                names.append(f"factor_coeff_d{degree}_x{pos}_v{value}")
    for value in range(p):
        names.append(f"all_factor_coeff_v{value}")
    return names


def features_from_necklaces(necklaces, max_factor_degree, p):
    degrees = factor_degrees_from_necklaces(necklaces)
    degree = sum(degrees)
    genus = (degree - 1) // 2 if degree % 2 else (degree - 2) // 2
    factor_counter = Counter(degrees)
    coords = []
    transitions = Counter()
    position_mod = Counter()
    for necklace in necklaces:
        flat = [int(c) for vector in necklace for c in vector]
        coords.extend(flat)
        for idx, value in enumerate(flat):
            position_mod[idx % p] += value
        for left, right in zip(flat, flat[1:] + flat[:1]):
            transitions[(left, right)] += 1

    coord_counter = Counter(coords)
    mean = float(np.mean(degrees)) if degrees else 0.0
    var = float(np.var(degrees)) if degrees else 0.0
    row = [
        genus,
        degree,
        int(degree == 2 * genus + 1),
        len(degrees),
        min(degrees) if degrees else 0,
        max(degrees) if degrees else 0,
        mean,
        var,
        factor_counter.get(1, 0),
        len(coords),
    ]
    row.extend(factor_counter.get(degree, 0) for degree in range(1, max_factor_degree + 1))
    row.extend(coord_counter.get(value, 0) for value in range(p))
    row.extend(transitions.get((a, b), 0) for a in range(p) for b in range(p))
    row.extend(position_mod.get(value, 0) for value in range(p))
    return row


def features_from_factor_polynomials(factors, max_factor_degree, p):
    degrees = [len(factor) - 1 for factor in factors]
    degree = sum(degrees)
    genus = (degree - 1) // 2 if degree % 2 else (degree - 2) // 2
    factor_counter = Counter(degrees)
    mean = float(np.mean(degrees)) if degrees else 0.0
    var = float(np.var(degrees)) if degrees else 0.0
    all_coeff_counter = Counter()
    coeff_counts = defaultdict(Counter)
    for factor in factors:
        factor_degree = len(factor) - 1
        for pos, coeff in enumerate(factor[:-1]):
            value = int(coeff) % p
            coeff_counts[(factor_degree, pos)][value] += 1
            all_coeff_counter[value] += 1

    row = [
        genus,
        degree,
        int(degree == 2 * genus + 1),
        len(degrees),
        min(degrees) if degrees else 0,
        max(degrees) if degrees else 0,
        mean,
        var,
        factor_counter.get(1, 0),
    ]
    row.extend(factor_counter.get(degree, 0) for degree in range(1, max_factor_degree + 1))
    for degree in range(1, max_factor_degree + 1):
        for pos in range(degree):
            counter = coeff_counts[(degree, pos)]
            row.extend(counter.get(value, 0) for value in range(p))
    row.extend(all_coeff_counter.get(value, 0) for value in range(p))
    return row


def factors_for_examples(examples, p, batch_size):
    out = []
    for start in range(0, len(examples), batch_size):
        chunk = examples[start : start + batch_size]
        response = _sage_necklace_request(
            {
                "op": "necklaces_to_factors_batch",
                "p": int(p),
                "necklaces_batch": [
                    [[[int(c) for c in vector] for vector in necklace] for necklace in necklaces]
                    for _, _, necklaces in chunk
                ],
            }
        )
        for factors in response["factors"]:
            if factors is None:
                raise RuntimeError("failed to convert necklaces to factor polynomials")
            out.append([[int(c) % p for c in factor] for factor in factors])
    return out


def load_positive_tokens(path, tok, use_all_positives, max_positives, min_genus, max_positive_genus, seed):
    points = pickle.load(open(path, "rb"))
    positives = []
    for point in points:
        genus = int(getattr(point, "genus", getattr(point, "N", 0)))
        if min_genus > 0 and genus < min_genus:
            continue
        if max_positive_genus > 0 and genus > max_positive_genus:
            continue
        tokens = cached_token(point)
        if tokens is None and use_all_positives:
            tokens = tok.encode(point).astype(int).tolist()
        if tokens is None:
            continue
        positives.append((point, tokens, parse_necklaces(tokens, tok)))
    rng = random.Random(seed)
    rng.shuffle(positives)
    if max_positives > 0:
        positives = positives[:max_positives]
    return positives


def replace_random_factor(necklaces, replacements, rng):
    mutated = list(necklaces)
    replaceable = [idx for idx, necklace in enumerate(mutated) if len(necklace) in replacements]
    if not replaceable:
        return None
    idx = rng.choice(replaceable)
    choices = replacements[len(mutated[idx])]
    mutated[idx] = rng.choice(choices)
    return tuple(sorted(mutated))


def generate_negative_tokens(positives, tok, p, attempts_per_positive, score_batch_size, seed, mutation_mode):
    rng = random.Random(seed)
    seen = set()
    positive_features = {point.features for point, _, _ in positives}
    replacements = {}
    if mutation_mode == "factor":
        degree_counts = Counter(
            len(necklace)
            for _, _, necklaces in positives
            for necklace in necklaces
        )
        for degree, count in degree_counts.items():
            batch_size = min(max(16, count), 256)
            response = _sage_necklace_request(
                {"op": "random_necklace_batch", "p": int(p), "degrees": [int(degree)] * batch_size}
            )
            replacements[int(degree)] = [
                tuple(tuple(int(c) for c in vector) for vector in necklace)
                for necklace in response["necklaces"]
            ]

    Hyperelliptic2DataPoint.SCORE_BATCH_SIZE = score_batch_size
    negatives_by_index = {}
    remaining = set(range(len(positives)))

    for _ in range(attempts_per_positive):
        if not remaining:
            break
        pass_candidates = []
        for idx in sorted(remaining):
            _, _, necklaces = positives[idx]
            if mutation_mode == "factor":
                mutated = replace_random_factor(necklaces, replacements, rng)
            else:
                mutated = mutate_necklaces(necklaces, p, rng)
            if mutated is None:
                continue
            tokens = necklaces_to_tokens(mutated, tok)
            key = tuple(tokens)
            if key in seen:
                continue
            seen.add(key)
            pass_candidates.append((idx, tokens, mutated))

        for start in range(0, len(pass_candidates), score_batch_size):
            chunk = pass_candidates[start : start + score_batch_size]
            conversion = _sage_necklace_request(
                {
                    "op": "necklaces_to_polynomial_batch",
                    "p": int(p),
                    "necklaces_batch": [
                        [[[int(c) for c in vector] for vector in necklace] for necklace in necklaces]
                        for _, _, necklaces in chunk
                    ],
                }
            )
            converted = [
                (idx, tokens, necklaces, coeffs)
                for (idx, tokens, necklaces), coeffs in zip(chunk, conversion["coefficients"])
                if coeffs is not None and idx in remaining
            ]
            if not converted:
                continue
            rows, _ = Hyperelliptic2DataPoint._score_arrays([coeffs for _, _, _, coeffs in converted], [p] * len(converted))
            for (idx, tokens, necklaces, coeffs), row in zip(converted, rows):
                if idx not in remaining:
                    continue
                positive_genus = int(positives[idx][0].genus)
                point = Hyperelliptic2DataPoint._from_coefficients(coeffs, p)
                if point is None or point.features in positive_features:
                    continue
                if int(point.genus) != positive_genus:
                    continue
                point._apply_score(row)
                if point.score < 0 and row["genus"] is not None:
                    negatives_by_index[idx] = (point, tokens, necklaces)
                    remaining.remove(idx)

    indices = sorted(negatives_by_index)
    return [positives[idx] for idx in indices], [negatives_by_index[idx] for idx in indices]


def evaluate_by_genus(y_true, y_prob, genera):
    out = {}
    for genus in sorted(set(genera)):
        idx = [i for i, value in enumerate(genera) if value == genus]
        if len(idx) < 10 or len({int(y_true[i]) for i in idx}) < 2:
            continue
        pred = [int(y_prob[i] >= 0.5) for i in idx]
        out[str(genus)] = {
            "n": len(idx),
            "accuracy": float(accuracy_score([y_true[i] for i in idx], pred)),
            "auc": float(roc_auc_score([y_true[i] for i in idx], [y_prob[i] for i in idx])),
        }
    return out


def plain_counter(counter):
    return {str(int(key)): int(value) for key, value in sorted(counter.items())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--positive-data", default="training_data/hyperelliptic2_pgl2_highgenus_g100/train_data.pkl")
    parser.add_argument("--p", type=int, default=3)
    parser.add_argument(
        "--max-genus",
        type=int,
        default=24,
        help="Tokenizer genus bound used by cached necklace tokens; use 24 for the current tracked cache.",
    )
    parser.add_argument("--max-positives", type=int, default=0)
    parser.add_argument("--min-positive-genus", type=int, default=0)
    parser.add_argument("--max-positive-genus", type=int, default=0)
    parser.add_argument("--use-all-positives", action="store_true")
    parser.add_argument("--attempts-per-positive", type=int, default=80)
    parser.add_argument("--score-batch-size", type=int, default=32)
    parser.add_argument("--negative-mutation", choices=["coordinate", "factor"], default="coordinate")
    parser.add_argument("--feature-mode", choices=["necklace", "factorized"], default="necklace")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--output-dir", default="results/hyperelliptic2_xgboost")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    Hyperelliptic2DataPoint.PRIME = args.p
    Hyperelliptic2DataPoint.MAX_GENUS = args.max_genus
    tok = tokenizer(args.max_genus, args.p)
    positives = load_positive_tokens(
        args.positive_data,
        tok,
        args.use_all_positives,
        args.max_positives,
        args.min_positive_genus,
        args.max_positive_genus,
        args.seed,
    )
    if not positives:
        raise RuntimeError("no positive necklace-token rows found")
    positives, negatives = generate_negative_tokens(
        positives=positives,
        tok=tok,
        p=args.p,
        attempts_per_positive=args.attempts_per_positive,
        score_batch_size=args.score_batch_size,
        seed=args.seed + 1,
        mutation_mode=args.negative_mutation,
    )
    if not negatives:
        raise RuntimeError("no verified non-trinomial negatives generated")

    max_factor_degree = 2 * args.max_genus + 2
    if args.feature_mode == "factorized":
        names = factorized_feature_names(max_factor_degree, args.p)
        positive_factor_rows = factors_for_examples(positives, args.p, args.score_batch_size)
        negative_factor_rows = factors_for_examples(negatives, args.p, args.score_batch_size)
    else:
        names = necklace_feature_names(max_factor_degree, args.p)
        positive_factor_rows = None
        negative_factor_rows = None
    rows = []
    labels = []
    genera = []
    for idx, (point, _, necklaces) in enumerate(positives):
        if args.feature_mode == "factorized":
            rows.append(features_from_factor_polynomials(positive_factor_rows[idx], max_factor_degree, args.p))
        else:
            rows.append(features_from_necklaces(necklaces, max_factor_degree, args.p))
        labels.append(1)
        genera.append(int(point.genus))
    for idx, (point, _, necklaces) in enumerate(negatives):
        if args.feature_mode == "factorized":
            rows.append(features_from_factor_polynomials(negative_factor_rows[idx], max_factor_degree, args.p))
        else:
            rows.append(features_from_necklaces(necklaces, max_factor_degree, args.p))
        labels.append(0)
        genera.append(int(point.genus))

    X = np.asarray(rows, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int32)
    genera = np.asarray(genera, dtype=np.int32)
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )
    model = XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=0.04,
        subsample=0.9,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=max(1, os.cpu_count() or 1),
        random_state=args.seed,
    )
    model.fit(X[train_idx], y[train_idx])
    prob = model.predict_proba(X[test_idx])[:, 1]
    pred = (prob >= 0.5).astype(np.int32)
    inverted_prob = 1.0 - prob
    inverted_pred = (inverted_prob >= 0.5).astype(np.int32)
    importances = model.feature_importances_
    top = sorted(
        ((float(value), names[idx]) for idx, value in enumerate(importances)),
        reverse=True,
    )[:30]

    report = {
        "positive_data": args.positive_data,
        "feature_mode": args.feature_mode,
        "negative_mutation": args.negative_mutation,
        "n_positive": int(sum(y == 1)),
        "n_negative": int(sum(y == 0)),
        "positive_genus_distribution": plain_counter(Counter(genera[y == 1])),
        "negative_genus_distribution": plain_counter(Counter(genera[y == 0])),
        "accuracy": float(accuracy_score(y[test_idx], pred)),
        "auc": float(roc_auc_score(y[test_idx], prob)),
        "average_precision": float(average_precision_score(y[test_idx], prob)),
        "confusion_matrix": confusion_matrix(y[test_idx], pred).tolist(),
        "inverted_accuracy": float(accuracy_score(y[test_idx], inverted_pred)),
        "inverted_auc": float(roc_auc_score(y[test_idx], inverted_prob)),
        "inverted_average_precision": float(average_precision_score(y[test_idx], inverted_prob)),
        "inverted_confusion_matrix": confusion_matrix(y[test_idx], inverted_pred).tolist(),
        "by_genus": evaluate_by_genus(y[test_idx], prob, genera[test_idx]),
        "top_features": [{"feature": name, "importance": value} for value, name in top],
    }

    run_dir = os.path.join(args.output_dir, time.strftime("%Y_%m_%d_%H_%M_%S"))
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")
    with open(os.path.join(run_dir, "model.pkl"), "wb") as f:
        pickle.dump({"model": model, "feature_names": names, "report": report}, f)

    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"wrote {run_dir}")


if __name__ == "__main__":
    main()
