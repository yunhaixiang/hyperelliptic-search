import ast
import json
import os
import pickle
import random
import shlex
import sqlite3
import subprocess
import math

import numpy as np

from src.envs.environment import BaseEnvironment, DataPoint
from src.envs.hyperelliptic2_mod2 import is_mod2_allowed_factor_degrees
from src.envs.tokenizers import Tokenizer
from src.utils import bool_flag


SUPPORTED_PRIMES = {3, 5, 7, 11}
SAGE_PYTHON = "/Applications/SageMath-10-6.app/Contents/MacOS/Python"
SAGE_DOT_DIR = os.environ.get("SAGE_DOT_DIR", os.environ.get("DOT_SAGE", "/private/tmp/sage-dot-cache"))
_SAGE_FACTOR_WORKER = None


def _genus_from_degree(degree):
    degree = int(degree)
    if degree < 3:
        return None
    if degree % 2 == 1:
        return (degree - 1) // 2
    return (degree - 2) // 2


def _sage_python_command(default=SAGE_PYTHON):
    command = os.environ.get("SAGE_PYTHON_CMD") or os.environ.get("SAGE_PYTHON") or default
    parts = shlex.split(command)
    if not parts:
        raise RuntimeError("empty Sage Python command")
    executable = parts[0]
    if os.path.sep in executable and not os.path.exists(executable):
        raise RuntimeError(f"Sage Python executable not found: {executable}")
    return parts


def _sage_subprocess_env():
    env = os.environ.copy()
    env["DOT_SAGE"] = SAGE_DOT_DIR

    sage_app_bin = "/Applications/SageMath-10-6.app/Contents/Frameworks/Sage.framework/Versions/Current/local/bin"
    if os.path.isdir(sage_app_bin):
        path = env.get("PATH", "")
        if sage_app_bin not in path.split(os.pathsep):
            env["PATH"] = sage_app_bin + (os.pathsep + path if path else "")
    return env


def _sage_factor_worker():
    global _SAGE_FACTOR_WORKER
    if _SAGE_FACTOR_WORKER is not None and _SAGE_FACTOR_WORKER.poll() is None:
        return _SAGE_FACTOR_WORKER
    os.makedirs(SAGE_DOT_DIR, exist_ok=True)
    worker_path = os.path.abspath("tools/sage_hyperelliptic_factorized_worker.py")
    _SAGE_FACTOR_WORKER = subprocess.Popen(
        _sage_python_command() + [worker_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=_sage_subprocess_env(),
    )
    return _SAGE_FACTOR_WORKER


def _sage_factor_request(request):
    worker = _sage_factor_worker()
    assert worker.stdin is not None and worker.stdout is not None
    worker.stdin.write(json.dumps(request) + "\n")
    worker.stdin.flush()
    line = worker.stdout.readline()
    if not line:
        stderr = worker.stderr.read() if worker.stderr is not None else ""
        raise RuntimeError(f"Sage factorized worker exited unexpectedly. stderr:\n{stderr}")
    response = json.loads(line)
    if "error" in response:
        raise RuntimeError(f"Sage factorized worker error: {response['error']}")
    return response


def _polynomial_to_factorization(poly, p):
    response = _sage_factor_request(
        {"op": "factor_polynomial", "p": int(p), "coefficients": [int(c) for c in poly]}
    )
    if response["factors"] is None:
        return None, None
    return (
        int(response["leading_coefficient"]) % int(p),
        tuple(tuple(int(c) % int(p) for c in factor) for factor in response["factors"]),
    )


def _normalize_factor_blocks(blocks, p, leading_coefficient=1):
    response = _sage_factor_request(
        {
            "op": "normalize_factor_blocks",
            "p": int(p),
            "leading_coefficient": int(leading_coefficient) % int(p),
            "blocks": [[int(c) for c in block] for block in blocks],
        }
    )
    return {
        "coefficients": [int(c) % int(p) for c in response["coefficients"]],
        "factors": tuple(tuple(int(c) % int(p) for c in factor) for factor in response["factors"]),
        "factor_degrees": [int(degree) for degree in response["factor_degrees"]],
    }


def _random_irreducible_factor(degree, p):
    response = _sage_factor_request({"op": "random_irreducible_factor", "p": int(p), "degree": int(degree)})
    return tuple(int(c) % int(p) for c in response["factor"])


def _pgl2_orbit_factorizations(poly, p):
    response = _sage_factor_request(
        {
            "op": "pgl2_orbit_factorizations",
            "p": int(p),
            "coefficients": [int(c) for c in poly],
        }
    )
    return response.get("rows", [])


def _random_squarefree_monic_polynomial(degree, p):
    response = _sage_factor_request(
        {"op": "random_squarefree_monic_polynomial", "p": int(p), "degree": int(degree)}
    )
    return np.asarray(response["coefficients"], dtype=np.int64)


def _total_factor_degree(factors):
    return sum(len(factor) - 1 for factor in factors)


class FixedFactorizedHyperellipticTokenizer(Tokenizer):
    """Tokenize fixed-genus curves as a scalar and proposed monic factor blocks."""

    def __init__(self, dataclass, genus, p, extra_symbols):
        self.dataclass = dataclass
        self.fixed_genus = int(genus)
        self.p = int(p)
        self.max_factor_degree = 2 * self.fixed_genus + 2
        self.extra_symbols = list(extra_symbols) + ["LC"]
        self.stoi = {}
        self.itos = {}

        for coeff in range(self.p):
            self.stoi[coeff] = coeff
            self.itos[coeff] = coeff

        offset = self.p
        for degree in range(1, self.max_factor_degree + 1):
            self.stoi[f"D{degree}"] = offset
            self.itos[offset] = f"D{degree}"
            offset += 1
        for symbol in self.extra_symbols:
            self.stoi[symbol] = offset
            self.itos[offset] = symbol
            offset += 1

    def encode(self, datapoint_to_encode):
        factors = getattr(datapoint_to_encode, "factors", None)
        leading_coefficient = getattr(datapoint_to_encode, "leading_coefficient", None)
        if factors is None:
            leading_coefficient, factors = _polynomial_to_factorization(datapoint_to_encode.data.tolist(), self.p)
        if factors is None:
            raise ValueError("cannot tokenize a non-squarefree polynomial")
        leading_coefficient = int(1 if leading_coefficient is None else leading_coefficient) % self.p
        if leading_coefficient == 0:
            raise ValueError("leading coefficient must be nonzero")

        tokens = [self.stoi["BOS"], self.stoi["LC"], leading_coefficient, self.stoi["SEP"]]
        sorted_factors = tuple(sorted(factors))
        for idx, factor in enumerate(sorted_factors):
            degree = len(factor) - 1
            tokens.append(self.stoi[f"D{degree}"])
            tokens.extend(int(c) % self.p for c in factor[:-1])
            if idx + 1 < len(sorted_factors):
                tokens.append(self.stoi["SEP"])
        tokens.append(self.stoi["EOS"])
        return np.array(tokens, dtype=np.int32)

    def cache_key_for_datapoint(self, datapoint, max_len):
        return (self.__class__.__name__, getattr(datapoint, "features", None))

    def decode(self, token_seq_to_decode):
        try:
            seq = [int(t) for t in token_seq_to_decode]
            if len(seq) < 5 or self.itos.get(seq[0]) != "BOS":
                return None
            if self.itos.get(seq[1]) != "LC":
                return None
            leading_decoded = self.itos.get(seq[2])
            if not isinstance(leading_decoded, int):
                return None
            leading_coefficient = int(leading_decoded) % self.p
            if leading_coefficient == 0:
                return None
            if self.itos.get(seq[3]) != "SEP":
                return None

            pos = 4
            blocks = []
            eos_pos = None
            while pos < len(seq):
                token = self.itos.get(seq[pos])
                if token == "EOS":
                    eos_pos = pos
                    break
                if not isinstance(token, str) or not token.startswith("D"):
                    return None
                degree = int(token[1:])
                if degree < 1 or degree > self.max_factor_degree:
                    return None
                pos += 1
                if pos + degree > len(seq):
                    return None
                coeffs = []
                for token_id in seq[pos : pos + degree]:
                    decoded = self.itos.get(token_id)
                    if decoded in self.extra_symbols or not isinstance(decoded, int):
                        return None
                    coeffs.append(int(decoded) % self.p)
                blocks.append(tuple(coeffs + [1]))
                pos += degree
                separator = self.itos.get(seq[pos]) if pos < len(seq) else None
                if separator == "SEP":
                    pos += 1
                elif separator != "EOS":
                    return None

            if eos_pos is None:
                return None
            degree = _total_factor_degree(blocks)
            if degree not in self.dataclass._allowed_degrees():
                return None
            genus = _genus_from_degree(degree)
            if genus != self.fixed_genus:
                return None
            normalized = _normalize_factor_blocks(blocks, self.p, leading_coefficient)
            if not is_mod2_allowed_factor_degrees(
                normalized["factor_degrees"],
                genus,
                degree == 2 * genus + 1,
            ):
                return None

            datapoint = self.dataclass(N=genus)
            datapoint.p = self.p
            datapoint.data = np.asarray(normalized["coefficients"], dtype=np.int64)
            datapoint.leading_coefficient = leading_coefficient
            datapoint.factors = normalized["factors"]
            datapoint.degree = len(datapoint.data) - 1
            datapoint.genus = genus
            datapoint.N = genus
            datapoint.calc_features()
            clean_tokens = self.encode(datapoint)
            datapoint._encoded_token_cache_key = self.cache_key_for_datapoint(datapoint, None)
            datapoint._encoded_token_cache_value = clean_tokens
            return datapoint
        except Exception:
            return None


class FixedFactorizedHyperellipticDataPoint(DataPoint):
    PRIME = 3
    GENUS = 8
    LOCAL_SEARCH_STEPS = 0
    LOCAL_SEARCH_BATCH_SIZE = 32
    LOCAL_SEARCH_SCORE_BIAS = False
    LOCAL_SEARCH_SCORE_BIAS_MAX_MULT = 4.0
    LOCAL_SEARCH_SCORE_BIAS_MIN_SCORE = 0.0
    LOCAL_SEARCH_SCORE_BIAS_BASE = 2.0
    LOCAL_SEARCH_SCORE_BIAS_TOP_ROUNDS = 0
    LOCAL_SEARCH_SAME_TYPE_WEIGHT = 0.80
    LOCAL_SEARCH_SPLIT_WEIGHT = 0.10
    LOCAL_SEARCH_MERGE_WEIGHT = 0.10
    LOCAL_SEARCH_PGL2_TOP_ORBIT = True
    SCORE_BATCH_SIZE = 32
    SCORE_TIEBREAK_MODE = "hasse_witt"
    SPARSITY_REJECT_THRESHOLD = -1
    SAGE_PYTHON = SAGE_PYTHON
    SAGE_DOT_DIR = SAGE_DOT_DIR
    LOCAL_SEARCH_CURRENT_MAX_SCORE = None
    _SAGE_SCORE_WORKER = None

    def __init__(self, N, init=False):
        super().__init__()
        self.N = int(N)
        self.genus = int(N)
        self.p = self.PRIME
        self.degree = 2 * self.genus + random.randint(1, 2)
        self.data = np.zeros(self.degree + 1, dtype=np.int64)
        self.leading_coefficient = 1
        self.factors = None
        self.lpoly = None
        self.middle = None
        self.target_coeffs = None
        self.hw_target_coeffs = None
        self.hw_zero_count = None
        self.lpoly_zero_count = None
        if init:
            self.genus = self.GENUS
            self.N = self.GENUS
            self.degree = 2 * self.GENUS + random.randint(1, 2)
            self.data = _random_squarefree_monic_polynomial(self.degree, self.p)
            leading_coefficient = random.randrange(1, self.p)
            if leading_coefficient != 1:
                self.data = (self.data * leading_coefficient) % self.p
            self.leading_coefficient, self.factors = _polynomial_to_factorization(self.data.tolist(), self.p)
            self.calc_features()

    @classmethod
    def _allowed_degrees(cls):
        return {2 * int(cls.GENUS) + 1, 2 * int(cls.GENUS) + 2}

    def calc_features(self):
        if self.factors is None:
            self.leading_coefficient, self.factors = _polynomial_to_factorization(self.data.tolist(), self.p)
        if self.factors is not None:
            factor_text = ";".join(",".join(str(int(c) % self.p) for c in factor) for factor in self.factors)
            self.leading_coefficient = int(self.leading_coefficient) % int(self.p)
            self.features = f"p={self.p};g={self.genus};lc={self.leading_coefficient};factors={factor_text}"
        else:
            coeffs = ",".join(str(int(c) % self.p) for c in self.data.tolist())
            self.features = f"p={self.p};g={self.genus};coeffs={coeffs}"

    def calc_score(self):
        if not self._passes_mod2_filter():
            self.score = -1.0
            return
        scored, _ = self._score_arrays([self.data], [self.p])
        self._apply_score(scored[0])

    def _passes_mod2_filter(self):
        if self.factors is None:
            self.leading_coefficient, self.factors = _polynomial_to_factorization(self.data.tolist(), self.p)
        if self.factors is None:
            return False
        degree = _total_factor_degree(self.factors)
        genus = _genus_from_degree(degree)
        if genus != self.GENUS or degree not in self._allowed_degrees():
            return False
        return is_mod2_allowed_factor_degrees(
            [len(factor) - 1 for factor in self.factors],
            genus,
            degree == 2 * genus + 1,
        )

    def _apply_score(self, row):
        self.score = float(row["score"])
        self.genus = int(row["genus"]) if row["genus"] is not None else self.genus
        self.N = self.genus
        self.degree = len(self.data) - 1
        self.lpoly = [int(v) for v in row["lpoly"]] if row["lpoly"] is not None else None
        self.middle = int(row["middle"]) if row["middle"] is not None else None
        self.target_coeffs = (
            [int(v) for v in row["target_coeffs"]] if row["target_coeffs"] is not None else None
        )
        self.hw_target_coeffs = (
            [int(v) for v in row.get("hw_target_coeffs")] if row.get("hw_target_coeffs") is not None else None
        )
        self.hw_zero_count = int(row["hw_zero_count"]) if row.get("hw_zero_count") is not None else None
        self.lpoly_zero_count = int(row["lpoly_zero_count"]) if row.get("lpoly_zero_count") is not None else None

    def _copy_score_metadata_from(self, other):
        self.score = float(other.score)
        self.lpoly = list(other.lpoly) if other.lpoly is not None else None
        self.middle = other.middle
        self.target_coeffs = list(other.target_coeffs) if other.target_coeffs is not None else None
        self.hw_target_coeffs = list(other.hw_target_coeffs) if other.hw_target_coeffs is not None else None
        self.hw_zero_count = other.hw_zero_count
        self.lpoly_zero_count = other.lpoly_zero_count

    def _is_true_trinomial_lpoly(self):
        target_len = int(self.GENUS) - 1
        if self.lpoly_zero_count is not None:
            return int(self.lpoly_zero_count) >= target_len
        if self.target_coeffs is None:
            return False
        return len(self.target_coeffs) == target_len and all(int(value) == 0 for value in self.target_coeffs)

    def _is_top_score(self):
        return self._is_true_trinomial_lpoly()

    def pgl2_orbit_datapoints(self):
        if not self.LOCAL_SEARCH_PGL2_TOP_ORBIT or not self._is_top_score():
            return []
        rows = _pgl2_orbit_factorizations(self.data.tolist(), self.p)
        out = []
        seen = set()
        for row in rows:
            coeffs = [int(c) % int(self.p) for c in row["coefficients"]]
            degree = len(coeffs) - 1
            genus = _genus_from_degree(degree)
            if genus != self.GENUS or degree not in self._allowed_degrees():
                continue
            factors = tuple(tuple(int(c) % int(self.p) for c in factor) for factor in row["factors"])
            factor_degrees = [len(factor) - 1 for factor in factors]
            if not is_mod2_allowed_factor_degrees(factor_degrees, genus, degree == 2 * genus + 1):
                continue
            d = self.__class__(N=genus)
            d.p = int(self.p)
            d.data = np.asarray(coeffs, dtype=np.int64)
            d.leading_coefficient = int(row["leading_coefficient"]) % int(self.p)
            d.factors = factors
            d.degree = degree
            d.genus = genus
            d.N = genus
            d.calc_features()
            if d.features in seen:
                continue
            seen.add(d.features)
            d._copy_score_metadata_from(self)
            d.from_pgl2_orbit = True
            out.append(d)
        return out

    @classmethod
    def _sage_score_worker(cls):
        if cls._SAGE_SCORE_WORKER is not None and cls._SAGE_SCORE_WORKER.poll() is None:
            return cls._SAGE_SCORE_WORKER
        os.makedirs(cls.SAGE_DOT_DIR, exist_ok=True)
        worker_path = os.path.abspath("tools/sage_hyperelliptic_factorized_fixed_score_worker.py")
        env = _sage_subprocess_env()
        env["DOT_SAGE"] = cls.SAGE_DOT_DIR
        cls._SAGE_SCORE_WORKER = subprocess.Popen(
            _sage_python_command(cls.SAGE_PYTHON) + [worker_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
        )
        return cls._SAGE_SCORE_WORKER

    @classmethod
    def _score_arrays(cls, data, primes):
        scores = []
        n_invalid = 0
        if len(data) == 0:
            return scores, n_invalid
        worker = cls._sage_score_worker()
        assert worker.stdin is not None and worker.stdout is not None

        arrays = [np.asarray(row, dtype=np.int64).astype(int).tolist() for row in data]
        primes = [int(p) for p in primes]
        for p in sorted(set(primes)):
            group = [idx for idx, value in enumerate(primes) if value == p]
            for start in range(0, len(group), cls.SCORE_BATCH_SIZE):
                indices = group[start : start + cls.SCORE_BATCH_SIZE]
                request = {
                    "p": p,
                    "genus": cls.GENUS,
                    "data": [arrays[idx] for idx in indices],
                    "score_tiebreak_mode": cls.SCORE_TIEBREAK_MODE,
                    "sparsity_reject_threshold": cls.SPARSITY_REJECT_THRESHOLD,
                }
                worker.stdin.write(json.dumps(request) + "\n")
                worker.stdin.flush()
                line = worker.stdout.readline()
                if not line:
                    stderr = worker.stderr.read() if worker.stderr is not None else ""
                    raise RuntimeError(f"Sage fixed factorized scorer exited unexpectedly. stderr:\n{stderr}")
                response = json.loads(line)
                if "error" in response:
                    raise RuntimeError(f"Sage fixed factorized scorer error: {response['error']}")
                for offset, idx in enumerate(indices):
                    while len(scores) <= idx:
                        scores.append(None)
                    row = {
                        "score": float(response["scores"][offset]),
                        "valid": bool(response["valid"][offset]),
                        "genus": response["genera"][offset],
                        "lpoly": response["lpolys"][offset],
                        "middle": response["middles"][offset],
                        "target_coeffs": response["target_coeffs"][offset],
                        "hw_target_coeffs": response["hw_target_coeffs"][offset],
                        "hw_zero_count": response["hw_zero_counts"][offset],
                        "lpoly_zero_count": response["lpoly_zero_counts"][offset],
                    }
                    if row["score"] < 0:
                        n_invalid += 1
                    scores[idx] = row
        return scores, n_invalid

    @classmethod
    def _from_factors(cls, factors, p, leading_coefficient=1):
        leading_coefficient = int(leading_coefficient) % int(p)
        if leading_coefficient == 0:
            return None
        total_degree = _total_factor_degree(factors)
        if total_degree not in cls._allowed_degrees():
            return None
        genus = _genus_from_degree(total_degree)
        if genus != cls.GENUS:
            return None
        blocks = tuple(tuple(int(c) % int(p) for c in factor) for factor in factors)
        try:
            normalized = _normalize_factor_blocks(blocks, p, leading_coefficient)
        except Exception:
            return None
        if not is_mod2_allowed_factor_degrees(
            normalized["factor_degrees"],
            genus,
            total_degree == 2 * genus + 1,
        ):
            return None
        d = cls(N=genus)
        d.p = int(p)
        d.data = np.asarray(normalized["coefficients"], dtype=np.int64)
        d.leading_coefficient = leading_coefficient
        d.factors = normalized["factors"]
        d.degree = len(d.data) - 1
        d.genus = genus
        d.N = genus
        d.calc_features()
        return d

    @classmethod
    def _from_coefficients(cls, coeffs, p):
        coeffs = [int(c) % int(p) for c in coeffs]
        degree = len(coeffs) - 1
        genus = _genus_from_degree(degree)
        if genus != cls.GENUS or degree not in cls._allowed_degrees():
            return None
        leading_coefficient, factors = _polynomial_to_factorization(coeffs, p)
        if factors is None:
            return None
        d = cls(N=genus)
        d.p = int(p)
        d.data = np.asarray(coeffs, dtype=np.int64)
        d.leading_coefficient = leading_coefficient
        d.factors = tuple(factors)
        d.degree = degree
        d.genus = genus
        d.N = genus
        d.calc_features()
        return d

    def _replace_same_degree_factor(self, factors):
        if not factors:
            return None
        mutated = list(factors)
        idx = random.randrange(len(mutated))
        mutated[idx] = _random_irreducible_factor(len(mutated[idx]) - 1, self.p)
        return tuple(sorted(mutated))

    def _split_factor(self, factors):
        splittable = [idx for idx, factor in enumerate(factors) if len(factor) - 1 >= 2]
        if not splittable:
            return None
        mutated = list(factors)
        idx = random.choice(splittable)
        degree = len(mutated.pop(idx)) - 1
        left = random.randint(1, degree - 1)
        mutated.extend([
            _random_irreducible_factor(left, self.p),
            _random_irreducible_factor(degree - left, self.p),
        ])
        return tuple(sorted(mutated))

    def _merge_factors(self, factors):
        if len(factors) < 2:
            return None
        mutated = list(factors)
        i, j = sorted(random.sample(range(len(mutated)), 2), reverse=True)
        degree = len(mutated.pop(i)) - 1 + len(mutated.pop(j)) - 1
        mutated.append(_random_irreducible_factor(degree, self.p))
        return tuple(sorted(mutated))

    def _toggle_odd_even_model(self, factors):
        total_degree = _total_factor_degree(factors)
        mutated = list(factors)
        if total_degree == 2 * self.GENUS + 1:
            mutated.append(_random_irreducible_factor(1, self.p))
            return tuple(sorted(mutated))
        if total_degree == 2 * self.GENUS + 2 and len(mutated) > 1:
            linear = [idx for idx, factor in enumerate(mutated) if len(factor) - 1 == 1]
            if not linear:
                return None
            mutated.pop(random.choice(linear))
            return tuple(sorted(mutated))
        return None

    def _mutate_factors(self, factors):
        moves = [
            (self._replace_same_degree_factor, max(0.0, float(self.LOCAL_SEARCH_SAME_TYPE_WEIGHT))),
            (self._split_factor, max(0.0, float(self.LOCAL_SEARCH_SPLIT_WEIGHT))),
            (self._merge_factors, max(0.0, float(self.LOCAL_SEARCH_MERGE_WEIGHT))),
        ]
        positive_moves = [(move, weight) for move, weight in moves if weight > 0.0]
        if positive_moves:
            move = random.choices(
                [item[0] for item in positive_moves],
                weights=[item[1] for item in positive_moves],
                k=1,
            )[0]
            mutated = move(factors)
            if mutated is not None and _total_factor_degree(mutated) in self._allowed_degrees():
                return mutated
        fallback_moves = [
            self._replace_same_degree_factor,
            self._split_factor,
            self._merge_factors,
        ]
        random.shuffle(fallback_moves)
        for move in fallback_moves:
            mutated = move(factors)
            if mutated is not None and _total_factor_degree(mutated) in self._allowed_degrees():
                return mutated
        return None

    def local_search(self, improve_with_local_search):
        if self.LOCAL_SEARCH_STEPS <= 0:
            self.calc_score()
            return
        if self.factors is None:
            self.leading_coefficient, self.factors = _polynomial_to_factorization(self.data.tolist(), self.p)
        best_factors = list(self.factors or [])
        best = self
        rounds = self.LOCAL_SEARCH_STEPS if improve_with_local_search else max(1, self.LOCAL_SEARCH_STEPS // 4)
        if improve_with_local_search and self.LOCAL_SEARCH_SCORE_BIAS and self.score >= self.LOCAL_SEARCH_SCORE_BIAS_MIN_SCORE:
            max_score = self.LOCAL_SEARCH_CURRENT_MAX_SCORE
            if max_score is None:
                max_score = max(1.0, float(self.GENUS - 1))
            max_score_bucket = max(0, math.floor(float(max_score)))
            score_bucket = max(0, math.floor(float(self.score)))
            base = max(1.0, float(self.LOCAL_SEARCH_SCORE_BIAS_BASE))
            top_rounds = int(self.LOCAL_SEARCH_SCORE_BIAS_TOP_ROUNDS)
            if top_rounds <= 0:
                top_rounds = max(1, int(round(rounds * max(1.0, float(self.LOCAL_SEARCH_SCORE_BIAS_MAX_MULT)))))
            score_gap = max(0, max_score_bucket - score_bucket)
            rounds = max(1, int(round(top_rounds / (base ** score_gap))))
        for _ in range(rounds):
            candidates = []
            for _ in range(self.LOCAL_SEARCH_BATCH_SIZE):
                mutated = self._mutate_factors(best_factors)
                if mutated is None:
                    continue
                candidate = self._from_factors(mutated, self.p, self.leading_coefficient)
                if candidate is not None:
                    candidates.append(candidate)
            if not candidates:
                continue
            scored, _ = self._score_arrays([c.data for c in candidates], [c.p for c in candidates])
            for candidate, row in zip(candidates, scored):
                candidate._apply_score(row)
            usable = [candidate for candidate in candidates if candidate.score >= 0]
            if not usable:
                continue
            next_best = max(usable, key=lambda d: (d.score, d.degree))
            if best.score >= 0 and (next_best.score, next_best.degree) <= (best.score, best.degree):
                continue
            best = next_best
            best_factors = list(best.factors or [])
        self.__dict__.update(best.__dict__)

    @classmethod
    def _update_class_params(cls, pars):
        cls.PRIME = int(pars["prime"])
        cls.GENUS = int(pars["genus"])
        cls.LOCAL_SEARCH_STEPS = int(pars.get("local_search_steps", cls.LOCAL_SEARCH_STEPS))
        cls.LOCAL_SEARCH_BATCH_SIZE = int(pars.get("local_search_batch_size", cls.LOCAL_SEARCH_BATCH_SIZE))
        cls.LOCAL_SEARCH_SCORE_BIAS = str(pars.get("local_search_score_bias", cls.LOCAL_SEARCH_SCORE_BIAS)).lower() in {"1", "true", "yes", "on"}
        cls.LOCAL_SEARCH_SCORE_BIAS_MAX_MULT = float(
            pars.get("local_search_score_bias_max_mult", cls.LOCAL_SEARCH_SCORE_BIAS_MAX_MULT)
        )
        cls.LOCAL_SEARCH_SCORE_BIAS_MIN_SCORE = float(
            pars.get("local_search_score_bias_min_score", cls.LOCAL_SEARCH_SCORE_BIAS_MIN_SCORE)
        )
        cls.LOCAL_SEARCH_SCORE_BIAS_BASE = float(
            pars.get("local_search_score_bias_base", cls.LOCAL_SEARCH_SCORE_BIAS_BASE)
        )
        cls.LOCAL_SEARCH_SCORE_BIAS_TOP_ROUNDS = int(
            pars.get("local_search_score_bias_top_rounds", cls.LOCAL_SEARCH_SCORE_BIAS_TOP_ROUNDS)
        )
        cls.LOCAL_SEARCH_SAME_TYPE_WEIGHT = float(
            pars.get("local_search_same_type_weight", cls.LOCAL_SEARCH_SAME_TYPE_WEIGHT)
        )
        cls.LOCAL_SEARCH_SPLIT_WEIGHT = float(
            pars.get("local_search_split_weight", cls.LOCAL_SEARCH_SPLIT_WEIGHT)
        )
        cls.LOCAL_SEARCH_MERGE_WEIGHT = float(
            pars.get("local_search_merge_weight", cls.LOCAL_SEARCH_MERGE_WEIGHT)
        )
        cls.LOCAL_SEARCH_PGL2_TOP_ORBIT = str(
            pars.get("local_search_pgl2_top_orbit", cls.LOCAL_SEARCH_PGL2_TOP_ORBIT)
        ).lower() in {"1", "true", "yes", "on"}
        cls.SCORE_BATCH_SIZE = int(pars.get("score_batch_size", cls.SCORE_BATCH_SIZE))
        cls.SCORE_TIEBREAK_MODE = str(pars.get("score_tiebreak_mode", cls.SCORE_TIEBREAK_MODE))
        cls.SPARSITY_REJECT_THRESHOLD = int(
            pars.get("sparsity_reject_threshold", cls.SPARSITY_REJECT_THRESHOLD)
        )

    @classmethod
    def _save_class_params(cls):
        return {
            "prime": cls.PRIME,
            "genus": cls.GENUS,
            "local_search_steps": cls.LOCAL_SEARCH_STEPS,
            "local_search_batch_size": cls.LOCAL_SEARCH_BATCH_SIZE,
            "local_search_score_bias": cls.LOCAL_SEARCH_SCORE_BIAS,
            "local_search_score_bias_max_mult": cls.LOCAL_SEARCH_SCORE_BIAS_MAX_MULT,
            "local_search_score_bias_min_score": cls.LOCAL_SEARCH_SCORE_BIAS_MIN_SCORE,
            "local_search_score_bias_base": cls.LOCAL_SEARCH_SCORE_BIAS_BASE,
            "local_search_score_bias_top_rounds": cls.LOCAL_SEARCH_SCORE_BIAS_TOP_ROUNDS,
            "local_search_same_type_weight": cls.LOCAL_SEARCH_SAME_TYPE_WEIGHT,
            "local_search_split_weight": cls.LOCAL_SEARCH_SPLIT_WEIGHT,
            "local_search_merge_weight": cls.LOCAL_SEARCH_MERGE_WEIGHT,
            "local_search_pgl2_top_orbit": cls.LOCAL_SEARCH_PGL2_TOP_ORBIT,
            "score_batch_size": cls.SCORE_BATCH_SIZE,
            "score_tiebreak_mode": cls.SCORE_TIEBREAK_MODE,
            "sparsity_reject_threshold": cls.SPARSITY_REJECT_THRESHOLD,
        }

    @classmethod
    def _load_seed_pickle(cls, path, max_rows):
        loaded = pickle.load(open(path, "rb"))
        out = []
        for item in loaded:
            coeffs = getattr(item, "data", None)
            p = int(getattr(item, "p", cls.PRIME))
            if coeffs is None or p != cls.PRIME:
                continue
            degree = len(coeffs) - 1
            genus = _genus_from_degree(degree)
            if genus != cls.GENUS or degree not in cls._allowed_degrees():
                continue
            factors = getattr(item, "factors", None)
            if factors is None:
                d = cls._from_coefficients(np.asarray(coeffs, dtype=np.int64).tolist(), p)
            else:
                d = cls(N=genus)
                d.p = p
                d.data = np.asarray(coeffs, dtype=np.int64)
                d.leading_coefficient = int(getattr(item, "leading_coefficient", int(d.data[-1]) % p)) % p
                d.factors = tuple(tuple(int(c) % p for c in factor) for factor in factors)
                d.degree = degree
                d.genus = genus
                d.N = genus
                d.calc_features()
            if d is not None:
                d.score = float(getattr(item, "score", -1.0))
                d.lpoly = getattr(item, "lpoly", None)
                d.middle = getattr(item, "middle", None)
                d.target_coeffs = getattr(item, "target_coeffs", None)
                d.hw_target_coeffs = getattr(item, "hw_target_coeffs", None)
                d.hw_zero_count = getattr(item, "hw_zero_count", None)
                d.lpoly_zero_count = getattr(item, "lpoly_zero_count", None)
                out.append(d)
                if max_rows and max_rows > 0 and len(out) >= max_rows:
                    break
        return out

    @classmethod
    def _load_seed_sqlite(cls, path, max_rows):
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table','view')")}
        if "canonical_classes" in tables:
            table = "canonical_classes"
            coeff_col = "representative_coefficients"
            where = f"prime = {int(cls.PRIME)} and genus = {int(cls.GENUS)} and sparsity = 0"
        elif "curves" in tables:
            table = "curves"
            columns = {row[1] for row in conn.execute("PRAGMA table_info(curves)")}
            coeff_col = "coefficients_json" if "coefficients_json" in columns else None
            where = "1=1"
            if coeff_col is None:
                raise ValueError(f"{path} curves table does not contain coefficients_json")
        else:
            raise ValueError(f"{path} does not contain canonical_classes or curves")
        limit = "" if max_rows <= 0 else f" LIMIT {int(max_rows)}"
        sql = f"SELECT prime, {coeff_col} AS coefficients FROM {table} WHERE {where} ORDER BY genus DESC, prime ASC{limit}"
        rows = list(conn.execute(sql))
        conn.close()
        out = []
        for row in rows:
            p = int(row["prime"])
            if p != cls.PRIME:
                continue
            d = cls._from_coefficients(ast.literal_eval(row["coefficients"]), p)
            if d is not None:
                out.append(d)
        return out

    @classmethod
    def load_initial_data(cls, path, N, max_rows=0, pars=None):
        if pars is not None:
            cls._update_class_params(pars)
        candidates = cls._load_seed_pickle(path, max_rows) if path.endswith(".pkl") else cls._load_seed_sqlite(path, max_rows)
        candidates = [
            d for d in candidates
            if int(d.p) == int(cls.PRIME)
            and int(d.genus) == int(cls.GENUS)
            and int(d.degree) in cls._allowed_degrees()
        ]
        if not candidates:
            return []
        rows, _ = cls._score_arrays([d.data for d in candidates], [d.p for d in candidates])
        out = []
        for d, row in zip(candidates, rows):
            d._apply_score(row)
            d.calc_features()
            if d.score >= 0:
                out.append(d)
        return out

    @classmethod
    def _batch_generate_and_score(cls, batch_size, N, pars=None):
        if pars is not None:
            cls._update_class_params(pars)
        data = [cls(N=cls.GENUS, init=True) for _ in range(batch_size)]
        scores, _ = cls._score_arrays([d.data for d in data], [d.p for d in data])
        out = []
        for d, row in zip(data, scores):
            d._apply_score(row)
            if d.score >= 0 and int(d.genus) == int(cls.GENUS):
                out.append(d)
        return out

    @classmethod
    def _batch_score_datapoints(cls, data, always_search=False, redeem_only=False, pars=None):
        if pars is not None:
            cls._update_class_params(pars)
        if not data:
            return [], 0
        scores = [None] * len(data)
        score_indices = []
        n_invalid = 0
        for idx, d in enumerate(data):
            d.calc_features()
            if d._passes_mod2_filter():
                score_indices.append(idx)
            else:
                scores[idx] = {
                    "score": -1.0,
                    "valid": False,
                    "genus": d.genus,
                    "lpoly": None,
                    "middle": None,
                    "target_coeffs": None,
                    "hw_target_coeffs": None,
                    "hw_zero_count": None,
                    "lpoly_zero_count": None,
                }
                n_invalid += 1
        if score_indices:
            scored, sage_invalid = cls._score_arrays(
                [data[idx].data for idx in score_indices],
                [data[idx].p for idx in score_indices],
            )
            n_invalid += sage_invalid
            for idx, row in zip(score_indices, scored):
                scores[idx] = row
        for d, row in zip(data, scores):
            d._apply_score(row)
        valid_scores = [float(d.score) for d in data if d.score >= 0]
        cls.LOCAL_SEARCH_CURRENT_MAX_SCORE = max(valid_scores) if valid_scores else None
        try:
            for d in data:
                if always_search or (d.score < 0 and redeem_only):
                    d.local_search(improve_with_local_search=always_search)
        finally:
            cls.LOCAL_SEARCH_CURRENT_MAX_SCORE = None
        if always_search and cls.LOCAL_SEARCH_PGL2_TOP_ORBIT:
            expanded = []
            seen = {d.features for d in data}
            for d in data:
                expanded.append(d)
                if not d._is_top_score():
                    continue
                for orbit_point in d.pgl2_orbit_datapoints():
                    if orbit_point.features in seen:
                        continue
                    seen.add(orbit_point.features)
                    expanded.append(orbit_point)
            data = expanded
        return data, n_invalid


class FixedFactorizedHyperellipticEnvironment(BaseEnvironment):
    data_class = FixedFactorizedHyperellipticDataPoint

    def __init__(self, params):
        super().__init__(params)
        if params.p not in SUPPORTED_PRIMES:
            raise ValueError("fixed factorized hyperelliptic supports --p in {3, 5, 7, 11}")
        if params.N < 1:
            raise ValueError("--N must be a positive exact genus")
        max_factor_degree = 2 * params.N + 2
        needed_len = 3 * max_factor_degree + 4
        if params.max_len < needed_len:
            raise ValueError(f"--max_len must be at least {needed_len} for fixed genus {params.N}")

        self.data_class.PRIME = params.p
        self.data_class.GENUS = params.N
        self.data_class.LOCAL_SEARCH_STEPS = params.local_search_steps
        self.data_class.LOCAL_SEARCH_BATCH_SIZE = params.local_search_batch_size
        self.data_class.LOCAL_SEARCH_SCORE_BIAS = params.local_search_score_bias
        self.data_class.LOCAL_SEARCH_SCORE_BIAS_MAX_MULT = params.local_search_score_bias_max_mult
        self.data_class.LOCAL_SEARCH_SCORE_BIAS_MIN_SCORE = params.local_search_score_bias_min_score
        self.data_class.LOCAL_SEARCH_SCORE_BIAS_BASE = params.local_search_score_bias_base
        self.data_class.LOCAL_SEARCH_SCORE_BIAS_TOP_ROUNDS = params.local_search_score_bias_top_rounds
        self.data_class.LOCAL_SEARCH_SAME_TYPE_WEIGHT = params.local_search_same_type_weight
        self.data_class.LOCAL_SEARCH_SPLIT_WEIGHT = params.local_search_split_weight
        self.data_class.LOCAL_SEARCH_MERGE_WEIGHT = params.local_search_merge_weight
        self.data_class.LOCAL_SEARCH_PGL2_TOP_ORBIT = params.local_search_pgl2_top_orbit
        self.data_class.SCORE_BATCH_SIZE = params.score_batch_size
        self.data_class.SCORE_TIEBREAK_MODE = params.score_tiebreak_mode
        self.data_class.SPARSITY_REJECT_THRESHOLD = params.sparsity_reject_threshold
        self.tokenizer = FixedFactorizedHyperellipticTokenizer(
            dataclass=self.data_class,
            genus=params.N,
            p=params.p,
            extra_symbols=self.SPECIAL_SYMBOLS,
        )

    @staticmethod
    def register_args(parser):
        parser.add_argument("--N", type=int, default=8, help="Exact genus")
        parser.add_argument("--p", type=int, default=3, help="Prime field characteristic")
        parser.add_argument("--local_search_steps", type=int, default=0, help="Fixed-genus factor local-search rounds")
        parser.add_argument("--local_search_batch_size", type=int, default=32, help="Factor mutations tested per round")
        parser.add_argument("--local_search_score_bias", type=bool_flag, default=False, help="If true, give higher-scoring candidates more local-search rounds")
        parser.add_argument("--local_search_score_bias_max_mult", type=float, default=4.0, help="Fallback multiplier for top-score local-search rounds when --local_search_score_bias_top_rounds is 0")
        parser.add_argument("--local_search_score_bias_min_score", type=float, default=0.0, help="Minimum score required before score-biased local-search scaling applies")
        parser.add_argument("--local_search_score_bias_base", type=float, default=2.0, help="Exponential local-search decay per score point below maximum")
        parser.add_argument("--local_search_score_bias_top_rounds", type=int, default=0, help="Local-search rounds assigned to maximum-score candidates; 0 uses --local_search_steps times --local_search_score_bias_max_mult")
        parser.add_argument("--local_search_same_type_weight", type=float, default=0.80, help="Local-search mutation weight for replacing a factor by another factor of the same degree")
        parser.add_argument("--local_search_split_weight", type=float, default=0.10, help="Local-search mutation weight for splitting one factor into two factors")
        parser.add_argument("--local_search_merge_weight", type=float, default=0.10, help="Local-search mutation weight for merging two factors")
        parser.add_argument("--local_search_pgl2_top_orbit", type=bool_flag, default=True, help="If true, append the full PGL2 orbit of top-score local-search results")
        parser.add_argument("--score_batch_size", type=int, default=32, help="Sage scorer batch size")
        parser.add_argument("--sparsity_reject_threshold", type=int, default=-1, help="Deprecated; sparsity is no longer rejected during fixed-factorized scoring")
        parser.add_argument(
            "--score_tiebreak_mode",
            type=str,
            default="hasse_witt",
            choices=["hasse_witt", "average", "archimedean"],
            help="Fractional tie-break for equal actual sparsity: hasse_witt uses mod-p sparsity, average/archimedean uses normalized average absolute L-polynomial coefficient size",
        )
        parser.add_argument("--initial_data_sqlite", type=str, default="", help="Comma-separated SQLite or pickle seed files")
        parser.add_argument("--initial_data_max_rows", type=int, default=0, help="Maximum seed rows loaded from each file; 0 means all")
