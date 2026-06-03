import ast
import json
import os
import pickle
import random
import shlex
import sqlite3
import subprocess
from functools import lru_cache

import numpy as np

from src.envs.environment import BaseEnvironment, DataPoint
from src.envs.tokenizers import Tokenizer
from src.utils import bool_flag


SUPPORTED_NORMAL_BASIS_PRIMES = {3, 5, 7, 11}
MAX_PRECOMPUTED_EXTENSION_DEGREE = 1999
NORMAL_BASIS_TABLE_PATH = os.path.join(os.path.dirname(__file__), "hyperelliptic2_normal_basis.json")
SAGE_PYTHON = "/Applications/SageMath-10-6.app/Contents/MacOS/Python"
SAGE_DOT_DIR = os.environ.get("SAGE_DOT_DIR", os.environ.get("DOT_SAGE", "/private/tmp/sage-dot-cache"))
_SAGE_NECKLACE_WORKER = None


def _sage_python_command(default=SAGE_PYTHON):
    command = os.environ.get("SAGE_PYTHON_CMD") or os.environ.get("SAGE_PYTHON") or default
    parts = shlex.split(command)
    if not parts:
        raise RuntimeError("empty Sage Python command")
    executable = parts[0]
    if os.path.sep in executable and not os.path.exists(executable):
        raise RuntimeError(f"Sage Python executable not found: {executable}")
    return parts


def _sage_necklace_worker():
    global _SAGE_NECKLACE_WORKER
    if _SAGE_NECKLACE_WORKER is not None and _SAGE_NECKLACE_WORKER.poll() is None:
        return _SAGE_NECKLACE_WORKER
    os.makedirs(SAGE_DOT_DIR, exist_ok=True)
    worker_path = os.path.abspath("tools/sage_hyperelliptic2_necklace_worker.py")
    env = os.environ.copy()
    env["DOT_SAGE"] = SAGE_DOT_DIR
    _SAGE_NECKLACE_WORKER = subprocess.Popen(
        _sage_python_command() + [worker_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=env,
    )
    return _SAGE_NECKLACE_WORKER


def _sage_necklace_request(request):
    worker = _sage_necklace_worker()
    assert worker.stdin is not None and worker.stdout is not None
    worker.stdin.write(json.dumps(request) + "\n")
    worker.stdin.flush()
    line = worker.stdout.readline()
    if not line:
        stderr = worker.stderr.read() if worker.stderr is not None else ""
        raise RuntimeError(f"Sage hyperelliptic2 necklace worker exited unexpectedly. stderr:\n{stderr}")
    response = json.loads(line)
    if "error" in response:
        raise RuntimeError(f"Sage hyperelliptic2 necklace worker error: {response['error']}")
    return response


@lru_cache(maxsize=1)
def _normal_basis_table():
    if not os.path.exists(NORMAL_BASIS_TABLE_PATH):
        raise RuntimeError(
            f"missing precomputed normal-basis table: {NORMAL_BASIS_TABLE_PATH}. "
            "Run tools/precompute_hyperelliptic2_normal_basis.sage.py first."
        )
    with open(NORMAL_BASIS_TABLE_PATH, "r") as f:
        return json.load(f)


def _normal_basis_record(degree, p):
    if p not in SUPPORTED_NORMAL_BASIS_PRIMES:
        raise ValueError(f"no precomputed normal-basis data for p={p}; supported primes are 3, 5, 7, 11")
    if degree < 1 or degree > MAX_PRECOMPUTED_EXTENSION_DEGREE:
        raise ValueError(f"normal-basis degree must be in [1, {MAX_PRECOMPUTED_EXTENSION_DEGREE}], got {degree}")

    table = _normal_basis_table()
    record = table.get("primes", {}).get(str(p), {}).get(str(degree))
    if record is None:
        raise RuntimeError(f"missing precomputed normal-basis polynomial B_{{{degree},{p}}}")
    polynomial = [int(c) % p for c in record["polynomial"]]
    if len(polynomial) != degree + 1 or polynomial[-1] != 1:
        raise RuntimeError(f"invalid precomputed B_{{{degree},{p}}}: expected monic degree {degree}")
    return record


def _ensure_normal_basis_coverage(p, max_degree):
    for degree in range(1, max_degree + 1):
        _normal_basis_record(degree, p)


def _is_aperiodic_necklace(necklace):
    degree = len(necklace)
    return all(necklace[shift:] + necklace[:shift] != necklace for shift in range(1, degree))


def _polynomial_to_necklaces(poly, p):
    response = _sage_necklace_request(
        {"op": "polynomial_to_necklaces", "p": int(p), "coefficients": [int(c) for c in poly]}
    )
    necklaces = response["necklaces"]
    if necklaces is None:
        return None
    return tuple(tuple(tuple(int(c) for c in vector) for vector in necklace) for necklace in necklaces)


def _necklaces_to_polynomial(necklaces, p):
    response = _sage_necklace_request(
        {
            "op": "necklaces_to_polynomial",
            "p": int(p),
            "necklaces": [
                [[int(c) for c in vector] for vector in necklace]
                for necklace in necklaces
            ],
        }
    )
    return [int(c) % p for c in response["coefficients"]]


def _random_necklace(degree, p):
    response = _sage_necklace_request({"op": "random_necklace", "p": int(p), "degree": int(degree)})
    return tuple(tuple(int(c) for c in vector) for vector in response["necklace"])


def _random_squarefree_monic_polynomial(degree, p):
    response = _sage_necklace_request(
        {"op": "random_squarefree_monic_polynomial", "p": int(p), "degree": int(degree)}
    )
    return np.asarray(response["coefficients"], dtype=np.int64)


def _total_necklace_degree(necklaces):
    return sum(len(necklace) for necklace in necklaces)


def _genus_from_degree(degree):
    degree = int(degree)
    if degree < 3:
        return None
    if degree % 2 == 1:
        return (degree - 1) // 2
    return (degree - 2) // 2


def _coerce_bool(value):
    if isinstance(value, bool):
        return value
    return bool_flag(str(value))


class Hyperelliptic2Tokenizer(Tokenizer):
    """Tokenize y^2=f(x) by the multiset of irreducible-factor necklaces."""

    def __init__(self, dataclass, max_genus, p, extra_symbols):
        self.dataclass = dataclass
        self.max_genus = int(max_genus)
        self.p = int(p)
        self.max_factor_degree = 2 * self.max_genus + 2
        self.extra_symbols = list(extra_symbols) + ["FACT"]
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
        necklaces = _polynomial_to_necklaces(datapoint_to_encode.data.tolist(), self.p)
        if necklaces is None:
            raise ValueError("cannot tokenize a non-squarefree polynomial")

        tokens = [self.stoi["BOS"], self.stoi["SEP"]]
        for idx, necklace in enumerate(necklaces):
            degree = len(necklace)
            tokens.append(self.stoi[f"D{degree}"])
            for vector in necklace:
                tokens.extend(int(c) % self.p for c in vector)
            if idx + 1 < len(necklaces):
                tokens.append(self.stoi["SEP"])
        tokens.append(self.stoi["EOS"])
        return np.array(tokens, dtype=np.int32)

    def decode(self, token_seq_to_decode):
        try:
            seq = [int(t) for t in token_seq_to_decode]
            if len(seq) < 3 or self.itos.get(seq[0]) != "BOS":
                return None
            if self.itos.get(seq[1]) != "SEP":
                return None
            pos = 2
            necklaces = []
            while pos < len(seq):
                token = self.itos.get(seq[pos])
                if token == "EOS":
                    break
                if not isinstance(token, str) or not token.startswith("D"):
                    return None
                degree = int(token[1:])
                if degree < 1 or degree > self.max_factor_degree:
                    return None
                pos += 1
                needed = degree * degree
                if pos + needed > len(seq):
                    return None
                raw = []
                for token_id in seq[pos : pos + needed]:
                    decoded = self.itos.get(token_id)
                    if decoded in self.extra_symbols or not isinstance(decoded, int):
                        return None
                    raw.append(decoded)
                necklace = tuple(tuple(raw[i : i + degree]) for i in range(0, needed, degree))
                if not _is_aperiodic_necklace(necklace):
                    return None
                necklaces.append(necklace)
                pos += needed
                separator = self.itos.get(seq[pos]) if pos < len(seq) else None
                if separator == "SEP":
                    pos += 1
                elif separator != "EOS":
                    return None

            if tuple(sorted(necklaces)) != tuple(necklaces):
                return None

            coeffs = _necklaces_to_polynomial(necklaces, self.p)
            degree = len(coeffs) - 1
            genus = _genus_from_degree(degree)
            if genus < 1 or genus > self.max_genus:
                return None

            datapoint = self.dataclass(N=genus)
            datapoint.data = np.asarray(coeffs, dtype=np.int64)
            datapoint.p = self.p
            datapoint.degree = degree
            datapoint.calc_features()
            return datapoint
        except Exception:
            return None


class Hyperelliptic2DataPoint(DataPoint):
    PRIME = 3
    MAX_GENUS = 8
    LOCAL_SEARCH_STEPS = 0
    LOCAL_SEARCH_BATCH_SIZE = 32
    LOCAL_SEARCH_HIGH_GENUS_BIAS = True
    LOCAL_SEARCH_GROW_PROB = 0.72
    LOCAL_SEARCH_REMOVE_PROB = 0.02
    SCORE_BATCH_SIZE = 32
    SAGE_PYTHON = "/Applications/SageMath-10-6.app/Contents/MacOS/Python"
    SAGE_DOT_DIR = os.environ.get("SAGE_DOT_DIR", os.environ.get("DOT_SAGE", "/private/tmp/sage-dot-cache"))
    _SAGE_WORKER = None

    def __init__(self, N, init=False):
        super().__init__()
        self.N = int(N)
        self.genus = int(N)
        self.p = self.PRIME
        self.degree = 2 * self.genus + random.randint(1, 2)
        self.data = np.zeros(self.degree + 1, dtype=np.int64)
        self.lpoly = None
        self.middle = None
        self.target_coeffs = None
        if init:
            self.genus = random.randint(1, self.MAX_GENUS)
            self.N = self.genus
            self.degree = 2 * self.genus + random.randint(1, 2)
            self.data = self._random_squarefree_monic_polynomial(self.degree)
            self.calc_features()

    @classmethod
    def _random_squarefree_monic_polynomial(cls, degree):
        return _random_squarefree_monic_polynomial(degree, cls.PRIME)

    def calc_features(self):
        coeffs = ",".join(str(int(c) % self.p) for c in self.data.tolist())
        self.features = f"p={self.p};g={self.genus};coeffs={coeffs}"

    def calc_score(self):
        scored, _ = self._score_arrays([self.data], [self.p])
        self._apply_score(scored[0])

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

    @classmethod
    def _sage_worker(cls):
        if cls._SAGE_WORKER is not None and cls._SAGE_WORKER.poll() is None:
            return cls._SAGE_WORKER
        os.makedirs(cls.SAGE_DOT_DIR, exist_ok=True)
        worker_path = os.path.abspath("tools/sage_hyperelliptic2_score_worker.py")
        env = os.environ.copy()
        env["DOT_SAGE"] = cls.SAGE_DOT_DIR
        cls._SAGE_WORKER = subprocess.Popen(
            _sage_python_command(cls.SAGE_PYTHON) + [worker_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
        )
        return cls._SAGE_WORKER

    @classmethod
    def _score_arrays(cls, data, primes):
        scores = []
        n_invalid = 0
        if len(data) == 0:
            return scores, n_invalid
        worker = cls._sage_worker()
        assert worker.stdin is not None and worker.stdout is not None

        arrays = [np.asarray(row, dtype=np.int64).astype(int).tolist() for row in data]
        primes = [int(p) for p in primes]
        for p in sorted(set(primes)):
            group = [idx for idx, value in enumerate(primes) if value == p]
            for start in range(0, len(group), cls.SCORE_BATCH_SIZE):
                indices = group[start : start + cls.SCORE_BATCH_SIZE]
                request = {"p": p, "data": [arrays[idx] for idx in indices]}
                worker.stdin.write(json.dumps(request) + "\n")
                worker.stdin.flush()
                line = worker.stdout.readline()
                if not line:
                    stderr = worker.stderr.read() if worker.stderr is not None else ""
                    raise RuntimeError(f"Sage hyperelliptic2 scorer exited unexpectedly. stderr:\n{stderr}")
                response = json.loads(line)
                if "error" in response:
                    raise RuntimeError(f"Sage hyperelliptic2 scorer error: {response['error']}")
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
                    }
                    if row["score"] < 0:
                        n_invalid += 1
                    scores[idx] = row
        return scores, n_invalid

    def _necklaces(self):
        return list(_polynomial_to_necklaces(self.data.tolist(), self.p) or [])

    @classmethod
    def _from_necklaces(cls, necklaces, p):
        total_degree = _total_necklace_degree(necklaces)
        if total_degree < 3 or total_degree > 2 * cls.MAX_GENUS + 2:
            return None
        if tuple(sorted(necklaces)) != tuple(necklaces):
            necklaces = sorted(necklaces)
        coeffs = _necklaces_to_polynomial(necklaces, p)
        genus = _genus_from_degree(len(coeffs) - 1)
        if genus < 1 or genus > cls.MAX_GENUS:
            return None
        d = cls(N=genus)
        d.p = int(p)
        d.data = np.asarray(coeffs, dtype=np.int64)
        d.degree = len(coeffs) - 1
        d.calc_features()
        return d

    def _random_growth_degree(self, lower, upper):
        if upper < lower:
            return None
        # Bias toward larger added/replacement factors without making the move deterministic.
        span = upper - lower + 1
        return lower + min(span - 1, int((random.random() ** 0.5) * span))

    def _add_higher_degree_factor(self, mutated, total_degree):
        max_degree = 2 * self.MAX_GENUS + 2
        remaining = max_degree - total_degree
        if remaining <= 0:
            return False
        lower = 2 if remaining >= 2 else 1
        degree = self._random_growth_degree(lower, remaining)
        if degree is None:
            return False
        mutated.append(_random_necklace(degree, self.p))
        return True

    def _replace_by_higher_degree_factor(self, mutated, total_degree):
        max_degree = 2 * self.MAX_GENUS + 2
        if total_degree + 1 > max_degree:
            return False
        expandable = [
            idx for idx, necklace in enumerate(mutated)
            if len(necklace) < max_degree - (total_degree - len(necklace))
        ]
        if not expandable:
            return False
        idx = random.choice(expandable)
        old_degree = len(mutated[idx])
        upper = max_degree - (total_degree - old_degree)
        new_degree = self._random_growth_degree(old_degree + 1, upper)
        if new_degree is None:
            return False
        mutated[idx] = _random_necklace(new_degree, self.p)
        return True

    def _mutate_necklaces(self, necklaces):
        if not necklaces:
            return None
        move = random.random()
        total_degree = _total_necklace_degree(necklaces)
        mutated = list(necklaces)

        if self.LOCAL_SEARCH_HIGH_GENUS_BIAS and move < self.LOCAL_SEARCH_GROW_PROB:
            if random.random() < 0.65:
                if self._add_higher_degree_factor(mutated, total_degree):
                    return tuple(sorted(mutated))
            if self._replace_by_higher_degree_factor(mutated, total_degree):
                return tuple(sorted(mutated))
            if self._add_higher_degree_factor(mutated, total_degree):
                return tuple(sorted(mutated))
            return None

        if self.LOCAL_SEARCH_HIGH_GENUS_BIAS:
            residual = random.random()
            remove_threshold = self.LOCAL_SEARCH_REMOVE_PROB
        else:
            residual = move
            remove_threshold = 0.10

        if residual < 0.40:
            idx = random.randrange(len(mutated))
            mutated[idx] = _random_necklace(len(mutated[idx]), self.p)
        elif residual < 0.62:
            splittable = [idx for idx, necklace in enumerate(mutated) if len(necklace) >= 2]
            if not splittable:
                return None
            idx = random.choice(splittable)
            degree = len(mutated.pop(idx))
            left = random.randint(1, degree - 1)
            mutated.extend([_random_necklace(left, self.p), _random_necklace(degree - left, self.p)])
        elif residual < 0.84:
            if len(mutated) < 2:
                return None
            i, j = sorted(random.sample(range(len(mutated)), 2), reverse=True)
            degree = len(mutated.pop(i)) + len(mutated.pop(j))
            mutated.append(_random_necklace(degree, self.p))
        elif residual < 1.0 - remove_threshold:
            if not self._add_higher_degree_factor(mutated, total_degree):
                return None
        else:
            if len(mutated) <= 1:
                return None
            mutated.pop(random.randrange(len(mutated)))
        return tuple(sorted(mutated))

    def local_search(self, improve_with_local_search):
        if self.LOCAL_SEARCH_STEPS <= 0:
            self.calc_score()
            return

        try:
            best_necklaces = self._necklaces()
        except Exception:
            best_necklaces = []
        best = self
        rounds = self.LOCAL_SEARCH_STEPS if improve_with_local_search else max(1, self.LOCAL_SEARCH_STEPS // 4)
        for _ in range(rounds):
            candidates = []
            for _ in range(self.LOCAL_SEARCH_BATCH_SIZE):
                mutated = self._mutate_necklaces(best_necklaces)
                if mutated is None:
                    continue
                candidate = self._from_necklaces(mutated, self.p)
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
            next_best = max(usable, key=lambda d: (d.score, d.genus, d.degree))
            if best.score >= 0 and (next_best.score, next_best.degree) <= (best.score, best.degree):
                continue
            best = next_best
            best_necklaces = best._necklaces()

        self.__dict__.update(best.__dict__)

    @classmethod
    def _update_class_params(cls, pars):
        cls.PRIME = int(pars["prime"])
        cls.MAX_GENUS = int(pars["max_genus"])
        cls.LOCAL_SEARCH_STEPS = int(pars.get("local_search_steps", cls.LOCAL_SEARCH_STEPS))
        cls.LOCAL_SEARCH_BATCH_SIZE = int(pars.get("local_search_batch_size", cls.LOCAL_SEARCH_BATCH_SIZE))
        cls.LOCAL_SEARCH_HIGH_GENUS_BIAS = _coerce_bool(
            pars.get("local_search_high_genus_bias", cls.LOCAL_SEARCH_HIGH_GENUS_BIAS)
        )
        cls.LOCAL_SEARCH_GROW_PROB = float(pars.get("local_search_grow_prob", cls.LOCAL_SEARCH_GROW_PROB))
        cls.LOCAL_SEARCH_REMOVE_PROB = float(pars.get("local_search_remove_prob", cls.LOCAL_SEARCH_REMOVE_PROB))
        cls.SCORE_BATCH_SIZE = int(pars.get("score_batch_size", cls.SCORE_BATCH_SIZE))

    @classmethod
    def _save_class_params(cls):
        return {
            "prime": cls.PRIME,
            "max_genus": cls.MAX_GENUS,
            "local_search_steps": cls.LOCAL_SEARCH_STEPS,
            "local_search_batch_size": cls.LOCAL_SEARCH_BATCH_SIZE,
            "local_search_high_genus_bias": cls.LOCAL_SEARCH_HIGH_GENUS_BIAS,
            "local_search_grow_prob": cls.LOCAL_SEARCH_GROW_PROB,
            "local_search_remove_prob": cls.LOCAL_SEARCH_REMOVE_PROB,
            "score_batch_size": cls.SCORE_BATCH_SIZE,
        }

    @classmethod
    def _make_datapoint(cls, N, row, score_row, p=None):
        genus = _genus_from_degree(len(row) - 1)
        d = cls(N=genus if genus is not None and genus >= 1 else N)
        d.p = int(cls.PRIME if p is None else p)
        d.data = np.asarray(row, dtype=np.int64)
        d.degree = len(row) - 1
        d.calc_features()
        d._apply_score(score_row)
        return d

    @classmethod
    def _batch_generate_and_score(cls, batch_size, N, pars=None):
        if pars is not None:
            cls._update_class_params(pars)
        data = [cls(N=random.randint(1, cls.MAX_GENUS), init=True) for _ in range(batch_size)]
        scores, _ = cls._score_arrays([d.data for d in data], [d.p for d in data])
        out = []
        for d, row in zip(data, scores):
            d._apply_score(row)
            if d.score >= 0:
                out.append(d)
        return out

    @classmethod
    def _batch_score_datapoints(cls, data, always_search=False, redeem_only=False, pars=None):
        if pars is not None:
            cls._update_class_params(pars)
        if not data:
            return [], 0
        scores, n_invalid = cls._score_arrays([d.data for d in data], [d.p for d in data])
        for d, row in zip(data, scores):
            d.calc_features()
            d._apply_score(row)
        for d in data:
            if always_search or (d.score < 0 and redeem_only):
                d.local_search(improve_with_local_search=always_search)
        return data, n_invalid

    @classmethod
    def _from_coefficients(cls, coeffs, p):
        coeffs = [int(c) % int(p) for c in coeffs]
        genus = _genus_from_degree(len(coeffs) - 1)
        if genus is None or genus < 1 or genus > cls.MAX_GENUS:
            return None
        d = cls(N=genus)
        d.p = int(p)
        d.data = np.asarray(coeffs, dtype=np.int64)
        d.degree = len(coeffs) - 1
        d.genus = genus
        d.N = genus
        d.calc_features()
        return d

    @classmethod
    def _load_seed_pickle(cls, path, max_rows):
        loaded = pickle.load(open(path, "rb"))
        rows = loaded[:max_rows] if max_rows and max_rows > 0 else loaded
        out = []
        for item in rows:
            coeffs = getattr(item, "data", None)
            p = int(getattr(item, "p", cls.PRIME))
            if coeffs is None or p != cls.PRIME:
                continue
            d = cls._from_coefficients(np.asarray(coeffs, dtype=np.int64).tolist(), p)
            if d is not None:
                out.append(d)
        return out

    @classmethod
    def _load_seed_sqlite(cls, path, max_rows):
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table','view')")}
        if "canonical_classes" in tables:
            table = "canonical_classes"
            coeff_col = "representative_coefficients"
            where = f"prime = {int(cls.PRIME)} and sparsity = 0"
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
            coeffs = ast.literal_eval(row["coefficients"])
            d = cls._from_coefficients(coeffs, p)
            if d is not None:
                out.append(d)
        return out

    @classmethod
    def load_initial_data(cls, path, N, max_rows=0, pars=None):
        if pars is not None:
            cls._update_class_params(pars)
        if path.endswith(".pkl"):
            candidates = cls._load_seed_pickle(path, max_rows)
        else:
            candidates = cls._load_seed_sqlite(path, max_rows)
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


class Hyperelliptic2Environment(BaseEnvironment):
    data_class = Hyperelliptic2DataPoint

    def __init__(self, params):
        super().__init__(params)
        if params.p not in SUPPORTED_NORMAL_BASIS_PRIMES:
            raise ValueError("hyperelliptic2 has precomputed normal-basis data only for --p in {3, 5, 7, 11}")
        if params.N < 1:
            raise ValueError("--N must be positive")

        # Worst case is one irreducible factor of degree 2N+2, represented by
        # (2N+2)^2 vector coordinates, plus BOS/SEP/D/EOS.
        max_factor_degree = 2 * params.N + 2
        if max_factor_degree > MAX_PRECOMPUTED_EXTENSION_DEGREE:
            raise ValueError(
                f"--N is too large for the precomputed normal-basis table: "
                f"2*N+2 must be <= {MAX_PRECOMPUTED_EXTENSION_DEGREE}"
            )
        _ensure_normal_basis_coverage(params.p, max_factor_degree)

        needed_len = (2 * params.N + 2) ** 2 + 4
        if params.max_len < needed_len:
            raise ValueError(f"--max_len must be at least {needed_len} for max genus {params.N}")

        self.data_class.PRIME = params.p
        self.data_class.MAX_GENUS = params.N
        self.data_class.LOCAL_SEARCH_STEPS = params.local_search_steps
        self.data_class.LOCAL_SEARCH_BATCH_SIZE = params.local_search_batch_size
        self.data_class.LOCAL_SEARCH_HIGH_GENUS_BIAS = _coerce_bool(params.local_search_high_genus_bias)
        self.data_class.LOCAL_SEARCH_GROW_PROB = params.local_search_grow_prob
        self.data_class.LOCAL_SEARCH_REMOVE_PROB = params.local_search_remove_prob
        self.data_class.SCORE_BATCH_SIZE = params.score_batch_size
        self.tokenizer = Hyperelliptic2Tokenizer(
            dataclass=self.data_class,
            max_genus=params.N,
            p=params.p,
            extra_symbols=self.SPECIAL_SYMBOLS,
        )

    @staticmethod
    def register_args(parser):
        parser.add_argument("--N", type=int, default=8, help="Maximum genus")
        parser.add_argument("--p", type=int, default=3, help="Prime field characteristic")
        parser.add_argument("--local_search_steps", type=int, default=0, help="Necklace-level local-search rounds")
        parser.add_argument("--local_search_batch_size", type=int, default=32, help="Necklace mutations tested per round")
        parser.add_argument("--local_search_high_genus_bias", type=bool_flag, default=True, help="Bias necklace mutations toward larger genus")
        parser.add_argument("--local_search_grow_prob", type=float, default=0.72, help="Probability of a genus-increasing local-search mutation")
        parser.add_argument("--local_search_remove_prob", type=float, default=0.02, help="Probability of a degree-decreasing local-search mutation")
        parser.add_argument("--score_batch_size", type=int, default=32, help="Sage scorer batch size")
        parser.add_argument("--initial_data_sqlite", type=str, default="", help="Comma-separated SQLite or pickle seed files")
        parser.add_argument("--initial_data_max_rows", type=int, default=0, help="Maximum seed rows loaded from each file; 0 means all")
        parser.add_argument("--make_object_canonical", type=bool_flag, default="false", help="Reserved for compatibility")
        parser.add_argument("--augment_data_representation", type=bool_flag, default="false", help="Reserved for compatibility")
