import ast
import json
import os
import pickle
import random
import shlex
import sqlite3
import subprocess

import numpy as np

from src.envs.environment import BaseEnvironment, DataPoint
from src.envs.hyperelliptic2 import (
    SAGE_DOT_DIR,
    SAGE_PYTHON,
    SUPPORTED_NORMAL_BASIS_PRIMES,
    _coerce_bool,
    _genus_from_degree,
    _sage_python_command,
)
from src.envs.hyperelliptic2_mod2 import is_mod2_allowed_factor_degrees
from src.envs.tokenizers import Tokenizer
from src.utils import bool_flag


_SAGE_FACTOR_WORKER = None


def _sage_factor_worker():
    global _SAGE_FACTOR_WORKER
    if _SAGE_FACTOR_WORKER is not None and _SAGE_FACTOR_WORKER.poll() is None:
        return _SAGE_FACTOR_WORKER
    os.makedirs(SAGE_DOT_DIR, exist_ok=True)
    worker_path = os.path.abspath("tools/sage_hyperelliptic_factorized_worker.py")
    env = os.environ.copy()
    env["DOT_SAGE"] = SAGE_DOT_DIR
    _SAGE_FACTOR_WORKER = subprocess.Popen(
        _sage_python_command(SAGE_PYTHON) + [worker_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=env,
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
    factors = response["factors"]
    if factors is None:
        return None, None
    leading_coefficient = int(response["leading_coefficient"]) % int(p)
    return leading_coefficient, tuple(tuple(int(c) % p for c in factor) for factor in factors)


def _polynomial_to_factors(poly, p):
    _, factors = _polynomial_to_factorization(poly, p)
    return factors


def _factors_to_polynomial(factors, p, leading_coefficient=1):
    response = _sage_factor_request(
        {
            "op": "factors_to_polynomial",
            "p": int(p),
            "leading_coefficient": int(leading_coefficient) % int(p),
            "factors": [[int(c) for c in factor] for factor in factors],
        }
    )
    return [int(c) % p for c in response["coefficients"]]


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
        "coefficients": [int(c) % p for c in response["coefficients"]],
        "factors": tuple(tuple(int(c) % p for c in factor) for factor in response["factors"]),
        "factor_degrees": [int(degree) for degree in response["factor_degrees"]],
    }


def _random_irreducible_factor(degree, p):
    response = _sage_factor_request({"op": "random_irreducible_factor", "p": int(p), "degree": int(degree)})
    return tuple(int(c) % p for c in response["factor"])


def _random_squarefree_monic_polynomial(degree, p):
    response = _sage_factor_request(
        {"op": "random_squarefree_monic_polynomial", "p": int(p), "degree": int(degree)}
    )
    return np.asarray(response["coefficients"], dtype=np.int64)


def _total_factor_degree(factors):
    return sum(len(factor) - 1 for factor in factors)


class FactorizedHyperellipticTokenizer(Tokenizer):
    """Tokenize y^2=f(x) by the multiset of monic irreducible factors of f."""

    def __init__(self, dataclass, max_genus, p, extra_symbols):
        self.dataclass = dataclass
        self.max_genus = int(max_genus)
        self.p = int(p)
        self.max_factor_degree = 2 * self.max_genus + 2
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
        for idx, factor in enumerate(tuple(sorted(factors))):
            degree = len(factor) - 1
            tokens.append(self.stoi[f"D{degree}"])
            tokens.extend(int(c) % self.p for c in factor[:-1])
            if idx + 1 < len(factors):
                tokens.append(self.stoi["SEP"])
        tokens.append(self.stoi["EOS"])
        return np.array(tokens, dtype=np.int32)

    def cache_key_for_datapoint(self, datapoint, max_len):
        return (self.__class__.__name__, getattr(datapoint, "features", None))

    def _completed_factor_degree(self, token_ids):
        pos = 0
        if pos >= len(token_ids) or self.itos.get(int(token_ids[pos])) != "BOS":
            return 0
        pos += 1
        if pos >= len(token_ids) or self.itos.get(int(token_ids[pos])) != "LC":
            return 0
        pos += 1
        if pos >= len(token_ids) or not isinstance(self.itos.get(int(token_ids[pos])), int):
            return 0
        pos += 1
        if pos >= len(token_ids) or self.itos.get(int(token_ids[pos])) != "SEP":
            return 0
        pos += 1
        total_degree = 0
        while pos < len(token_ids):
            token = self.itos.get(int(token_ids[pos]))
            if token in ("EOS", "PAD"):
                break
            if not isinstance(token, str) or not token.startswith("D"):
                break
            degree = int(token[1:])
            pos += 1
            if pos + degree > len(token_ids):
                break
            for token_id in token_ids[pos : pos + degree]:
                decoded = self.itos.get(int(token_id))
                if decoded in self.extra_symbols or not isinstance(decoded, int):
                    return total_degree
            pos += degree
            total_degree += degree
            if pos >= len(token_ids):
                break
            separator = self.itos.get(int(token_ids[pos]))
            if separator == "SEP":
                pos += 1
                continue
            if separator == "EOS":
                break
            break
        return total_degree

    def make_eos_min_genus_processor(self, min_genus):
        min_genus = int(min_genus)
        threshold_degree = 2 * min_genus + 1
        eos_token_id = self.stoi["EOS"]

        def processor(logits, idx):
            idx_cpu = idx.detach().cpu().numpy()
            blocked = []
            for row in range(idx_cpu.shape[0]):
                completed_degree = self._completed_factor_degree(idx_cpu[row].tolist())
                if completed_degree < threshold_degree:
                    blocked.append(row)
            if blocked:
                logits[blocked, eos_token_id] = -float("inf")
            return logits

        return processor

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
            genus = _genus_from_degree(degree)
            if genus is None or genus < 1 or genus > self.max_genus:
                return None
            normalized = _normalize_factor_blocks(blocks, self.p, leading_coefficient)
            if not is_mod2_allowed_factor_degrees(
                normalized["factor_degrees"],
                genus,
                degree == 2 * genus + 1,
            ):
                return None
            coeffs = normalized["coefficients"]
            factors = normalized["factors"]

            datapoint = self.dataclass(N=genus)
            datapoint.data = np.asarray(coeffs, dtype=np.int64)
            datapoint.factors = tuple(factors)
            datapoint.leading_coefficient = leading_coefficient
            datapoint.p = self.p
            datapoint.degree = degree
            datapoint.calc_features()
            clean_tokens = self.encode(datapoint)
            datapoint._encoded_token_cache_key = self.cache_key_for_datapoint(datapoint, None)
            datapoint._encoded_token_cache_value = clean_tokens
            return datapoint
        except Exception:
            return None


class FactorizedHyperellipticDataPoint(DataPoint):
    PRIME = 3
    MAX_GENUS = 8
    LOCAL_SEARCH_STEPS = 0
    LOCAL_SEARCH_BATCH_SIZE = 32
    LOCAL_SEARCH_HIGH_GENUS_BIAS = True
    LOCAL_SEARCH_GROW_PROB = 0.72
    LOCAL_SEARCH_REMOVE_PROB = 0.02
    SCORE_BATCH_SIZE = 32

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
        if init:
            self.genus = random.randint(1, self.MAX_GENUS)
            self.N = self.genus
            self.degree = 2 * self.genus + random.randint(1, 2)
            self.data = _random_squarefree_monic_polynomial(self.degree, self.p)
            leading_coefficient = random.randrange(1, self.p)
            if leading_coefficient != 1:
                self.data = (self.data * leading_coefficient) % self.p
            self.leading_coefficient, self.factors = _polynomial_to_factorization(self.data.tolist(), self.p)
            self.calc_features()

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
        if genus is None or genus < 1 or genus > self.MAX_GENUS:
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

    @classmethod
    def _score_arrays(cls, data, primes):
        from src.envs.hyperelliptic2 import Hyperelliptic2DataPoint

        old_prime = Hyperelliptic2DataPoint.PRIME
        old_max_genus = Hyperelliptic2DataPoint.MAX_GENUS
        old_score_batch_size = Hyperelliptic2DataPoint.SCORE_BATCH_SIZE
        Hyperelliptic2DataPoint.PRIME = cls.PRIME
        Hyperelliptic2DataPoint.MAX_GENUS = cls.MAX_GENUS
        Hyperelliptic2DataPoint.SCORE_BATCH_SIZE = cls.SCORE_BATCH_SIZE
        try:
            return Hyperelliptic2DataPoint._score_arrays(data, primes)
        finally:
            Hyperelliptic2DataPoint.PRIME = old_prime
            Hyperelliptic2DataPoint.MAX_GENUS = old_max_genus
            Hyperelliptic2DataPoint.SCORE_BATCH_SIZE = old_score_batch_size

    @classmethod
    def _from_factors(cls, factors, p, leading_coefficient=1):
        leading_coefficient = int(leading_coefficient) % int(p)
        if leading_coefficient == 0:
            return None
        total_degree = _total_factor_degree(factors)
        if total_degree < 3 or total_degree > 2 * cls.MAX_GENUS + 2:
            return None
        blocks = tuple(sorted(tuple(int(c) % int(p) for c in factor) for factor in factors))
        genus = _genus_from_degree(total_degree)
        if genus is None or genus < 1 or genus > cls.MAX_GENUS:
            return None
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
        coeffs = normalized["coefficients"]
        d = cls(N=genus)
        d.p = int(p)
        d.data = np.asarray(coeffs, dtype=np.int64)
        d.leading_coefficient = leading_coefficient
        d.factors = normalized["factors"]
        d.degree = len(coeffs) - 1
        d.calc_features()
        return d

    def _random_growth_degree(self, lower, upper):
        if upper < lower:
            return None
        span = upper - lower + 1
        return lower + min(span - 1, int((random.random() ** 0.5) * span))

    def _add_higher_degree_factor(self, mutated, total_degree):
        remaining = 2 * self.MAX_GENUS + 2 - total_degree
        if remaining <= 0:
            return False
        degree = self._random_growth_degree(2 if remaining >= 2 else 1, remaining)
        if degree is None:
            return False
        mutated.append(_random_irreducible_factor(degree, self.p))
        return True

    def _replace_by_higher_degree_factor(self, mutated, total_degree):
        max_degree = 2 * self.MAX_GENUS + 2
        expandable = [
            idx for idx, factor in enumerate(mutated)
            if len(factor) - 1 < max_degree - (total_degree - (len(factor) - 1))
        ]
        if not expandable:
            return False
        idx = random.choice(expandable)
        old_degree = len(mutated[idx]) - 1
        upper = max_degree - (total_degree - old_degree)
        new_degree = self._random_growth_degree(old_degree + 1, upper)
        if new_degree is None:
            return False
        mutated[idx] = _random_irreducible_factor(new_degree, self.p)
        return True

    def _mutate_factors(self, factors):
        if not factors:
            return None
        move = random.random()
        total_degree = _total_factor_degree(factors)
        mutated = list(factors)

        if self.LOCAL_SEARCH_HIGH_GENUS_BIAS and move < self.LOCAL_SEARCH_GROW_PROB:
            if random.random() < 0.65 and self._add_higher_degree_factor(mutated, total_degree):
                return tuple(sorted(mutated))
            if self._replace_by_higher_degree_factor(mutated, total_degree):
                return tuple(sorted(mutated))
            if self._add_higher_degree_factor(mutated, total_degree):
                return tuple(sorted(mutated))
            return None

        residual = random.random() if self.LOCAL_SEARCH_HIGH_GENUS_BIAS else move
        remove_threshold = self.LOCAL_SEARCH_REMOVE_PROB if self.LOCAL_SEARCH_HIGH_GENUS_BIAS else 0.10
        if residual < 0.45:
            idx = random.randrange(len(mutated))
            mutated[idx] = _random_irreducible_factor(len(mutated[idx]) - 1, self.p)
        elif residual < 0.70:
            splittable = [idx for idx, factor in enumerate(mutated) if len(factor) - 1 >= 2]
            if not splittable:
                return None
            idx = random.choice(splittable)
            degree = len(mutated.pop(idx)) - 1
            left = random.randint(1, degree - 1)
            mutated.extend([
                _random_irreducible_factor(left, self.p),
                _random_irreducible_factor(degree - left, self.p),
            ])
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
        if self.factors is None:
            self.factors = _polynomial_to_factors(self.data.tolist(), self.p)
        best_factors = list(self.factors or [])
        best = self
        rounds = self.LOCAL_SEARCH_STEPS if improve_with_local_search else max(1, self.LOCAL_SEARCH_STEPS // 4)
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
            next_best = max(usable, key=lambda d: (d.score, d.genus, d.degree))
            if best.score >= 0 and (next_best.score, next_best.degree) <= (best.score, best.degree):
                continue
            best = next_best
            best_factors = list(best.factors or [])
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
    def _from_coefficients(cls, coeffs, p):
        coeffs = [int(c) % int(p) for c in coeffs]
        genus = _genus_from_degree(len(coeffs) - 1)
        if genus is None or genus < 1 or genus > cls.MAX_GENUS:
            return None
        leading_coefficient, factors = _polynomial_to_factorization(coeffs, p)
        if factors is None:
            return None
        d = cls(N=genus)
        d.p = int(p)
        d.data = np.asarray(coeffs, dtype=np.int64)
        d.leading_coefficient = leading_coefficient
        d.factors = tuple(factors)
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
            factors = getattr(item, "factors", None)
            if factors is not None:
                genus = _genus_from_degree(len(coeffs) - 1)
                d = cls(N=genus if genus is not None else 1)
                d.p = int(p)
                d.data = np.asarray(coeffs, dtype=np.int64)
                d.leading_coefficient = int(getattr(item, "leading_coefficient", int(d.data[-1]) % p)) % p
                d.factors = tuple(tuple(int(c) % p for c in factor) for factor in factors)
                d.degree = len(d.data) - 1
                d.genus = genus
                d.N = genus
                d.calc_features()
            else:
                d = cls._from_coefficients(np.asarray(coeffs, dtype=np.int64).tolist(), p)
            if d is not None:
                d.score = float(getattr(item, "score", -1.0))
                d.lpoly = getattr(item, "lpoly", None)
                d.middle = getattr(item, "middle", None)
                d.target_coeffs = getattr(item, "target_coeffs", None)
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
            d = cls._from_coefficients(ast.literal_eval(row["coefficients"]), p)
            if d is not None:
                out.append(d)
        return out

    @classmethod
    def load_initial_data(cls, path, N, max_rows=0, pars=None):
        if pars is not None:
            cls._update_class_params(pars)
        candidates = cls._load_seed_pickle(path, max_rows) if path.endswith(".pkl") else cls._load_seed_sqlite(path, max_rows)
        if not candidates:
            return []
        if path.endswith(".pkl") and all(getattr(d, "score", -1.0) >= 0 for d in candidates):
            for d in candidates:
                d.calc_features()
            return candidates
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
        for d in data:
            if always_search or (d.score < 0 and redeem_only):
                d.local_search(improve_with_local_search=always_search)
        return data, n_invalid


class FactorizedHyperellipticEnvironment(BaseEnvironment):
    data_class = FactorizedHyperellipticDataPoint

    def __init__(self, params):
        super().__init__(params)
        if params.p not in SUPPORTED_NORMAL_BASIS_PRIMES:
            raise ValueError("factorized hyperelliptic supports --p in {3, 5, 7, 11}")
        if params.N < 1:
            raise ValueError("--N must be positive")
        max_factor_degree = 2 * params.N + 2
        # Worst case is a squarefree product of max_factor_degree linear
        # factors: BOS LC coeff SEP, then each D1+coefficient, separators
        # between factors, and EOS.
        needed_len = 3 * max_factor_degree + 4
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
        self.tokenizer = FactorizedHyperellipticTokenizer(
            dataclass=self.data_class,
            max_genus=params.N,
            p=params.p,
            extra_symbols=self.SPECIAL_SYMBOLS,
        )

    @staticmethod
    def register_args(parser):
        parser.add_argument("--N", type=int, default=8, help="Maximum genus")
        parser.add_argument("--p", type=int, default=3, help="Prime field characteristic")
        parser.add_argument("--local_search_steps", type=int, default=0, help="Factor-level local-search rounds")
        parser.add_argument("--local_search_batch_size", type=int, default=32, help="Factor mutations tested per round")
        parser.add_argument("--local_search_high_genus_bias", type=bool_flag, default=True, help="Bias factor mutations toward larger genus")
        parser.add_argument("--local_search_grow_prob", type=float, default=0.72, help="Probability of a genus-increasing local-search mutation")
        parser.add_argument("--local_search_remove_prob", type=float, default=0.02, help="Probability of a degree-decreasing local-search mutation")
        parser.add_argument("--score_batch_size", type=int, default=32, help="Sage scorer batch size")
        parser.add_argument("--initial_data_sqlite", type=str, default="", help="Comma-separated SQLite or pickle seed files")
        parser.add_argument("--initial_data_max_rows", type=int, default=0, help="Maximum seed rows loaded from each file; 0 means all")
        parser.add_argument("--make_object_canonical", type=bool_flag, default="false", help="Reserved for compatibility")
        parser.add_argument("--augment_data_representation", type=bool_flag, default="false", help="Reserved for compatibility")
