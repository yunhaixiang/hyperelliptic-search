import math
import json
import os
import random
import sqlite3
import subprocess

import numpy as np

from src.envs.environment import BaseEnvironment, DataPoint
from src.envs.tokenizers import Tokenizer
from src.utils import bool_flag


def is_prime(n):
    if n < 2:
        return False
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    for q in small_primes:
        if n == q:
            return True
        if n % q == 0:
            return False

    d = n - 1
    s = 0
    while d % 2 == 0:
        s += 1
        d //= 2

    # Deterministic Miller-Rabin bases for unsigned 64-bit integers.
    for a in [2, 325, 9375, 28178, 450775, 9780504, 1795265022]:
        if a >= n:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
            x = (x * x) % n
            if x == n - 1:
                break
        else:
            return False
    return True


def poly_trim(poly):
    poly = [int(c) for c in poly]
    while len(poly) > 1 and poly[-1] == 0:
        poly.pop()
    return poly


def poly_mod(a, b, p):
    a = poly_trim([c % p for c in a])
    b = poly_trim([c % p for c in b])
    if len(b) == 1 and b[0] == 0:
        raise ZeroDivisionError("polynomial division by zero")
    inv_lead = pow(b[-1], p - 2, p)
    while len(a) >= len(b) and not (len(a) == 1 and a[0] == 0):
        factor = a[-1] * inv_lead % p
        shift = len(a) - len(b)
        for i, coeff in enumerate(b):
            a[shift + i] = (a[shift + i] - factor * coeff) % p
        a = poly_trim(a)
    return a


def poly_gcd(a, b, p):
    a = poly_trim([c % p for c in a])
    b = poly_trim([c % p for c in b])
    while not (len(b) == 1 and b[0] == 0):
        a, b = b, poly_mod(a, b, p)
    if len(a) == 1 and a[0] == 0:
        return a
    inv_lead = pow(a[-1], p - 2, p)
    return [(c * inv_lead) % p for c in a]


class HyperellipticCoefficientDigitTokenizer(Tokenizer):
    def __init__(self, dataclass, genus, p, base, digits, extra_symbols):
        self.dataclass = dataclass
        self.genus = genus
        self.p = p
        self.base = base
        self.digits = digits
        self.prime_digits = digits
        self.extra_symbols = extra_symbols
        self.stoi = {}
        self.itos = {}

        for token in range(base):
            self.stoi[token] = token
            self.itos[token] = token

        offset = len(self.stoi)
        for idx, symbol in enumerate(extra_symbols):
            self.stoi[symbol] = offset + idx
            self.itos[offset + idx] = symbol

        offset = len(self.stoi)
        self.stoi["P"] = offset
        self.itos[offset] = "P"
        offset += 1
        for idx in range(dataclass.NUM_COEFFICIENTS):
            symbol = f"C{idx}"
            self.stoi[symbol] = offset + idx
            self.itos[offset + idx] = symbol

    def _encode_int(self, value, digits):
        value = int(value)
        tokens = []
        for _ in range(digits):
            tokens.append(self.stoi[value % self.base])
            value //= self.base
        return tokens

    def _decode_int(self, tokens, bound=None):
        value = 0
        place = 1
        for token in tokens:
            digit = self.itos[int(token)]
            if digit in self.extra_symbols or not isinstance(digit, int):
                return None
            value += digit * place
            place *= self.base
        if bound is not None and value >= bound:
            return None
        return value

    def _encode_coeff(self, coeff):
        return self._encode_int(coeff, self.digits)

    def _decode_coeff(self, tokens, p):
        return self._decode_int(tokens, bound=p)

    def encode(self, datapoint_to_encode):
        tokens = [self.stoi["BOS"]]
        tokens.append(self.stoi["P"])
        tokens.extend(self._encode_int(datapoint_to_encode.p, self.prime_digits))
        tokens.append(self.stoi["SEP"])
        for idx, coeff in enumerate(datapoint_to_encode.data):
            tokens.append(self.stoi[f"C{idx}"])
            tokens.extend(self._encode_coeff(coeff))
            if idx + 1 < len(datapoint_to_encode.data):
                tokens.append(self.stoi["SEP"])
        tokens.append(self.stoi["EOS"])
        return np.array(tokens, dtype=np.int32)

    def decode(self, token_seq_to_decode):
        try:
            seq = [int(t) for t in token_seq_to_decode]
            if not seq or self.itos.get(seq[0]) != "BOS":
                return None
            pos = 1
            if pos >= len(seq) or self.itos.get(seq[pos]) != "P":
                return None
            pos += 1
            if pos + self.prime_digits > len(seq):
                return None
            p = self._decode_int(seq[pos : pos + self.prime_digits])
            if p is None or p < 3:
                return None
            pos += self.prime_digits
            if pos >= len(seq) or self.itos.get(seq[pos]) != "SEP":
                return None
            pos += 1
            coeffs = []
            for idx in range(self.dataclass.NUM_COEFFICIENTS):
                if pos >= len(seq) or self.itos.get(seq[pos]) != f"C{idx}":
                    return None
                pos += 1
                if pos + self.digits > len(seq):
                    return None
                coeff = self._decode_coeff(seq[pos : pos + self.digits], p)
                if coeff is None:
                    return None
                coeffs.append(coeff)
                pos += self.digits
                if idx + 1 < self.dataclass.NUM_COEFFICIENTS:
                    if pos >= len(seq) or self.itos.get(seq[pos]) != "SEP":
                        return None
                    pos += 1
            if pos >= len(seq) or self.itos.get(seq[pos]) != "EOS":
                return None

            datapoint = self.dataclass(N=self.genus)
            datapoint.p = p
            datapoint.data = np.array(coeffs, dtype=np.int64)
            return datapoint
        except Exception:
            return None


class HyperellipticDataPoint(DataPoint):
    PRIME = 3
    TARGET_PRIME = 3
    DEGREE_MODEL = "odd"
    MONIC = True
    DEPRESSED = True
    NUM_COEFFICIENTS = 4
    LOCAL_SEARCH_STEPS = 0
    LOCAL_SEARCH_BATCH_SIZE = 32
    MUTATION_RADIUS = 16
    LOCAL_SEARCH_RANDOM_REPLACEMENT = 0.2
    SCORE_BATCH_SIZE = 32
    NON_TARGET_RANK_CAP = 0.5
    SAGE_PYTHON = "/Applications/SageMath-10-6.app/Contents/MacOS/Python"
    SAGE_DOT_DIR = "/private/tmp/sage-dot-cache"
    _SAGE_WORKER = None

    def __init__(self, N, init=False):
        super().__init__()
        if N != 2:
            raise ValueError("The implemented hyperelliptic scorer currently supports only genus N=2.")
        if self.DEGREE_MODEL != "odd" or not self.MONIC:
            raise ValueError("The implemented hyperelliptic scorer currently supports monic odd-degree genus-2 models.")

        self.N = N
        self.genus = N
        self.p = self.PRIME
        self.degree = 5
        self.num_coefficients = self.NUM_COEFFICIENTS
        self.c1 = None
        self.c2 = None
        self.lpoly = None
        self.data = np.zeros(self.num_coefficients, dtype=np.int64)

        if init:
            self.data = np.random.randint(0, self.p, size=self.num_coefficients, dtype=np.int64)
            self.calc_features()
            self.calc_score()

    @classmethod
    def _configure_num_coefficients(cls):
        cls.NUM_COEFFICIENTS = 4 if cls.DEPRESSED else 5

    def _full_coefficients_low_to_high(self, data=None):
        data = self.data if data is None else data
        data = [int(c) % self.p for c in data]
        if self.DEPRESSED:
            a0, a1, a2, a3 = data
            return [a0, a1, a2, a3, 0, 1]
        a0, a1, a2, a3, a4 = data
        return [a0, a1, a2, a3, a4, 1]

    def is_squarefree(self, data=None):
        coeffs = self._full_coefficients_low_to_high(data)
        derivative = [(i * coeffs[i]) % self.p for i in range(1, len(coeffs))]
        return len(poly_gcd(coeffs, derivative, self.p)) == 1

    @classmethod
    def _score_from_c1(cls, c1):
        c1 = np.asarray(c1, dtype=np.int64)
        scores = 1.0 - np.abs(c1) / (4.0 * math.sqrt(cls.PRIME))
        return np.where(c1 == 0, 10.0, scores)

    @classmethod
    def _score_arrays(cls, data):
        return cls._score_arrays_sage(data)

    @classmethod
    def _sage_worker(cls):
        if cls._SAGE_WORKER is not None and cls._SAGE_WORKER.poll() is None:
            return cls._SAGE_WORKER
        if not os.path.exists(cls.SAGE_PYTHON):
            raise RuntimeError(f"Sage Python not found: {cls.SAGE_PYTHON}")
        os.makedirs(cls.SAGE_DOT_DIR, exist_ok=True)
        worker_path = os.path.abspath("tools/sage_score_worker.py")
        env = os.environ.copy()
        env["DOT_SAGE"] = cls.SAGE_DOT_DIR
        cls._SAGE_WORKER = subprocess.Popen(
            [cls.SAGE_PYTHON, worker_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
        )
        return cls._SAGE_WORKER

    @classmethod
    def _score_arrays_sage(cls, data):
        data = np.asarray(data, dtype=np.int64)
        primes = np.full(data.shape[0], cls.PRIME, dtype=np.int64)
        return cls._score_arrays_sage_with_primes(data, primes)

    @classmethod
    def _score_arrays_sage_with_primes(cls, data, primes):
        data = np.asarray(data, dtype=np.int64)
        primes = np.asarray(primes, dtype=np.int64)
        scores = np.full(data.shape[0], -1.0, dtype=np.float64)
        c1s = np.full(data.shape[0], 0, dtype=np.int64)
        c2s = np.full(data.shape[0], 0, dtype=np.int64)
        valid = np.zeros(data.shape[0], dtype=bool)
        lpolys = [None] * data.shape[0]
        if data.size == 0:
            return scores, c1s, c2s, valid, lpolys

        worker = cls._sage_worker()
        assert worker.stdin is not None and worker.stdout is not None
        for p in np.unique(primes):
            group_indices = np.flatnonzero(primes == p)
            for offset_start in range(0, len(group_indices), cls.SCORE_BATCH_SIZE):
                indices = group_indices[offset_start : offset_start + cls.SCORE_BATCH_SIZE]
                chunk = data[indices]
                request = {
                    "p": int(p),
                    "depressed": cls.DEPRESSED,
                    "mode": "c1",
                    "data": chunk.astype(int).tolist(),
                }
                worker.stdin.write(json.dumps(request) + "\n")
                worker.stdin.flush()
                line = worker.stdout.readline()
                if not line:
                    stderr = worker.stderr.read() if worker.stderr is not None else ""
                    raise RuntimeError(f"Sage scorer exited unexpectedly. stderr:\n{stderr}")
                response = json.loads(line)
                if "error" in response:
                    raise RuntimeError(f"Sage scorer error: {response['error']}")
                for offset, score in enumerate(response["scores"]):
                    idx = indices[offset]
                    scores[idx] = float(score)
                    valid[idx] = bool(response["valid"][offset])
                    if valid[idx]:
                        c1s[idx] = int(response["c1s"][offset])
                        if response["c2s"][offset] is not None:
                            c2s[idx] = int(response["c2s"][offset])
                        if response["lpolys"][offset] is not None:
                            lpolys[idx] = [int(v) for v in response["lpolys"][offset]]
        return scores, c1s, c2s, valid, lpolys

    def calc_score(self):
        scores, c1s, c2s, valid, lpolys = self._score_arrays_sage_with_primes(
            np.array([self.data], dtype=np.int64),
            np.array([self.p], dtype=np.int64),
        )
        self._apply_score(scores[0], c1s[0], valid[0], c2s[0], lpolys[0])

    def _apply_score(self, score, c1, valid, c2=None, lpoly=None):
        self.score = float(score)
        self.c1 = int(c1) if valid else None
        self.c2 = int(c2) if valid and c2 is not None else None
        self.lpoly = [int(v) for v in lpoly] if valid and lpoly is not None else None

    def selection_score(self):
        if self.score < 0:
            return self.score
        if self.p == self.TARGET_PRIME:
            return self.score
        return min(self.score, self.NON_TARGET_RANK_CAP)

    def calc_features(self):
        self.features = ",".join([str(int(self.p))] + [str(int(c)) for c in self.data.tolist()])

    def _random_neighbor(self, center):
        candidate = center.copy()
        idx = np.random.randint(len(candidate))
        if random.random() < self.LOCAL_SEARCH_RANDOM_REPLACEMENT:
            candidate[idx] = np.random.randint(0, self.p)
            return candidate

        delta = 0
        while delta == 0:
            delta = np.random.randint(-self.MUTATION_RADIUS, self.MUTATION_RADIUS + 1)
        candidate[idx] = (int(candidate[idx]) + delta) % self.p
        return candidate

    def local_search(self, improve_with_local_search):
        self.p = self.TARGET_PRIME
        self.data %= self.p
        if self.LOCAL_SEARCH_STEPS <= 0:
            self.calc_score()
            return

        best_data = self.data.copy()
        if self.score < 0:
            best_c1 = None
            best_abs = float("inf")
        else:
            best_c1 = self.c1
            best_abs = abs(best_c1)
            if best_abs == 0:
                return

        rounds = self.LOCAL_SEARCH_STEPS if improve_with_local_search else max(1, self.LOCAL_SEARCH_STEPS // 4)
        for _ in range(rounds):
            candidates = np.array([self._random_neighbor(best_data) for _ in range(self.LOCAL_SEARCH_BATCH_SIZE)], dtype=np.int64)
            scores, c1s, c2s, valid, lpolys = self._score_arrays_sage_with_primes(
                candidates,
                np.full(candidates.shape[0], self.p, dtype=np.int64),
            )
            usable = np.flatnonzero(valid)
            accepted = False
            if len(usable) > 0:
                c1_abs = np.abs(c1s[usable])
                best_idx = usable[int(np.argmin(c1_abs))]
                if c1_abs.min() < best_abs:
                    best_data = candidates[best_idx]
                    best_c1 = int(c1s[best_idx])
                    best_abs = int(c1_abs.min())
                    accepted = True
            if best_abs == 0:
                break
            if not accepted and best_c1 is None:
                # Keep moving if the initial model was singular and this batch did not repair it.
                best_data = self._random_neighbor(best_data)

        self.data = best_data
        self.calc_features()
        scores, c1s, c2s, valid, lpolys = self._score_arrays_sage_with_primes(
            np.array([self.data], dtype=np.int64),
            np.array([self.p], dtype=np.int64),
        )
        self._apply_score(scores[0], c1s[0], valid[0], c2s[0], lpolys[0])

    @classmethod
    def _make_datapoint(cls, N, row, score, c1, valid, c2=None, lpoly=None, p=None):
        d = cls(N=N)
        d.p = int(cls.PRIME if p is None else p)
        d.data = np.asarray(row, dtype=np.int64)
        d.calc_features()
        d._apply_score(score, c1, valid, c2, lpoly)
        return d

    @classmethod
    def _batch_generate_and_score(cls, batch_size, N, pars=None):
        if pars is not None:
            cls._update_class_params(pars)
        data = np.random.randint(0, cls.PRIME, size=(batch_size, cls.NUM_COEFFICIENTS), dtype=np.int64)
        primes = np.full(batch_size, cls.PRIME, dtype=np.int64)
        scores, c1s, c2s, valid, lpolys = cls._score_arrays_sage_with_primes(data, primes)
        out = []
        for idx in np.flatnonzero(scores >= 0):
            out.append(cls._make_datapoint(N, data[idx], scores[idx], c1s[idx], valid[idx], c2s[idx], lpolys[idx], p=primes[idx]))
        return out

    @classmethod
    def _batch_score_datapoints(cls, data, always_search=False, redeem_only=False, pars=None):
        if pars is not None:
            cls._update_class_params(pars)
        if len(data) == 0:
            return [], 0

        for d in data:
            if d.p != cls.TARGET_PRIME:
                d.p = cls.TARGET_PRIME
                d.data %= d.p

        arrays = np.array([d.data for d in data], dtype=np.int64)
        primes = np.array([d.p for d in data], dtype=np.int64)
        scores, c1s, c2s, valid, lpolys = cls._score_arrays_sage_with_primes(arrays, primes)
        n_invalid = int((scores < 0).sum())

        for idx, d in enumerate(data):
            d.calc_features()
            d._apply_score(scores[idx], c1s[idx], valid[idx], c2s[idx], lpolys[idx])

        for d in data:
            if always_search or (d.score < 0 and redeem_only):
                d.local_search(improve_with_local_search=always_search)

        return data, n_invalid

    @classmethod
    def load_initial_data(cls, path, N, max_rows=0, pars=None):
        if pars is not None:
            cls._update_class_params(pars)
        if not path:
            return []

        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "curves" not in tables:
            raise ValueError(f"{path} does not contain a curves table")

        columns = {row[1] for row in conn.execute("PRAGMA table_info(curves)")}
        required = {"a0", "a1", "a2", "a3"}
        if not required.issubset(columns):
            raise ValueError(f"{path} curves table must contain columns {sorted(required)}")

        metadata = {}
        if "run_metadata" in tables:
            metadata = dict(conn.execute("SELECT key, value FROM run_metadata"))

        default_prime = int(metadata.get("prime", cls.PRIME))
        has_p_column = "p" in columns or "prime" in columns
        p_column = "p" if "p" in columns else ("prime" if "prime" in columns else None)
        limit = "" if max_rows <= 0 else f" LIMIT {int(max_rows)}"
        select_cols = f"{p_column}, a0, a1, a2, a3" if has_p_column else "a0, a1, a2, a3"
        rows = list(conn.execute(f"SELECT {select_cols} FROM curves ORDER BY score DESC{limit}"))
        conn.close()

        data = []
        primes = []
        for row in rows:
            if has_p_column:
                p = int(row[0])
                coeffs = [int(row[1]), int(row[2]), int(row[3]), int(row[4])]
            else:
                p = default_prime
                coeffs = [int(row["a0"]), int(row["a1"]), int(row["a2"]), int(row["a3"])]
            if not is_prime(p) or p == 2:
                continue
            coeffs = [c % p for c in coeffs]
            data.append(coeffs)
            primes.append(p)

        if not data:
            return []

        arrays = np.array(data, dtype=np.int64)
        prime_array = np.array(primes, dtype=np.int64)
        scores, c1s, c2s, valid, lpolys = cls._score_arrays_sage_with_primes(arrays, prime_array)
        out = []
        for idx in np.flatnonzero(scores >= 0):
            out.append(cls._make_datapoint(N, arrays[idx], scores[idx], c1s[idx], valid[idx], c2s[idx], lpolys[idx], p=prime_array[idx]))
        return out

    @classmethod
    def _update_class_params(cls, pars):
        cls.PRIME = pars["prime"]
        cls.TARGET_PRIME = pars["target_prime"]
        cls.DEGREE_MODEL = pars["degree_model"]
        cls.MONIC = pars["monic"]
        cls.DEPRESSED = pars["depressed"]
        cls.LOCAL_SEARCH_STEPS = pars["local_search_steps"]
        cls.LOCAL_SEARCH_BATCH_SIZE = pars["local_search_batch_size"]
        cls.MUTATION_RADIUS = pars["mutation_radius"]
        cls.LOCAL_SEARCH_RANDOM_REPLACEMENT = pars["local_search_random_replacement"]
        cls.SCORE_BATCH_SIZE = pars["score_batch_size"]
        cls.NON_TARGET_RANK_CAP = pars["non_target_rank_cap"]
        cls.SAGE_PYTHON = pars["sage_python"]
        cls.SAGE_DOT_DIR = pars["sage_dot_dir"]
        cls._configure_num_coefficients()

    @classmethod
    def _save_class_params(cls):
        return {
            "prime": cls.PRIME,
            "target_prime": cls.TARGET_PRIME,
            "degree_model": cls.DEGREE_MODEL,
            "monic": cls.MONIC,
            "depressed": cls.DEPRESSED,
            "local_search_steps": cls.LOCAL_SEARCH_STEPS,
            "local_search_batch_size": cls.LOCAL_SEARCH_BATCH_SIZE,
            "mutation_radius": cls.MUTATION_RADIUS,
            "local_search_random_replacement": cls.LOCAL_SEARCH_RANDOM_REPLACEMENT,
            "score_batch_size": cls.SCORE_BATCH_SIZE,
            "non_target_rank_cap": cls.NON_TARGET_RANK_CAP,
            "sage_python": cls.SAGE_PYTHON,
            "sage_dot_dir": cls.SAGE_DOT_DIR,
        }


class HyperellipticEnvironment(BaseEnvironment):
    data_class = HyperellipticDataPoint

    def __init__(self, params):
        super().__init__(params)
        if params.encoding_tokens != "coefficient_digits":
            raise ValueError("hyperelliptic currently supports --encoding_tokens coefficient_digits")
        if params.N != 2:
            raise ValueError("The implemented hyperelliptic environment currently supports only --N 2")
        if not is_prime(params.p) or params.p == 2:
            raise ValueError("--p must be an odd prime")
        if params.degree_model != "odd" or not params.monic:
            raise ValueError("The implemented hyperelliptic environment currently supports --degree_model odd --monic true")
        if params.coefficient_base < 2:
            raise ValueError("--coefficient_base must be at least 2")
        if params.mutation_radius < 1:
            raise ValueError("--mutation_radius must be positive")
        if not 0.0 <= params.local_search_random_replacement <= 1.0:
            raise ValueError("--local_search_random_replacement must be between 0 and 1")

        self.data_class.PRIME = params.p
        self.data_class.TARGET_PRIME = params.p
        self.data_class.DEGREE_MODEL = params.degree_model
        self.data_class.MONIC = params.monic
        self.data_class.DEPRESSED = params.depressed
        self.data_class.LOCAL_SEARCH_STEPS = params.local_search_steps
        self.data_class.LOCAL_SEARCH_BATCH_SIZE = params.local_search_batch_size
        self.data_class.MUTATION_RADIUS = params.mutation_radius
        self.data_class.LOCAL_SEARCH_RANDOM_REPLACEMENT = params.local_search_random_replacement
        self.data_class.SCORE_BATCH_SIZE = params.score_batch_size
        self.data_class.NON_TARGET_RANK_CAP = params.non_target_rank_cap
        self.data_class.SAGE_PYTHON = params.sage_python
        self.data_class.SAGE_DOT_DIR = params.sage_dot_dir
        self.data_class._configure_num_coefficients()

        digits = params.coefficient_digits
        if digits <= 0:
            digits = math.ceil(math.log(params.p, params.coefficient_base))
        if params.coefficient_base**digits <= params.p - 1:
            raise ValueError("--coefficient_digits is too small to encode all elements of F_p")

        self.tokenizer = HyperellipticCoefficientDigitTokenizer(
            self.data_class,
            params.N,
            params.p,
            params.coefficient_base,
            digits,
            self.SPECIAL_SYMBOLS,
        )

    @staticmethod
    def register_args(parser):
        parser.add_argument("--N", type=int, default=2, help="Genus of the hyperelliptic curve; currently only genus 2 is implemented")
        parser.add_argument("--p", type=int, default=101, help="Odd prime defining the base field F_p")
        parser.add_argument("--degree_model", type=str, default="odd", choices=["odd"], help="Polynomial degree model")
        parser.add_argument("--monic", type=bool_flag, default="true", help="Represent only monic polynomial models")
        parser.add_argument("--depressed", type=bool_flag, default="true", help="Use x^5 + a3*x^3 + a2*x^2 + a1*x + a0")
        parser.add_argument("--encoding_tokens", type=str, default="coefficient_digits", help="Use fixed-base coefficient digit tokenization")
        parser.add_argument("--coefficient_base", type=int, default=256, help="Base used to tokenize each F_p coefficient")
        parser.add_argument("--coefficient_digits", type=int, default=0, help="Digits per coefficient; 0 chooses the minimum width for p")
        parser.add_argument("--local_search_steps", type=int, default=0, help="Local-search rounds per sample")
        parser.add_argument("--local_search_batch_size", type=int, default=32, help="Neighbor mutations tested per local-search round")
        parser.add_argument("--mutation_radius", type=int, default=16, help="Small coefficient mutation radius")
        parser.add_argument("--local_search_random_replacement", type=float, default=0.2, help="Probability of replacing one coefficient randomly")
        parser.add_argument("--score_batch_size", type=int, default=32, help="Candidate batch size for Sage c1 scoring")
        parser.add_argument("--non_target_rank_cap", type=float, default=0.5, help="Maximum replay-buffer ranking score for examples whose p is not --p")
        parser.add_argument("--sage_python", type=str, default="/Applications/SageMath-10-6.app/Contents/MacOS/Python", help="Sage Python executable")
        parser.add_argument("--sage_dot_dir", type=str, default="/private/tmp/sage-dot-cache", help="Writable DOT_SAGE directory for Sage")
        parser.add_argument("--initial_data_sqlite", type=str, default="", help="Optional comma-separated SQLite curves tables used to seed initial training data")
        parser.add_argument("--initial_data_max_rows", type=int, default=0, help="Maximum rows loaded from each --initial_data_sqlite file; 0 means all rows")
        parser.add_argument("--make_object_canonical", type=bool_flag, default="false", help="Reserved for future canonicalization support")
        parser.add_argument("--augment_data_representation", type=bool_flag, default="false", help="Reserved for future data augmentation")
