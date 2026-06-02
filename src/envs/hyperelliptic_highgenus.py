import json
import os
import random
import subprocess

import numpy as np
import sympy as sp

from src.envs.environment import BaseEnvironment, DataPoint
from src.envs.hyperelliptic import is_prime, poly_gcd
from src.envs.tokenizers import Tokenizer
from src.utils import bool_flag


class HighGenusCoefficientTokenizer(Tokenizer):
    def __init__(self, dataclass, genus, p, extra_symbols):
        self.dataclass = dataclass
        self.genus = genus
        self.p = p
        self.extra_symbols = extra_symbols
        self.stoi = {}
        self.itos = {}

        for token in range(p):
            self.stoi[token] = token
            self.itos[token] = token

        offset = len(self.stoi)
        for idx, symbol in enumerate(extra_symbols):
            self.stoi[symbol] = offset + idx
            self.itos[offset + idx] = symbol

        self.allowed_token_ids_by_pos = [None] * (2 * genus + 3)
        for pos in range(1, 2 * genus + 1):
            self.allowed_token_ids_by_pos[pos] = list(range(p))
        self.allowed_token_ids_by_pos[2 * genus + 1] = [self.stoi["EOS"]]

    def encode(self, datapoint_to_encode):
        tokens = [self.stoi["BOS"]]
        tokens.extend(int(c) for c in datapoint_to_encode.data.tolist())
        tokens.append(self.stoi["EOS"])
        return np.array(tokens, dtype=np.int32)

    def decode(self, token_seq_to_decode):
        try:
            seq = [int(t) for t in token_seq_to_decode]
            if not seq or self.itos.get(seq[0]) != "BOS":
                return None

            coeffs = []
            for token in seq[1:]:
                decoded = self.itos.get(token)
                if decoded == "EOS":
                    if len(coeffs) == self.dataclass.NUM_COEFFICIENTS:
                        break
                    return None
                if decoded in self.extra_symbols or not isinstance(decoded, int):
                    return None
                coeffs.append(decoded)
                if len(coeffs) == self.dataclass.NUM_COEFFICIENTS:
                    break

            if len(coeffs) != self.dataclass.NUM_COEFFICIENTS:
                return None

            datapoint = self.dataclass(N=self.genus)
            datapoint.data = np.array(coeffs, dtype=np.int64)
            datapoint.p = self.p
            return datapoint
        except Exception:
            return None


def poly_mul_mod_p(a, b, p):
    out = [0] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        if ai == 0:
            continue
        for j, bj in enumerate(b):
            if bj:
                out[i + j] = (out[i + j] + ai * bj) % p
    return out


def poly_pow_small_mod_p(poly, exponent, p):
    out = [1]
    base = [int(c) % p for c in poly]
    e = int(exponent)
    while e:
        if e & 1:
            out = poly_mul_mod_p(out, base, p)
        e >>= 1
        if e:
            base = poly_mul_mod_p(base, base, p)
    return out


def hasse_witt_mod_coeffs(row, p, genus):
    coeffs = [int(c) % p for c in row] + [0, 1]
    powered = poly_pow_small_mod_p(coeffs, (p - 1) // 2, p)
    matrix = []
    for i in range(1, genus + 1):
        row_vals = []
        for j in range(1, genus + 1):
            idx = p * i - j
            row_vals.append(powered[idx] % p if 0 <= idx < len(powered) else 0)
        matrix.append(row_vals)

    x = sp.symbols("x")
    char_coeffs = sp.Poly(sp.Matrix(matrix).charpoly(x).as_expr(), x).all_coeffs()
    return [int(c) % p for c in char_coeffs[1:genus]]


def score_from_sparsity(genus, sparsity):
    if sparsity == 0:
        return 1000.0 * genus + 100.0
    if sparsity == 1:
        return 1000.0 * genus
    return -float(sparsity)


class HighGenusHyperellipticDataPoint(DataPoint):
    PRIME = 3
    GENUS = 3
    NUM_COEFFICIENTS = 6
    LOCAL_SEARCH_ROUNDS = 3
    TWO_FLIP_TRIALS = 128
    COORDINATE_SWEEP = True
    COORDINATE_SAMPLE_SIZE = 32
    SCORE_BATCH_SIZE = 64
    SAGE_PYTHON = "/Applications/SageMath-10-6.app/Contents/MacOS/Python"
    SAGE_DOT_DIR = "/private/tmp/sage-dot-cache"
    _SAGE_WORKER = None

    def __init__(self, N, init=False):
        super().__init__()
        if N != self.GENUS:
            raise ValueError(f"High-genus hyperelliptic environment was configured for genus {self.GENUS}, got {N}")
        self.N = N
        self.genus = N
        self.p = self.PRIME
        self.num_coefficients = self.NUM_COEFFICIENTS
        self.data = np.zeros(self.num_coefficients, dtype=np.int64)
        self.sparsity = None
        self.mod_sparsity = None
        self.target_coeffs = None
        self.sage_called = False

        if init:
            self.data = np.random.randint(0, self.p, size=self.num_coefficients, dtype=np.int64)
            self.calc_features()
            self.calc_score()

    @classmethod
    def _configure(cls):
        cls.NUM_COEFFICIENTS = 2 * cls.GENUS

    def _full_coefficients_low_to_high(self, data=None):
        data = self.data if data is None else data
        return [int(c) % self.p for c in data] + [0, 1]

    def is_squarefree(self, data=None):
        coeffs = self._full_coefficients_low_to_high(data)
        derivative = [(i * coeffs[i]) % self.p for i in range(1, len(coeffs))]
        return len(poly_gcd(coeffs, derivative, self.p)) == 1

    @classmethod
    def _sage_worker(cls):
        if cls._SAGE_WORKER is not None and cls._SAGE_WORKER.poll() is None:
            return cls._SAGE_WORKER
        if not os.path.exists(cls.SAGE_PYTHON):
            raise RuntimeError(f"Sage Python not found: {cls.SAGE_PYTHON}")
        os.makedirs(cls.SAGE_DOT_DIR, exist_ok=True)
        worker_path = os.path.abspath("tools/sage_highgenus_score_worker.py")
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
    def _exact_score_arrays_sage(cls, data):
        data = np.asarray(data, dtype=np.int64)
        scores = np.full(data.shape[0], -1.0, dtype=np.float64)
        valid = np.zeros(data.shape[0], dtype=bool)
        sparsities = np.full(data.shape[0], -1, dtype=np.int64)
        target_coeffs = [None] * data.shape[0]
        if data.size == 0:
            return scores, valid, sparsities, target_coeffs

        worker = cls._sage_worker()
        assert worker.stdin is not None and worker.stdout is not None
        for start in range(0, data.shape[0], cls.SCORE_BATCH_SIZE):
            chunk = data[start : start + cls.SCORE_BATCH_SIZE]
            request = {
                "p": int(cls.PRIME),
                "genus": int(cls.GENUS),
                "data": chunk.astype(int).tolist(),
            }
            worker.stdin.write(json.dumps(request) + "\n")
            worker.stdin.flush()
            line = worker.stdout.readline()
            if not line:
                stderr = worker.stderr.read() if worker.stderr is not None else ""
                raise RuntimeError(f"Sage high-genus scorer exited unexpectedly. stderr:\n{stderr}")
            response = json.loads(line)
            if "error" in response:
                raise RuntimeError(f"Sage high-genus scorer error: {response['error']}")
            for offset, score in enumerate(response["scores"]):
                idx = start + offset
                scores[idx] = float(score)
                valid[idx] = bool(response["valid"][offset])
                if valid[idx]:
                    sparsities[idx] = int(response["sparsities"][offset])
                    target_coeffs[idx] = [int(v) for v in response["target_coeffs"][offset]]
        return scores, valid, sparsities, target_coeffs

    @classmethod
    def _score_arrays(cls, data):
        data = np.asarray(data, dtype=np.int64) % cls.PRIME
        scores = np.full(data.shape[0], -1.0, dtype=np.float64)
        valid = np.zeros(data.shape[0], dtype=bool)
        sparsities = np.full(data.shape[0], -1, dtype=np.int64)
        mod_sparsities = np.full(data.shape[0], -1, dtype=np.int64)
        target_coeffs = [None] * data.shape[0]
        sage_called = np.zeros(data.shape[0], dtype=bool)

        sage_indices = []
        for idx, row in enumerate(data):
            if not cls._is_squarefree_array(row):
                continue
            mod_coeffs = hasse_witt_mod_coeffs(row, cls.PRIME, cls.GENUS)
            mod_s = sum(1 for c in mod_coeffs if c % cls.PRIME != 0)
            mod_sparsities[idx] = mod_s
            valid[idx] = True
            if mod_s >= 2:
                scores[idx] = -float(mod_s)
                sparsities[idx] = mod_s
            else:
                sage_indices.append(idx)

        if sage_indices:
            exact_scores, exact_valid, exact_sparsities, exact_coeffs = cls._exact_score_arrays_sage(data[sage_indices])
            for offset, idx in enumerate(sage_indices):
                valid[idx] = bool(exact_valid[offset])
                scores[idx] = float(exact_scores[offset])
                sage_called[idx] = True
                if exact_valid[offset]:
                    sparsities[idx] = int(exact_sparsities[offset])
                    target_coeffs[idx] = exact_coeffs[offset]

        return scores, valid, sparsities, mod_sparsities, target_coeffs, sage_called

    @classmethod
    def _is_squarefree_array(cls, row):
        coeffs = [int(c) % cls.PRIME for c in row] + [0, 1]
        derivative = [(i * coeffs[i]) % cls.PRIME for i in range(1, len(coeffs))]
        return len(poly_gcd(coeffs, derivative, cls.PRIME)) == 1

    def calc_score(self):
        scores, valid, sparsities, mod_sparsities, target_coeffs, sage_called = self._score_arrays(np.array([self.data], dtype=np.int64))
        self._apply_score(scores[0], valid[0], sparsities[0], mod_sparsities[0], target_coeffs[0], sage_called[0])

    def _apply_score(self, score, valid, sparsity, mod_sparsity, target_coeffs, sage_called):
        self.score = float(score)
        self.sparsity = int(sparsity) if valid and sparsity >= 0 else None
        self.mod_sparsity = int(mod_sparsity) if mod_sparsity >= 0 else None
        self.target_coeffs = [int(v) for v in target_coeffs] if target_coeffs is not None else None
        self.sage_called = bool(sage_called)

    def calc_features(self):
        self.features = ",".join(str(int(c)) for c in self.data.tolist())

    def _coordinate_neighbors(self, center):
        if self.COORDINATE_SWEEP or self.COORDINATE_SAMPLE_SIZE >= len(center):
            positions = range(len(center))
        else:
            positions = np.random.choice(len(center), size=self.COORDINATE_SAMPLE_SIZE, replace=False)
        out = []
        for pos in positions:
            old = int(center[pos])
            for value in range(self.p):
                if value == old:
                    continue
                candidate = center.copy()
                candidate[pos] = value
                out.append(candidate)
        return out

    def _two_flip_neighbors(self, center):
        out = []
        if len(center) < 2 or self.TWO_FLIP_TRIALS <= 0:
            return out
        for _ in range(self.TWO_FLIP_TRIALS):
            i, j = np.random.choice(len(center), size=2, replace=False)
            candidate = center.copy()
            for pos in [i, j]:
                old = int(candidate[pos])
                value = old
                while value == old:
                    value = np.random.randint(0, self.p)
                candidate[pos] = value
            out.append(candidate)
        return out

    def local_search(self, improve_with_local_search):
        if self.LOCAL_SEARCH_ROUNDS <= 0:
            self.calc_score()
            return

        self.data %= self.p
        best_data = self.data.copy()
        self.calc_score()
        best_score = self.score

        rounds = self.LOCAL_SEARCH_ROUNDS if improve_with_local_search else max(1, self.LOCAL_SEARCH_ROUNDS // 2)
        for _ in range(rounds):
            neighbors = self._coordinate_neighbors(best_data)
            neighbors.extend(self._two_flip_neighbors(best_data))
            if not neighbors:
                break
            candidates = np.array(neighbors, dtype=np.int64)
            scores, valid, sparsities, mod_sparsities, target_coeffs, sage_called = self._score_arrays(candidates)
            usable = np.flatnonzero(valid)
            if len(usable) == 0:
                break
            best_idx = usable[int(np.argmax(scores[usable]))]
            if scores[best_idx] <= best_score:
                break
            best_data = candidates[best_idx].copy()
            best_score = float(scores[best_idx])
            self._apply_score(scores[best_idx], valid[best_idx], sparsities[best_idx], mod_sparsities[best_idx], target_coeffs[best_idx], sage_called[best_idx])
            if self.sparsity == 0:
                break

        self.data = best_data
        self.calc_features()
        self.calc_score()

    @classmethod
    def _make_datapoint(cls, N, row, score, valid, sparsity, mod_sparsity, target_coeffs, sage_called):
        d = cls(N=N)
        d.data = np.asarray(row, dtype=np.int64) % cls.PRIME
        d.calc_features()
        d._apply_score(score, valid, sparsity, mod_sparsity, target_coeffs, sage_called)
        return d

    @classmethod
    def _batch_generate_and_score(cls, batch_size, N, pars=None):
        if pars is not None:
            cls._update_class_params(pars)
        data = np.random.randint(0, cls.PRIME, size=(batch_size, cls.NUM_COEFFICIENTS), dtype=np.int64)
        scores, valid, sparsities, mod_sparsities, target_coeffs, sage_called = cls._score_arrays(data)
        out = []
        for idx in np.flatnonzero(scores >= 0):
            out.append(cls._make_datapoint(N, data[idx], scores[idx], valid[idx], sparsities[idx], mod_sparsities[idx], target_coeffs[idx], sage_called[idx]))
        return out

    @classmethod
    def _batch_score_datapoints(cls, data, always_search=False, redeem_only=False, pars=None):
        if pars is not None:
            cls._update_class_params(pars)
        if len(data) == 0:
            return [], 0

        for d in data:
            d.data %= cls.PRIME
            d.p = cls.PRIME
        arrays = np.array([d.data for d in data], dtype=np.int64)
        scores, valid, sparsities, mod_sparsities, target_coeffs, sage_called = cls._score_arrays(arrays)
        n_invalid = int((scores < 0).sum())
        for idx, d in enumerate(data):
            d.calc_features()
            d._apply_score(scores[idx], valid[idx], sparsities[idx], mod_sparsities[idx], target_coeffs[idx], sage_called[idx])

        for d in data:
            if always_search or (d.score < 0 and redeem_only):
                d.local_search(improve_with_local_search=always_search)

        return data, n_invalid

    @classmethod
    def _update_class_params(cls, pars):
        cls.PRIME = pars["prime"]
        cls.GENUS = pars["genus"]
        cls.LOCAL_SEARCH_ROUNDS = pars["local_search_rounds"]
        cls.TWO_FLIP_TRIALS = pars["two_flip_trials"]
        cls.COORDINATE_SWEEP = pars["coordinate_sweep"]
        cls.COORDINATE_SAMPLE_SIZE = pars["coordinate_sample_size"]
        cls.SCORE_BATCH_SIZE = pars["score_batch_size"]
        cls.SAGE_PYTHON = pars["sage_python"]
        cls.SAGE_DOT_DIR = pars["sage_dot_dir"]
        cls._configure()

    @classmethod
    def _save_class_params(cls):
        return {
            "prime": cls.PRIME,
            "genus": cls.GENUS,
            "local_search_rounds": cls.LOCAL_SEARCH_ROUNDS,
            "two_flip_trials": cls.TWO_FLIP_TRIALS,
            "coordinate_sweep": cls.COORDINATE_SWEEP,
            "coordinate_sample_size": cls.COORDINATE_SAMPLE_SIZE,
            "score_batch_size": cls.SCORE_BATCH_SIZE,
            "sage_python": cls.SAGE_PYTHON,
            "sage_dot_dir": cls.SAGE_DOT_DIR,
        }


class HighGenusHyperellipticEnvironment(BaseEnvironment):
    data_class = HighGenusHyperellipticDataPoint

    def __init__(self, params):
        super().__init__(params)
        if not is_prime(params.p) or params.p not in (3, 5, 7):
            raise ValueError("--p must be one of the small odd primes 3, 5, 7")
        if params.N < 2:
            raise ValueError("--N must be at least 2")
        if params.max_len < 2 * params.N + 1:
            raise ValueError(f"--max_len must be at least {2 * params.N + 1} for genus {params.N}")

        self.data_class.PRIME = params.p
        self.data_class.GENUS = params.N
        self.data_class.LOCAL_SEARCH_ROUNDS = params.local_search_rounds
        self.data_class.TWO_FLIP_TRIALS = params.two_flip_trials
        self.data_class.COORDINATE_SWEEP = params.coordinate_sweep
        self.data_class.COORDINATE_SAMPLE_SIZE = params.coordinate_sample_size
        self.data_class.SCORE_BATCH_SIZE = params.score_batch_size
        self.data_class.SAGE_PYTHON = params.sage_python
        self.data_class.SAGE_DOT_DIR = params.sage_dot_dir
        self.data_class._configure()

        self.tokenizer = HighGenusCoefficientTokenizer(
            dataclass=self.data_class,
            genus=params.N,
            p=params.p,
            extra_symbols=self.SPECIAL_SYMBOLS,
        )

    @staticmethod
    def register_args(parser):
        parser.add_argument("--N", type=int, default=3, help="Genus")
        parser.add_argument("--p", type=int, default=3, help="Fixed small prime; supported values are 3, 5, 7")
        parser.add_argument("--local_search_rounds", type=int, default=3, help="Coordinate/two-flip local-search rounds per candidate")
        parser.add_argument("--two_flip_trials", type=int, default=128, help="Random two-coordinate mutations per local-search round")
        parser.add_argument("--coordinate_sweep", type=bool_flag, default="true", help="Try every one-coordinate finite-field mutation")
        parser.add_argument("--coordinate_sample_size", type=int, default=32, help="Coordinate positions sampled when --coordinate_sweep false")
        parser.add_argument("--score_batch_size", type=int, default=32, help="Sage exact scoring batch size after Hasse-Witt gate")
        parser.add_argument("--sage_python", type=str, default="/Applications/SageMath-10-6.app/Contents/MacOS/Python", help="Sage Python executable")
        parser.add_argument("--sage_dot_dir", type=str, default="/private/tmp/sage-dot-cache", help="Writable DOT_SAGE directory for Sage")
        parser.add_argument("--make_object_canonical", type=bool_flag, default="false", help="Reserved for compatibility")
        parser.add_argument("--augment_data_representation", type=bool_flag, default="false", help="Reserved for compatibility")
