import ast
import json
import os
import random
import sqlite3
import subprocess

import numpy as np

from src.envs.environment import BaseEnvironment, DataPoint
from src.envs.hyperelliptic import is_prime, poly_gcd
from src.envs.hyperelliptic_highgenus import hasse_witt_mod_coeffs
from src.envs.tokenizers import Tokenizer
from src.utils import bool_flag


class MultiGenusCoefficientTokenizer(Tokenizer):
    def __init__(self, dataclass, min_genus, max_genus, p, extra_symbols):
        self.dataclass = dataclass
        self.min_genus = min_genus
        self.max_genus = max_genus
        self.p = p
        self.extra_symbols = extra_symbols
        self.stoi = {}
        self.itos = {}

        for token in range(p):
            self.stoi[token] = token
            self.itos[token] = token

        offset = len(self.stoi)
        for genus in range(min_genus, max_genus + 1):
            key = f"G{genus}"
            self.stoi[key] = offset
            self.itos[offset] = key
            offset += 1

        for symbol in extra_symbols:
            self.stoi[symbol] = offset
            self.itos[offset] = symbol
            offset += 1

        genus_ids = [self.stoi[f"G{g}"] for g in range(min_genus, max_genus + 1)]
        coefficient_or_eos = list(range(p)) + [self.stoi["EOS"]]
        self.allowed_token_ids_by_pos = [None] * (2 * max_genus + 7)
        self.allowed_token_ids_by_pos[1] = genus_ids
        self.allowed_token_ids_by_pos[2] = [self.stoi["SEP"]]
        for pos in range(3, len(self.allowed_token_ids_by_pos)):
            self.allowed_token_ids_by_pos[pos] = coefficient_or_eos

    def encode(self, datapoint_to_encode):
        tokens = [self.stoi["BOS"], self.stoi[f"G{datapoint_to_encode.genus}"], self.stoi["SEP"]]
        tokens.extend(int(c) for c in datapoint_to_encode.data.tolist())
        tokens.append(self.stoi["EOS"])
        return np.array(tokens, dtype=np.int32)

    def decode(self, token_seq_to_decode):
        try:
            seq = [int(t) for t in token_seq_to_decode]
            if len(seq) < 4 or self.itos.get(seq[0]) != "BOS":
                return None

            genus_token = self.itos.get(seq[1])
            if not isinstance(genus_token, str) or not genus_token.startswith("G"):
                return None
            genus = int(genus_token[1:])
            if genus < self.min_genus or genus > self.max_genus:
                return None
            if self.itos.get(seq[2]) != "SEP":
                return None

            coeffs = []
            for token in seq[3:]:
                decoded = self.itos.get(token)
                if decoded == "EOS":
                    break
                if decoded in self.extra_symbols or not isinstance(decoded, int):
                    return None
                coeffs.append(decoded)
                if len(coeffs) > 2 * genus + 3:
                    return None

            if len(coeffs) not in (2 * genus + 2, 2 * genus + 3):
                return None
            if coeffs[-1] % self.p == 0:
                return None

            datapoint = self.dataclass(N=genus)
            datapoint.data = np.array(coeffs, dtype=np.int64)
            datapoint.p = self.p
            datapoint.calc_features()
            return datapoint
        except Exception:
            return None


class MultiGenusHyperellipticDataPoint(DataPoint):
    PRIME = 3
    MIN_GENUS = 2
    MAX_GENUS = 16
    LOCAL_SEARCH_ROUNDS = 2
    TWO_FLIP_TRIALS = 64
    COORDINATE_SWEEP = True
    COORDINATE_SAMPLE_SIZE = 32
    SCORE_BATCH_SIZE = 64
    SPARSITY0_BONUS = 500.0
    SAGE_PYTHON = "/Applications/SageMath-10-6.app/Contents/MacOS/Python"
    SAGE_DOT_DIR = "/private/tmp/sage-dot-cache"
    _SAGE_WORKER = None

    def __init__(self, N, init=False):
        super().__init__()
        if N < self.MIN_GENUS or N > self.MAX_GENUS:
            raise ValueError(f"genus must be in [{self.MIN_GENUS}, {self.MAX_GENUS}], got {N}")
        self.N = N
        self.genus = N
        self.p = self.PRIME
        self.num_coefficients = 2 * self.genus + 2
        self.data = np.zeros(self.num_coefficients, dtype=np.int64)
        self.sparsity = None
        self.mod_sparsity = None
        self.target_coeffs = None
        self.sage_called = False

        if init:
            self.genus = random.randint(self.MIN_GENUS, self.MAX_GENUS)
            self.N = self.genus
            self.num_coefficients = random.choice((2 * self.genus + 2, 2 * self.genus + 3))
            self.data = np.random.randint(0, self.p, size=self.num_coefficients, dtype=np.int64)
            self.data[-1] = np.random.randint(1, self.p)
            self.calc_features()
            self.calc_score()

    @classmethod
    def _sage_worker(cls):
        if cls._SAGE_WORKER is not None and cls._SAGE_WORKER.poll() is None:
            return cls._SAGE_WORKER
        if not os.path.exists(cls.SAGE_PYTHON):
            raise RuntimeError(f"Sage Python not found: {cls.SAGE_PYTHON}")
        os.makedirs(cls.SAGE_DOT_DIR, exist_ok=True)
        worker_path = os.path.abspath("tools/sage_multigenus_score_worker.py")
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
    def _exact_score_sage(cls, data):
        if len(data) == 0:
            return [], [], [], []
        worker = cls._sage_worker()
        assert worker.stdin is not None and worker.stdout is not None
        scores, valid, sparsities, target_coeffs = [], [], [], []
        for start in range(0, len(data), cls.SCORE_BATCH_SIZE):
            chunk = data[start : start + cls.SCORE_BATCH_SIZE]
            request = {
                "p": int(cls.PRIME),
                "sparsity0_bonus": float(cls.SPARSITY0_BONUS),
                "items": [{"genus": int(d.genus), "coefficients": d.data.astype(int).tolist()} for d in chunk],
            }
            worker.stdin.write(json.dumps(request) + "\n")
            worker.stdin.flush()
            line = worker.stdout.readline()
            if not line:
                stderr = worker.stderr.read() if worker.stderr is not None else ""
                raise RuntimeError(f"Sage multigenus scorer exited unexpectedly. stderr:\n{stderr}")
            response = json.loads(line)
            if "error" in response:
                raise RuntimeError(f"Sage multigenus scorer error: {response['error']}")
            scores.extend(float(s) for s in response["scores"])
            valid.extend(bool(v) for v in response["valid"])
            sparsities.extend(-1 if s is None else int(s) for s in response["sparsities"])
            target_coeffs.extend(response["target_coeffs"])
        return scores, valid, sparsities, target_coeffs

    @staticmethod
    def _is_squarefree(row, p):
        coeffs = [int(c) % p for c in row]
        if not coeffs or coeffs[-1] == 0:
            return False
        derivative = [(i * coeffs[i]) % p for i in range(1, len(coeffs))]
        return len(poly_gcd(coeffs, derivative, p)) == 1

    @classmethod
    def _mod_sparsity(cls, row, genus):
        coeffs = [int(c) % cls.PRIME for c in row]
        if len(coeffs) == 2 * genus + 2 and coeffs[2 * genus] == 0 and coeffs[-1] == 1:
            hw_coeffs = hasse_witt_mod_coeffs(coeffs[: 2 * genus], cls.PRIME, genus)
        else:
            powered = [1]
            exponent = (cls.PRIME - 1) // 2
            for _ in range(exponent):
                next_power = [0] * (len(powered) + len(coeffs) - 1)
                for i, ai in enumerate(powered):
                    if ai == 0:
                        continue
                    for j, bj in enumerate(coeffs):
                        if bj:
                            next_power[i + j] = (next_power[i + j] + ai * bj) % cls.PRIME
                powered = next_power
            matrix = []
            for i in range(1, genus + 1):
                matrix.append([powered[cls.PRIME * i - j] % cls.PRIME if 0 <= cls.PRIME * i - j < len(powered) else 0 for j in range(1, genus + 1)])
            import sympy as sp

            x = sp.symbols("x")
            hw_coeffs = [int(c) % cls.PRIME for c in sp.Poly(sp.Matrix(matrix).charpoly(x).as_expr(), x).all_coeffs()[1:genus]]
        return sum(1 for c in hw_coeffs if c % cls.PRIME != 0)

    def calc_score(self):
        scored, _, _ = self._score_datapoints([self])
        if scored:
            self.__dict__.update(scored[0].__dict__)
        else:
            self.score = -1.0

    @classmethod
    def score_from_sparsity(cls, genus, sparsity):
        if sparsity == 0:
            return 1000.0 * genus + float(cls.SPARSITY0_BONUS)
        if sparsity == 1:
            return 1000.0 * genus
        return -float(sparsity)

    def _apply_score(self, score, sparsity=None, mod_sparsity=None, target_coeffs=None, sage_called=False):
        self.score = float(score)
        self.sparsity = int(sparsity) if sparsity is not None and sparsity >= 0 else None
        self.mod_sparsity = int(mod_sparsity) if mod_sparsity is not None and mod_sparsity >= 0 else None
        self.target_coeffs = [int(v) for v in target_coeffs] if target_coeffs is not None else None
        self.sage_called = bool(sage_called)

    def calc_features(self):
        coeffs = ",".join(str(int(c)) for c in self.data.tolist())
        self.features = f"p={self.p};g={self.genus};coeffs={coeffs}"

    def _coordinate_neighbors(self, center):
        if self.COORDINATE_SWEEP or self.COORDINATE_SAMPLE_SIZE >= len(center):
            positions = range(len(center))
        else:
            positions = np.random.choice(len(center), size=self.COORDINATE_SAMPLE_SIZE, replace=False)
        out = []
        for pos in positions:
            old = int(center[pos])
            values = range(1, self.p) if pos == len(center) - 1 else range(self.p)
            for value in values:
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
            candidate = center.copy()
            for pos in np.random.choice(len(center), size=2, replace=False):
                old = int(candidate[pos])
                value = old
                while value == old:
                    value = np.random.randint(1, self.p) if pos == len(center) - 1 else np.random.randint(0, self.p)
                candidate[pos] = value
            out.append(candidate)
        return out

    def local_search(self, improve_with_local_search):
        if self.LOCAL_SEARCH_ROUNDS <= 0:
            self.calc_score()
            return

        self.data %= self.p
        if self.data[-1] == 0:
            self.data[-1] = 1
        self.calc_score()
        best_data = self.data.copy()
        best_score = self.score

        rounds = self.LOCAL_SEARCH_ROUNDS if improve_with_local_search else max(1, self.LOCAL_SEARCH_ROUNDS // 2)
        for _ in range(rounds):
            neighbors = self._coordinate_neighbors(best_data)
            neighbors.extend(self._two_flip_neighbors(best_data))
            candidates = [self._from_coefficients(self.genus, row) for row in neighbors]
            scored, _, _ = self._score_datapoints(candidates)
            if not scored:
                break
            best = max(scored, key=lambda d: d.score)
            if best.score <= best_score:
                break
            best_data = best.data.copy()
            best_score = best.score
            self.__dict__.update(best.__dict__)
            if self.sparsity == 0:
                break

        self.data = best_data
        self.calc_features()
        self.calc_score()

    @classmethod
    def _from_coefficients(cls, genus, coefficients, score=None, sparsity=None):
        d = cls(N=int(genus))
        d.data = np.asarray(coefficients, dtype=np.int64) % cls.PRIME
        d.p = cls.PRIME
        d.calc_features()
        if score is not None:
            d._apply_score(score, sparsity=sparsity, sage_called=True)
        return d

    @classmethod
    def _score_datapoints(cls, data):
        candidates = []
        n_invalid = 0
        for d in data:
            d.p = cls.PRIME
            d.data = np.asarray(d.data, dtype=np.int64) % cls.PRIME
            if len(d.data) not in (2 * d.genus + 2, 2 * d.genus + 3) or d.data[-1] == 0 or not cls._is_squarefree(d.data, cls.PRIME):
                d._apply_score(-1.0)
                n_invalid += 1
                continue
            mod_s = cls._mod_sparsity(d.data, d.genus)
            if mod_s >= 2:
                d._apply_score(-float(mod_s), sparsity=mod_s, mod_sparsity=mod_s)
                n_invalid += 1
                continue
            d.mod_sparsity = mod_s
            candidates.append(d)

        scores, valid, sparsities, target_coeffs = cls._exact_score_sage(candidates)
        out = []
        for d, score, ok, sparsity, coeffs in zip(candidates, scores, valid, sparsities, target_coeffs):
            if ok and score >= 0:
                d._apply_score(score, sparsity=sparsity, mod_sparsity=d.mod_sparsity, target_coeffs=coeffs, sage_called=True)
                d.calc_features()
                out.append(d)
            else:
                d._apply_score(-1.0)
                n_invalid += 1
        return out, n_invalid, data

    @classmethod
    def _batch_generate_and_score(cls, batch_size, N, pars=None):
        if pars is not None:
            cls._update_class_params(pars)
        data = [cls(N=N, init=True) for _ in range(batch_size)]
        scored, _, _ = cls._score_datapoints(data)
        return scored

    @classmethod
    def _batch_score_datapoints(cls, data, always_search=False, redeem_only=False, pars=None):
        if pars is not None:
            cls._update_class_params(pars)
        scored, n_invalid, processed = cls._score_datapoints(data)
        if always_search or redeem_only:
            for d in processed:
                if always_search or d.score < 0:
                    d.local_search(improve_with_local_search=always_search)
            scored = [d for d in processed if d.score >= 0]
        return scored, n_invalid

    @classmethod
    def load_initial_data(cls, path, N, max_rows=0, pars=None):
        if pars is not None:
            cls._update_class_params(pars)
        data = []
        with sqlite3.connect(path) as conn:
            tables = {row[0] for row in conn.execute("select name from sqlite_master where type in ('table', 'view')")}
            if "orbit_presentations" in tables:
                sql = (
                    "select prime, genus, coefficients, sparsity "
                    "from orbit_presentations "
                    "where prime = ? and genus between ? and ? "
                    "order by genus desc, sparsity asc"
                )
                params = [cls.PRIME, cls.MIN_GENUS, cls.MAX_GENUS]
                if max_rows and max_rows > 0:
                    sql += " limit ?"
                    params.append(int(max_rows))
                for prime, genus, coeff_text, sparsity in conn.execute(sql, params):
                    coeffs = [int(c) % cls.PRIME for c in ast.literal_eval(coeff_text)]
                    if len(coeffs) not in (2 * int(genus) + 2, 2 * int(genus) + 3) or coeffs[-1] == 0:
                        continue
                    score = cls.score_from_sparsity(int(genus), int(sparsity))
                    data.append(cls._from_coefficients(int(genus), coeffs, score=score, sparsity=int(sparsity)))
                return data

            if "canonical_classes" in tables:
                sql = (
                    "select prime, genus, representative_coefficients, sparsity "
                    "from canonical_classes "
                    "where prime = ? and genus between ? and ? "
                    "order by genus desc, sparsity asc"
                )
                params = [cls.PRIME, cls.MIN_GENUS, cls.MAX_GENUS]
                if max_rows and max_rows > 0:
                    sql += " limit ?"
                    params.append(int(max_rows))
                for prime, genus, coeff_text, sparsity in conn.execute(sql, params):
                    coeffs = [int(c) % cls.PRIME for c in ast.literal_eval(coeff_text)]
                    if len(coeffs) not in (2 * int(genus) + 2, 2 * int(genus) + 3) or coeffs[-1] == 0:
                        continue
                    score = cls.score_from_sparsity(int(genus), int(sparsity))
                    data.append(cls._from_coefficients(int(genus), coeffs, score=score, sparsity=int(sparsity)))
                return data

            summary = conn.execute("select prime, genus from enumeration_summary where id = 1").fetchone()
            if summary is None:
                raise ValueError(f"{path} has no enumeration_summary row")
            prime, genus = int(summary[0]), int(summary[1])
            if prime != cls.PRIME:
                return []
            if genus < cls.MIN_GENUS or genus > cls.MAX_GENUS:
                return []
            sql = "select coefficients, sparsity from sparse_curves order by sparsity asc"
            if max_rows and max_rows > 0:
                sql += f" limit {int(max_rows)}"
            for coeff_text, sparsity in conn.execute(sql):
                coeffs = [int(c) % cls.PRIME for c in ast.literal_eval(coeff_text)]
                if len(coeffs) not in (2 * genus + 2, 2 * genus + 3) or coeffs[-1] == 0:
                    continue
                score = cls.score_from_sparsity(genus, int(sparsity))
                data.append(cls._from_coefficients(genus, coeffs, score=score, sparsity=int(sparsity)))
        return data

    @classmethod
    def _update_class_params(cls, pars):
        cls.PRIME = pars["prime"]
        cls.MIN_GENUS = pars["min_genus"]
        cls.MAX_GENUS = pars["max_genus"]
        cls.LOCAL_SEARCH_ROUNDS = pars["local_search_rounds"]
        cls.TWO_FLIP_TRIALS = pars["two_flip_trials"]
        cls.COORDINATE_SWEEP = pars["coordinate_sweep"]
        cls.COORDINATE_SAMPLE_SIZE = pars["coordinate_sample_size"]
        cls.SCORE_BATCH_SIZE = pars["score_batch_size"]
        cls.SPARSITY0_BONUS = pars["sparsity0_bonus"]
        cls.SAGE_PYTHON = pars["sage_python"]
        cls.SAGE_DOT_DIR = pars["sage_dot_dir"]

    @classmethod
    def _save_class_params(cls):
        return {
            "prime": cls.PRIME,
            "min_genus": cls.MIN_GENUS,
            "max_genus": cls.MAX_GENUS,
            "local_search_rounds": cls.LOCAL_SEARCH_ROUNDS,
            "two_flip_trials": cls.TWO_FLIP_TRIALS,
            "coordinate_sweep": cls.COORDINATE_SWEEP,
            "coordinate_sample_size": cls.COORDINATE_SAMPLE_SIZE,
            "score_batch_size": cls.SCORE_BATCH_SIZE,
            "sparsity0_bonus": cls.SPARSITY0_BONUS,
            "sage_python": cls.SAGE_PYTHON,
            "sage_dot_dir": cls.SAGE_DOT_DIR,
        }


class MultiGenusHyperellipticEnvironment(BaseEnvironment):
    data_class = MultiGenusHyperellipticDataPoint

    def __init__(self, params):
        super().__init__(params)
        if not is_prime(params.p) or params.p not in (3, 5, 7):
            raise ValueError("--p must be one of 3, 5, 7")
        if params.min_genus < 1 or params.N < params.min_genus:
            raise ValueError("--N must be at least --min_genus")
        if params.max_len < 2 * params.N + 5:
            raise ValueError(f"--max_len must be at least {2 * params.N + 5} for multigenus max genus {params.N}")

        self.data_class.PRIME = params.p
        self.data_class.MIN_GENUS = params.min_genus
        self.data_class.MAX_GENUS = params.N
        self.data_class.LOCAL_SEARCH_ROUNDS = params.local_search_rounds
        self.data_class.TWO_FLIP_TRIALS = params.two_flip_trials
        self.data_class.COORDINATE_SWEEP = params.coordinate_sweep
        self.data_class.COORDINATE_SAMPLE_SIZE = params.coordinate_sample_size
        self.data_class.SCORE_BATCH_SIZE = params.score_batch_size
        self.data_class.SPARSITY0_BONUS = params.sparsity0_bonus
        self.data_class.SAGE_PYTHON = params.sage_python
        self.data_class.SAGE_DOT_DIR = params.sage_dot_dir

        self.tokenizer = MultiGenusCoefficientTokenizer(
            dataclass=self.data_class,
            min_genus=params.min_genus,
            max_genus=params.N,
            p=params.p,
            extra_symbols=self.SPECIAL_SYMBOLS,
        )

    @staticmethod
    def register_args(parser):
        parser.add_argument("--N", type=int, default=16, help="Maximum genus to train/generate")
        parser.add_argument("--min_genus", type=int, default=2, help="Minimum genus included in the multigenus token vocabulary")
        parser.add_argument("--p", type=int, default=3, help="Fixed small prime; supported values are 3, 5, 7")
        parser.add_argument("--initial_data_sqlite", type=str, default="", help="Comma-separated SQLite files or glob patterns used as seed data")
        parser.add_argument("--initial_data_max_rows", type=int, default=0, help="Maximum rows loaded from each seed file; 0 means all rows")
        parser.add_argument("--local_search_rounds", type=int, default=2, help="Coordinate/two-flip local-search rounds per candidate")
        parser.add_argument("--two_flip_trials", type=int, default=64, help="Random two-coordinate mutations per local-search round")
        parser.add_argument("--coordinate_sweep", type=bool_flag, default="true", help="Try every one-coordinate finite-field mutation")
        parser.add_argument("--coordinate_sample_size", type=int, default=32, help="Coordinate positions sampled when --coordinate_sweep false")
        parser.add_argument("--score_batch_size", type=int, default=32, help="Sage exact scoring batch size after Hasse-Witt gate")
        parser.add_argument("--sparsity0_bonus", type=float, default=500.0, help="Extra score added for exact sparsity 0")
        parser.add_argument("--sage_python", type=str, default="/Applications/SageMath-10-6.app/Contents/MacOS/Python", help="Sage Python executable")
        parser.add_argument("--sage_dot_dir", type=str, default="/private/tmp/sage-dot-cache", help="Writable DOT_SAGE directory for Sage")
        parser.add_argument("--make_object_canonical", type=bool_flag, default="false", help="Reserved for compatibility")
        parser.add_argument("--augment_data_representation", type=bool_flag, default="false", help="Reserved for compatibility")
