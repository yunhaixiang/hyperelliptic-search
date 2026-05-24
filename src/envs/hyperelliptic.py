import math
import random

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
        for idx in range(dataclass.NUM_COEFFICIENTS):
            symbol = f"C{idx}"
            self.stoi[symbol] = offset + idx
            self.itos[offset + idx] = symbol

    def _encode_coeff(self, coeff):
        coeff = int(coeff)
        tokens = []
        for _ in range(self.digits):
            tokens.append(self.stoi[coeff % self.base])
            coeff //= self.base
        return tokens

    def _decode_coeff(self, tokens):
        coeff = 0
        place = 1
        for token in tokens:
            value = self.itos[int(token)]
            if value in self.extra_symbols or not isinstance(value, int):
                return None
            coeff += value * place
            place *= self.base
        if coeff >= self.p:
            return None
        return coeff

    def encode(self, datapoint_to_encode):
        tokens = [self.stoi["BOS"]]
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
            coeffs = []
            for idx in range(self.dataclass.NUM_COEFFICIENTS):
                if pos >= len(seq) or self.itos.get(seq[pos]) != f"C{idx}":
                    return None
                pos += 1
                if pos + self.digits > len(seq):
                    return None
                coeff = self._decode_coeff(seq[pos : pos + self.digits])
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
            datapoint.data = np.array(coeffs, dtype=np.int64)
            return datapoint
        except Exception:
            return None


class HyperellipticDataPoint(DataPoint):
    PRIME = 3
    DEGREE_MODEL = "odd"
    MONIC = True
    DEPRESSED = True
    NUM_COEFFICIENTS = 4
    LOCAL_SEARCH_STEPS = 0
    LOCAL_SEARCH_BATCH_SIZE = 32
    MUTATION_RADIUS = 16
    LOCAL_SEARCH_RANDOM_REPLACEMENT = 0.2
    MAX_LEGENDRE_TABLE_P = 10_000_000
    SCORE_BATCH_SIZE = 32
    SCORE_X_CHUNK_SIZE = 65_536

    _LEGENDRE_P = None
    _LEGENDRE_TABLE = None
    _X_P = None
    _X_VALUES = None

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
        self.data = np.zeros(self.num_coefficients, dtype=np.int64)

        if init:
            self.data = np.random.randint(0, self.p, size=self.num_coefficients, dtype=np.int64)
            self.calc_features()
            self.calc_score()

    @classmethod
    def _configure_num_coefficients(cls):
        cls.NUM_COEFFICIENTS = 4 if cls.DEPRESSED else 5

    @classmethod
    def _x_values(cls):
        if cls._X_P != cls.PRIME or cls._X_VALUES is None:
            cls._X_VALUES = np.arange(cls.PRIME, dtype=np.int64)
            cls._X_P = cls.PRIME
        return cls._X_VALUES

    @classmethod
    def _legendre_table(cls):
        if cls.PRIME > cls.MAX_LEGENDRE_TABLE_P:
            raise ValueError(
                f"p={cls.PRIME} is above --max_legendre_table_p={cls.MAX_LEGENDRE_TABLE_P}; "
                "use a smaller prime for the NumPy scorer or add a C++/FLINT batch scorer."
            )
        if cls._LEGENDRE_P == cls.PRIME and cls._LEGENDRE_TABLE is not None:
            return cls._LEGENDRE_TABLE

        residues = (np.arange(cls.PRIME, dtype=np.int64) ** 2) % cls.PRIME
        table = np.full(cls.PRIME, -1, dtype=np.int8)
        table[residues] = 1
        table[0] = 0
        cls._LEGENDRE_P = cls.PRIME
        cls._LEGENDRE_TABLE = table
        return table

    def _full_coefficients_low_to_high(self, data=None):
        data = self.data if data is None else data
        data = [int(c) % self.p for c in data]
        if self.DEPRESSED:
            a0, a1, a2, a3 = data
            return [a0, a1, a2, a3, 0, 1]
        a0, a1, a2, a3, a4 = data
        return [a0, a1, a2, a3, a4, 1]

    @classmethod
    def _full_coefficients_matrix_low_to_high(cls, data):
        data = np.asarray(data, dtype=np.int64) % cls.PRIME
        if cls.DEPRESSED:
            zeros = np.zeros((data.shape[0], 1), dtype=np.int64)
            ones = np.ones((data.shape[0], 1), dtype=np.int64)
            return np.concatenate([data, zeros, ones], axis=1)
        ones = np.ones((data.shape[0], 1), dtype=np.int64)
        return np.concatenate([data, ones], axis=1)

    def is_squarefree(self, data=None):
        coeffs = self._full_coefficients_low_to_high(data)
        derivative = [(i * coeffs[i]) % self.p for i in range(1, len(coeffs))]
        return len(poly_gcd(coeffs, derivative, self.p)) == 1

    def compute_c1(self, data=None):
        coeffs = self._full_coefficients_low_to_high(data)
        x = self._x_values()
        values = np.full_like(x, coeffs[-1], dtype=np.int64)
        for coeff in reversed(coeffs[:-1]):
            values *= x
            values += coeff
            values %= self.p
        return int(self._legendre_table()[values].sum())

    @classmethod
    def compute_c1_batch(cls, data):
        data = np.asarray(data, dtype=np.int64)
        if data.size == 0:
            return np.array([], dtype=np.int64)

        coeffs = cls._full_coefficients_matrix_low_to_high(data)
        table = cls._legendre_table()
        out = np.zeros(coeffs.shape[0], dtype=np.int64)
        x_all = cls._x_values()

        for x_start in range(0, cls.PRIME, cls.SCORE_X_CHUNK_SIZE):
            x = x_all[x_start : x_start + cls.SCORE_X_CHUNK_SIZE]
            values = np.full((coeffs.shape[0], x.shape[0]), coeffs[:, -1, None], dtype=np.int64)
            for coeff_col in range(coeffs.shape[1] - 2, -1, -1):
                values *= x[None, :]
                values += coeffs[:, coeff_col, None]
                values %= cls.PRIME
            out += table[values].sum(axis=1)
        return out

    @classmethod
    def _score_from_c1(cls, c1):
        c1 = np.asarray(c1, dtype=np.int64)
        scores = 1.0 - np.abs(c1) / (4.0 * math.sqrt(cls.PRIME))
        return np.where(c1 == 0, 10.0, scores)

    @classmethod
    def _score_arrays(cls, data):
        data = np.asarray(data, dtype=np.int64)
        scores = np.full(data.shape[0], -1.0, dtype=np.float64)
        c1s = np.full(data.shape[0], 0, dtype=np.int64)
        valid = np.zeros(data.shape[0], dtype=bool)

        prototype = cls(N=2)
        for idx, row in enumerate(data):
            valid[idx] = prototype.is_squarefree(row)

        valid_indices = np.flatnonzero(valid)
        for start in range(0, len(valid_indices), cls.SCORE_BATCH_SIZE):
            indices = valid_indices[start : start + cls.SCORE_BATCH_SIZE]
            c1_batch = cls.compute_c1_batch(data[indices])
            c1s[indices] = c1_batch
            scores[indices] = cls._score_from_c1(c1_batch)

        return scores, c1s, valid

    def calc_score(self):
        if not self.is_squarefree():
            self.c1 = None
            self.score = -1
            return

        self.c1 = self.compute_c1()
        if self.c1 == 0:
            self.score = 10.0
        else:
            self.score = 1.0 - abs(self.c1) / (4.0 * math.sqrt(self.p))

    def _apply_score(self, score, c1, valid):
        self.score = float(score)
        self.c1 = int(c1) if valid else None

    def calc_features(self):
        self.features = ",".join(map(str, [int(c) for c in self.data.tolist()]))

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
        if self.LOCAL_SEARCH_STEPS <= 0:
            return

        best_data = self.data.copy()
        if self.score < 0:
            best_c1 = None
            best_abs = float("inf")
        else:
            best_c1 = self.c1 if self.c1 is not None else self.compute_c1(best_data)
            best_abs = abs(best_c1)
            if best_abs == 0:
                return

        rounds = self.LOCAL_SEARCH_STEPS if improve_with_local_search else max(1, self.LOCAL_SEARCH_STEPS // 4)
        for _ in range(rounds):
            candidates = np.array([self._random_neighbor(best_data) for _ in range(self.LOCAL_SEARCH_BATCH_SIZE)], dtype=np.int64)
            scores, c1s, valid = self._score_arrays(candidates)
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
        self.calc_score()

    @classmethod
    def _make_datapoint(cls, N, row, score, c1, valid):
        d = cls(N=N)
        d.data = np.asarray(row, dtype=np.int64)
        d.calc_features()
        d._apply_score(score, c1, valid)
        return d

    @classmethod
    def _batch_generate_and_score(cls, batch_size, N, pars=None):
        if pars is not None:
            cls._update_class_params(pars)
        data = np.random.randint(0, cls.PRIME, size=(batch_size, cls.NUM_COEFFICIENTS), dtype=np.int64)
        scores, c1s, valid = cls._score_arrays(data)
        out = []
        for idx in np.flatnonzero(scores >= 0):
            out.append(cls._make_datapoint(N, data[idx], scores[idx], c1s[idx], valid[idx]))
        return out

    @classmethod
    def _batch_score_datapoints(cls, data, always_search=False, redeem_only=False, pars=None):
        if pars is not None:
            cls._update_class_params(pars)
        if len(data) == 0:
            return [], 0

        arrays = np.array([d.data for d in data], dtype=np.int64)
        scores, c1s, valid = cls._score_arrays(arrays)
        n_invalid = int((scores < 0).sum())

        for idx, d in enumerate(data):
            d.calc_features()
            d._apply_score(scores[idx], c1s[idx], valid[idx])

        for d in data:
            if always_search or (d.score < 0 and redeem_only):
                d.local_search(improve_with_local_search=always_search)

        return data, n_invalid

    @classmethod
    def _update_class_params(cls, pars):
        cls.PRIME = pars["prime"]
        cls.DEGREE_MODEL = pars["degree_model"]
        cls.MONIC = pars["monic"]
        cls.DEPRESSED = pars["depressed"]
        cls.LOCAL_SEARCH_STEPS = pars["local_search_steps"]
        cls.LOCAL_SEARCH_BATCH_SIZE = pars["local_search_batch_size"]
        cls.MUTATION_RADIUS = pars["mutation_radius"]
        cls.LOCAL_SEARCH_RANDOM_REPLACEMENT = pars["local_search_random_replacement"]
        cls.MAX_LEGENDRE_TABLE_P = pars["max_legendre_table_p"]
        cls.SCORE_BATCH_SIZE = pars["score_batch_size"]
        cls.SCORE_X_CHUNK_SIZE = pars["score_x_chunk_size"]
        cls._configure_num_coefficients()

    @classmethod
    def _save_class_params(cls):
        return {
            "prime": cls.PRIME,
            "degree_model": cls.DEGREE_MODEL,
            "monic": cls.MONIC,
            "depressed": cls.DEPRESSED,
            "local_search_steps": cls.LOCAL_SEARCH_STEPS,
            "local_search_batch_size": cls.LOCAL_SEARCH_BATCH_SIZE,
            "mutation_radius": cls.MUTATION_RADIUS,
            "local_search_random_replacement": cls.LOCAL_SEARCH_RANDOM_REPLACEMENT,
            "max_legendre_table_p": cls.MAX_LEGENDRE_TABLE_P,
            "score_batch_size": cls.SCORE_BATCH_SIZE,
            "score_x_chunk_size": cls.SCORE_X_CHUNK_SIZE,
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
        self.data_class.DEGREE_MODEL = params.degree_model
        self.data_class.MONIC = params.monic
        self.data_class.DEPRESSED = params.depressed
        self.data_class.LOCAL_SEARCH_STEPS = params.local_search_steps
        self.data_class.LOCAL_SEARCH_BATCH_SIZE = params.local_search_batch_size
        self.data_class.MUTATION_RADIUS = params.mutation_radius
        self.data_class.LOCAL_SEARCH_RANDOM_REPLACEMENT = params.local_search_random_replacement
        self.data_class.MAX_LEGENDRE_TABLE_P = params.max_legendre_table_p
        self.data_class.SCORE_BATCH_SIZE = params.score_batch_size
        self.data_class.SCORE_X_CHUNK_SIZE = params.score_x_chunk_size
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
        parser.add_argument("--max_legendre_table_p", type=int, default=10_000_000, help="Largest p allowed for the NumPy Legendre-table scorer")
        parser.add_argument("--score_batch_size", type=int, default=32, help="Candidate batch size for vectorized c1 scoring")
        parser.add_argument("--score_x_chunk_size", type=int, default=65536, help="Number of field elements per vectorized scoring chunk")
        parser.add_argument("--make_object_canonical", type=bool_flag, default="false", help="Reserved for future canonicalization support")
        parser.add_argument("--augment_data_representation", type=bool_flag, default="false", help="Reserved for future data augmentation")
