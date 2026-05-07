import numpy as np

from src.envs.environment import BaseEnvironment, DataPoint
from src.envs.tokenizers import Tokenizer
from src.utils import bool_flag


class HyperellipticCoefficientTokenizer(Tokenizer):
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

    def encode(self, datapoint_to_encode):
        tokens = [self.stoi["BOS"]]
        tokens.extend(int(c) for c in datapoint_to_encode.data)
        tokens.append(self.stoi["EOS"])
        return np.array(tokens, dtype=np.int32)

    def decode(self, token_seq_to_decode):
        try:
            coeffs = []
            for token in token_seq_to_decode[1:]:
                value = self.itos[int(token)]
                if value in self.extra_symbols:
                    break
                coeffs.append(value)

            datapoint = self.dataclass(N=self.genus)
            if len(coeffs) != datapoint.num_coefficients:
                return None
            datapoint.data = np.array(coeffs, dtype=np.int16)
            return datapoint
        except Exception:
            return None


class HyperellipticDataPoint(DataPoint):
    PRIME = 3
    DEGREE_MODEL = "odd"
    MONIC = True

    def __init__(self, N, init=False):
        super().__init__()
        self.N = N
        self.genus = N
        self.p = self.PRIME
        self.degree = 2 * self.genus + 1 if self.DEGREE_MODEL == "odd" else 2 * self.genus + 2
        self.num_coefficients = self.degree if self.MONIC else self.degree + 1
        self.data = np.zeros(self.num_coefficients, dtype=np.int16)

        if init:
            self.data = np.random.randint(0, self.p, size=self.num_coefficients, dtype=np.int16)
            self.calc_features()

    def calc_score(self):
        raise NotImplementedError("Hyperelliptic scoring will be provided by the external C++ backend.")

    def calc_features(self):
        self.features = ",".join(map(str, self.data.tolist()))

    def local_search(self, improve_with_local_search):
        raise NotImplementedError("Hyperelliptic local search will be provided by the external C++ backend.")

    @classmethod
    def _update_class_params(cls, pars):
        cls.PRIME = pars["prime"]
        cls.DEGREE_MODEL = pars["degree_model"]
        cls.MONIC = pars["monic"]

    @classmethod
    def _save_class_params(cls):
        return {"prime": cls.PRIME, "degree_model": cls.DEGREE_MODEL, "monic": cls.MONIC}


class HyperellipticEnvironment(BaseEnvironment):
    data_class = HyperellipticDataPoint

    def __init__(self, params):
        super().__init__(params)
        if params.encoding_tokens != "coefficients":
            raise ValueError("hyperelliptic currently supports only --encoding_tokens coefficients")
        if params.p < 3 or params.p % 2 == 0:
            raise ValueError("--p must be an odd prime")

        self.data_class.PRIME = params.p
        self.data_class.DEGREE_MODEL = params.degree_model
        self.data_class.MONIC = params.monic
        self.tokenizer = HyperellipticCoefficientTokenizer(self.data_class, params.N, params.p, self.SPECIAL_SYMBOLS)

    @staticmethod
    def register_args(parser):
        parser.add_argument("--N", type=int, default=3, help="Genus of the hyperelliptic curve")
        parser.add_argument("--p", type=int, default=3, help="Small odd prime defining the base field F_p")
        parser.add_argument("--degree_model", type=str, default="odd", choices=["odd", "even"], help="Polynomial degree model")
        parser.add_argument("--monic", type=bool_flag, default="true", help="Represent only monic polynomial models")
        parser.add_argument("--encoding_tokens", type=str, default="coefficients", help="Only coefficient tokenization is supported")
        parser.add_argument("--make_object_canonical", type=bool_flag, default="false", help="Reserved for future canonicalization support")
        parser.add_argument("--augment_data_representation", type=bool_flag, default="false", help="Reserved for future data augmentation")
