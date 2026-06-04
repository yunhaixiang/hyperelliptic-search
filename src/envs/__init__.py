from importlib import import_module


_ENV_SPECS = {
    "square": ("src.envs.cycle", "SquareEnvironment"),
    "isosceles": ("src.envs.isosceles", "IsoscelesEnvironment"),
    "sphere": ("src.envs.sphere", "SphereEnvironment"),
    "hyperelliptic": ("src.envs.hyperelliptic", "HyperellipticEnvironment"),
    "hyperelliptic_highgenus": ("src.envs.hyperelliptic_highgenus", "HighGenusHyperellipticEnvironment"),
    "hyperelliptic_multigenus": ("src.envs.hyperelliptic_multigenus", "MultiGenusHyperellipticEnvironment"),
    "hyperelliptic2": ("src.envs.hyperelliptic2", "Hyperelliptic2Environment"),
    "hyperelliptic_factorized": ("src.envs.hyperelliptic_factorized", "FactorizedHyperellipticEnvironment"),
    "hyperelliptic_factorized_fixed": (
        "src.envs.hyperelliptic_factorized_fixed",
        "FixedFactorizedHyperellipticEnvironment",
    ),
}


class LazyEnvironmentRegistry:
    def __init__(self, specs):
        self.specs = dict(specs)
        self.cache = {}

    def __contains__(self, key):
        return key in self.specs

    def __getitem__(self, key):
        if key not in self.specs:
            raise KeyError(key)
        if key not in self.cache:
            module_name, class_name = self.specs[key]
            self.cache[key] = getattr(import_module(module_name), class_name)
        return self.cache[key]

    def keys(self):
        return self.specs.keys()

    def items(self):
        for key in self.specs:
            yield key, self[key]


ENVS = LazyEnvironmentRegistry(_ENV_SPECS)


def build_env(params):
    """
    Build environment.
    """
    env = ENVS[params.env_name](params)
    return env
