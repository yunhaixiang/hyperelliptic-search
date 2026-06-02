from src.envs.cycle import SquareEnvironment
from src.envs.hyperelliptic import HyperellipticEnvironment
from src.envs.hyperelliptic_highgenus import HighGenusHyperellipticEnvironment
from src.envs.hyperelliptic_multigenus import MultiGenusHyperellipticEnvironment
from src.envs.hyperelliptic2 import Hyperelliptic2Environment
from src.envs.isosceles import IsoscelesEnvironment
from src.envs.sphere import SphereEnvironment

ENVS = {
    "square": SquareEnvironment,
    "isosceles": IsoscelesEnvironment,
    "sphere": SphereEnvironment,
    "hyperelliptic": HyperellipticEnvironment,
    "hyperelliptic_highgenus": HighGenusHyperellipticEnvironment,
    "hyperelliptic_multigenus": MultiGenusHyperellipticEnvironment,
    "hyperelliptic2": Hyperelliptic2Environment,
}


def build_env(params):
    """
    Build environment.
    """
    env = ENVS[params.env_name](params)
    return env
