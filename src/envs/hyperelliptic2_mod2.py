import json
import os
from functools import lru_cache


MOD2_FORMS_PATH = os.path.join(os.path.dirname(__file__), "hyperelliptic2_mod2_forms.json")


@lru_cache(maxsize=1)
def _mod2_forms_table():
    if not os.path.exists(MOD2_FORMS_PATH):
        raise RuntimeError(
            f"missing precomputed hyperelliptic2 mod-2 forms table: {MOD2_FORMS_PATH}. "
            "Run tools/precompute_hyperelliptic2_mod2_forms.py first."
        )
    with open(MOD2_FORMS_PATH, "r") as f:
        return json.load(f)


def _finite_branch_prefix_mask(factor_degrees, prefix_bits):
    mask_limit = (1 << int(prefix_bits)) - 1
    product = 1
    for degree in factor_degrees:
        degree = int(degree)
        if degree < prefix_bits:
            product ^= (product << degree) & mask_limit
    return product & mask_limit


def is_mod2_allowed_factor_degrees(factor_degrees, genus, infinity_branch):
    genus = int(genus)
    if genus < 1:
        return False
    finite_degree = sum(int(degree) for degree in factor_degrees)
    table = _mod2_forms_table()
    if genus > int(table["max_genus"]):
        raise RuntimeError(
            f"mod-2 forms table only covers genus <= {table['max_genus']}; got genus {genus}"
        )
    record = table["forms"].get(str(genus), {}).get(str(finite_degree))
    if record is None or bool(record["infinity_branch"]) != bool(infinity_branch):
        return False
    prefix_bits = int(record["prefix_bits"])
    finite_mask = _finite_branch_prefix_mask(factor_degrees, prefix_bits)
    return finite_mask in {int(mask) for mask in record["allowed_finite_prefix_masks"]}
