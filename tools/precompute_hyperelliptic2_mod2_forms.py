#!/usr/bin/env python3
"""Precompute mod-2 branch-cycle prefixes for hyperelliptic2.

For a trinomial L-polynomial, the mod-2 zeta constraint forces

    prod_orbits (1 + T^degree) / (1 + T)^2

to have zero coefficients in degrees 1,...,g-1. Equivalently, the low
branch polynomial prefix is the prefix of (1+T)^2. This table stores the
allowed finite-branch prefixes for the odd and even degree models.
"""

import argparse
import json
import os


def branch_target_prefix_mask(genus):
    mask = 1
    if genus > 2:
        mask |= 1 << 2
    return mask


def finite_mask_with_infinity(genus):
    """Solve finite_prefix * (1+T) = target mod T^genus over F_2."""
    target = branch_target_prefix_mask(genus)
    out = 0
    previous = 0
    for degree in range(genus):
        target_bit = (target >> degree) & 1
        bit = target_bit ^ previous
        if bit:
            out |= 1 << degree
        previous = bit
    return out


def build_table(max_genus):
    forms = {}
    for genus in range(1, max_genus + 1):
        odd_finite_degree = 2 * genus + 1
        even_finite_degree = 2 * genus + 2
        forms[str(genus)] = {
            str(odd_finite_degree): {
                "finite_degree": odd_finite_degree,
                "infinity_branch": True,
                "prefix_bits": genus,
                "allowed_finite_prefix_masks": [finite_mask_with_infinity(genus)],
            },
            str(even_finite_degree): {
                "finite_degree": even_finite_degree,
                "infinity_branch": False,
                "prefix_bits": genus,
                "allowed_finite_prefix_masks": [branch_target_prefix_mask(genus)],
            },
        }
    return {
        "format": "hyperelliptic2_mod2_forms_v1",
        "max_genus": int(max_genus),
        "description": (
            "Allowed finite branch-polynomial prefixes modulo 2 for the "
            "trinomial L-polynomial branch-cycle prefilter."
        ),
        "forms": forms,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-genus", type=int, default=100)
    parser.add_argument(
        "--output",
        default=os.path.join("src", "envs", "hyperelliptic2_mod2_forms.json"),
    )
    args = parser.parse_args()

    table = build_table(args.max_genus)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(table, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"wrote {args.output} for genus <= {args.max_genus}")


if __name__ == "__main__":
    main()
