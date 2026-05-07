from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from itertools import product
import json
from math import comb
from pathlib import Path
import sqlite3
from time import perf_counter
from typing import Iterable, Optional


def _is_odd_prime(value: int) -> bool:
    if value < 3 or value % 2 == 0:
        return False
    divisor = 3
    while divisor * divisor <= value:
        if value % divisor == 0:
            return False
        divisor += 2
    return True


@dataclass
class CanonicalRecord:
    canonical_key: tuple[int, ...]
    coefficients: tuple[int, ...]
    rational_branch_count: int
    hasse_witt_lpoly_mod_p: Optional[tuple[int, ...]] = None
    status_by_max_sparsity: dict[int, str] = field(default_factory=dict)
    sparsity_by_max_sparsity: dict[int, Optional[int]] = field(default_factory=dict)
    exact_lpoly_by_max_sparsity: dict[int, Optional[tuple[int, ...]]] = field(default_factory=dict)


@dataclass(frozen=True)
class PrimeField:
    prime: int

    def __post_init__(self) -> None:
        if not _is_odd_prime(self.prime):
            raise ValueError("prime field characteristic must be an odd prime")

    def normalize(self, value: int) -> int:
        return value % self.prime

    def inverse(self, value: int) -> int:
        value %= self.prime
        if value == 0:
            raise ValueError("zero has no inverse")
        return pow(value, -1, self.prime)


def _trim(coefficients: list[int]) -> list[int]:
    while len(coefficients) > 1 and coefficients[-1] == 0:
        coefficients.pop()
    return coefficients


def _poly_trim_mod(coefficients: list[int], p: int) -> tuple[int, ...]:
    reduced = [coefficient % p for coefficient in coefficients]
    return tuple(_trim(reduced))


def _poly_add_mod(lhs: tuple[int, ...], rhs: tuple[int, ...], p: int) -> tuple[int, ...]:
    length = max(len(lhs), len(rhs))
    return _poly_trim_mod(
        [(lhs[i] if i < len(lhs) else 0) + (rhs[i] if i < len(rhs) else 0) for i in range(length)],
        p,
    )


def _poly_sub_mod(lhs: tuple[int, ...], rhs: tuple[int, ...], p: int) -> tuple[int, ...]:
    length = max(len(lhs), len(rhs))
    return _poly_trim_mod(
        [(lhs[i] if i < len(lhs) else 0) - (rhs[i] if i < len(rhs) else 0) for i in range(length)],
        p,
    )


def _poly_mul_mod(lhs: tuple[int, ...], rhs: tuple[int, ...], p: int) -> tuple[int, ...]:
    if lhs == (0,) or rhs == (0,):
        return (0,)
    result = [0] * (len(lhs) + len(rhs) - 1)
    for i, lhs_coefficient in enumerate(lhs):
        if lhs_coefficient == 0:
            continue
        for j, rhs_coefficient in enumerate(rhs):
            if rhs_coefficient == 0:
                continue
            result[i + j] += lhs_coefficient * rhs_coefficient
    return _poly_trim_mod(result, p)


def _poly_pow_mod(base: tuple[int, ...], exponent: int, p: int) -> tuple[int, ...]:
    result = (1,)
    while exponent > 0:
        if exponent % 2 == 1:
            result = _poly_mul_mod(result, base, p)
        base = _poly_mul_mod(base, base, p)
        exponent //= 2
    return result


def _poly_exact_div_mod(numerator: tuple[int, ...], denominator: tuple[int, ...], p: int) -> tuple[int, ...]:
    if denominator == (0,):
        raise ZeroDivisionError("division by zero polynomial")
    if numerator == (0,):
        return (0,)

    rem = list(numerator)
    quotient = [0] * max(1, len(numerator) - len(denominator) + 1)
    denominator_degree = len(denominator) - 1
    denominator_leading_inverse = pow(denominator[-1], -1, p)

    while len(rem) >= len(denominator) and rem != [0]:
        shift = len(rem) - len(denominator)
        scale = rem[-1] * denominator_leading_inverse % p
        quotient[shift] = scale
        for i in range(denominator_degree + 1):
            rem[shift + i] = (rem[shift + i] - scale * denominator[i]) % p
        _trim(rem)

    if rem != [0]:
        raise ValueError("polynomial division was not exact")
    return _poly_trim_mod(quotient, p)


def _poly_scalar_mul_mod(polynomial: tuple[int, ...], scalar: int, p: int) -> tuple[int, ...]:
    return _poly_trim_mod([scalar * coefficient for coefficient in polynomial], p)


def _poly_exact_quotient_mod(numerator: tuple[int, ...], denominator: tuple[int, ...], p: int) -> tuple[int, ...]:
    return _poly_exact_div_mod(_poly_trim_mod(list(numerator), p), _poly_trim_mod(list(denominator), p), p)


def _poly_remainder_mod(numerator: tuple[int, ...], denominator: tuple[int, ...], p: int) -> tuple[int, ...]:
    if denominator == (0,):
        raise ZeroDivisionError("division by zero polynomial")
    rem = list(_poly_trim_mod(list(numerator), p))
    denominator = _poly_trim_mod(list(denominator), p)
    denominator_degree = len(denominator) - 1
    denominator_leading_inverse = pow(denominator[-1], -1, p)

    while len(rem) >= len(denominator) and rem != [0]:
        shift = len(rem) - len(denominator)
        scale = rem[-1] * denominator_leading_inverse % p
        for i in range(denominator_degree + 1):
            rem[shift + i] = (rem[shift + i] - scale * denominator[i]) % p
        _trim(rem)

    return _poly_trim_mod(rem, p)


def _poly_monic_mod(polynomial: tuple[int, ...], p: int) -> tuple[int, ...]:
    polynomial = _poly_trim_mod(list(polynomial), p)
    if polynomial == (0,):
        return polynomial
    leading_inverse = pow(polynomial[-1], -1, p)
    return _poly_scalar_mul_mod(polynomial, leading_inverse, p)


def _poly_gcd_mod(lhs: tuple[int, ...], rhs: tuple[int, ...], p: int) -> tuple[int, ...]:
    lhs = _poly_trim_mod(list(lhs), p)
    rhs = _poly_trim_mod(list(rhs), p)
    while rhs != (0,):
        lhs, rhs = rhs, _poly_remainder_mod(lhs, rhs, p)
    return _poly_monic_mod(lhs, p)


def _poly_mul_remainder_mod(lhs: tuple[int, ...], rhs: tuple[int, ...], modulus: tuple[int, ...], p: int) -> tuple[int, ...]:
    return _poly_remainder_mod(_poly_mul_mod(lhs, rhs, p), modulus, p)


def _poly_pow_remainder_mod(base: tuple[int, ...], exponent: int, modulus: tuple[int, ...], p: int) -> tuple[int, ...]:
    result = (1,)
    base = _poly_remainder_mod(base, modulus, p)
    while exponent > 0:
        if exponent % 2 == 1:
            result = _poly_mul_remainder_mod(result, base, modulus, p)
        base = _poly_mul_remainder_mod(base, base, modulus, p)
        exponent //= 2
    return result


def _determinant_polynomial_mod(matrix: list[list[tuple[int, ...]]], p: int) -> tuple[int, ...]:
    n = len(matrix)
    if n == 0:
        return (1,)
    if n == 1:
        return matrix[0][0]

    working = [[entry for entry in row] for row in matrix]
    previous_pivot = (1,)
    sign = 1

    for k in range(n - 1):
        pivot_row = None
        pivot_column = None
        for i in range(k, n):
            for j in range(k, n):
                if working[i][j] != (0,):
                    pivot_row = i
                    pivot_column = j
                    break
            if pivot_row is not None:
                break

        if pivot_row is None or pivot_column is None:
            return (0,)

        if pivot_row != k:
            working[k], working[pivot_row] = working[pivot_row], working[k]
            sign = -sign
        if pivot_column != k:
            for row in working:
                row[k], row[pivot_column] = row[pivot_column], row[k]
            sign = -sign

        pivot = working[k][k]
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                numerator = _poly_sub_mod(
                    _poly_mul_mod(working[i][j], pivot, p),
                    _poly_mul_mod(working[i][k], working[k][j], p),
                    p,
                )
                working[i][j] = _poly_exact_div_mod(numerator, previous_pivot, p)

        previous_pivot = pivot

    determinant = working[n - 1][n - 1]
    if sign == -1:
        determinant = _poly_scalar_mul_mod(determinant, -1, p)
    return determinant


def _affine_polynomial_to_binary_form(polynomial: Polynomial, binary_degree: int) -> tuple[int, ...]:
    if polynomial.degree > binary_degree:
        raise ValueError("polynomial degree exceeds binary form degree")
    return tuple(polynomial.coefficient(i) for i in range(binary_degree + 1))


def _precompute_pgl2(prime: int) -> tuple[tuple[int, int, int, int], ...]:
    representatives = {}
    for a, b, c, d in product(range(prime), repeat=4):
        determinant = (a * d - b * c) % prime
        if determinant == 0:
            continue

        entries = (a, b, c, d)
        first_nonzero = next(entry for entry in entries if entry != 0)
        scale = pow(first_nonzero, -1, prime)
        representative = tuple(entry * scale % prime for entry in entries)
        representatives[representative] = representative

    return tuple(sorted(representatives))


def _linear_power_coefficients(alpha: int, beta: int, exponent: int, prime: int) -> tuple[int, ...]:
    return tuple(
        comb(exponent, k) * pow(alpha, k, prime) * pow(beta, exponent - k, prime) % prime
        for k in range(exponent + 1)
    )


def _transform_binary_form(binary_form: tuple[int, ...], matrix: tuple[int, int, int, int], prime: int) -> tuple[int, ...]:
    a, b, c, d = matrix
    degree = len(binary_form) - 1
    result = [0] * (degree + 1)

    powers_ax_bz = [_linear_power_coefficients(a, b, exponent, prime) for exponent in range(degree + 1)]
    powers_cx_dz = [_linear_power_coefficients(c, d, exponent, prime) for exponent in range(degree + 1)]

    for i, coefficient in enumerate(binary_form):
        if coefficient == 0:
            continue
        left = powers_ax_bz[i]
        right = powers_cx_dz[degree - i]
        for left_x_degree, left_coefficient in enumerate(left):
            if left_coefficient == 0:
                continue
            for right_x_degree, right_coefficient in enumerate(right):
                if right_coefficient == 0:
                    continue
                x_degree = left_x_degree + right_x_degree
                result[x_degree] = (result[x_degree] + coefficient * left_coefficient * right_coefficient) % prime

    return tuple(result)


def _pgl2_action_matrix(matrix: tuple[int, int, int, int], degree: int, prime: int) -> tuple[tuple[int, ...], ...]:
    a, b, c, d = matrix
    columns = []

    powers_ax_bz = [_linear_power_coefficients(a, b, exponent, prime) for exponent in range(degree + 1)]
    powers_cx_dz = [_linear_power_coefficients(c, d, exponent, prime) for exponent in range(degree + 1)]

    for i in range(degree + 1):
        column = [0] * (degree + 1)
        left = powers_ax_bz[i]
        right = powers_cx_dz[degree - i]
        for left_x_degree, left_coefficient in enumerate(left):
            if left_coefficient == 0:
                continue
            for right_x_degree, right_coefficient in enumerate(right):
                if right_coefficient == 0:
                    continue
                x_degree = left_x_degree + right_x_degree
                column[x_degree] = (column[x_degree] + left_coefficient * right_coefficient) % prime
        columns.append(tuple(column))

    return tuple(columns)


def _apply_binary_form_action_matrix(binary_form: tuple[int, ...], action_matrix: tuple[tuple[int, ...], ...], prime: int) -> tuple[int, ...]:
    result = [0] * len(binary_form)
    for coefficient, column in zip(binary_form, action_matrix):
        if coefficient == 0:
            continue
        for j, matrix_entry in enumerate(column):
            if matrix_entry != 0:
                result[j] = (result[j] + coefficient * matrix_entry) % prime
    return tuple(result)


def _normalize_binary_form_up_to_square_scalar(binary_form: tuple[int, ...], prime: int) -> tuple[int, ...]:
    if all(coefficient == 0 for coefficient in binary_form):
        raise ValueError("zero binary form cannot be normalized")

    square_scalars = sorted({value * value % prime for value in range(1, prime)})
    return min(tuple(scalar * coefficient % prime for coefficient in binary_form) for scalar in square_scalars)


def _pack_int_tuple(values: tuple[int, ...] | list[int]) -> str:
    return json.dumps(list(values), separators=(",", ":"))


def _unpack_int_tuple(text: str | bytes) -> tuple[int, ...]:
    if isinstance(text, bytes):
        text = text.decode("ascii")
    return tuple(json.loads(text))


def _parse_max_sparsity_from_sqlite_path(path: Optional[Path]) -> Optional[int]:
    if path is None:
        return None
    stem = path.stem
    marker = "_s_"
    if marker not in stem:
        marker = "_sparsity"
        if marker not in stem:
            return None
    suffix = stem.rsplit(marker, 1)[1]
    digits = []
    for character in suffix:
        if not character.isdigit():
            break
        digits.append(character)
    return int("".join(digits)) if digits else None


@dataclass(frozen=True)
class Polynomial:
    field: PrimeField
    coefficients: tuple[int, ...]

    def __init__(self, field: PrimeField, coefficients: list[int] | tuple[int, ...]) -> None:
        if not coefficients:
            coefficients = [0]
        if any(coefficient < 0 or coefficient >= field.prime for coefficient in coefficients):
            raise ValueError(f"polynomial coefficients must be in 0..{field.prime - 1}")
        object.__setattr__(self, "field", field)
        object.__setattr__(self, "coefficients", tuple(_trim(list(coefficients))))

    def coefficient(self, index: int) -> int:
        return self.coefficients[index] if index < len(self.coefficients) else 0

    @property
    def degree(self) -> int:
        return len(self.coefficients) - 1

    def is_zero(self) -> bool:
        return self.coefficients == (0,)

    def is_monic(self) -> bool:
        return not self.is_zero() and self.coefficients[-1] == 1

    def derivative(self) -> Polynomial:
        if self.degree == 0:
            return Polynomial(self.field, [0])
        return Polynomial(
            self.field,
            [self.field.normalize(i * self.coefficients[i]) for i in range(1, len(self.coefficients))],
        )

    def remainder(self, divisor: Polynomial) -> Polynomial:
        self._require_same_field(divisor)
        if divisor.is_zero():
            raise ValueError("division by zero polynomial")

        rem = list(self.coefficients)
        divisor_degree = divisor.degree
        divisor_leading_inverse = self.field.inverse(divisor.coefficients[-1])

        while len(rem) - 1 >= divisor_degree and rem != [0]:
            shift = len(rem) - 1 - divisor_degree
            scale = self.field.normalize(rem[-1] * divisor_leading_inverse)
            for i in range(divisor_degree + 1):
                rem[shift + i] = self.field.normalize(rem[shift + i] - scale * divisor.coefficient(i))
            _trim(rem)

        return Polynomial(self.field, rem)

    def monic(self) -> Polynomial:
        if self.is_zero():
            return self
        leading_inverse = self.field.inverse(self.coefficients[-1])
        return Polynomial(self.field, [self.field.normalize(coefficient * leading_inverse) for coefficient in self.coefficients])

    def gcd(self, other: Polynomial) -> Polynomial:
        self._require_same_field(other)
        current = self
        while not other.is_zero():
            current, other = other, current.remainder(other)
        return current.monic()

    def is_squarefree(self) -> bool:
        return not self.is_zero() and self.gcd(self.derivative()).degree == 0

    def _require_same_field(self, other: Polynomial) -> None:
        if self.field.prime != other.field.prime:
            raise ValueError("polynomials are over different fields")


class FiniteExtension:
    def __init__(self, prime: int, degree: int) -> None:
        if degree < 1:
            raise ValueError("extension degree must be positive")
        self.prime = prime
        self.degree = degree
        self.modulus = self._find_irreducible_polynomial(prime, degree)
        self._elements: Optional[tuple[tuple[int, ...], ...]] = None
        self._squares: Optional[set[tuple[int, ...]]] = None
        self._point_contributions: Optional[dict[tuple[int, ...], int]] = None

    @property
    def size(self) -> int:
        return self.prime**self.degree

    def zero(self) -> tuple[int, ...]:
        return (0,) * self.degree

    def one(self) -> tuple[int, ...]:
        return (1,) + (0,) * (self.degree - 1)

    def constant(self, value: int) -> tuple[int, ...]:
        return (value % self.prime,) + (0,) * (self.degree - 1)

    def elements(self) -> tuple[tuple[int, ...], ...]:
        if self._elements is None:
            self._elements = tuple(product(range(self.prime), repeat=self.degree))
        return self._elements

    def is_zero(self, element: tuple[int, ...]) -> bool:
        return all(coefficient == 0 for coefficient in element)

    def is_one(self, element: tuple[int, ...]) -> bool:
        return element == self.one()

    def is_square(self, element: tuple[int, ...]) -> bool:
        return element in self.squares()

    def squares(self) -> set[tuple[int, ...]]:
        if self._squares is None:
            self._squares = {self.multiply(element, element) for element in self.elements()}
        return self._squares

    def point_contributions(self) -> dict[tuple[int, ...], int]:
        if self._point_contributions is None:
            zero = self.zero()
            squares = self.squares()
            self._point_contributions = {
                element: 1 if element == zero else 2 if element in squares else 0
                for element in self.elements()
            }
        return self._point_contributions

    def add(self, lhs: tuple[int, ...], rhs: tuple[int, ...]) -> tuple[int, ...]:
        return tuple((a + b) % self.prime for a, b in zip(lhs, rhs))

    def multiply(self, lhs: tuple[int, ...], rhs: tuple[int, ...]) -> tuple[int, ...]:
        product_coefficients = [0] * (2 * self.degree - 1)
        for i, lhs_coefficient in enumerate(lhs):
            for j, rhs_coefficient in enumerate(rhs):
                product_coefficients[i + j] = (product_coefficients[i + j] + lhs_coefficient * rhs_coefficient) % self.prime

        for d in range(len(product_coefficients) - 1, self.degree - 1, -1):
            coefficient = product_coefficients[d]
            if coefficient == 0:
                continue
            for j in range(self.degree):
                index = d - self.degree + j
                product_coefficients[index] = (product_coefficients[index] - coefficient * self.modulus[j]) % self.prime

        return tuple(product_coefficients[: self.degree])

    def pow(self, base: tuple[int, ...], exponent: int) -> tuple[int, ...]:
        result = self.one()
        while exponent > 0:
            if exponent % 2 == 1:
                result = self.multiply(result, base)
            base = self.multiply(base, base)
            exponent //= 2
        return result

    @staticmethod
    def _find_irreducible_polynomial(prime: int, degree: int) -> tuple[int, ...]:
        if degree == 1:
            return (0, 1)

        field = PrimeField(prime)
        for low_coefficients in product(range(prime), repeat=degree):
            coefficients = (*low_coefficients, 1)
            polynomial = Polynomial(field, coefficients)
            if _is_irreducible(polynomial):
                return coefficients

        raise RuntimeError("failed to find irreducible polynomial for finite extension")


def _is_irreducible(polynomial: Polynomial) -> bool:
    if polynomial.degree <= 0:
        return False
    field = polynomial.field
    for divisor_degree in range(1, polynomial.degree // 2 + 1):
        for low_coefficients in product(range(field.prime), repeat=divisor_degree):
            divisor = Polynomial(field, (*low_coefficients, 1))
            if polynomial.remainder(divisor).is_zero():
                return False
    return True


class PointCountingContext:
    def __init__(self, field: PrimeField, polynomial_degree: int) -> None:
        if polynomial_degree < 0:
            raise ValueError("polynomial degree must be nonnegative")
        self.field = field
        self.polynomial_degree = polynomial_degree
        self._extension_cache: dict[int, FiniteExtension] = {}
        self._power_cache: dict[int, tuple[tuple[tuple[int, ...], tuple[tuple[int, ...], ...]], ...]] = {}
        self._ground_powers: Optional[tuple[tuple[int, tuple[int, ...]], ...]] = None
        self._ground_value_table: Optional[tuple[tuple[tuple[int, ...], ...], ...]] = None
        self._ground_quadratic_residues: Optional[set[int]] = None
        self._ground_point_contributions: Optional[tuple[int, ...]] = None

    def require_compatible_polynomial(self, polynomial: Polynomial) -> None:
        if polynomial.field.prime != self.field.prime:
            raise ValueError("point-counting context is over a different field")
        if polynomial.degree > self.polynomial_degree:
            raise ValueError("point-counting context polynomial degree is too small")

    def extension(self, extension_degree: int) -> FiniteExtension:
        if extension_degree < 1:
            raise ValueError("extension degree must be positive")
        extension = self._extension_cache.get(extension_degree)
        if extension is None:
            extension = FiniteExtension(self.field.prime, extension_degree)
            self._extension_cache[extension_degree] = extension
        return extension

    def extension_powers(self, extension_degree: int) -> tuple[tuple[tuple[int, ...], tuple[tuple[int, ...], ...]], ...]:
        cached = self._power_cache.get(extension_degree)
        if cached is not None:
            return cached

        extension = self.extension(extension_degree)
        one = extension.one()
        rows = []
        for x in extension.elements():
            powers = [one]
            for _ in range(self.polynomial_degree):
                powers.append(extension.multiply(powers[-1], x))
            rows.append((x, tuple(powers)))

        cached = tuple(rows)
        self._power_cache[extension_degree] = cached
        return cached

    def ground_powers(self) -> tuple[tuple[int, tuple[int, ...]], ...]:
        if self._ground_powers is not None:
            return self._ground_powers

        rows = []
        for x in range(self.field.prime):
            powers = [1]
            for _ in range(self.polynomial_degree):
                powers.append((powers[-1] * x) % self.field.prime)
            rows.append((x, tuple(powers)))

        self._ground_powers = tuple(rows)
        return self._ground_powers

    def ground_value_table(self) -> tuple[tuple[tuple[int, ...], ...], ...]:
        if self._ground_value_table is not None:
            return self._ground_value_table

        powers_by_x = self.ground_powers()
        table = []
        for degree in range(self.polynomial_degree + 1):
            values_by_coefficient = []
            for coefficient in range(self.field.prime):
                values_by_coefficient.append(
                    tuple(coefficient * powers[degree] % self.field.prime for _, powers in powers_by_x)
                )
            table.append(tuple(values_by_coefficient))

        self._ground_value_table = tuple(table)
        return self._ground_value_table

    def ground_quadratic_residues(self) -> set[int]:
        if self._ground_quadratic_residues is None:
            self._ground_quadratic_residues = {y * y % self.field.prime for y in range(self.field.prime)}
        return self._ground_quadratic_residues

    def ground_point_contributions(self) -> tuple[int, ...]:
        if self._ground_point_contributions is None:
            contributions = [0] * self.field.prime
            contributions[0] = 1
            for square in self.ground_quadratic_residues():
                if square != 0:
                    contributions[square] = 2
            self._ground_point_contributions = tuple(contributions)
        return self._ground_point_contributions


@dataclass(frozen=True)
class HyperellipticCurve:
    defining_polynomial: Polynomial
    point_counting_context: Optional[PointCountingContext] = field(default=None, repr=False, compare=False)
    _hasse_witt_matrix: Optional[tuple[tuple[int, ...], ...]] = field(default=None, init=False, repr=False, compare=False)
    _l_polynomial_coefficients_mod_p: Optional[tuple[int, ...]] = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._validate_model()
        if self.point_counting_context is None:
            object.__setattr__(self, "point_counting_context", PointCountingContext(self.field, self.degree))
        self.point_counting_context.require_compatible_polynomial(self.defining_polynomial)

    @property
    def field(self) -> PrimeField:
        return self.defining_polynomial.field

    @property
    def degree(self) -> int:
        return self.defining_polynomial.degree

    @property
    def genus(self) -> int:
        return (self.degree - 1) // 2

    @property
    def degree_model(self) -> str:
        return "odd" if self.degree % 2 == 1 else "even"

    def is_monic_model(self) -> bool:
        return self.defining_polynomial.is_monic()

    def point_count_over_extension(self, extension_degree: int) -> int:
        if extension_degree < 1:
            raise ValueError("extension degree must be positive")
        if extension_degree == 1:
            return self.point_count_over_ground_field()

        extension = self.point_counting_context.extension(extension_degree)
        count = self._points_at_infinity(extension)

        point_contributions = extension.point_contributions()
        for x, powers in self.point_counting_context.extension_powers(extension_degree):
            rhs = self._evaluate_defining_polynomial_from_powers(extension, powers)
            count += point_contributions[rhs]

        return count

    def point_count_over_ground_field(self) -> int:
        p = self.field.prime
        count = 1 if self.degree_model == "odd" else self._points_at_infinity_over_ground_field()
        point_contributions = self.point_counting_context.ground_point_contributions()
        coefficients = self.defining_polynomial.coefficients
        value_table = self.point_counting_context.ground_value_table()

        for x_index in range(p):
            value = sum(value_table[degree][coefficient][x_index] for degree, coefficient in enumerate(coefficients)) % p
            count += point_contributions[value]

        return count

    def l_polynomial_coefficients(self) -> list[int]:
        coefficients = self._compute_l_polynomial_coefficients(max_sparsity=None)
        if coefficients is None:
            raise RuntimeError("unlimited L-polynomial computation unexpectedly failed")
        return coefficients

    def l_polynomial_coefficients_with_sparsity_limit(self, max_sparsity: int) -> Optional[list[int]]:
        if max_sparsity < 0:
            raise ValueError("max sparsity must be nonnegative")
        if not self.passes_hasse_witt_sparsity_filter(max_sparsity):
            return None
        return self._compute_l_polynomial_coefficients(max_sparsity=max_sparsity)

    def hasse_witt_matrix(self) -> tuple[tuple[int, ...], ...]:
        if self._hasse_witt_matrix is not None:
            return self._hasse_witt_matrix

        p = self.field.prime
        h = _poly_pow_mod(self.defining_polynomial.coefficients, (p - 1) // 2, p)
        rows = []
        for i in range(1, self.genus + 1):
            row = []
            for j in range(1, self.genus + 1):
                coefficient_index = p * i - j
                row.append(h[coefficient_index] if coefficient_index < len(h) else 0)
            rows.append(tuple(row))

        matrix = tuple(rows)
        object.__setattr__(self, "_hasse_witt_matrix", matrix)
        return matrix

    def l_polynomial_coefficients_mod_p(self) -> list[int]:
        if self._l_polynomial_coefficients_mod_p is not None:
            return list(self._l_polynomial_coefficients_mod_p)

        p = self.field.prime
        matrix = self.hasse_witt_matrix()
        polynomial_matrix = []
        for i, row in enumerate(matrix):
            polynomial_row = []
            for j, entry in enumerate(row):
                if i == j:
                    polynomial_row.append(_poly_trim_mod([1, -entry], p))
                else:
                    polynomial_row.append(_poly_trim_mod([0, -entry], p))
            polynomial_matrix.append(polynomial_row)

        determinant = _determinant_polynomial_mod(polynomial_matrix, p)
        coefficients = tuple(determinant[i] if i < len(determinant) else 0 for i in range(1, self.genus + 1))
        object.__setattr__(self, "_l_polynomial_coefficients_mod_p", coefficients)
        return list(coefficients)

    def passes_hasse_witt_sparsity_filter(self, max_sparsity: int) -> bool:
        if max_sparsity < 0:
            raise ValueError("max sparsity must be nonnegative")
        sparsity_mod_p = sum(1 for coefficient in self.l_polynomial_coefficients_mod_p()[:-1] if coefficient != 0)
        return sparsity_mod_p <= max_sparsity

    def _compute_l_polynomial_coefficients(self, max_sparsity: Optional[int]) -> Optional[list[int]]:
        power_sums = [0] * (self.genus + 1)
        coefficients = [0] * (self.genus + 1)
        coefficients[0] = 1
        sparsity = 0
        q = 1

        for k in range(1, self.genus + 1):
            q *= self.field.prime
            power_sums[k] = q + 1 - self.point_count_over_extension(k)
            total = sum(coefficients[k - i] * power_sums[i] for i in range(1, k + 1))
            if total % k != 0:
                raise RuntimeError("Newton identity produced a nonintegral coefficient")
            coefficients[k] = -total // k

            if k < self.genus and coefficients[k] != 0:
                sparsity += 1
                if max_sparsity is not None and sparsity > max_sparsity:
                    return None

        return coefficients[1:]

    def _evaluate_defining_polynomial(self, extension: FiniteExtension, x: tuple[int, ...]) -> tuple[int, ...]:
        result = extension.zero()
        for i in range(self.degree, -1, -1):
            result = extension.add(
                extension.multiply(result, x),
                extension.constant(self.defining_polynomial.coefficient(i)),
            )
        return result

    def _evaluate_defining_polynomial_from_powers(
        self,
        extension: FiniteExtension,
        powers: tuple[tuple[int, ...], ...],
    ) -> tuple[int, ...]:
        result = [0] * extension.degree
        for coefficient, power in zip(self.defining_polynomial.coefficients, powers):
            if coefficient == 0:
                continue
            for i, power_coefficient in enumerate(power):
                result[i] = (result[i] + coefficient * power_coefficient) % extension.prime
        return tuple(result)

    def _points_at_infinity(self, extension: FiniteExtension) -> int:
        if self.degree_model == "odd":
            return 1

        leading_coefficient = extension.constant(self.defining_polynomial.coefficient(self.degree))
        return 2 if extension.is_square(leading_coefficient) else 0

    def _points_at_infinity_over_ground_field(self) -> int:
        leading_coefficient = self.defining_polynomial.coefficient(self.degree)
        return 2 if leading_coefficient in self.point_counting_context.ground_quadratic_residues() else 0

    def _validate_model(self) -> None:
        if self.defining_polynomial.is_zero():
            raise ValueError("defining polynomial must be nonzero")
        if self.degree < 3:
            raise ValueError("defining polynomial degree must be at least 3")
        if not self.defining_polynomial.is_squarefree():
            raise ValueError("defining polynomial must be squarefree")


class EnumerationContext:
    def __init__(
        self,
        prime: int,
        genus: int,
        sqlite_path: str | Path | None = None,
        search_mode: str = "exhaustive-search",
        hasse_witt_before_canonicalization: Optional[bool] = None,
    ) -> None:
        if genus < 1:
            raise ValueError("genus must be positive")
        if search_mode not in {"exhaustive-search", "sparse-search"}:
            raise ValueError("search mode must be 'exhaustive-search' or 'sparse-search'")
        if hasse_witt_before_canonicalization is not None:
            search_mode = "sparse-search" if hasse_witt_before_canonicalization else search_mode
        self._created_at = perf_counter()
        self._processed_polynomials = 0
        self._processing_seconds = 0.0
        self._sqlite_load_seconds = 0.0
        self._sqlite_write_seconds = 0.0
        self._status_counts: dict[str, int] = {}
        self.field = PrimeField(prime)
        self.genus = genus
        self.binary_degree = 2 * genus + 2
        self.search_mode = search_mode
        self.point_counting_context = PointCountingContext(self.field, self.binary_degree)
        self.pgl2 = _precompute_pgl2(prime)
        self.pgl2_action_matrices = tuple(_pgl2_action_matrix(matrix, self.binary_degree, prime) for matrix in self.pgl2)
        self.seen_keys: set[tuple[int, ...]] = set()
        self.canonical_key_cache: dict[tuple[int, ...], tuple[int, ...]] = {}
        self.canonical_records: dict[tuple[int, ...], CanonicalRecord] = {}
        self.index_by_rational_branch_count: dict[int, set[tuple[int, ...]]] = {}
        self.l_polynomial_mod_p_cache: dict[tuple[int, ...], list[int]] = {}
        self.exact_l_polynomial_cache: dict[tuple[tuple[int, ...], Optional[int]], Optional[list[int]]] = {}
        self._irreducible_factor_cache: dict[int, tuple[tuple[int, ...], ...]] = {}
        self.sqlite_path = Path(sqlite_path) if sqlite_path is not None else None
        self._sqlite_max_sparsity: Optional[int] = _parse_max_sparsity_from_sqlite_path(self.sqlite_path)
        self.sqlite_connection: Optional[sqlite3.Connection] = None
        if self.sqlite_path is not None:
            self.open_sqlite(self.sqlite_path)

    def open_sqlite(self, sqlite_path: str | Path) -> None:
        self.sqlite_path = Path(sqlite_path)
        self._sqlite_max_sparsity = _parse_max_sparsity_from_sqlite_path(self.sqlite_path)
        self.sqlite_connection = sqlite3.connect(self.sqlite_path)
        self.sqlite_connection.execute("PRAGMA journal_mode=WAL")
        self.sqlite_connection.execute("PRAGMA synchronous=NORMAL")
        self._initialize_sqlite_schema()
        started_at = perf_counter()
        self._load_sqlite_records()
        self._sqlite_load_seconds += perf_counter() - started_at

    def close_sqlite(self) -> None:
        if self.sqlite_connection is not None:
            self.sqlite_connection.close()
            self.sqlite_connection = None

    def elapsed_seconds(self) -> float:
        return perf_counter() - self._created_at

    def timing_summary(self) -> dict[str, object]:
        return {
            "elapsed_seconds": self.elapsed_seconds(),
            "processed_polynomials": self._processed_polynomials,
            "processing_seconds": self._processing_seconds,
            "sqlite_load_seconds": self._sqlite_load_seconds,
            "sqlite_write_seconds": self._sqlite_write_seconds,
            "other_seconds": max(
                0.0,
                self.elapsed_seconds()
                - self._processing_seconds
                - self._sqlite_load_seconds
                - self._sqlite_write_seconds,
            ),
            "status_counts": dict(self._status_counts),
        }

    def _initialize_sqlite_schema(self) -> None:
        if self.sqlite_connection is None:
            raise ValueError("sqlite connection is not open")
        self.sqlite_connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS curve_cache (
                canonical_key TEXT NOT NULL,
                rational_branch_count INTEGER NOT NULL,
                coefficients TEXT NOT NULL,
                lpoly_mod_p TEXT,
                exact_lpoly TEXT,
                sparsity INTEGER,
                status TEXT NOT NULL,
                PRIMARY KEY (canonical_key)
            );

            CREATE TABLE IF NOT EXISTS sparse_curves (
                canonical_key TEXT NOT NULL,
                coefficients TEXT NOT NULL,
                lpoly TEXT NOT NULL,
                sparsity INTEGER NOT NULL,
                rational_branch_count INTEGER NOT NULL,
                PRIMARY KEY (canonical_key)
            );

            CREATE TABLE IF NOT EXISTS orbit_cache (
                rational_branch_count INTEGER NOT NULL,
                orbit_key BLOB NOT NULL,
                canonical_key BLOB NOT NULL,
                PRIMARY KEY (rational_branch_count, orbit_key)
            );
            """
        )
        self.sqlite_connection.commit()

    def _load_sqlite_records(self) -> None:
        if self.sqlite_connection is None:
            raise ValueError("sqlite connection is not open")

        columns = {
            row[1]
            for row in self.sqlite_connection.execute("PRAGMA table_info(curve_cache)").fetchall()
        }
        has_hasse_witt_column = "hasse_witt_lpoly_mod_p" in columns
        has_max_sparsity_column = "max_sparsity" in columns
        rows = self.sqlite_connection.execute(
            f"""
            SELECT
                canonical_key,
                rational_branch_count,
                coefficients,
                lpoly_mod_p,
                exact_lpoly,
                sparsity,
                status
                {', max_sparsity' if has_max_sparsity_column else ''}
                {', hasse_witt_lpoly_mod_p' if has_hasse_witt_column else ''}
            FROM curve_cache
            """
        ).fetchall()

        for row in rows:
            canonical_key_text = row[0]
            rational_branch_count = row[1]
            coefficients_text = row[2]
            lpoly_mod_p_text = row[3]
            exact_lpoly_text = row[4]
            sparsity = row[5]
            status = row[6]
            next_index = 7
            max_sparsity = row[next_index] if has_max_sparsity_column else self._sqlite_max_sparsity
            next_index += 1 if has_max_sparsity_column else 0
            hasse_witt_lpoly_mod_p_text = row[next_index] if has_hasse_witt_column else None
            if max_sparsity is None:
                continue
            canonical_key = _unpack_int_tuple(canonical_key_text)
            coefficients = _unpack_int_tuple(coefficients_text)
            hasse_witt_lpoly_mod_p = (
                _unpack_int_tuple(hasse_witt_lpoly_mod_p_text)
                if hasse_witt_lpoly_mod_p_text is not None
                else None
            )
            lpoly_mod_p = _unpack_int_tuple(lpoly_mod_p_text) if lpoly_mod_p_text is not None else None
            exact_lpoly = _unpack_int_tuple(exact_lpoly_text) if exact_lpoly_text is not None else None

            record = self._register_or_update_record(
                canonical_key=canonical_key,
                coefficients=coefficients,
                rational_branch_count=rational_branch_count,
                hasse_witt_lpoly_mod_p=hasse_witt_lpoly_mod_p or lpoly_mod_p,
            )
            record.status_by_max_sparsity[max_sparsity] = status
            record.sparsity_by_max_sparsity[max_sparsity] = sparsity
            record.exact_lpoly_by_max_sparsity[max_sparsity] = exact_lpoly

            self.seen_keys.add(canonical_key)
            self.canonical_key_cache[coefficients] = canonical_key
            if lpoly_mod_p is not None:
                self.l_polynomial_mod_p_cache[canonical_key] = list(lpoly_mod_p)
            if exact_lpoly is not None:
                self.exact_l_polynomial_cache[(canonical_key, max_sparsity)] = list(exact_lpoly)
            elif status in {"rejected_hasse_witt", "rejected_exact"}:
                self.exact_l_polynomial_cache[(canonical_key, max_sparsity)] = None

    def _pack_field_tuple_blob(self, values: tuple[int, ...]) -> bytes:
        if self.field.prime < 256:
            return bytes(values)
        return json.dumps(list(values), separators=(",", ":")).encode("ascii")

    def _unpack_field_tuple_blob(self, data: bytes) -> tuple[int, ...]:
        if self.field.prime < 256:
            return tuple(data)
        return tuple(json.loads(data.decode("ascii")))

    def polynomial(self, coefficients: list[int] | tuple[int, ...]) -> Polynomial:
        return Polynomial(self.field, coefficients)

    def curve(self, polynomial_or_coefficients: Polynomial | list[int] | tuple[int, ...]) -> HyperellipticCurve:
        polynomial = self._coerce_polynomial(polynomial_or_coefficients)
        curve = HyperellipticCurve(polynomial, point_counting_context=self.point_counting_context)
        if curve.genus != self.genus:
            raise ValueError("curve genus does not match enumeration context")
        return curve

    def rational_branch_count(self, polynomial_or_coefficients: Polynomial | list[int] | tuple[int, ...]) -> int:
        polynomial = self._coerce_polynomial(polynomial_or_coefficients)
        self._require_compatible_polynomial(polynomial)
        count = 1 if polynomial.degree == 2 * self.genus + 1 else 0
        for x in range(self.field.prime):
            value = 0
            for coefficient in reversed(polynomial.coefficients):
                value = (value * x + coefficient) % self.field.prime
            if value == 0:
                count += 1
        return count

    def factorization_pattern(self, polynomial_or_coefficients: Polynomial | list[int] | tuple[int, ...]) -> tuple[int, ...]:
        polynomial = self._coerce_polynomial(polynomial_or_coefficients)
        self._require_compatible_polynomial(polynomial)
        remaining = _poly_monic_mod(polynomial.coefficients, self.field.prime)
        pattern = [1] if polynomial.degree == 2 * self.genus + 1 else []
        x_polynomial = (0, 1)
        frobenius_power = x_polynomial
        factor_degree = 1

        while 2 * factor_degree <= len(remaining) - 1:
            frobenius_power = _poly_pow_remainder_mod(frobenius_power, self.field.prime, remaining, self.field.prime)
            degree_factor = _poly_gcd_mod(remaining, _poly_sub_mod(frobenius_power, x_polynomial, self.field.prime), self.field.prime)
            if degree_factor != (1,):
                factor_count = (len(degree_factor) - 1) // factor_degree
                pattern.extend([factor_degree] * factor_count)
                remaining = _poly_exact_quotient_mod(remaining, degree_factor, self.field.prime)
                frobenius_power = _poly_remainder_mod(frobenius_power, remaining, self.field.prime)
            factor_degree += 1

        if len(remaining) > 1:
            pattern.append(len(remaining) - 1)

        return tuple(sorted(pattern))

    def _index_record(self, record: CanonicalRecord) -> None:
        self.index_by_rational_branch_count.setdefault(record.rational_branch_count, set()).add(record.canonical_key)

    def _register_or_update_record(
        self,
        canonical_key: tuple[int, ...],
        coefficients: tuple[int, ...],
        rational_branch_count: int,
        hasse_witt_lpoly_mod_p: Optional[tuple[int, ...]],
    ) -> CanonicalRecord:
        record = self.canonical_records.get(canonical_key)
        if record is None:
            record = CanonicalRecord(
                canonical_key=canonical_key,
                coefficients=coefficients,
                rational_branch_count=rational_branch_count,
                hasse_witt_lpoly_mod_p=hasse_witt_lpoly_mod_p,
            )
            self.canonical_records[canonical_key] = record
            self._index_record(record)
            return record

        if record.hasse_witt_lpoly_mod_p is None and hasse_witt_lpoly_mod_p is not None:
            record.hasse_witt_lpoly_mod_p = hasse_witt_lpoly_mod_p
        return record

    def _result_from_record(self, record: CanonicalRecord, max_sparsity: int) -> Optional[dict[str, object]]:
        status = record.status_by_max_sparsity.get(max_sparsity)
        if status is None:
            return None

        result: dict[str, object] = {
            "status": status,
            "canonical_key": record.canonical_key,
            "max_sparsity": max_sparsity,
            "coefficients": record.coefficients,
            "sparsity": record.sparsity_by_max_sparsity.get(max_sparsity),
        }
        lpoly_mod_p = self.l_polynomial_mod_p_cache.get(record.canonical_key)
        if lpoly_mod_p is not None:
            result["lpoly_mod_p"] = list(lpoly_mod_p)
        exact_lpoly = record.exact_lpoly_by_max_sparsity.get(max_sparsity)
        if exact_lpoly is not None:
            result["lpoly"] = list(exact_lpoly)
        return result

    def _normalized_binary_form_key(self, polynomial: Polynomial) -> tuple[int, ...]:
        binary_form = _affine_polynomial_to_binary_form(polynomial, self.binary_degree)
        return _normalize_binary_form_up_to_square_scalar(binary_form, self.field.prime)

    def _canonical_key_and_orbit(self, polynomial: Polynomial) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]:
        binary_form = _affine_polynomial_to_binary_form(polynomial, self.binary_degree)
        orbit = tuple(
            _normalize_binary_form_up_to_square_scalar(
                _apply_binary_form_action_matrix(binary_form, action_matrix, self.field.prime),
                self.field.prime,
            )
            for action_matrix in self.pgl2_action_matrices
        )
        key = min(orbit)
        self.canonical_key_cache[polynomial.coefficients] = key
        return key, orbit

    def canonical_key(self, polynomial_or_coefficients: Polynomial | list[int] | tuple[int, ...]) -> tuple[int, ...]:
        polynomial = self._coerce_polynomial(polynomial_or_coefficients)
        self._require_compatible_polynomial(polynomial)

        cached = self.canonical_key_cache.get(polynomial.coefficients)
        if cached is not None:
            return cached

        key, _ = self._canonical_key_and_orbit(polynomial)
        return key

    def is_seen(self, polynomial_or_coefficients: Polynomial | list[int] | tuple[int, ...]) -> bool:
        return self.canonical_key(polynomial_or_coefficients) in self.seen_keys

    def mark_seen(self, polynomial_or_coefficients: Polynomial | list[int] | tuple[int, ...]) -> tuple[int, ...]:
        key = self.canonical_key(polynomial_or_coefficients)
        self.seen_keys.add(key)
        return key

    def is_new_isomorphism_class(self, polynomial_or_coefficients: Polynomial | list[int] | tuple[int, ...]) -> bool:
        key = self.canonical_key(polynomial_or_coefficients)
        if key in self.seen_keys:
            return False
        self.seen_keys.add(key)
        return True

    def l_polynomial_coefficients_mod_p(self, polynomial_or_coefficients: Polynomial | list[int] | tuple[int, ...]) -> list[int]:
        polynomial = self._coerce_polynomial(polynomial_or_coefficients)
        key = self.canonical_key(polynomial)
        cached = self.l_polynomial_mod_p_cache.get(key)
        if cached is None:
            cached = self.curve(polynomial).l_polynomial_coefficients_mod_p()
            self.l_polynomial_mod_p_cache[key] = cached
        return list(cached)

    def passes_hasse_witt_sparsity_filter(
        self,
        polynomial_or_coefficients: Polynomial | list[int] | tuple[int, ...],
        max_sparsity: int,
    ) -> bool:
        if max_sparsity < 0:
            raise ValueError("max sparsity must be nonnegative")
        sparsity_mod_p = sum(1 for coefficient in self.l_polynomial_coefficients_mod_p(polynomial_or_coefficients)[:-1] if coefficient != 0)
        return sparsity_mod_p <= max_sparsity

    def l_polynomial_coefficients(self, polynomial_or_coefficients: Polynomial | list[int] | tuple[int, ...]) -> list[int]:
        polynomial = self._coerce_polynomial(polynomial_or_coefficients)
        key = self.canonical_key(polynomial)
        cache_key = (key, None)
        cached = self.exact_l_polynomial_cache.get(cache_key)
        if cached is None:
            cached = self.curve(polynomial).l_polynomial_coefficients()
            self.exact_l_polynomial_cache[cache_key] = cached
        return list(cached)

    def l_polynomial_coefficients_with_sparsity_limit(
        self,
        polynomial_or_coefficients: Polynomial | list[int] | tuple[int, ...],
        max_sparsity: int,
    ) -> Optional[list[int]]:
        if max_sparsity < 0:
            raise ValueError("max sparsity must be nonnegative")

        polynomial = self._coerce_polynomial(polynomial_or_coefficients)
        key = self.canonical_key(polynomial)
        cache_key = (key, max_sparsity)
        if cache_key not in self.exact_l_polynomial_cache:
            if not self.passes_hasse_witt_sparsity_filter(polynomial, max_sparsity):
                self.exact_l_polynomial_cache[cache_key] = None
            else:
                self.exact_l_polynomial_cache[cache_key] = self.curve(polynomial).l_polynomial_coefficients_with_sparsity_limit(max_sparsity)

        cached = self.exact_l_polynomial_cache[cache_key]
        return None if cached is None else list(cached)

    def process_polynomial_for_output(
        self,
        polynomial_or_coefficients: Polynomial | list[int] | tuple[int, ...],
        max_sparsity: int,
    ) -> dict[str, object]:
        started_at = perf_counter()
        try:
            result = self._process_polynomial_for_output(polynomial_or_coefficients, max_sparsity)
        finally:
            self._processing_seconds += perf_counter() - started_at

        self._processed_polynomials += 1
        status = str(result["status"])
        self._status_counts[status] = self._status_counts.get(status, 0) + 1
        return result

    def _process_polynomial_for_output(
        self,
        polynomial_or_coefficients: Polynomial | list[int] | tuple[int, ...],
        max_sparsity: int,
    ) -> dict[str, object]:
        if max_sparsity < 0:
            raise ValueError("max sparsity must be nonnegative")

        polynomial = self._coerce_polynomial(polynomial_or_coefficients)
        self._require_compatible_polynomial(polynomial)
        if not polynomial.is_squarefree():
            return {"status": "singular", "coefficients": polynomial.coefficients}

        precomputed_lpoly_mod_p: Optional[list[int]] = None
        if self.search_mode == "sparse-search":
            precomputed_lpoly_mod_p = self.curve(polynomial).l_polynomial_coefficients_mod_p()
            sparsity_mod_p = sum(1 for coefficient in precomputed_lpoly_mod_p[:-1] if coefficient != 0)
            if sparsity_mod_p > max_sparsity:
                return {
                    "status": "rejected_hasse_witt_uncanonicalized",
                    "coefficients": polynomial.coefficients,
                    "lpoly_mod_p": precomputed_lpoly_mod_p,
                }

        if self.search_mode == "sparse-search":
            rational_branch_count = self.rational_branch_count(polynomial)
            hasse_witt_lpoly_mod_p = tuple(precomputed_lpoly_mod_p) if precomputed_lpoly_mod_p is not None else None
            orbit_key = self._normalized_binary_form_key(polynomial)
            canonical_key = self._lookup_orbit_cache(rational_branch_count, orbit_key)
            if canonical_key is None:
                canonical_key, orbit = self._canonical_key_and_orbit(polynomial)
                self._insert_orbit_cache(rational_branch_count, orbit, canonical_key)
            else:
                self.canonical_key_cache[polynomial.coefficients] = canonical_key
        else:
            rational_branch_count = self.rational_branch_count(polynomial)
            hasse_witt_lpoly_mod_p = None
            orbit_key = self._normalized_binary_form_key(polynomial)
            canonical_key = self._lookup_orbit_cache(rational_branch_count, orbit_key)
            if canonical_key is None:
                canonical_key, orbit = self._canonical_key_and_orbit(polynomial)
                self._insert_orbit_cache(rational_branch_count, orbit, canonical_key)
            else:
                self.canonical_key_cache[polynomial.coefficients] = canonical_key
        existing_record = self.canonical_records.get(canonical_key)
        record = self._register_or_update_record(
            canonical_key=canonical_key,
            coefficients=polynomial.coefficients,
            rational_branch_count=rational_branch_count,
            hasse_witt_lpoly_mod_p=hasse_witt_lpoly_mod_p,
        )

        cached_result = self._result_from_record(record, max_sparsity)
        if cached_result is not None:
            if existing_record is not None:
                cached_result = dict(cached_result)
                cached_result["previous_status"] = cached_result["status"]
                cached_result["status"] = "duplicate"
            return cached_result

        lpoly_mod_p = (
            precomputed_lpoly_mod_p
            if precomputed_lpoly_mod_p is not None
            else list(record.hasse_witt_lpoly_mod_p)
            if record.hasse_witt_lpoly_mod_p is not None
            else self.l_polynomial_coefficients_mod_p(polynomial)
        )
        self.l_polynomial_mod_p_cache[canonical_key] = lpoly_mod_p
        record.hasse_witt_lpoly_mod_p = tuple(lpoly_mod_p)
        self._index_record(record)
        sparsity_mod_p = sum(1 for coefficient in lpoly_mod_p[:-1] if coefficient != 0)
        if sparsity_mod_p > max_sparsity:
            record.status_by_max_sparsity[max_sparsity] = "rejected_hasse_witt"
            record.sparsity_by_max_sparsity[max_sparsity] = None
            record.exact_lpoly_by_max_sparsity[max_sparsity] = None
            self._insert_curve_cache(
                canonical_key=canonical_key,
                rational_branch_count=rational_branch_count,
                coefficients=polynomial.coefficients,
                lpoly_mod_p=lpoly_mod_p,
                exact_lpoly=None,
                sparsity=None,
                status="rejected_hasse_witt",
            )
            return {
                "status": "rejected_hasse_witt",
                "canonical_key": canonical_key,
                "coefficients": polynomial.coefficients,
                "lpoly_mod_p": lpoly_mod_p,
            }

        exact_cache_key = (canonical_key, max_sparsity)
        if exact_cache_key not in self.exact_l_polynomial_cache:
            self.exact_l_polynomial_cache[exact_cache_key] = self.curve(polynomial)._compute_l_polynomial_coefficients(
                max_sparsity=max_sparsity,
            )
        exact_lpoly = self.exact_l_polynomial_cache[exact_cache_key]
        if exact_lpoly is None:
            record.status_by_max_sparsity[max_sparsity] = "rejected_exact"
            record.sparsity_by_max_sparsity[max_sparsity] = None
            record.exact_lpoly_by_max_sparsity[max_sparsity] = None
            self._insert_curve_cache(
                canonical_key=canonical_key,
                rational_branch_count=rational_branch_count,
                coefficients=polynomial.coefficients,
                lpoly_mod_p=lpoly_mod_p,
                exact_lpoly=None,
                sparsity=None,
                status="rejected_exact",
            )
            return {
                "status": "rejected_exact",
                "canonical_key": canonical_key,
                "coefficients": polynomial.coefficients,
                "lpoly_mod_p": lpoly_mod_p,
            }

        sparsity = sum(1 for coefficient in exact_lpoly[:-1] if coefficient != 0)
        record.status_by_max_sparsity[max_sparsity] = "sparse"
        record.sparsity_by_max_sparsity[max_sparsity] = sparsity
        record.exact_lpoly_by_max_sparsity[max_sparsity] = tuple(exact_lpoly)
        self._insert_curve_cache(
            canonical_key=canonical_key,
            rational_branch_count=rational_branch_count,
            coefficients=polynomial.coefficients,
            lpoly_mod_p=lpoly_mod_p,
            exact_lpoly=exact_lpoly,
            sparsity=sparsity,
            status="sparse",
        )
        self._insert_sparse_curve(
            canonical_key=canonical_key,
            coefficients=polynomial.coefficients,
            lpoly=exact_lpoly,
            sparsity=sparsity,
            rational_branch_count=rational_branch_count,
        )
        return {
            "status": "sparse",
            "canonical_key": canonical_key,
            "coefficients": polynomial.coefficients,
            "lpoly_mod_p": lpoly_mod_p,
            "lpoly": exact_lpoly,
            "sparsity": sparsity,
        }

    def process_polynomials_for_output(
        self,
        polynomials_or_coefficients: Iterable[Polynomial | list[int] | tuple[int, ...]],
        max_sparsity: int,
    ) -> dict[str, int]:
        stats: dict[str, int] = {"processed": 0}
        for polynomial_or_coefficients in polynomials_or_coefficients:
            result = self.process_polynomial_for_output(polynomial_or_coefficients, max_sparsity)
            status = str(result["status"])
            stats["processed"] += 1
            stats[status] = stats.get(status, 0) + 1
        return stats

    def _coerce_polynomial(self, polynomial_or_coefficients: Polynomial | list[int] | tuple[int, ...]) -> Polynomial:
        if isinstance(polynomial_or_coefficients, Polynomial):
            return polynomial_or_coefficients
        return self.polynomial(polynomial_or_coefficients)

    def _require_compatible_polynomial(self, polynomial: Polynomial) -> None:
        if polynomial.field.prime != self.field.prime:
            raise ValueError("polynomial is over a different field")
        if polynomial.degree not in {2 * self.genus + 1, 2 * self.genus + 2}:
            raise ValueError("polynomial degree does not match enumeration genus")

    def _irreducible_polynomials(self, degree: int) -> tuple[tuple[int, ...], ...]:
        cached = self._irreducible_factor_cache.get(degree)
        if cached is not None:
            return cached

        factors = []
        for coefficients in product(range(self.field.prime), repeat=degree):
            polynomial = Polynomial(self.field, (*coefficients, 1))
            if _is_irreducible(polynomial):
                factors.append(polynomial.coefficients)

        cached = tuple(factors)
        self._irreducible_factor_cache[degree] = cached
        return cached

    def _lookup_orbit_cache(self, rational_branch_count: int, orbit_key: tuple[int, ...]) -> Optional[tuple[int, ...]]:
        if self.sqlite_connection is None:
            return None

        row = self.sqlite_connection.execute(
            """
            SELECT canonical_key
            FROM orbit_cache
            WHERE rational_branch_count = ?
              AND orbit_key = ?
            """,
            (rational_branch_count, self._pack_field_tuple_blob(orbit_key)),
        ).fetchone()
        return None if row is None else self._unpack_field_tuple_blob(row[0])

    def _insert_orbit_cache(
        self,
        rational_branch_count: int,
        orbit: tuple[tuple[int, ...], ...],
        canonical_key: tuple[int, ...],
    ) -> None:
        if self.sqlite_connection is None:
            return

        started_at = perf_counter()
        packed_canonical_key = self._pack_field_tuple_blob(canonical_key)
        rows = [
            (rational_branch_count, self._pack_field_tuple_blob(orbit_key), packed_canonical_key)
            for orbit_key in set(orbit)
        ]
        self.sqlite_connection.executemany(
            """
            INSERT OR IGNORE INTO orbit_cache (
                rational_branch_count,
                orbit_key,
                canonical_key
            )
            VALUES (?, ?, ?)
            """,
            rows,
        )
        self.sqlite_connection.commit()
        self._sqlite_write_seconds += perf_counter() - started_at

    def _insert_curve_cache(
        self,
        canonical_key: tuple[int, ...],
        rational_branch_count: int,
        coefficients: tuple[int, ...],
        lpoly_mod_p: Optional[list[int]],
        exact_lpoly: Optional[list[int]],
        sparsity: Optional[int],
        status: str,
    ) -> None:
        if self.sqlite_connection is None:
            return

        started_at = perf_counter()
        self.sqlite_connection.execute(
            """
            INSERT OR REPLACE INTO curve_cache (
                canonical_key,
                rational_branch_count,
                coefficients,
                lpoly_mod_p,
                exact_lpoly,
                sparsity,
                status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _pack_int_tuple(canonical_key),
                rational_branch_count,
                _pack_int_tuple(coefficients),
                _pack_int_tuple(lpoly_mod_p) if lpoly_mod_p is not None else None,
                _pack_int_tuple(exact_lpoly) if exact_lpoly is not None else None,
                sparsity,
                status,
            ),
        )
        self.sqlite_connection.commit()
        self._sqlite_write_seconds += perf_counter() - started_at

    def _insert_sparse_curve(
        self,
        canonical_key: tuple[int, ...],
        coefficients: tuple[int, ...],
        lpoly: list[int],
        sparsity: int,
        rational_branch_count: int,
    ) -> None:
        if self.sqlite_connection is None:
            return

        started_at = perf_counter()
        self.sqlite_connection.execute(
            """
            INSERT OR REPLACE INTO sparse_curves (
                canonical_key,
                coefficients,
                lpoly,
                sparsity,
                rational_branch_count
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                _pack_int_tuple(canonical_key),
                _pack_int_tuple(coefficients),
                _pack_int_tuple(lpoly),
                sparsity,
                rational_branch_count,
            ),
        )
        self.sqlite_connection.commit()
        self._sqlite_write_seconds += perf_counter() - started_at


def coefficient_vectors(
    prime: int,
    genus: int,
    degree_model: str,
    monic: bool,
    limit: Optional[int],
) -> Iterable[tuple[int, ...]]:
    degrees = []
    if degree_model in {"odd", "both"}:
        degrees.append(2 * genus + 1)
    if degree_model in {"even", "both"}:
        degrees.append(2 * genus + 2)

    produced = 0
    for degree in degrees:
        leading_coefficients = (1,) if monic else range(1, prime)
        for coefficients in product(range(prime), repeat=degree):
            for leading_coefficient in leading_coefficients:
                yield (*coefficients, leading_coefficient)
                produced += 1
                if limit is not None and produced >= limit:
                    return


def default_sqlite_path(prime: int, genus: int, max_sparsity: int) -> Path:
    return Path("data_gen") / "results" / f"p{prime}_g{genus}_s_{max_sparsity}.sqlite"


def total_coefficient_vectors(prime: int, genus: int, degree_model: str, monic: bool, limit: Optional[int]) -> int:
    total = 0
    leading_count = 1 if monic else prime - 1
    if degree_model in {"odd", "both"}:
        total += prime ** (2 * genus + 1) * leading_count
    if degree_model in {"even", "both"}:
        total += prime ** (2 * genus + 2) * leading_count
    return min(total, limit) if limit is not None else total


def sparse_isomorphism_classes(context: EnumerationContext, max_sparsity: int) -> int:
    return sum(
        1
        for record in context.canonical_records.values()
        if record.status_by_max_sparsity.get(max_sparsity) == "sparse"
    )


def should_print_progress(processed: int, total: int, interval: int) -> bool:
    if processed == total:
        return True
    if interval <= 0:
        return False
    return processed % interval == 0


def progress_line(
    processed: int,
    total: int,
    sparse_presentations: int,
    context: EnumerationContext,
    max_sparsity: int,
) -> str:
    fields = [
        f"progress: {processed}/{total}",
        f"sparse_presentations={sparse_presentations}",
        f"sparse_isomorphism_classes={sparse_isomorphism_classes(context, max_sparsity)}",
    ]
    if context.search_mode == "exhaustive-search":
        fields.append(f"total_isomorphism_classes={len(context.canonical_records)}")
    else:
        fields.append(f"canonicalized_isomorphism_classes={len(context.canonical_records)}")
    return " ".join(fields)


def parse_enumeration_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enumerate hyperelliptic curves and write sparse L-polynomial data to SQLite.")
    parser.add_argument("--p", type=int, required=True, help="Odd prime field characteristic.")
    parser.add_argument("--genus", "-g", type=int, required=True, help="Curve genus.")
    parser.add_argument("--max-sparsity", type=int, required=True, help="Maximum allowed sparsity among a_1, ..., a_{g-1}.")
    parser.add_argument(
        "--degree-model",
        choices=("odd", "even", "both"),
        default="both",
        help="Enumerate degree 2g+1 models, degree 2g+2 models, or both.",
    )
    parser.add_argument("--monic", action="store_true", help="Only enumerate monic models.")
    parser.add_argument("--out", type=Path, help="SQLite output path. Defaults to data_gen/results/p{p}_g{g}_s_{max}.sqlite.")
    parser.add_argument("--limit", type=int, help="Optional maximum number of raw coefficient vectors to process.")
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=1000,
        help="Print progress every N raw coefficient vectors. Use 0 to print only final output.",
    )
    parser.add_argument(
        "--mode",
        choices=("exhaustive-search", "sparse-search"),
        default="exhaustive-search",
        help="exhaustive-search uses SQLite orbit lookup; sparse-search applies Hasse-Witt before canonicalization.",
    )
    return parser.parse_args()


def run_enumeration_from_args(args: argparse.Namespace) -> None:
    sqlite_path = args.out if args.out is not None else default_sqlite_path(args.p, args.genus, args.max_sparsity)
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)

    context = EnumerationContext(
        prime=args.p,
        genus=args.genus,
        sqlite_path=sqlite_path,
        search_mode=args.mode,
    )
    try:
        total = total_coefficient_vectors(
            prime=args.p,
            genus=args.genus,
            degree_model=args.degree_model,
            monic=args.monic,
            limit=args.limit,
        )
        vectors = coefficient_vectors(
            prime=args.p,
            genus=args.genus,
            degree_model=args.degree_model,
            monic=args.monic,
            limit=args.limit,
        )

        stats: dict[str, int] = {"processed": 0}
        sparse_presentations = 0
        print(progress_line(0, total, 0, context, args.max_sparsity), flush=True)
        for vector in vectors:
            result = context.process_polynomial_for_output(vector, max_sparsity=args.max_sparsity)
            status = str(result["status"])
            stats["processed"] += 1
            stats[status] = stats.get(status, 0) + 1

            if status == "sparse" or (status == "duplicate" and result.get("previous_status") == "sparse"):
                sparse_presentations += 1

            if should_print_progress(stats["processed"], total, args.progress_interval):
                print(
                    progress_line(stats["processed"], total, sparse_presentations, context, args.max_sparsity),
                    flush=True,
                )

        timing = context.timing_summary()
    finally:
        context.close_sqlite()

    print(f"output: {sqlite_path}")
    print(f"stats: {stats}")
    print(f"timing: {timing}")


def main() -> None:
    run_enumeration_from_args(parse_enumeration_args())


if __name__ == "__main__":
    main()
