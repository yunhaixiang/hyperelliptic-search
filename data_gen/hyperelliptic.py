from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Optional


def _is_odd_prime(value: int) -> bool:
    if value < 3 or value % 2 == 0:
        return False
    divisor = 3
    while divisor * divisor <= value:
        if value % divisor == 0:
            return False
        divisor += 2
    return True


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

    @property
    def size(self) -> int:
        return self.prime**self.degree

    def zero(self) -> tuple[int, ...]:
        return (0,) * self.degree

    def one(self) -> tuple[int, ...]:
        return (1,) + (0,) * (self.degree - 1)

    def constant(self, value: int) -> tuple[int, ...]:
        return (value % self.prime,) + (0,) * (self.degree - 1)

    def elements(self):
        for coefficients in product(range(self.prime), repeat=self.degree):
            yield coefficients

    def is_zero(self, element: tuple[int, ...]) -> bool:
        return all(coefficient == 0 for coefficient in element)

    def is_one(self, element: tuple[int, ...]) -> bool:
        return element == self.one()

    def is_square(self, element: tuple[int, ...]) -> bool:
        if self.is_zero(element):
            return True
        return self.is_one(self.pow(element, (self.size - 1) // 2))

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


@dataclass(frozen=True)
class HyperellipticCurve:
    defining_polynomial: Polynomial

    def __post_init__(self) -> None:
        self._validate_model()

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
        extension = FiniteExtension(self.field.prime, extension_degree)
        count = self._points_at_infinity(extension)

        for x in extension.elements():
            rhs = self._evaluate_defining_polynomial(extension, x)
            if extension.is_zero(rhs):
                count += 1
            elif extension.is_square(rhs):
                count += 2

        return count

    def l_polynomial_coefficients(self) -> list[int]:
        coefficients = self._compute_l_polynomial_coefficients(max_sparsity=None)
        if coefficients is None:
            raise RuntimeError("unlimited L-polynomial computation unexpectedly failed")
        return coefficients

    def l_polynomial_coefficients_with_sparsity_limit(self, max_sparsity: int) -> Optional[list[int]]:
        if max_sparsity < 0:
            raise ValueError("max sparsity must be nonnegative")
        return self._compute_l_polynomial_coefficients(max_sparsity=max_sparsity)

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

    def _points_at_infinity(self, extension: FiniteExtension) -> int:
        if self.degree_model == "odd":
            return 1

        leading_coefficient = extension.constant(self.defining_polynomial.coefficient(self.degree))
        return 2 if extension.is_square(leading_coefficient) else 0

    def _validate_model(self) -> None:
        if self.defining_polynomial.is_zero():
            raise ValueError("defining polynomial must be nonzero")
        if self.degree < 3:
            raise ValueError("defining polynomial degree must be at least 3")
        if not self.defining_polynomial.is_squarefree():
            raise ValueError("defining polynomial must be squarefree")
