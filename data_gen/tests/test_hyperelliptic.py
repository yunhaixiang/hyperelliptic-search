import unittest

from data_gen.hyperelliptic import HyperellipticCurve, Polynomial, PrimeField


class HyperellipticTests(unittest.TestCase):
    def test_polynomial_coefficients_must_be_canonical(self):
        field = PrimeField(5)
        with self.assertRaises(ValueError):
            Polynomial(field, [1, 5])
        with self.assertRaises(ValueError):
            Polynomial(field, [-1, 1])

    def test_rejects_repeated_root_model(self):
        field = PrimeField(5)
        with self.assertRaises(ValueError):
            HyperellipticCurve(Polynomial(field, [1, 0, 0, 0, 0, 1]))

    def test_genus_one_l_polynomial_coefficient(self):
        field = PrimeField(5)
        curve = HyperellipticCurve(Polynomial(field, [1, 1, 0, 1]))

        self.assertEqual(curve.point_count_over_extension(1), 9)
        self.assertEqual(curve.l_polynomial_coefficients(), [3])
        self.assertEqual(curve.l_polynomial_coefficients_with_sparsity_limit(0), [3])

    def test_genus_two_l_polynomial_coefficients(self):
        field = PrimeField(5)
        curve = HyperellipticCurve(Polynomial(field, [1, 1, 0, 0, 0, 1]))

        self.assertEqual(curve.point_count_over_extension(1), 6)
        self.assertEqual(curve.point_count_over_extension(2), 46)
        self.assertEqual(curve.l_polynomial_coefficients(), [0, 10])
        self.assertEqual(curve.l_polynomial_coefficients_with_sparsity_limit(0), [0, 10])

    def test_sparsity_limit_stops_when_exceeded(self):
        field = PrimeField(5)
        curve = HyperellipticCurve(Polynomial(field, [1, 4, 0, 0, 0, 1]))

        self.assertIsNone(curve.l_polynomial_coefficients_with_sparsity_limit(0))


if __name__ == "__main__":
    unittest.main()
