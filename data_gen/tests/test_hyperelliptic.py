import sqlite3
import subprocess
import sys
import tempfile
import unittest

from data_gen.hyperelliptic import (
    EnumerationContext,
    FiniteExtension,
    HyperellipticCurve,
    PointCountingContext,
    Polynomial,
    PrimeField,
    _affine_polynomial_to_binary_form,
    _apply_binary_form_action_matrix,
    _transform_binary_form,
)


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

    def test_point_contribution_tables(self):
        field = PrimeField(5)
        context = PointCountingContext(field, polynomial_degree=3)
        self.assertEqual(context.ground_point_contributions(), (1, 2, 0, 0, 2))

        extension = FiniteExtension(5, 2)
        contributions = extension.point_contributions()
        self.assertEqual(contributions[extension.zero()], 1)
        self.assertEqual(sum(contributions.values()), extension.size)

    def test_ground_value_table(self):
        field = PrimeField(5)
        context = PointCountingContext(field, polynomial_degree=3)
        value_table = context.ground_value_table()

        self.assertEqual(value_table[0][3], (3, 3, 3, 3, 3))
        self.assertEqual(value_table[2][2], (0, 2, 3, 3, 2))

        curve = HyperellipticCurve(Polynomial(field, [1, 1, 0, 1]), point_counting_context=context)
        self.assertEqual(curve.point_count_over_ground_field(), 9)
        self.assertIs(context.ground_value_table(), value_table)

    def test_genus_one_l_polynomial_coefficient(self):
        field = PrimeField(5)
        curve = HyperellipticCurve(Polynomial(field, [1, 1, 0, 1]))

        self.assertEqual(curve.point_count_over_ground_field(), 9)
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

    def test_hasse_witt_matrix_and_l_polynomial_mod_p(self):
        field = PrimeField(5)
        elliptic_curve = HyperellipticCurve(Polynomial(field, [1, 1, 0, 1]))
        self.assertEqual(elliptic_curve.hasse_witt_matrix(), ((2,),))
        self.assertEqual(elliptic_curve.l_polynomial_coefficients_mod_p(), [3])

        genus_two_curve = HyperellipticCurve(Polynomial(field, [1, 1, 0, 0, 0, 1]))
        self.assertEqual(genus_two_curve.hasse_witt_matrix(), ((0, 0), (0, 0)))
        self.assertEqual(genus_two_curve.l_polynomial_coefficients_mod_p(), [0, 0])

    def test_hasse_witt_sparsity_filter(self):
        field = PrimeField(5)
        context = PointCountingContext(field, polynomial_degree=5)
        curve = HyperellipticCurve(Polynomial(field, [1, 0, 1, 0, 0, 1]), point_counting_context=context)

        self.assertEqual(curve.l_polynomial_coefficients(), [-1, 0])
        self.assertEqual(curve.l_polynomial_coefficients_mod_p(), [4, 0])
        self.assertFalse(curve.passes_hasse_witt_sparsity_filter(0))
        self.assertTrue(curve.passes_hasse_witt_sparsity_filter(1))

        context = PointCountingContext(field, polynomial_degree=5)
        curve = HyperellipticCurve(Polynomial(field, [1, 0, 1, 0, 0, 1]), point_counting_context=context)
        self.assertIsNone(curve.l_polynomial_coefficients_with_sparsity_limit(0))
        self.assertEqual(context._extension_cache, {})

    def test_sparsity_limit_stops_when_exceeded(self):
        field = PrimeField(5)
        curve = HyperellipticCurve(Polynomial(field, [1, 4, 0, 0, 0, 1]))

        self.assertIsNone(curve.l_polynomial_coefficients_with_sparsity_limit(0))

    def test_point_counting_context_can_be_shared(self):
        field = PrimeField(5)
        context = PointCountingContext(field, polynomial_degree=5)
        sparse_curve = HyperellipticCurve(Polynomial(field, [1, 1, 0, 0, 0, 1]), point_counting_context=context)
        nonsparse_curve = HyperellipticCurve(Polynomial(field, [1, 4, 0, 0, 0, 1]), point_counting_context=context)

        self.assertEqual(sparse_curve.l_polynomial_coefficients(), [0, 10])
        self.assertIsNone(nonsparse_curve.l_polynomial_coefficients_with_sparsity_limit(0))
        self.assertIn(2, context._extension_cache)
        self.assertIn(2, context._power_cache)

    def test_point_counting_context_rejects_incompatible_curves(self):
        context = PointCountingContext(PrimeField(5), polynomial_degree=3)

        with self.assertRaises(ValueError):
            HyperellipticCurve(Polynomial(PrimeField(7), [1, 1, 0, 1]), point_counting_context=context)
        with self.assertRaises(ValueError):
            HyperellipticCurve(Polynomial(PrimeField(5), [1, 1, 0, 0, 0, 1]), point_counting_context=context)

    def test_enumeration_context_canonicalizes_isomorphic_polynomials(self):
        context = EnumerationContext(prime=5, genus=2)
        f = context.polynomial([1, 1, 0, 0, 0, 1])
        translated_f = context.polynomial([3, 1, 0, 0, 0, 1])
        square_scaled_f = context.polynomial([4, 4, 0, 0, 0, 4])

        self.assertEqual(context.canonical_key(f), context.canonical_key(translated_f))
        self.assertEqual(context.canonical_key(f), context.canonical_key(square_scaled_f))
        self.assertTrue(context.is_new_isomorphism_class(f))
        self.assertFalse(context.is_new_isomorphism_class(translated_f))
        self.assertEqual(len(context.pgl2_action_matrices), len(context.pgl2))

    def test_cached_pgl2_action_matrices_match_direct_transforms(self):
        context = EnumerationContext(prime=5, genus=2)
        binary_form = _affine_polynomial_to_binary_form(context.polynomial([1, 1, 0, 0, 0, 1]), context.binary_degree)

        for matrix, action_matrix in zip(context.pgl2, context.pgl2_action_matrices):
            self.assertEqual(
                _apply_binary_form_action_matrix(binary_form, action_matrix, context.field.prime),
                _transform_binary_form(binary_form, matrix, context.field.prime),
            )

    def test_enumeration_context_reuses_curve_level_caches(self):
        context = EnumerationContext(prime=5, genus=2)
        f = context.polynomial([1, 1, 0, 0, 0, 1])
        translated_f = context.polynomial([3, 1, 0, 0, 0, 1])

        self.assertEqual(context.l_polynomial_coefficients_mod_p(f), [0, 0])
        self.assertEqual(context.l_polynomial_coefficients_mod_p(translated_f), [0, 0])
        self.assertEqual(len(context.l_polynomial_mod_p_cache), 1)

        self.assertEqual(context.l_polynomial_coefficients(f), [0, 10])
        self.assertEqual(context.l_polynomial_coefficients(translated_f), [0, 10])
        self.assertEqual(len(context.exact_l_polynomial_cache), 1)

    def test_enumeration_context_rejects_incompatible_polynomials(self):
        context = EnumerationContext(prime=5, genus=2)
        with self.assertRaises(ValueError):
            context.canonical_key(Polynomial(PrimeField(7), [1, 1, 0, 0, 0, 1]))
        with self.assertRaises(ValueError):
            context.canonical_key([1, 1, 0, 1])

    def test_exhaustive_search_uses_sqlite_orbit_lookup(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = f"{tmpdir}/curves.sqlite"
            context = EnumerationContext(prime=5, genus=2, sqlite_path=db_path)

            result = context.process_polynomial_for_output([1, 1, 0, 0, 0, 1], max_sparsity=0)
            self.assertEqual(result["status"], "sparse")
            self.assertEqual(result["lpoly"], [0, 10])

            translated_result = context.process_polynomial_for_output([3, 1, 0, 0, 0, 1], max_sparsity=0)
            self.assertEqual(translated_result["status"], "duplicate")
            self.assertEqual(translated_result["previous_status"], "sparse")
            self.assertEqual(translated_result["lpoly"], [0, 10])
            self.assertEqual(translated_result["canonical_key"], result["canonical_key"])

            connection = sqlite3.connect(db_path)
            self.assertGreater(connection.execute("SELECT COUNT(*) FROM orbit_cache").fetchone()[0], 0)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM curve_cache").fetchone()[0], 1)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM sparse_curves").fetchone()[0], 1)
            curve_cache_columns = {
                row[1]
                for row in connection.execute("PRAGMA table_info(curve_cache)").fetchall()
            }
            self.assertNotIn("ground_point_count", curve_cache_columns)
            self.assertNotIn("factorization_pattern", curve_cache_columns)
            self.assertNotIn("max_sparsity", curve_cache_columns)
            self.assertEqual(connection.execute("SELECT typeof(coefficients) FROM curve_cache").fetchone()[0], "text")
            self.assertEqual(connection.execute("SELECT typeof(canonical_key) FROM curve_cache").fetchone()[0], "blob")
            self.assertEqual(connection.execute("SELECT typeof(lpoly_mod_p) FROM curve_cache").fetchone()[0], "blob")
            self.assertEqual(connection.execute("SELECT typeof(exact_lpoly) FROM curve_cache").fetchone()[0], "blob")
            self.assertEqual(connection.execute("SELECT typeof(coefficients) FROM sparse_curves").fetchone()[0], "text")
            self.assertEqual(connection.execute("SELECT typeof(lpoly) FROM sparse_curves").fetchone()[0], "text")
            self.assertEqual(connection.execute("SELECT typeof(canonical_key) FROM sparse_curves").fetchone()[0], "blob")
            self.assertEqual(len(context.canonical_records), 1)
            record = context.canonical_records[result["canonical_key"]]
            self.assertIn(result["canonical_key"], context.index_by_rational_branch_count[record.rational_branch_count])
            connection.close()
            context.close_sqlite()

    def test_sqlite_enumeration_output_records_hasse_witt_rejection(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = f"{tmpdir}/curves.sqlite"
            context = EnumerationContext(prime=5, genus=2, sqlite_path=db_path)

            result = context.process_polynomial_for_output([1, 0, 1, 0, 0, 1], max_sparsity=0)
            self.assertEqual(result["status"], "rejected_hasse_witt")
            stats = context.process_polynomials_for_output(([1, 0, 1, 0, 0, 1],), max_sparsity=0)
            self.assertEqual(stats, {"processed": 1, "duplicate": 1})

            connection = sqlite3.connect(db_path)
            self.assertEqual(connection.execute("SELECT status FROM curve_cache").fetchone()[0], "rejected_hasse_witt")
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM sparse_curves").fetchone()[0], 0)
            connection.close()
            context.close_sqlite()

    def test_hasse_witt_prefilter_skips_canonicalization_for_mod_p_rejections(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = f"{tmpdir}/curves.sqlite"
            context = EnumerationContext(
                prime=5,
                genus=2,
                sqlite_path=db_path,
                hasse_witt_prefilter=True,
            )

            result = context.process_polynomial_for_output([1, 0, 1, 0, 0, 1], max_sparsity=0)
            self.assertEqual(result["status"], "rejected_hasse_witt_uncanonicalized")
            self.assertEqual(result["lpoly_mod_p"], [4, 0])
            self.assertEqual(len(context.canonical_records), 0)

            connection = sqlite3.connect(db_path)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM curve_cache").fetchone()[0], 0)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM sparse_curves").fetchone()[0], 0)
            connection.close()
            context.close_sqlite()

    def test_hasse_witt_prefilter_uses_sqlite_orbit_lookup_for_survivors(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = f"{tmpdir}/curves.sqlite"
            context = EnumerationContext(
                prime=5,
                genus=2,
                sqlite_path=db_path,
                hasse_witt_prefilter=True,
            )

            result = context.process_polynomial_for_output([1, 1, 0, 0, 0, 1], max_sparsity=0)
            self.assertEqual(result["status"], "sparse")

            translated_result = context.process_polynomial_for_output([3, 1, 0, 0, 0, 1], max_sparsity=0)
            self.assertEqual(translated_result["status"], "duplicate")
            self.assertEqual(translated_result["previous_status"], "sparse")
            self.assertEqual(translated_result["canonical_key"], result["canonical_key"])

            connection = sqlite3.connect(db_path)
            self.assertGreater(connection.execute("SELECT COUNT(*) FROM orbit_cache").fetchone()[0], 0)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM curve_cache").fetchone()[0], 1)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM sparse_curves").fetchone()[0], 1)
            connection.close()
            context.close_sqlite()

    def test_sqlite_resume_loads_canonical_records_into_memory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = f"{tmpdir}/p5_g2_s_0.sqlite"
            context = EnumerationContext(prime=5, genus=2, sqlite_path=db_path)
            result = context.process_polynomial_for_output([1, 1, 0, 0, 0, 1], max_sparsity=0)
            context.close_sqlite()

            resumed_context = EnumerationContext(prime=5, genus=2, sqlite_path=db_path)
            self.assertEqual(len(resumed_context.canonical_records), 1)
            self.assertIn(result["canonical_key"], resumed_context.canonical_records)
            self.assertIn(result["canonical_key"], resumed_context.seen_keys)

            translated_result = resumed_context.process_polynomial_for_output([3, 1, 0, 0, 0, 1], max_sparsity=0)
            self.assertEqual(translated_result["status"], "duplicate")
            self.assertEqual(translated_result["previous_status"], "sparse")
            self.assertEqual(translated_result["lpoly"], [0, 10])

            connection = sqlite3.connect(db_path)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM curve_cache").fetchone()[0], 1)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM sparse_curves").fetchone()[0], 1)
            connection.close()
            resumed_context.close_sqlite()

    def test_enumeration_context_records_timing_summary(self):
        context = EnumerationContext(prime=5, genus=2)
        context.process_polynomial_for_output([1, 1, 0, 0, 0, 1], max_sparsity=0)
        context.process_polynomial_for_output([3, 1, 0, 0, 0, 1], max_sparsity=0)

        timing = context.timing_summary()
        self.assertEqual(timing["processed_polynomials"], 2)
        self.assertEqual(timing["status_counts"], {"sparse": 1, "duplicate": 1})
        self.assertGreaterEqual(timing["elapsed_seconds"], timing["processing_seconds"])
        self.assertGreater(timing["processing_seconds"], 0.0)
        self.assertEqual(timing["sqlite_load_seconds"], 0.0)
        self.assertEqual(timing["sqlite_write_seconds"], 0.0)

    def test_enumeration_cli_prints_progress(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = f"{tmpdir}/curves.sqlite"
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "data_gen.hyperelliptic",
                    "--p",
                    "5",
                    "--genus",
                    "2",
                    "--max-sparsity",
                    "2",
                    "--limit",
                    "3",
                    "--progress-interval",
                    "1",
                    "--hasse-witt-prefilter",
                    "--out",
                    db_path,
                ],
                check=True,
                capture_output=True,
                text=True,
            )

        self.assertIn("progress: 0/3 sparse_presentations=0", completed.stdout)
        self.assertIn("progress: 3/3", completed.stdout)
        self.assertIn("sparse_isomorphism_classes=", completed.stdout)
        self.assertIn("canonicalized_isomorphism_classes=", completed.stdout)
        self.assertNotIn("total_isomorphism_classes=", completed.stdout)


if __name__ == "__main__":
    unittest.main()
