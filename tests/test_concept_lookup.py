import unittest

from omcp.concept_lookup import (
    build_concepts_by_id_sql,
    build_lookup_concept_sql,
    build_lookup_condition_sql,
    build_lookup_drug_sql,
)


class BuildLookupConceptSqlTests(unittest.TestCase):
    def test_builds_a_domain_filtered_query(self):
        sql = build_lookup_concept_sql("base", "sofa score", "measurement")
        self.assertIn("FROM base.concept c", sql)
        self.assertIn("c.domain_id = 'Measurement'", sql)
        self.assertIn("LOWER(c.concept_name) LIKE LOWER('%sofa score%')", sql)
        self.assertIn("c.standard_concept = 'S'", sql)
        self.assertIn("c.invalid_reason IS NULL", sql)
        self.assertIn("LIMIT 10 OFFSET 0", sql)

    def test_matches_by_synonym_as_well_as_name(self):
        """Regression for gap #5: a plain concept_name LIKE alone misses
        abbreviations/alternate names recorded only in concept_synonym."""
        sql = build_lookup_concept_sql("base", "sofa score", "measurement")
        self.assertIn("LEFT JOIN base.concept_synonym cs ON cs.concept_id = c.concept_id", sql)
        self.assertIn("LOWER(cs.concept_synonym_name) LIKE LOWER('%sofa score%')", sql)
        self.assertIn("SELECT DISTINCT", sql)

    def test_orders_by_vocabulary_first_so_mixed_results_group_together(self):
        sql = build_lookup_concept_sql("base", "creatinine", "measurement")
        self.assertIn("ORDER BY c.vocabulary_id, LENGTH(c.concept_name), c.concept_name", sql)

    def test_vocabulary_filter_is_applied_only_when_given(self):
        without = build_lookup_concept_sql("base", "creatinine", "measurement")
        self.assertNotIn("c.vocabulary_id =", without)
        with_vocab = build_lookup_concept_sql("base", "creatinine", "measurement", vocabulary_id="LOINC")
        self.assertIn("c.vocabulary_id = 'LOINC'", with_vocab)

    def test_rejects_an_unknown_domain(self):
        with self.assertRaises(ValueError):
            build_lookup_concept_sql("base", "term", "drug")
        with self.assertRaises(ValueError):
            build_lookup_concept_sql("base", "term", "not_a_domain")

    def test_rejects_a_non_alphanumeric_vocabulary_id(self):
        with self.assertRaises(ValueError):
            build_lookup_concept_sql("base", "term", "measurement", vocabulary_id="LOINC'; DROP TABLE x; --")

    def test_escapes_single_quotes_in_term(self):
        sql = build_lookup_concept_sql("base", "patient's weight", "observation")
        self.assertIn("patient''s weight", sql)
        self.assertNotIn("patient's weight", sql)

    def test_limit_is_applied(self):
        sql = build_lookup_concept_sql("base", "term", "procedure", limit=5)
        self.assertIn("LIMIT 5 OFFSET 0", sql)

    def test_offset_pages_past_the_first_results(self):
        """Regression for gap #5: results were capped with no way to see
        past the first page."""
        sql = build_lookup_concept_sql("base", "term", "procedure", limit=10, offset=10)
        self.assertIn("LIMIT 10 OFFSET 10", sql)


class BuildLookupDrugSqlTests(unittest.TestCase):
    def test_builds_the_same_query_shape_as_before(self):
        sql = build_lookup_drug_sql("base", "albumin")
        self.assertIn("FROM base.concept c", sql)
        self.assertIn("c.domain_id = 'Drug'", sql)
        self.assertIn("c.vocabulary_id = 'RxNorm'", sql)
        self.assertIn("LOWER(c.concept_name) LIKE LOWER('%albumin%')", sql)
        self.assertIn("c.standard_concept = 'S'", sql)
        self.assertIn("c.invalid_reason IS NULL", sql)
        self.assertIn("LIMIT 10 OFFSET 0", sql)

    def test_matches_by_synonym_as_well_as_name(self):
        sql = build_lookup_drug_sql("base", "albumin")
        self.assertIn("LEFT JOIN base.concept_synonym cs ON cs.concept_id = c.concept_id", sql)
        self.assertIn("LOWER(cs.concept_synonym_name) LIKE LOWER('%albumin%')", sql)

    def test_escapes_single_quotes_in_term(self):
        """Regression: `term` used to be interpolated unescaped -- a
        SQL-injection-shaped bug. A single quote in a legitimate search term
        (or a deliberately malicious one) must not break out of the LIKE
        string literal."""
        sql = build_lookup_drug_sql("base", "patient's albumin")
        self.assertIn("patient''s albumin", sql)
        self.assertNotIn("patient's albumin", sql)

    def test_limit_is_int_cast(self):
        sql = build_lookup_drug_sql("base", "albumin", limit=5)
        self.assertIn("LIMIT 5 OFFSET 0", sql)

    def test_offset_pages_past_the_first_results(self):
        sql = build_lookup_drug_sql("base", "albumin", limit=10, offset=20)
        self.assertIn("LIMIT 10 OFFSET 20", sql)


class BuildLookupConditionSqlTests(unittest.TestCase):
    def test_builds_the_same_query_shape_as_before(self):
        sql = build_lookup_condition_sql("base", "sepsis")
        self.assertIn("FROM base.concept c", sql)
        self.assertIn("c.domain_id = 'Condition'", sql)
        self.assertIn("c.vocabulary_id = 'SNOMED'", sql)
        self.assertIn("LOWER(c.concept_name) LIKE LOWER('%sepsis%')", sql)
        self.assertIn("c.standard_concept = 'S'", sql)
        self.assertIn("c.invalid_reason IS NULL", sql)
        self.assertIn("LIMIT 10 OFFSET 0", sql)

    def test_matches_by_synonym_as_well_as_name(self):
        sql = build_lookup_condition_sql("base", "sepsis")
        self.assertIn("LEFT JOIN base.concept_synonym cs ON cs.concept_id = c.concept_id", sql)
        self.assertIn("LOWER(cs.concept_synonym_name) LIKE LOWER('%sepsis%')", sql)

    def test_escapes_single_quotes_in_term(self):
        sql = build_lookup_condition_sql("base", "Sjogren's syndrome")
        self.assertIn("Sjogren''s syndrome", sql)
        self.assertNotIn("Sjogren's syndrome", sql)

    def test_limit_is_int_cast(self):
        sql = build_lookup_condition_sql("base", "sepsis", limit=5)
        self.assertIn("LIMIT 5 OFFSET 0", sql)

    def test_offset_pages_past_the_first_results(self):
        sql = build_lookup_condition_sql("base", "sepsis", limit=10, offset=10)
        self.assertIn("LIMIT 10 OFFSET 10", sql)


class BuildConceptsByIdSqlTests(unittest.TestCase):
    def test_builds_an_id_filtered_query(self):
        sql = build_concepts_by_id_sql("base", (4301868, 3004249))
        self.assertIn("FROM base.concept", sql)
        self.assertIn("WHERE concept_id IN (4301868, 3004249)", sql)
        self.assertIn("SELECT concept_id, concept_name, concept_code, vocabulary_id, domain_id", sql)

    def test_concept_ids_are_int_cast(self):
        """No string ever reaches this query -- ids are trusted literals
        only after an explicit int() cast, matching every other builder
        here."""
        sql = build_concepts_by_id_sql("base", (4301868,))
        self.assertNotIn("'", sql)

    def test_rejects_empty_concept_ids(self):
        with self.assertRaises(ValueError):
            build_concepts_by_id_sql("base", ())


if __name__ == "__main__":
    unittest.main()
