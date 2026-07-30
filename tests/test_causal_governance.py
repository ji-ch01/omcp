import unittest

from omcp.governance import (
    CausalAuthorizationError,
    CausalExecutionContext,
    authorize_causal_operation,
)
from omcp.sql_validator import SQLValidator


def context(**overrides):
    values = {
        "run_id": "study-run_123",
        "study_id": "study_123",
        "mode": "exploratory",
        "clinical_mapping_status": "provisional",
        "analysis_plan_status": "pending",
        "aggregate_feasibility_allowed": True,
        "analytic_dataset_build_allowed": False,
        "causal_estimation_allowed": False,
    }
    values.update(overrides)
    return CausalExecutionContext.from_mapping(values)


class CausalGovernanceTests(unittest.TestCase):
    def test_provisional_context_allows_only_aggregate_feasibility(self):
        authorize_causal_operation("aggregate_feasibility", context())
        with self.assertRaises(CausalAuthorizationError):
            authorize_causal_operation("cohort_build", context())
        with self.assertRaises(CausalAuthorizationError):
            authorize_causal_operation("causal_estimation", context())

    def test_approved_readiness_allows_estimation(self):
        approved = context(
            clinical_mapping_status="approved",
            analysis_plan_status="approved",
            analytic_dataset_build_allowed=True,
            causal_estimation_allowed=True,
        )
        authorize_causal_operation("cohort_build", approved)
        authorize_causal_operation("diagnostics", approved)
        authorize_causal_operation("causal_estimation", approved)

    def test_context_requires_boolean_flags(self):
        with self.assertRaises(ValueError):
            context(causal_estimation_allowed="yes")

    def test_bigquery_qualified_omop_table_is_validated(self):
        query = (
            "SELECT COUNT(*) AS n FROM "
            "`project-with-hyphens.dataset.condition_occurrence`"
        )
        self.assertEqual(SQLValidator(from_dialect="bigquery").validate_sql(query), [])


if __name__ == "__main__":
    unittest.main()
