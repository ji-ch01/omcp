import unittest

from omcp.governance import (
    CausalAuthorizationError,
    CausalExecutionContext,
    authorize_causal_operation,
    sign_causal_context,
    verify_causal_context_signature,
)
from omcp.sql_validator import SQLValidator


def context_fields(**overrides):
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
    return values


def context(**overrides):
    return CausalExecutionContext.from_mapping(context_fields(**overrides))


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


class CausalContextSignatureTests(unittest.TestCase):
    def test_valid_signature_verifies(self):
        fields = context_fields()
        signed = {**fields, "signature": sign_causal_context(fields, "secret")}
        self.assertTrue(verify_causal_context_signature(signed, "secret"))

    def test_wrong_secret_is_rejected(self):
        fields = context_fields()
        signed = {**fields, "signature": sign_causal_context(fields, "secret")}
        self.assertFalse(verify_causal_context_signature(signed, "other-secret"))

    def test_missing_signature_is_rejected(self):
        self.assertFalse(verify_causal_context_signature(context_fields(), "secret"))

    def test_tampering_with_an_allowed_flag_invalidates_the_signature(self):
        """A client cannot self-upgrade the readiness snapshot it was given:
        flipping any signed field after signing must break verification,
        which is the whole point of signing over `authorize_causal_operation`
        alone (see the client -> server trust discussion in `governance.py`)."""
        fields = context_fields()
        signed = {**fields, "signature": sign_causal_context(fields, "secret")}
        signed["causal_estimation_allowed"] = True
        self.assertFalse(verify_causal_context_signature(signed, "secret"))


if __name__ == "__main__":
    unittest.main()
