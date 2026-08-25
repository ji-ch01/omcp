"""Causal execution policy shared by OMCP clients.

FastOMOP can keep using the descriptive tools without a causal context.
CausalOMOP must use the governed causal tool and provide a readiness snapshot.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping


CAUSAL_OPERATIONS = (
    "aggregate_feasibility",
    "cohort_build",
    "diagnostics",
    "causal_estimation",
)
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{2,127}$")

# Fields the caller cannot self-assert without knowing OMCP_CONTEXT_SECRET.
# Signing (not just shape-validating) `causal_context` closes the gap where
# any client able to reach this tool -- e.g. over the optional SSE transport
# -- could otherwise hand-craft {"causal_estimation_allowed": true, ...} and
# pass `authorize_causal_operation` outright, since `from_mapping` below only
# checks types/enums, never whether the run actually earned that state.
SIGNED_CONTEXT_FIELDS = (
    "run_id",
    "study_id",
    "mode",
    "clinical_mapping_status",
    "analysis_plan_status",
    "aggregate_feasibility_allowed",
    "analytic_dataset_build_allowed",
    "causal_estimation_allowed",
)


def _canonical_context_bytes(context: Mapping[str, Any]) -> bytes:
    ordered = {name: context[name] for name in SIGNED_CONTEXT_FIELDS}
    return json.dumps(ordered, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sign_causal_context(context: Mapping[str, Any], secret: str) -> str:
    """HMAC-SHA256 over the canonical readiness fields, keyed by a secret
    shared out-of-band between whoever computed the readiness snapshot and
    this server -- never derived from the context itself."""
    return hmac.new(
        secret.encode("utf-8"), _canonical_context_bytes(context), hashlib.sha256
    ).hexdigest()


def verify_causal_context_signature(context: Mapping[str, Any], secret: str) -> bool:
    signature = context.get("signature")
    if not isinstance(signature, str):
        return False
    try:
        expected = sign_causal_context(context, secret)
    except (KeyError, TypeError):
        return False
    return hmac.compare_digest(signature, expected)


class CausalAuthorizationError(PermissionError):
    """Raised when an MCP caller requests an unauthorized causal operation."""


@dataclass(frozen=True)
class CausalExecutionContext:
    run_id: str
    study_id: str
    mode: str
    clinical_mapping_status: str
    analysis_plan_status: str
    aggregate_feasibility_allowed: bool
    analytic_dataset_build_allowed: bool
    causal_estimation_allowed: bool

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CausalExecutionContext":
        if not isinstance(value, Mapping):
            raise ValueError("causal_context must be an object")
        required = {
            "run_id",
            "study_id",
            "mode",
            "clinical_mapping_status",
            "analysis_plan_status",
            "aggregate_feasibility_allowed",
            "analytic_dataset_build_allowed",
            "causal_estimation_allowed",
        }
        missing = sorted(required - set(value))
        if missing:
            raise ValueError(f"Missing causal context fields: {', '.join(missing)}")
        context = cls(**{name: value[name] for name in required})
        context.validate()
        return context

    def validate(self) -> None:
        if not _ID.fullmatch(self.run_id) or not _ID.fullmatch(self.study_id):
            raise ValueError("Invalid run_id or study_id")
        if self.mode not in {"exploratory", "final"}:
            raise ValueError("mode must be exploratory or final")
        statuses = {"pending", "approved", "rejected", "provisional"}
        if self.clinical_mapping_status not in statuses:
            raise ValueError("Invalid clinical_mapping_status")
        if self.analysis_plan_status not in statuses:
            raise ValueError("Invalid analysis_plan_status")
        for name in (
            "aggregate_feasibility_allowed",
            "analytic_dataset_build_allowed",
            "causal_estimation_allowed",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")


def authorize_causal_operation(
    operation: str, context: CausalExecutionContext
) -> None:
    """Fail closed when the readiness snapshot does not permit an operation."""
    if operation not in CAUSAL_OPERATIONS:
        raise ValueError(f"Unsupported causal operation: {operation}")

    if operation == "aggregate_feasibility":
        if not context.aggregate_feasibility_allowed:
            raise CausalAuthorizationError("Aggregate feasibility is not authorized")
        if context.mode == "final" and context.clinical_mapping_status != "approved":
            raise CausalAuthorizationError("Final-mode mapping must be approved")
        return

    if context.analysis_plan_status != "approved":
        raise CausalAuthorizationError("The analysis-plan gate must be approved")

    if operation in {"cohort_build", "diagnostics"}:
        if not context.analytic_dataset_build_allowed:
            raise CausalAuthorizationError("Analytic dataset construction is not authorized")
        return

    if context.clinical_mapping_status != "approved":
        raise CausalAuthorizationError("Clinical mapping must be approved")
    if not context.causal_estimation_allowed:
        raise CausalAuthorizationError("Causal estimation is not authorized")
