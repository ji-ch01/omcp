"""Generalized OMOP concept-candidate search, shared by `Lookup_Concept`.

`Lookup_Drug`/`Lookup_Condition` (in `main.py`) each hardcode one domain and
one canonical standard vocabulary (RxNorm, SNOMED) -- appropriate there since
OMOP recommends exactly one standard vocabulary for those two domains. The
other three domains a phenotype/covariate can declare (procedure, measurement,
observation) do not have a single canonical vocabulary in OMOP the same way,
so this module keeps `vocabulary_id` optional rather than hardcoding one.

The SQL-building logic lives here, separate from the `@mcp_app.tool`-decorated
function in `main.py`, so it can be unit-tested without a live database
connection -- the same split `governance.py` already uses for
`authorize_causal_operation`.
"""

from __future__ import annotations


# CausalOMOP-side lowercase domain name -> OMOP `concept.domain_id` value.
# `condition`/`drug` are intentionally absent: those already have their own
# dedicated, long-standing tools (`Lookup_Condition`/`Lookup_Drug`) and are
# not expected to be routed through this generalized tool.
DOMAIN_IDS = {
    "procedure": "Procedure",
    "measurement": "Measurement",
    "observation": "Observation",
}


def _escape_term(term: str) -> str:
    """Quote-escape a search term before it is interpolated into a LIKE
    clause. Shared by every `build_lookup_*_sql` function in this module so
    the escaping rule lives in exactly one place."""
    return term.replace("'", "''")


def _name_or_synonym_match_sql(
    schema: str,
    escaped_term: str,
    where_extra: str,
    order_by: str,
    limit: int,
    offset: int,
) -> str:
    """Shared query shape for every name-search lookup tool: match either a
    concept's own name or any of its `concept_synonym` rows (abbreviations,
    prior/alternate names -- a plain `concept_name LIKE` alone misses these),
    deduplicated since a concept with several matching synonyms would
    otherwise come back more than once. `LIMIT`/`OFFSET` let a caller page
    through results instead of being stuck with only the first `limit`.
    """
    return f"""
        SELECT DISTINCT c.concept_id, c.concept_name, c.concept_code, c.vocabulary_id, c.domain_id
        FROM {schema}.concept c
        LEFT JOIN {schema}.concept_synonym cs ON cs.concept_id = c.concept_id
        WHERE (
            LOWER(c.concept_name) LIKE LOWER('%{escaped_term}%')
            OR LOWER(cs.concept_synonym_name) LIKE LOWER('%{escaped_term}%')
          )
          AND c.standard_concept = 'S'
          AND c.invalid_reason IS NULL
          {where_extra}
        ORDER BY {order_by}
        LIMIT {int(limit)} OFFSET {int(offset)}
        """


def build_lookup_concept_sql(
    schema: str,
    term: str,
    domain: str,
    vocabulary_id: str | None = None,
    limit: int = 10,
    offset: int = 0,
) -> str:
    """Build the same shape of query `Lookup_Drug`/`Lookup_Condition` already
    use (case-insensitive partial match on name or synonym, standard/valid
    concepts only, shortest-name-first), generalized to a caller-supplied
    domain.

    `domain`/`vocabulary_id` are validated against fixed allow-lists (never
    interpolated freely) since they end up directly in the SQL string;
    `term` is quote-escaped for the same reason. Results are ordered by
    `vocabulary_id` first (then name length) so that, when `vocabulary_id`
    is left unset and results genuinely mix vocabularies (e.g. LOINC and
    SNOMED both matching for a measurement), same-vocabulary concepts group
    together instead of interleaving -- easier for a reviewer to scan.
    """
    if domain not in DOMAIN_IDS:
        raise ValueError(f"domain must be one of {tuple(DOMAIN_IDS)}")
    if vocabulary_id is not None and not vocabulary_id.replace("_", "").isalnum():
        raise ValueError("vocabulary_id must be alphanumeric")
    domain_id = DOMAIN_IDS[domain]
    escaped_term = _escape_term(term)
    vocabulary_clause = f"AND c.vocabulary_id = '{vocabulary_id}'" if vocabulary_id else ""
    return _name_or_synonym_match_sql(
        schema, escaped_term,
        where_extra=f"AND c.domain_id = '{domain_id}' {vocabulary_clause}",
        order_by="c.vocabulary_id, LENGTH(c.concept_name), c.concept_name",
        limit=limit, offset=offset,
    )


def build_lookup_drug_sql(schema: str, term: str, limit: int = 10, offset: int = 0) -> str:
    """Build the same query `Lookup_Drug` already returns (RxNorm standard
    drug concepts, now also matched by synonym -- see
    `_name_or_synonym_match_sql`), with `term` quote-escaped -- it was
    previously interpolated unescaped into the LIKE clause, a
    SQL-injection-shaped bug fixed here the same way `build_lookup_concept_sql`
    already handles it."""
    escaped_term = _escape_term(term)
    return _name_or_synonym_match_sql(
        schema, escaped_term,
        where_extra="AND c.domain_id = 'Drug' AND c.vocabulary_id = 'RxNorm'",
        order_by="LENGTH(c.concept_name), c.concept_name",
        limit=limit, offset=offset,
    )


def build_lookup_condition_sql(schema: str, term: str, limit: int = 10, offset: int = 0) -> str:
    """Build the same query `Lookup_Condition` already returns (SNOMED
    standard condition concepts, now also matched by synonym), with `term`
    quote-escaped -- see `build_lookup_drug_sql`."""
    escaped_term = _escape_term(term)
    return _name_or_synonym_match_sql(
        schema, escaped_term,
        where_extra="AND c.domain_id = 'Condition' AND c.vocabulary_id = 'SNOMED'",
        order_by="LENGTH(c.concept_name), c.concept_name",
        limit=limit, offset=offset,
    )


def build_concepts_by_id_sql(schema: str, concept_ids: tuple[int, ...]) -> str:
    """Look up concept_id -> concept_name/vocabulary_id/domain_id for a
    caller-supplied set of concept_ids (not a name search like every other
    `build_*` function here).

    Exists so a reviewer-approved `concept_id` can be checked against the
    OMOP domain a covariate was declared to belong to (see
    `causal.orchestrator.workflow.map_covariate_concepts`) -- a reviewer can
    type a concept_id directly rather than picking one of the search
    results, so nothing otherwise confirms it actually belongs to the
    expected domain.

    `concept_ids` are cast to `int` before interpolation -- the same
    trusted-literal contract every other builder in this module uses; no
    string ever reaches this query.
    """
    if not concept_ids:
        raise ValueError("concept_ids must not be empty")
    ids = ", ".join(str(int(concept_id)) for concept_id in concept_ids)
    return f"""
        SELECT concept_id, concept_name, concept_code, vocabulary_id, domain_id
        FROM {schema}.concept
        WHERE concept_id IN ({ids})
        """
