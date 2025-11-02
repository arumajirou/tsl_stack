# ADR 0001: Database as Single Source of Truth (SoT)

**Date:** 2025-11-02

## Context
We ingest canonical time-series into a relational DB and treat it as the source of truth.
Artifacts (predictions, configs) are stored on FS and referenced by the DB.

## Decision
- Keep ingestion log-driven, DB as SoT.
- Optional MLflow/W&B integrations behind feature flags.

## Consequences
- Stronger reproducibility and observability.
- Requires migrations management and constraints.
