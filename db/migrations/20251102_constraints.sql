\
-- 2025-11-02 hardening constraints
CREATE TABLE IF NOT EXISTS nf_runs(
  run_id TEXT PRIMARY KEY,
  started_at TIMESTAMPTZ,
  finished_at TIMESTAMPTZ,
  duration_sec DOUBLE PRECISION,
  status TEXT,
  model_name TEXT,
  backend TEXT,
  run_dir TEXT,
  model_dir TEXT,
  pred_rows INTEGER,
  ray_experiment_dir TEXT,
  extra JSONB
);
CREATE TABLE IF NOT EXISTS nf_artifacts(
  artifact_id BIGSERIAL PRIMARY KEY,
  run_id TEXT NOT NULL REFERENCES nf_runs(run_id) ON DELETE CASCADE,
  kind TEXT NOT NULL,
  rel_path TEXT NOT NULL,
  size_bytes BIGINT,
  extra JSONB,
  UNIQUE(run_id, kind, rel_path)
);
-- Constraints
ALTER TABLE nf_runs
  ADD CONSTRAINT nf_runs_time_chk CHECK (finished_at IS NULL OR started_at IS NULL OR finished_at >= started_at),
  ADD CONSTRAINT nf_runs_pred_rows_chk CHECK (pred_rows IS NULL OR pred_rows >= 0);
