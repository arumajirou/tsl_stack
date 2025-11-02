-- /mnt/e/env/ts/zip/tsl_stack/db/migrations/20251102_nf_errors.sql
CREATE TABLE IF NOT EXISTS nf_errors (
  id           INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at   TEXT NOT NULL,
  run_id       TEXT,
  model_name   TEXT,
  data_csv     TEXT,
  device       TEXT,
  h            INTEGER,
  backend      TEXT,
  error_type   TEXT,
  message      TEXT,
  traceback    TEXT,
  extra_json   TEXT
);

CREATE INDEX IF NOT EXISTS idx_nf_errors_created_at ON nf_errors(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_nf_errors_runid ON nf_errors(run_id);
