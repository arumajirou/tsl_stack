\
-- Superset: simple metrics over nf_runs
SELECT
  date_trunc('day', finished_at) AS day,
  count(*) AS runs,
  avg(duration_sec) AS avg_dur
FROM nf_runs
GROUP BY 1
ORDER BY 1 DESC
LIMIT 30;
