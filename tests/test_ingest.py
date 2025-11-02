\
import pandas as pd, tempfile, os
from sqlalchemy import create_engine, text
from tsl.ingest.pipeline import ingest_path

def test_ingest_roundtrip(tmp_path):
    df = pd.DataFrame({
        "unique_id": ["A"]*5 + ["B"]*5,
        "ds": pd.date_range("2020-01-01", periods=10, freq="D"),
        "y": [float(i) for i in range(10)]
    })
    csvp = tmp_path/"data.csv"; df.to_csv(csvp, index=False)
    eng = create_engine("sqlite+pysqlite:///:memory:", future=True)
    res = ingest_path(path=str(csvp), engine=eng, dataset_name="toy", dry_run=False)
    assert res["status"] == "ok"
    with eng.begin() as cx:
        n = cx.execute(text("SELECT count(*) FROM tsl_observations")).scalar()
        assert n == 10
