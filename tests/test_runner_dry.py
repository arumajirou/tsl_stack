\
import os, pandas as pd, tempfile, pytest
from tsl.runner.auto_runner import run_neuralforecast_auto, NotSupportedError

def test_runner_dry():
    df = pd.DataFrame({
        "unique_id":["A"]*20,
        "ds": pd.date_range("2020-01-01", periods=20, freq="D"),
        "y": list(range(20))
    })
    import tempfile
    tmp = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False)
    df.to_csv(tmp.name, index=False); tmp.close()
    res = run_neuralforecast_auto({"data_csv": tmp.name, "dry_run": True})
    assert res["status"] == "dry-run"

@pytest.mark.skipif(True, reason="NeuralForecast not installed in test env")
def test_runner_real(tmp_path):
    # This test is off by default to keep CI light.
    pass
