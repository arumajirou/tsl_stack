import json, os, sys, pathlib, subprocess, pytest

@pytest.mark.skipif("torch" not in sys.modules and False, reason="import gate")
def test_gpu_smoke_one(tmp_path):
    try:
        import torch
    except Exception as e:
        pytest.skip(f"torch未インストール: {e}")
    if not torch.cuda.is_available():
        pytest.skip("CUDAが利用不可のためskip")

    repo = pathlib.Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{repo/'src'}:{env.get('PYTHONPATH','')}"
    env["CUDA_VISIBLE_DEVICES"] = "0"

    cmd = [sys.executable, "-m", "tsl.cli.tsl", "run-auto",
           "--data-csv", str(repo/"gpu_smoke.csv"),
           "--gpu-smoke", "--num-samples", "1"]
    p = subprocess.run(cmd, cwd=repo, text=True, capture_output=True, env=env)
    assert p.returncode == 0, p.stderr or p.stdout
    last = p.stdout.strip().splitlines()[-1]
    j = json.loads(last)
    assert j["device"] == "cuda"
    assert j.get("cuda_error") in (None, "null")
    pred = pathlib.Path(j["pred_csv"])
    assert pred.exists(), f"pred.csv missing: {pred}"
