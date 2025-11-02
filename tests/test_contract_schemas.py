\
import json, pandas as pd, tempfile
from pathlib import Path

# Minimal schema stubs
forecast_request_schema = {
  "type":"object",
  "required":["unique_id","ds","y"],
  "properties":{
    "unique_id":{"type":"string"},
    "ds":{"type":"string"},
    "y":{"type":"number"}
  },
  "additionalProperties": True
}

def test_contract_examples(tmp_path):
    # single row matches schema (toy check)
    row = {"unique_id":"A","ds":"2020-01-01","y":1.0}
    def _validate(r):
        assert isinstance(r["unique_id"], str)
        float(r["y"])
    _validate(row)
