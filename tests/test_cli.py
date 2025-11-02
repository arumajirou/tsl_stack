\
from tsl.cli.tsl import main

def test_cli_help(capsys):
    try:
        main(["--help"])
    except SystemExit:
        pass  # argparse exits after help
    out = capsys.readouterr().out
    assert "diagnose" in out and "ingest" in out and "run-auto" in out
