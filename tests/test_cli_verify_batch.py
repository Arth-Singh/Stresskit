import json

from stresskit.cli import main


def _write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_verify_reports_unsupported_schema_and_still_verifies_the_rest(
    tmp_path, capsys
):
    """One card the installed version cannot parse must not abort the batch.

    A newer StressKit writes a higher schema_version; an auditor running an
    older one over a directory still needs a verdict for every other card.
    """
    good = json.loads(
        (
            __import__("pathlib").Path(__file__).resolve().parents[1]
            / "references" / "cards" / "ioi_gpt2_small.json"
        ).read_text()
    )
    _write(tmp_path / "good.json", good)
    future = dict(good, schema_version="99.0")
    _write(tmp_path / "future.json", future)

    code = main(["verify", str(tmp_path)])
    out = capsys.readouterr().out

    assert code == 1
    assert "OK: " in out and "good.json" in out
    assert "future.json" in out and "cannot be verified" in out
    assert "1 verified, 1 failed" in out
