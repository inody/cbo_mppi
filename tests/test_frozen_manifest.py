import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/frozen"


def test_frozen_manifest_matches_files():
    manifest = json.loads((DATA / "manifest.json").read_text())
    assert manifest["schema_version"] == 1
    listed = {record["path"] for record in manifest["files"]}
    actual = {
        path.relative_to(DATA).as_posix()
        for path in DATA.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    assert listed == actual
    for record in manifest["files"]:
        path = DATA / record["path"]
        assert path.is_file()
        assert path.stat().st_size == record["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
